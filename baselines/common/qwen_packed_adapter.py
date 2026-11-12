"""Qwen2.5-Omni/Qwen3 adapters for native KIVI and OTT packed caches.

The multimodal encoders and the Qwen text model remain the upstream
Transformers implementation.  Only text self-attention is wrapped:

* prompt prefill calls upstream FlashAttention-2 on BF16 Q/K/V;
* the resulting K/V tensors are packed with the baseline's native Triton
  quantizer;
* single-token decode either reads the packed cache through the baseline's CUDA
  GEMV or explicitly dequantizes it for an SDPA/FA2 correctness backend.

Both backends use the baseline's native quantizer and persistent packed cache.
The reference backend deliberately materializes K/V for PyTorch SDPA and is
not an efficient packed-decode result. KIVI defaults to the historical
Qwen-compatible cache state machine used by our reported 3B experiments; its
released state machine remains available explicitly.

This file implements only Qwen projection, multimodal RoPE, GQA/cache-layout
translation, and cache bookkeeping. It intentionally imports KIVI/OTT's
released quantizers and CUDA operators instead of altering their algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib
import math
import sys
from types import MethodType
from typing import Any, Callable

import torch
from torch import nn
from torch.nn import functional as F
from r2omni.sspa.integration.qwen_sspa_runtime import allocated_tensor_storage_bytes
from transformers.cache_utils import DynamicCache
try:
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
        ALL_ATTENTION_FUNCTIONS as QWEN25_ATTENTION_FUNCTIONS,
        apply_multimodal_rotary_pos_emb,
    )
except ImportError:  # MiniCPM's pinned Transformers build only needs Qwen3.
    QWEN25_ATTENTION_FUNCTIONS = None
    apply_multimodal_rotary_pos_emb = None
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS as QWEN3_ATTENTION_FUNCTIONS,
    apply_rotary_pos_emb,
    repeat_kv,
)


@dataclass
class PackedLayerState:
    key_quant: torch.Tensor | None = None
    key_full: torch.Tensor | None = None
    key_scale: torch.Tensor | None = None
    key_min: torch.Tensor | None = None
    value_quant: torch.Tensor | None = None
    value_full: torch.Tensor | None = None
    value_scale: torch.Tensor | None = None
    value_min: torch.Tensor | None = None
    sink_ids: torch.Tensor | None = None
    sink_keys: torch.Tensor | None = None
    sink_values: torch.Tensor | None = None
    sink_scores: torch.Tensor | None = None
    seq_len: int = 0


class NativePackedCache(DynamicCache):
    """Cache API shell whose storage is owned by native packed layer states."""

    def __init__(self, adapter: "NativeKVAdapter") -> None:
        self.packed_layers: dict[int, PackedLayerState] = {}
        self._key_cache_shell: list[torch.Tensor] = []
        self._value_cache_shell: list[torch.Tensor] = []
        super().__init__()
        self.adapter = adapter

    def update(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError("native packed attention owns cache updates")

    def get_seq_length(self, layer_idx: int = 0) -> int:
        state = self.packed_layers.get(layer_idx)
        return 0 if state is None else int(state.seq_len)

    def get_mask_sizes(
        self, cache_position: torch.Tensor, layer_idx: int
    ) -> tuple[int, int]:
        return self.get_seq_length(layer_idx) + int(cache_position.shape[0]), 0

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        return -1

    @property
    def is_compileable(self) -> bool:
        return False

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        if int(beam_idx.numel()) != 1:
            raise NotImplementedError("native packed adapters support greedy batch=1 only")

    @property
    def key_cache(self) -> list[torch.Tensor]:
        packed = [
            state.key_full
            for _, state in sorted(self.packed_layers.items())
            if torch.is_tensor(state.key_full)
        ]
        return packed if packed else self._key_cache_shell

    @key_cache.setter
    def key_cache(self, value: list[torch.Tensor]) -> None:
        self._key_cache_shell = value

    @property
    def value_cache(self) -> list[torch.Tensor]:
        packed = [
            state.value_full
            for _, state in sorted(self.packed_layers.items())
            if torch.is_tensor(state.value_full)
        ]
        return packed if packed else self._value_cache_shell

    @value_cache.setter
    def value_cache(self, value: list[torch.Tensor]) -> None:
        self._value_cache_shell = value

    @property
    def key_cache_pack(self) -> list[torch.Tensor]:
        return [
            state.key_quant
            for _, state in sorted(self.packed_layers.items())
            if torch.is_tensor(state.key_quant)
        ]

    @property
    def value_cache_pack(self) -> list[torch.Tensor]:
        return [
            state.value_quant
            for _, state in sorted(self.packed_layers.items())
            if torch.is_tensor(state.value_quant)
        ]

    @property
    def key_cache_params(self) -> list[torch.Tensor]:
        result: list[torch.Tensor] = []
        for _, state in sorted(self.packed_layers.items()):
            result.extend(
                tensor
                for tensor in (
                    state.key_scale,
                    state.key_min,
                    state.sink_ids,
                    state.sink_keys,
                    state.sink_scores,
                )
                if torch.is_tensor(tensor)
            )
        return result

    @property
    def value_cache_params(self) -> list[torch.Tensor]:
        result: list[torch.Tensor] = []
        for _, state in sorted(self.packed_layers.items()):
            result.extend(
                tensor
                for tensor in (state.value_scale, state.value_min, state.sink_values)
                if torch.is_tensor(tensor)
            )
        return result


class NativeKVAdapter:
    def __init__(
        self,
        *,
        method: str,
        baseline_root: str | Path,
        native_extension_root: str | Path | None = None,
        kivi_implementation: str = "legacy",
        k_bits: int,
        v_bits: int,
        group_size: int,
        residual_length: int,
        fp_prefix_layers: int = 0,
        readout_backend: str = "native_packed",
        sink_num: int = 5,
        sink_max_size: int = 32,
    ) -> None:
        if method not in {"kivi", "ott"}:
            raise ValueError(f"unsupported native KV method: {method}")
        if kivi_implementation not in {"legacy", "official"}:
            raise ValueError(
                "kivi_implementation must be either 'legacy' or 'official'"
            )
        if k_bits not in {2, 4} or v_bits not in {2, 4}:
            raise ValueError("native CUDA GEMV supports only INT2/INT4")
        if group_size <= 0 or residual_length <= 0:
            raise ValueError("group_size and residual_length must be positive")
        if fp_prefix_layers < 0:
            raise ValueError("fp_prefix_layers must be non-negative")
        if readout_backend not in {
            "native_packed",
            "dequant_sdpa_reference",
            "dequant_fa2_reference",
        }:
            raise ValueError("unsupported KIVI/OTT readout backend")
        self.method = method
        self.kivi_implementation = str(kivi_implementation)
        self.baseline_root = Path(baseline_root).resolve()
        self.native_extension_root = (
            Path(native_extension_root).resolve()
            if native_extension_root is not None
            else None
        )
        self.k_bits = int(k_bits)
        self.v_bits = int(v_bits)
        self.group_size = int(group_size)
        self.residual_length = int(residual_length)
        self.fp_prefix_layers = int(fp_prefix_layers)
        self.readout_backend = str(readout_backend)
        self.sink_num = int(sink_num)
        self.sink_max_size = int(sink_max_size)
        self.quantize, self.packed_bmm, self.ott_quant_step = self._load_native_ops()

    def _load_native_ops(
        self,
    ) -> tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any] | None]:
        root = str(self.baseline_root)
        quant_root = str(self.baseline_root / "quant")
        if self.native_extension_root is not None:
            extension_root = str(self.native_extension_root)
            if extension_root not in sys.path:
                sys.path.insert(0, extension_root)
        if quant_root not in sys.path:
            sys.path.insert(0, quant_root)
        if root not in sys.path:
            sys.path.insert(0, root)
        quantizer = importlib.import_module("quant.new_pack")
        matmul = importlib.import_module("quant.matmul")
        ott_quant_step = None
        if self.method == "ott":
            # Load the released algorithm directly from the pinned GitHub
            # checkout without importing it through a potentially conflicting
            # top-level ``models`` package.  The Qwen wrapper only translates
            # cache state to and from the official tuple representation.
            spec = importlib.util.spec_from_file_location(
                "_r2omni_official_ott_utils",
                self.baseline_root / "models" / "utils_OTT.py",
            )
            if spec is None or spec.loader is None:
                raise ImportError("cannot load official OTT models/utils_OTT.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            ott_quant_step = module.quant_step
        return (
            quantizer.triton_quantize_and_pack_along_last_dim,
            matmul.cuda_bmm_fA_qB_outer,
            ott_quant_step,
        )

    def new_cache(self) -> NativePackedCache:
        return NativePackedCache(self)

    @staticmethod
    def _append(current: torch.Tensor | None, new: torch.Tensor, dim: int) -> torch.Tensor:
        return new if current is None else torch.cat((current, new), dim=dim)

    def _pack_key(self, keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.quantize(
            keys.transpose(2, 3).contiguous().half(),
            self.group_size,
            self.k_bits,
        )

    def _pack_value(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.quantize(values.contiguous().half(), self.group_size, self.v_bits)

    def _sink_count(self, layer_idx: int) -> int:
        return 0 if self.method != "ott" or layer_idx in {0, 1} else self.sink_num

    def _keeps_full_precision(self, layer_idx: int) -> bool:
        return int(layer_idx) < self.fp_prefix_layers

    def _flush_official_ott(self, state: PackedLayerState, layer_idx: int) -> None:
        if self.ott_quant_step is None:
            raise RuntimeError("official OTT quant_step was not loaded")
        sink_count = self._sink_count(layer_idx)
        fields: tuple[Any, ...] = (
            state.key_quant,
            state.key_full,
            state.key_scale,
            state.key_min,
            state.value_quant,
            state.value_full,
            state.value_scale,
            state.value_min,
        )
        if sink_count:
            fields += (
                state.sink_ids,
                state.sink_keys,
                state.sink_values,
                state.sink_scores,
            )
        fields += (state.seq_len,)
        updated = self.ott_quant_step(
            self.k_bits,
            self.v_bits,
            self.group_size,
            self.residual_length,
            sink_count,
            self.sink_max_size,
            fields,
            state.seq_len,
        )
        (
            state.key_quant,
            state.key_full,
            state.key_scale,
            state.key_min,
            state.value_quant,
            state.value_full,
            state.value_scale,
            state.value_min,
        ) = updated[:8]
        if sink_count:
            (
                state.sink_ids,
                state.sink_keys,
                state.sink_values,
                state.sink_scores,
            ) = updated[8:12]

    def _flush_ott(self, state: PackedLayerState, count: int, layer_idx: int) -> None:
        if count <= 0:
            return
        if self.method != "ott":
            raise RuntimeError("synchronous flush is only used by official OTT")
        if state.key_full is None:
            raise RuntimeError("official OTT flush requires a residual key cache")
        before = int(state.key_full.shape[2])
        self._flush_official_ott(state, layer_idx)
        if state.key_full is None:
            flushed = before
        else:
            flushed = before - int(state.key_full.shape[2])
        if flushed != count:
            raise RuntimeError(
                f"official OTT flush mismatch: expected {count}, got {flushed}"
            )

    def _initialize_official_kivi(self, state: PackedLayerState) -> None:
        """Match the released KIVI prefill cache partition exactly.

        KIVI keeps an incomplete ``residual_length`` block for keys, while
        values always keep the most recent ``residual_length`` tokens.  The
        two boundaries are intentionally independent.
        """
        if state.key_full is None or state.value_full is None:
            raise RuntimeError("KIVI prefill requires full key and value states")

        sequence_length = int(state.seq_len)
        key_full_length = sequence_length % self.residual_length
        key_quant_length = sequence_length - key_full_length
        if key_quant_length:
            keys = state.key_full[:, :, :key_quant_length, :].contiguous().half()
            state.key_quant, state.key_scale, state.key_min = self._pack_key(keys)
        state.key_full = (
            state.key_full[:, :, key_quant_length:, :].contiguous()
            if key_full_length
            else None
        )

        value_full_length = min(sequence_length, self.residual_length)
        value_quant_length = sequence_length - value_full_length
        if value_quant_length:
            values = state.value_full[:, :, :value_quant_length, :].contiguous().half()
            state.value_quant, state.value_scale, state.value_min = self._pack_value(
                values
            )
        state.value_full = state.value_full[
            :, :, value_quant_length:, :
        ].contiguous()

    def _flush_legacy_kivi(self, state: PackedLayerState, count: int) -> None:
        """Flush aligned K/V groups used by the original Qwen adapter.

        The historical adapter that produced our reported KIVI results keeps
        keys and values behind the same residual boundary.  Complete
        ``group_size`` chunks before that boundary are quantized together.
        """
        if count <= 0:
            return
        if state.key_full is None or state.value_full is None:
            raise RuntimeError("legacy KIVI flush requires residual K/V states")
        keys = state.key_full[:, :, :count, :].contiguous().half()
        values = state.value_full[:, :, :count, :].contiguous().half()
        state.key_full = state.key_full[:, :, count:, :].contiguous()
        state.value_full = state.value_full[:, :, count:, :].contiguous()
        key_q, key_s, key_m = self._pack_key(keys)
        value_q, value_s, value_m = self._pack_value(values)
        state.key_quant = self._append(state.key_quant, key_q, 3)
        state.key_scale = self._append(state.key_scale, key_s, 3)
        state.key_min = self._append(state.key_min, key_m, 3)
        state.value_quant = self._append(state.value_quant, value_q, 2)
        state.value_scale = self._append(state.value_scale, value_s, 2)
        state.value_min = self._append(state.value_min, value_m, 2)

    def _flush_official_kivi_decode(self, state: PackedLayerState) -> None:
        """Apply the released KIVI single-token decode transitions."""
        if state.key_full is not None:
            key_full_length = int(state.key_full.shape[2])
            if key_full_length > self.residual_length:
                raise RuntimeError(
                    "KIVI key residual exceeded its official block length"
                )
            if key_full_length == self.residual_length:
                key_q, key_s, key_m = self._pack_key(state.key_full)
                state.key_quant = self._append(state.key_quant, key_q, 3)
                state.key_scale = self._append(state.key_scale, key_s, 3)
                state.key_min = self._append(state.key_min, key_m, 3)
                state.key_full = None

        if state.value_full is None:
            raise RuntimeError("KIVI requires a residual value cache")
        value_full_length = int(state.value_full.shape[2])
        if value_full_length > self.residual_length + 1:
            raise RuntimeError(
                "KIVI value residual exceeded the official sliding window"
            )
        if value_full_length == self.residual_length + 1:
            value_q, value_s, value_m = self._pack_value(
                state.value_full[:, :, :1, :].contiguous()
            )
            state.value_quant = self._append(state.value_quant, value_q, 2)
            state.value_scale = self._append(state.value_scale, value_s, 2)
            state.value_min = self._append(state.value_min, value_m, 2)
            state.value_full = state.value_full[:, :, 1:, :].contiguous()

    def initialize(
        self, cache: NativePackedCache, layer_idx: int, keys: torch.Tensor, values: torch.Tensor
    ) -> None:
        keep_bf16 = self._keeps_full_precision(layer_idx)
        state = PackedLayerState(
            key_full=(keys.contiguous() if keep_bf16 else keys.contiguous().half()),
            value_full=(values.contiguous() if keep_bf16 else values.contiguous().half()),
            seq_len=int(keys.shape[2]),
        )
        cache.packed_layers[layer_idx] = state
        if keep_bf16:
            return
        if self.method == "kivi":
            if self.kivi_implementation == "official":
                self._initialize_official_kivi(state)
            else:
                excess = int(keys.shape[2]) - self.residual_length
                count = max(0, (excess // self.group_size) * self.group_size)
                self._flush_legacy_kivi(state, count)
            return
        excess = int(keys.shape[2]) - self.residual_length
        count = max(0, (excess // self.group_size) * self.group_size)
        self._flush_ott(state, count, layer_idx)

    def append_decode(
        self, cache: NativePackedCache, layer_idx: int, keys: torch.Tensor, values: torch.Tensor
    ) -> PackedLayerState:
        state = cache.packed_layers[layer_idx]
        if self._keeps_full_precision(layer_idx):
            new_keys, new_values = keys.contiguous(), values.contiguous()
        else:
            new_keys, new_values = keys.contiguous().half(), values.contiguous().half()
        state.key_full = self._append(state.key_full, new_keys, 2)
        state.value_full = self._append(state.value_full, new_values, 2)
        state.seq_len += int(keys.shape[2])
        return state

    def maybe_flush_decode(self, state: PackedLayerState, layer_idx: int) -> None:
        if self._keeps_full_precision(layer_idx):
            return
        if self.method == "kivi":
            if self.kivi_implementation == "official":
                self._flush_official_kivi_decode(state)
            else:
                if state.key_full is None:
                    raise RuntimeError("legacy KIVI requires a residual key cache")
                excess = int(state.key_full.shape[2]) - self.residual_length
                count = max(0, (excess // self.group_size) * self.group_size)
                self._flush_legacy_kivi(state, count)
            return
        if state.key_full is None:
            raise RuntimeError("official OTT requires a residual key cache")
        excess = int(state.key_full.shape[2]) - self.residual_length
        count = max(0, (excess // self.group_size) * self.group_size)
        self._flush_ott(state, count, layer_idx)

    def packed_attention(
        self,
        module: nn.Module,
        state: PackedLayerState,
        query_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        layer_idx: int,
    ) -> torch.Tensor:
        query = (
            query_states.contiguous()
            if self._keeps_full_precision(layer_idx)
            else query_states.half().contiguous()
        )
        query_heads = int(query.shape[1])
        kv_heads = int(module.num_key_value_heads)
        if query_heads % kv_heads:
            raise ValueError(
                f"invalid Qwen GQA geometry: {query_heads} query heads / "
                f"{kv_heads} KV heads"
            )
        if int(module.num_key_value_groups) != query_heads // kv_heads:
            raise ValueError("Qwen GQA group count does not match tensor geometry")
        if self.readout_backend == "dequant_sdpa_reference":
            return self._dequant_sdpa_attention(
                module, state, query, attention_mask, layer_idx
            ).to(query_states.dtype)
        if self.readout_backend == "dequant_fa2_reference":
            return self._dequant_fa2_attention(
                module, state, query, attention_mask, layer_idx
            ).to(query_states.dtype)

        def native_bmm(
            left: torch.Tensor,
            packed: torch.Tensor,
            scale: torch.Tensor,
            minimum: torch.Tensor,
            bits: int,
        ) -> torch.Tensor:
            if self.method == "ott":
                # The released OTT extension predates KIVI's native GQA
                # arguments and expects the Python wrapper to repeat packed
                # KV heads. Preserve that official convention exactly.
                packed = repeat_kv(packed, module.num_key_value_groups)
                scale = repeat_kv(scale, module.num_key_value_groups)
                minimum = repeat_kv(minimum, module.num_key_value_groups)
            return self.packed_bmm(
                self.group_size, left, packed, scale, minimum, bits
            )

        # Scale Q before the FP16 QK readout.  This is algebraically identical
        # to scaling the completed scores, but avoids overflowing the native
        # GEMV (and the full-precision suffix matmul) on high-magnitude deep
        # Qwen2.5-Omni-7B layers.  The packed CUDA kernels and cache format are
        # unchanged; only the Qwen host-side placement of 1/sqrt(d) differs.
        score_query = query * float(module.scaling)
        full_scores = None
        if state.key_full is not None:
            full_keys = repeat_kv(state.key_full, module.num_key_value_groups)
            full_scores = torch.matmul(score_query, full_keys.transpose(2, 3))
        if state.key_quant is None:
            if full_scores is None:
                raise RuntimeError("native packed cache contains no key states")
            scores = full_scores
        else:
            quant_scores = native_bmm(
                score_query,
                state.key_quant,
                state.key_scale,
                state.key_min,
                self.k_bits,
            )
            if self._sink_count(layer_idx) and state.sink_ids is not None:
                sink_keys = repeat_kv(
                    state.sink_keys.unsqueeze(0), module.num_key_value_groups
                )
                quant_scores[:, :, :, state.sink_ids] = torch.matmul(
                    score_query, sink_keys.transpose(2, 3)
                )
            scores = (
                quant_scores
                if full_scores is None
                else torch.cat((quant_scores, full_scores), dim=-1)
            )
        if attention_mask is not None:
            if attention_mask.ndim == 4:
                scores = scores + attention_mask[..., : scores.shape[-1]].to(scores.dtype)
            elif attention_mask.ndim == 2 and bool((attention_mask == 0).any()):
                padding = attention_mask[:, None, None, : scores.shape[-1]] == 0
                scores = scores.masked_fill(padding, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        full_length = (
            int(state.value_full.shape[2]) if state.value_full is not None else 0
        )
        full_values = (
            repeat_kv(state.value_full, module.num_key_value_groups)
            if state.value_full is not None
            else None
        )
        if state.value_quant is None:
            if full_values is None:
                raise RuntimeError("native packed cache contains no value states")
            output = torch.matmul(weights, full_values)
        else:
            quant_weights = (
                weights[..., :-full_length] if full_length else weights
            )
            if self._sink_count(layer_idx) and state.sink_ids is not None:
                sink_weights = quant_weights[..., state.sink_ids].clone()
                quant_weights[..., state.sink_ids] = 0
            output = native_bmm(
                quant_weights.contiguous(),
                state.value_quant,
                state.value_scale,
                state.value_min,
                self.v_bits,
            )
            if full_length:
                output += torch.matmul(weights[..., -full_length:], full_values)
            if self._sink_count(layer_idx) and state.sink_ids is not None:
                sink_values = repeat_kv(
                    state.sink_values.unsqueeze(0), module.num_key_value_groups
                )
                output += torch.matmul(sink_weights, sink_values)
        output = output.to(query_states.dtype)
        return output

    @staticmethod
    def _dequant_last_dim(
        packed: torch.Tensor,
        scale: torch.Tensor,
        minimum: torch.Tensor,
        bits: int,
        group_size: int,
    ) -> torch.Tensor:
        """Explicit affine reconstruction used only by the reference backend."""
        factor = 32 // bits
        shifts = torch.arange(
            factor, device=packed.device, dtype=torch.int32
        ) * bits
        values = ((packed[..., None] >> shifts) & (2**bits - 1)).flatten(-2)
        groups = int(values.shape[-1]) // group_size
        values = values.view(*values.shape[:-1], groups, group_size).to(scale.dtype)
        return (
            values * scale.unsqueeze(-1) + minimum.unsqueeze(-1)
        ).flatten(-2)

    def _materialize_kv(
        self,
        state: PackedLayerState,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialize the persistent packed cache in KV-head order."""
        key_parts: list[torch.Tensor] = []
        if state.key_quant is not None:
            key_quant = self._dequant_last_dim(
                state.key_quant,
                state.key_scale,
                state.key_min,
                self.k_bits,
                self.group_size,
            ).transpose(2, 3).contiguous()
            if self._sink_count(layer_idx) and state.sink_ids is not None:
                key_quant[:, :, state.sink_ids, :] = state.sink_keys.unsqueeze(0)
            key_parts.append(key_quant)
        if state.key_full is not None:
            key_parts.append(state.key_full)

        value_parts: list[torch.Tensor] = []
        if state.value_quant is not None:
            value_quant = self._dequant_last_dim(
                state.value_quant,
                state.value_scale,
                state.value_min,
                self.v_bits,
                self.group_size,
            )
            if self._sink_count(layer_idx) and state.sink_ids is not None:
                value_quant[:, :, state.sink_ids, :] = state.sink_values.unsqueeze(0)
            value_parts.append(value_quant)
        if state.value_full is not None:
            value_parts.append(state.value_full)
        if not key_parts or not value_parts:
            raise RuntimeError("reference readout received an empty cache")

        keys = torch.cat(key_parts, dim=2)
        values = torch.cat(value_parts, dim=2)
        if int(keys.shape[2]) != int(values.shape[2]):
            raise RuntimeError("materialized K/V sequence lengths do not match")
        return keys, values

    def _dequant_sdpa_attention(
        self,
        module: nn.Module,
        state: PackedLayerState,
        query: torch.Tensor,
        attention_mask: torch.Tensor | None,
        layer_idx: int,
    ) -> torch.Tensor:
        """Correctness-only Qwen readout using dense PyTorch SDPA."""
        keys, values = self._materialize_kv(state, layer_idx)
        keys = repeat_kv(keys, module.num_key_value_groups)
        values = repeat_kv(values, module.num_key_value_groups)

        mask = None
        if attention_mask is not None:
            if attention_mask.ndim == 4:
                mask = attention_mask[..., : keys.shape[2]].to(query.dtype)
            elif attention_mask.ndim == 2 and bool((attention_mask == 0).any()):
                mask = attention_mask[:, None, None, : keys.shape[2]].to(torch.bool)
        return F.scaled_dot_product_attention(
            query,
            keys,
            values,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False,
            scale=float(module.scaling),
        )

    def _dequant_fa2_attention(
        self,
        module: nn.Module,
        state: PackedLayerState,
        query: torch.Tensor,
        attention_mask: torch.Tensor | None,
        layer_idx: int,
    ) -> torch.Tensor:
        """Correctness readout using dense FA2 without repeating GQA KV heads."""
        keys, values = self._materialize_kv(state, layer_idx)
        mask = attention_mask
        if mask is not None and mask.ndim == 4:
            mask = mask[:, 0, 0, : keys.shape[2]] >= 0
            if bool(mask.all()):
                mask = None
        attention_functions = (
            QWEN3_ATTENTION_FUNCTIONS
            if type(module).__name__ == "Qwen3Attention"
            else QWEN25_ATTENTION_FUNCTIONS
        )
        if attention_functions is None:
            raise RuntimeError("Qwen2.5-Omni is unavailable in this environment")
        attention = attention_functions["flash_attention_2"]
        output, _ = attention(
            module,
            query,
            keys,
            values,
            mask,
            dropout=0.0,
            scaling=module.scaling,
            sliding_window=module.sliding_window,
        )
        # Transformers FA2 returns [batch, query, heads, dim], whereas the
        # packed adapter contract is [batch, heads, query, dim].
        return output.transpose(1, 2).contiguous()

    def storage_metadata(self, cache: NativePackedCache) -> dict[str, Any]:
        packed = residual = sink = 0
        tensors = [value for state in cache.packed_layers.values() for value in vars(state).values() if torch.is_tensor(value)]
        for state in cache.packed_layers.values():
            for tensor in (
                state.key_quant,
                state.key_scale,
                state.key_min,
                state.value_quant,
                state.value_scale,
                state.value_min,
            ):
                if torch.is_tensor(tensor):
                    packed += tensor.numel() * tensor.element_size()
            for tensor in (state.key_full, state.value_full):
                if torch.is_tensor(tensor):
                    residual += tensor.numel() * tensor.element_size()
            for tensor in (state.sink_ids, state.sink_keys, state.sink_values, state.sink_scores):
                if torch.is_tensor(tensor):
                    sink += tensor.numel() * tensor.element_size()
        return {
            "method": self.method,
            "implementation": (
                f"{self.kivi_implementation}_qwen_adapter"
                if self.method == "kivi"
                else "official_qwen_adapter"
            ),
            "source_root": str(self.baseline_root),
            "packed_bytes": packed,
            "residual_bytes": residual,
            "sink_bytes": sink,
            "actual_cache_storage_bytes": packed + residual + sink,
            "active_persistent_cache_bytes": packed + residual + sink,
            "allocated_persistent_cache_bytes": allocated_tensor_storage_bytes(tensors),
            "allocated_storage_scope": "unique reachable cache backing storage; excludes allocator reserved pool and working buffers",
            "k_bits": self.k_bits,
            "v_bits": self.v_bits,
            "group_size": self.group_size,
            "residual_length": self.residual_length,
            "readout_backend": self.readout_backend,
            **(
                {"fp_prefix_layers": self.fp_prefix_layers}
                if self.fp_prefix_layers
                else {}
            ),
            "sink_num": self.sink_num if self.method == "ott" else 0,
            "sink_max_size": self.sink_max_size if self.method == "ott" else 0,
            "prefill_runtime": "transformers_flash_attention_2_bf16",
            "decode_runtime": {
                "native_packed": "native_cuda_packed_gemv_fp16_accumulation",
                "dequant_sdpa_reference": (
                    "explicit_dequantization_pytorch_sdpa_reference"
                ),
                "dequant_fa2_reference": (
                    "explicit_dequantization_flash_attention_2_reference"
                ),
            }[self.readout_backend],
        }


def install_native_qwen_adapter(model: Any, adapter: NativeKVAdapter) -> None:
    """Patch the loaded upstream Qwen attention modules in place."""
    text_model = getattr(model, "thinker", getattr(model, "llm", model))

    def native_forward(
        module: nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Any | None = None,
        past_key_values: Any | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        if output_attentions:
            raise ValueError("native packed adapters do not return attention weights")
        cache = past_key_value if past_key_value is not None else past_key_values
        if not isinstance(cache, NativePackedCache):
            raise TypeError("native packed attention requires NativePackedCache")
        batch, query_length, _ = hidden_states.shape
        query_states = module.q_proj(hidden_states).view(
            batch, query_length, -1, module.head_dim
        ).transpose(1, 2)
        key_states = module.k_proj(hidden_states).view(
            batch, query_length, -1, module.head_dim
        ).transpose(1, 2)
        value_states = module.v_proj(hidden_states).view(
            batch, query_length, -1, module.head_dim
        ).transpose(1, 2)
        is_qwen3 = type(module).__name__ == "Qwen3Attention"
        if is_qwen3:
            # Qwen3 applies per-head Q/K RMSNorm before standard RoPE.
            query_states = module.q_norm(query_states.transpose(1, 2)).transpose(1, 2)
            key_states = module.k_norm(key_states.transpose(1, 2)).transpose(1, 2)
        if position_embeddings is None:
            raise ValueError("Qwen rotary embeddings are required")
        cos, sin = position_embeddings
        if is_qwen3:
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )
        else:
            if apply_multimodal_rotary_pos_emb is None:
                raise RuntimeError("Qwen2.5-Omni is unavailable in this environment")
            query_states, key_states = apply_multimodal_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
                module.rope_scaling["mrope_section"],
            )
        state = cache.packed_layers.get(module.layer_idx)
        if state is None:
            attention_functions = (
                QWEN3_ATTENTION_FUNCTIONS if is_qwen3 else QWEN25_ATTENTION_FUNCTIONS
            )
            if attention_functions is None:
                raise RuntimeError("Qwen2.5-Omni is unavailable in this environment")
            attention = attention_functions["flash_attention_2"]
            output, _ = attention(
                module,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0,
                scaling=module.scaling,
                sliding_window=module.sliding_window,
                position_ids=position_ids,
                **kwargs,
            )
            adapter.initialize(cache, module.layer_idx, key_states, value_states)
        else:
            if query_length != 1:
                raise ValueError(
                    "native packed decode requires one query token; "
                    f"received {query_length} with cached length {state.seq_len} "
                    f"at layer {module.layer_idx}; cache_position="
                    f"{None if cache_position is None else cache_position.detach().cpu().tolist()}"
                )
            state = adapter.append_decode(
                cache, module.layer_idx, key_states, value_states
            )
            output = adapter.packed_attention(
                module, state, query_states, attention_mask, module.layer_idx
            )
            output = output.transpose(1, 2).contiguous()
            adapter.maybe_flush_decode(state, module.layer_idx)
        output = output.reshape(batch, query_length, -1).contiguous()
        return module.o_proj(output), None

    patched = 0
    for layer in text_model.model.layers:
        attention = layer.self_attn
        if type(attention).__name__ not in {"Qwen2_5OmniAttention", "Qwen3Attention"}:
            raise TypeError(f"unexpected text attention class: {type(attention).__name__}")
        if type(attention).__name__ == "Qwen3Attention":
            attention.num_heads = int(attention.config.num_attention_heads)
            attention.num_key_value_heads = int(
                attention.config.num_key_value_heads
            )
        attention.forward = MethodType(native_forward, attention)
        patched += 1
    model._r2omni_cache_factory = adapter.new_cache
    text_model._r2omni_cache_factory = adapter.new_cache
    model._r2omni_native_kv_adapter = adapter
    model._r2omni_native_kv_patched_layers = patched
