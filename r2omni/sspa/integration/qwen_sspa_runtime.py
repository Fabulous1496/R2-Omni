from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from types import MethodType
from typing import Any

import torch

from ..policy.modality_metadata import TEXT
from ..runtime.packed_kv_layout import PackedKVLayer, PackedTensor


@dataclass
class _LayerState:
    packed: PackedKVLayer | None = None
    bits: torch.Tensor | None = None
    migrated: bool = False
    last_cache_position: int | None = None
    original_full_bytes: int = 0
    num_heads: int = 0
    head_dim: int = 0
    calls: int = 0
    scoring_ms: float = 0.0
    alignment_ms: float = 0.0
    sensitivity_ms: float = 0.0
    quadrant_assignment_ms: float = 0.0
    sorting_ms: float = 0.0
    packing_ms: float = 0.0
    kernel_ms: float = 0.0
    cache_append_ms: float = 0.0
    generated_recompress_ms: float = 0.0
    generated_recompress_count: int = 0
    fallback_latency_ms: float = 0.0
    fallback_count: int = 0
    kernel_calls: int = 0
    append_count: int = 0
    duplicate_append_avoided: int = 0
    migration_count: int = 0
    prefill_finalize_ms: float = 0.0
    prefill_warmup_ms: float = 0.0
    prefill_finalize_count: int = 0
    prefill_warmup_count: int = 0
    original_seq_len: int = 0
    kv_object_id: int | None = None
    errors: list[str] = field(default_factory=list)
    pending_timings: list[tuple[str, Any, Any]] = field(default_factory=list)
    timing_samples_ms: dict[str, list[float]] = field(default_factory=dict)
    flush_decode_steps: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class _RuntimeLayerPlan:
    layer_id: int
    mode: str
    score_at_this_layer: bool
    assignment_source_layer: int
    group_start_layer: int | None
    group_end_layer: int | None


def _tensor_bytes(tensor: torch.Tensor | None) -> int:
    return 0 if tensor is None else int(tensor.numel() * tensor.element_size())


def allocated_tensor_storage_bytes(tensors: Any) -> int:
    """Count unique backing storages, including capacity retained by views.

    This is storage reachable from cache tensors, not CUDA allocator reserved
    memory, peak GPU memory, model weights, or temporary working buffers.
    """
    storages: dict[tuple[str, int], int] = {}
    for tensor in tensors:
        if not torch.is_tensor(tensor) or tensor.numel() == 0:
            continue
        storage = tensor.untyped_storage()
        storages[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
    return sum(storages.values())


def cache_tensor_storage_bytes(cache: Any | None) -> int:
    if cache is None:
        return 0
    states = getattr(cache, "packed_layers", None)
    if isinstance(states, dict):
        return allocated_tensor_storage_bytes(
            value for state in states.values() for value in vars(state).values()
        )
    tensors: list[Any] = []
    for name in ("key_cache", "value_cache", "key_cache_pack", "value_cache_pack",
                 "key_cache_params", "value_cache_params", "key_cache_norm", "key_cache_norm_pack"):
        tensors.extend(getattr(cache, name, []) or [])
    for layer in getattr(cache, "layers", []) or []:
        tensors.extend((getattr(layer, "keys", None), getattr(layer, "values", None)))
    return allocated_tensor_storage_bytes(tensors)


def _packed_tensor_bytes(tensor: PackedTensor | None) -> tuple[int, int]:
    if tensor is None:
        return 0, 0
    return _tensor_bytes(tensor.packed), _tensor_bytes(tensor.scale) + _tensor_bytes(tensor.zero_point)


def _json_metadata_bytes(value: Any) -> int:
    def clean(item: Any) -> Any:
        if torch.is_tensor(item):
            return {"shape": list(item.shape), "dtype": str(item.dtype)}
        if isinstance(item, dict):
            return {str(key): clean(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(val) for val in item]
        if isinstance(item, (str, int, float, bool)) or item is None:
            return item
        return repr(item)
    return len(json.dumps(clean(value), sort_keys=True).encode("utf-8"))


def calculate_actual_cache_storage_breakdown(
    cache: Any | None,
    *,
    packed_layers: dict[int, PackedKVLayer] | None = None,
    shared_assignments: dict[int, torch.Tensor] | None = None,
    partial_buffer_bytes: int = 0,
    metadata: Any | None = None,
) -> dict[str, int]:
    result = {
        "full_k_bytes": 0,
        "full_v_bytes": 0,
        "fp_k_bytes": 0,
        "fp_v_bytes": 0,
        "int4_packed_k_bytes": 0,
        "int4_packed_v_bytes": 0,
        "int2_packed_k_bytes": 0,
        "int2_packed_v_bytes": 0,
        "k_scales_bytes": 0,
        "v_scales_bytes": 0,
        "token_index_bytes": 0,
        "assignment_bytes": 0,
        "metadata_bytes": 0,
        "partial_buffer_bytes": int(partial_buffer_bytes),
    }
    if cache is not None:
        key_cache = list(getattr(cache, "key_cache", []) or [])
        value_cache = list(getattr(cache, "value_cache", []) or [])
        for tensor in key_cache:
            result["full_k_bytes"] += _tensor_bytes(tensor if torch.is_tensor(tensor) else None)
        for tensor in value_cache:
            result["full_v_bytes"] += _tensor_bytes(tensor if torch.is_tensor(tensor) else None)
        # Transformers >=4.57 stores DynamicCache tensors on layer objects.
        if not key_cache and not value_cache:
            for layer in getattr(cache, "layers", []) or []:
                result["full_k_bytes"] += _tensor_bytes(
                    getattr(layer, "keys", None)
                )
                result["full_v_bytes"] += _tensor_bytes(
                    getattr(layer, "values", None)
                )
        # Count optional packed payloads when a cache implementation exposes them.
        for tensor in getattr(cache, "key_cache_pack", []):
            result["int2_packed_k_bytes"] += _tensor_bytes(tensor if torch.is_tensor(tensor) else None)
        for tensor in getattr(cache, "value_cache_pack", []):
            result["int2_packed_v_bytes"] += _tensor_bytes(tensor if torch.is_tensor(tensor) else None)
        for name in (
            "key_cache_params",
            "value_cache_params",
            "key_cache_norm",
            "key_cache_norm_pack",
        ):
            for tensor in getattr(cache, name, []):
                result["metadata_bytes"] += _tensor_bytes(
                    tensor if torch.is_tensor(tensor) else None
                )
    for layer in (packed_layers or {}).values():
        result["fp_k_bytes"] += _tensor_bytes(layer.fp_k_buffer if layer.fp_k_buffer is not None else layer.fp_k)
        result["fp_v_bytes"] += _tensor_bytes(layer.fp_v_buffer if layer.fp_v_buffer is not None else layer.fp_v)
        packed, scales = _packed_tensor_bytes(layer.int4_k)
        result["int4_packed_k_bytes"] += packed
        result["k_scales_bytes"] += scales
        packed, scales = _packed_tensor_bytes(layer.int4_v)
        result["int4_packed_v_bytes"] += packed
        result["v_scales_bytes"] += scales
        fp_indices = layer.fp_index_buffer if layer.fp_index_buffer is not None else layer.keep_indices_fp
        for indices in (fp_indices, layer.keep_indices_int4, layer.keep_indices_int2, layer.evicted_indices):
            result["token_index_bytes"] += _tensor_bytes(indices)
        result["metadata_bytes"] += _json_metadata_bytes(layer.metadata)
    for assignment in (shared_assignments or {}).values():
        result["assignment_bytes"] += _tensor_bytes(assignment)
    result["metadata_bytes"] += _json_metadata_bytes(metadata) if metadata is not None else 0
    result["persistent_cache_bytes"] = sum(
        value for key, value in result.items() if key not in {"partial_buffer_bytes", "persistent_cache_bytes"}
    )
    result["total_cache_and_working_bytes"] = result["persistent_cache_bytes"] + result["partial_buffer_bytes"]
    result["total_cache_bytes"] = result["persistent_cache_bytes"]
    result["active_persistent_cache_bytes"] = result["persistent_cache_bytes"]
    result["allocated_persistent_cache_bytes"] = max(
        result["persistent_cache_bytes"], cache_tensor_storage_bytes(cache)
    )
    return result
class _QwenSSPARuntime:
    """Shared Qwen hooks and cache utilities used by the SSPA manager."""

    def __init__(
        self,
        model: Any,
        layer_strategy: dict[int, dict[str, Any]],
        assignment_config: Any,
        kernel_backend: str = "triton_mixed_v2",
        fallback_on_error: bool = True,
        verbose: bool = True,
        collect_timings: bool = True,
        validate_kernel_output: bool = True,
        allow_layer0_compression: bool = False,
    ) -> None:
        self.model = model
        # Qwen2.5-Omni exposes the text model as ``thinker``; MiniCPM-o wraps
        # an upstream Qwen3 model as ``llm``.
        self.thinker = getattr(model, "thinker", getattr(model, "llm", model))
        self.layer_strategy = layer_strategy
        self.assignment_config = assignment_config
        self.kernel_backend = kernel_backend
        self.fallback_on_error = bool(fallback_on_error)
        self.verbose = bool(verbose)
        self.collect_timings = bool(collect_timings)
        self.validate_kernel_output = bool(validate_kernel_output)
        if layer_strategy.get(0, {}).get("enabled") and not allow_layer0_compression:
            raise ValueError("layer 0 must remain baseline because DynamicCache uses it for global sequence length")
        if kernel_backend not in {"triton_mixed_v2", "reference"}:
            raise ValueError("active kernel_backend must be triton_mixed_v2 or reference")
        self._modules = self._resolve_modules()
        self._runtime_plans = {
            layer_id: _RuntimeLayerPlan(
                layer_id=layer_id,
                mode=str(self.layer_strategy[layer_id]["mode"]),
                score_at_this_layer=bool(self.layer_strategy[layer_id]["score_at_this_layer"]),
                assignment_source_layer=int(self.layer_strategy[layer_id]["assignment_source_layer"]),
                group_start_layer=self.layer_strategy[layer_id]["group_start_layer"],
                group_end_layer=self.layer_strategy[layer_id]["group_end_layer"],
            )
            for layer_id in self._modules
        }
        self._original_forwards: dict[int, Any] = {}
        self._installed = False
        self._prefill_modality: torch.Tensor | None = None
        self._modality_metadata: dict[str, Any] = {}
        self._states: dict[int, _LayerState] = {}
        self._group_assignments: dict[int, torch.Tensor] = {}
        self._fallback_groups: set[int] = set()
        self._last_cache: Any | None = None
        self._sample_id: Any = None
        self._kernel_fallback_count = 0
        self._original_fallback_count = 0
        self._survivor_state: Any | None = None
        self._unit_records: list[dict[str, Any]] = []
        self._scoring_layers = tuple(
            layer for layer, metadata in sorted(self.layer_strategy.items()) if metadata["score_at_this_layer"]
        )
        self._mask_validated = False
        self._stc_state: Any | None = None
        self._stc_compaction_applied = False
        self._prepared_modality_length = 0

    def bind_stc_state(self, state: Any | None) -> None:
        """Bind STC so modality metadata follows its compaction mask."""
        self._stc_state = state

    def _synchronize_stc_compaction(self) -> None:
        if self._stc_compaction_applied:
            return
        state = self._stc_state
        keep_mask = getattr(state, "last_keep_mask", None) if state is not None else None
        if keep_mask is None or self._prefill_modality is None:
            return
        keep = keep_mask.flatten().to(device=self._prefill_modality.device, dtype=torch.bool)
        if int(keep.numel()) != int(self._prefill_modality.numel()):
            raise ValueError("STC keep mask does not match the prepared modality vector")
        self._prefill_modality = self._prefill_modality[keep]
        for key, value in tuple(self._modality_metadata.items()):
            if torch.is_tensor(value) and int(value.numel()) == int(keep.numel()):
                self._modality_metadata[key] = value.flatten().to(keep.device)[keep]
        self._stc_compaction_applied = True

    def _measure(self, state: _LayerState, field_name: str, device: torch.device, fn: Any) -> Any:
        if not self.collect_timings:
            return fn()
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = fn()
            end.record()
            state.pending_timings.append((field_name, start, end))
            return result
        started = time.perf_counter()
        result = fn()
        elapsed = (time.perf_counter() - started) * 1000.0
        setattr(state, field_name, getattr(state, field_name) + elapsed)
        state.timing_samples_ms.setdefault(field_name, []).append(elapsed)
        return result

    def _resolve_pending_timings(self) -> None:
        pending = [item for state in self._states.values() for item in state.pending_timings]
        if pending and torch.cuda.is_available():
            torch.cuda.synchronize()
        for state in self._states.values():
            resolved_sorting = False
            for field_name, start, end in state.pending_timings:
                elapsed = float(start.elapsed_time(end))
                setattr(state, field_name, getattr(state, field_name) + elapsed)
                state.timing_samples_ms.setdefault(field_name, []).append(elapsed)
                resolved_sorting |= field_name == "sorting_ms"
            state.pending_timings.clear()
            if resolved_sorting:
                state.quadrant_assignment_ms = max(0.0, state.quadrant_assignment_ms - state.sorting_ms)

    def _resolve_modules(self) -> dict[int, Any]:
        enabled = {layer for layer, metadata in self.layer_strategy.items() if metadata["enabled"]}
        found: dict[int, Any] = {}
        for module in self.thinker.modules():
            layer_id = getattr(module, "layer_idx", None)
            if layer_id is None or int(layer_id) not in enabled:
                continue
            if type(module).__name__ == "Qwen3Attention":
                # Transformers 4.51 keeps these dimensions only on config,
                # whereas the released kernel planner uses module attributes.
                module.num_heads = int(module.config.num_attention_heads)
                module.num_key_value_heads = int(module.config.num_key_value_heads)
            if all(hasattr(module, name) for name in ("q_proj", "k_proj", "v_proj", "o_proj", "head_dim", "num_heads", "num_key_value_heads")):
                found[int(layer_id)] = module
        missing = sorted(enabled - set(found))
        if missing:
            raise RuntimeError(f"could not locate Qwen attention modules for enabled layers {missing}")
        return found

    @property
    def installed(self) -> bool:
        return self._installed

    def install(self) -> "_QwenSSPARuntime":
        if self._installed:
            return self
        for layer_id, module in self._modules.items():
            original = module.forward
            self._original_forwards[layer_id] = original
            plan = self._runtime_plans[layer_id]

            def replacement(module_self: Any, *args: Any, _plan: _RuntimeLayerPlan = plan, **kwargs: Any) -> Any:
                return self._dispatch(_plan, module_self, *args, **kwargs)

            module.forward = MethodType(replacement, module)
        self._installed = True
        return self

    def uninstall(self) -> None:
        for layer_id, original in self._original_forwards.items():
            self._modules[layer_id].forward = original
        self._original_forwards.clear()
        self._installed = False

    def __enter__(self) -> "_QwenSSPARuntime":
        return self.install()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.uninstall()

    def prepare(self, modality: torch.Tensor, *, modality_metadata: dict[str, Any] | None = None, sample_id: Any = None) -> None:
        if modality.ndim != 1:
            modality = modality.flatten()
        self._prefill_modality = modality.detach()
        self._prepared_modality_length = int(self._prefill_modality.numel())
        self._modality_metadata = dict(modality_metadata or {})
        self._states = {layer_id: _LayerState() for layer_id in self._modules}
        self._group_assignments.clear()
        self._fallback_groups.clear()
        self._last_cache = None
        self._sample_id = sample_id
        self._kernel_fallback_count = 0
        self._original_fallback_count = 0
        self._survivor_state = None
        self._unit_records.clear()
        self._mask_validated = False
        self._stc_compaction_applied = False

    def _call_original(self, layer_id: int, *args: Any, **kwargs: Any) -> Any:
        return self._original_forwards[layer_id](*args, **kwargs)

    def _fallback_original(self, layer_id: int, error: Exception, *args: Any, **kwargs: Any) -> Any:
        metadata = self.layer_strategy[layer_id]
        source = int(metadata["assignment_source_layer"])
        self._fallback_groups.add(source)
        state = self._states.setdefault(layer_id, _LayerState())
        state.fallback_count += 1
        state.errors.append(f"{type(error).__name__}: {error}")
        self._original_fallback_count += 1
        device = next(self._modules[layer_id].parameters()).device
        result = self._measure(
            state, "fallback_latency_ms", device, lambda: self._call_original(layer_id, *args, **kwargs)
        )
        return result

    def _dispatch(self, plan: _RuntimeLayerPlan, module: Any, *args: Any, **kwargs: Any) -> Any:
        layer_id = plan.layer_id
        hidden_states = kwargs.get("hidden_states", args[0] if args else None)
        if not torch.is_tensor(hidden_states):
            return self._call_original(layer_id, *args, **kwargs)
        if int(hidden_states.shape[1]) != 1 or self._prefill_modality is None:
            return self._call_original(layer_id, *args, **kwargs)
        source = plan.assignment_source_layer
        if source in self._fallback_groups:
            return self._fallback_original(layer_id, RuntimeError(f"assignment group {source} already fell back"), *args, **kwargs)
        try:
            return self._active_forward(plan, module, *args, **kwargs)
        except Exception as exc:
            state = self._states.setdefault(layer_id, _LayerState())
            if not self.fallback_on_error or state.migrated:
                raise
            return self._fallback_original(layer_id, exc, *args, **kwargs)


    @staticmethod
    def _cache_position(cache: Any, fallback: int) -> int:
        # Layer 0 is deliberately baseline and has already appended the current
        # token before any enabled layer runs. Tensor shape is host metadata;
        # unlike cache_position.item(), this does not synchronize the CUDA stream.
        if hasattr(cache, "get_seq_length"):
            global_length = int(cache.get_seq_length(0))
            if global_length > 0:
                return global_length - 1
        return int(fallback)

    @staticmethod
    def _validate_mask(attention_mask: torch.Tensor | None, seq_len: int) -> None:
        if attention_mask is None:
            return
        mask = attention_mask[..., -1:, :seq_len]
        if mask.dtype == torch.bool:
            if not bool(mask.all()):
                raise ValueError("active mixed kernel does not support masked decode source positions")
        elif bool((mask < -1e3).any()):
            raise ValueError("active mixed kernel does not support masked decode source positions")

    def _modality(self, seq_len: int, device: torch.device) -> torch.Tensor:
        assert self._prefill_modality is not None
        self._synchronize_stc_compaction()
        prefix = self._prefill_modality
        prefix = prefix.to(device=device, dtype=torch.long)
        if int(prefix.numel()) > seq_len:
            raise ValueError(f"modality length {prefix.numel()} exceeds cache length {seq_len}")
        return torch.cat((prefix, torch.full((seq_len - int(prefix.numel()),), TEXT, dtype=torch.long, device=device)))

    def _effective_prefill_token_count(self) -> int:
        return int(self._prefill_modality.numel()) if self._prefill_modality is not None else 0



    @staticmethod
    def _cache_layer_tensors(cache: Any, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one mutable DynamicCache layer across Transformers versions."""
        if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
            if layer_id >= len(cache.key_cache) or layer_id >= len(cache.value_cache):
                raise IndexError(f"DynamicCache has no layer {layer_id}")
            return cache.key_cache[layer_id], cache.value_cache[layer_id]
        layers = getattr(cache, "layers", None)
        if layers is not None and layer_id < len(layers):
            layer = layers[layer_id]
            if hasattr(layer, "keys") and hasattr(layer, "values"):
                return layer.keys, layer.values
        raise TypeError("active mixed attention requires a mutable HuggingFace DynamicCache")

    @staticmethod
    def _set_cache_layer_tensors(
        cache: Any,
        layer_id: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Replace one DynamicCache layer across Transformers versions."""
        if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
            if layer_id >= len(cache.key_cache) or layer_id >= len(cache.value_cache):
                raise IndexError(f"DynamicCache has no layer {layer_id}")
            cache.key_cache[layer_id] = key
            cache.value_cache[layer_id] = value
            return
        layers = getattr(cache, "layers", None)
        if layers is not None and layer_id < len(layers):
            layer = layers[layer_id]
            if hasattr(layer, "keys") and hasattr(layer, "values"):
                layer.keys = key
                layer.values = value
                layer.is_initialized = True
                layer.dtype = key.dtype
                layer.device = key.device
                return
        raise TypeError("active mixed attention requires a mutable HuggingFace DynamicCache")

    @classmethod
    def _clear_layer_cache(cls, cache: Any, layer_id: int, device: torch.device, dtype: torch.dtype) -> None:
        empty = torch.empty(0, device=device, dtype=dtype)
        cls._set_cache_layer_tensors(cache, layer_id, empty, empty)
