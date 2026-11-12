"""OScaR's native Qwen3 attention connected to MiniCPM-o."""

from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Callable

import torch
from torch import nn


def _normalized_hadamard(size: int, device: torch.device) -> torch.Tensor:
    matrix = torch.eye(size, device=device, dtype=torch.float32)
    width = 1
    while width < size:
        view = matrix.view(size, size // (2 * width), 2, width)
        even, odd = view[:, :, 0], view[:, :, 1]
        matrix = torch.stack((even + odd, even - odd), dim=2).reshape(size, size)
        width *= 2
    return matrix * (size**-0.5)


@torch.no_grad()
def _apply_offline_v_hadamard(llm: Any) -> int:
    """Apply OScaR's released V/O weight-equivalent Hadamard rotation."""
    head_dim = int(getattr(llm.config, "head_dim", 0))
    if head_dim <= 0 or head_dim & (head_dim - 1):
        raise ValueError(f"OScaR requires a power-of-two head_dim, got {head_dim}")
    hadamard = _normalized_hadamard(
        head_dim, llm.model.layers[0].self_attn.v_proj.weight.device
    )
    for layer in llm.model.layers:
        attention = layer.self_attn
        v_weight = attention.v_proj.weight.data
        v_float = v_weight.float().reshape(
            attention.v_proj.out_features // head_dim,
            head_dim,
            attention.v_proj.in_features,
        )
        attention.v_proj.weight.data = torch.matmul(
            hadamard, v_float
        ).reshape_as(v_weight).to(v_weight.dtype)

        o_weight = attention.o_proj.weight.data
        o_float = o_weight.float().reshape(
            attention.o_proj.out_features,
            attention.o_proj.in_features // head_dim,
            head_dim,
        )
        attention.o_proj.weight.data = torch.matmul(
            o_float, hadamard
        ).reshape_as(o_weight).to(o_weight.dtype)
    return head_dim


def install_oscar_om(
    model: Any,
    *,
    source_root: str | Path,
    extension_root: str | Path | None = None,
    num_bits: int = 2,
    group_size: int = 32,
    residual_block_size: int = 128,
    kv_rotation: str = "hadamard",
    kv_norm: str = "1",
    offline_v_hadamard: bool = True,
) -> Callable[[], Any]:
    """Install the upstream OScaR Qwen3 module without editing its source.

    MiniCPM-o-4.5 uses an upstream Qwen3 causal LM internally.  Replacing only
    each self-attention module preserves MiniCPM's multimodal wrappers while
    routing packed decode through OScaR's released CUDA implementation.
    """
    root = Path(source_root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if extension_root is not None:
        extension = str(Path(extension_root).resolve())
        if extension in sys.path:
            sys.path.remove(extension)
        sys.path.insert(0, extension)

    # OScaR vendors a Qwen3 model generated with Transformers 4.52, while
    # MiniCPM-o-4.5 pins 4.51. Only two definition-time helpers are absent in
    # 4.51; neither participates in Qwen3Oscar attention execution.
    import transformers.utils as transformers_utils

    if not hasattr(transformers_utils, "auto_docstring"):
        def auto_docstring(obj: Any | None = None, *args: Any, **kwargs: Any) -> Any:
            return (lambda value: value) if obj is None else obj

        transformers_utils.auto_docstring = auto_docstring
    try:
        import transformers.modeling_layers  # noqa: F401
    except ImportError:
        compatibility = ModuleType("transformers.modeling_layers")
        compatibility.GradientCheckpointingLayer = nn.Module
        sys.modules["transformers.modeling_layers"] = compatibility

    from evaluation.qwen3 import Qwen3Oscar
    from oscar import Cache as OscarCache, DynamicCache as OscarDynamicCache
    from transformers.cache_utils import (
        Cache as TransformersCache,
        DynamicCache as TransformersDynamicCache,
    )
    from transformers.models.qwen3 import modeling_qwen3

    # Let MiniCPM's upstream Qwen3 model create and accept OScaR's cache. The
    # first generation call still starts with ``past_key_values=None``, which
    # is required by GenerationMixin's initial cache-position setup.
    modeling_qwen3.Cache = (TransformersCache, OscarCache)
    modeling_qwen3.DynamicCache = OscarDynamicCache

    # GenerationMixin recognizes only Transformers Cache subclasses when it
    # initializes cache_position.  This thin model-side bridge preserves the
    # upstream OScaR cache implementation while making that type check pass.
    class OscarOmDynamicCache(OscarDynamicCache, TransformersDynamicCache):
        def get_seq_length(self, layer_idx: int = 0) -> int:
            packed_length = 0
            if len(self.value_cache_pack) > layer_idx:
                packed = self.value_cache_pack[layer_idx]
                if torch.is_tensor(packed) and packed.numel():
                    packed_length = int(packed.shape[1])
                    if len(self.key_cache_norm_pack) > layer_idx:
                        norms = self.key_cache_norm_pack[layer_idx]
                        if torch.is_tensor(norms) and norms.numel():
                            packed_length = int(norms.shape[-1])

            residual_length = 0
            if len(self.value_cache) > layer_idx:
                residual = self.value_cache[layer_idx]
                if torch.is_tensor(residual) and residual.numel():
                    residual_length = int(residual.shape[1])
            logical_length = packed_length + residual_length
            self._r2omni_packed_length = packed_length
            self._r2omni_residual_length = residual_length
            self._r2omni_logical_length = logical_length
            return logical_length

    llm = model.llm
    config = llm.config
    config.attn_backend = "oscar"
    config.num_bits = int(num_bits)
    config.quant_mode = "k-channel"
    config.group_size = int(group_size)
    config.kv_rotation = str(kv_rotation)
    config.kv_norm = str(kv_norm)
    config.residual_block_size = int(residual_block_size)

    if offline_v_hadamard:
        _apply_offline_v_hadamard(llm)

    for layer_idx, layer in enumerate(llm.model.layers):
        old = layer.self_attn
        replacement = Qwen3Oscar(config, layer_idx)
        replacement.load_state_dict(old.state_dict(), strict=True)
        replacement.to(device=old.q_proj.weight.device, dtype=old.q_proj.weight.dtype)
        replacement.eval()
        layer.self_attn = replacement

    model._r2omni_oscar_source = str(root)
    model._r2omni_oscar_layers = len(llm.model.layers)
    model._r2omni_oscar_offline_v_hadamard = bool(offline_v_hadamard)
    return OscarOmDynamicCache


__all__ = ["install_oscar_om"]
