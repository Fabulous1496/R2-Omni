"""KIVI-OM: KIVI's packed cache connected to Qwen2.5-Omni attention."""

from pathlib import Path
from typing import Any

from baselines.common.qwen_packed_adapter import (
    NativeKVAdapter,
    install_native_qwen_adapter,
)


def install_kivi_om(
    model: Any,
    *,
    source_root: str | Path,
    extension_root: str | Path | None = None,
    implementation: str = "official",
    k_bits: int = 2,
    v_bits: int = 2,
    group_size: int = 128,
    residual_length: int = 128,
    fp_prefix_layers: int = 0,
    readout_backend: str = "native_packed",
) -> NativeKVAdapter:
    """Install the model-side KIVI adapter without changing KIVI's core."""
    adapter = NativeKVAdapter(
        method="kivi",
        baseline_root=source_root,
        native_extension_root=extension_root,
        kivi_implementation=implementation,
        k_bits=k_bits,
        v_bits=v_bits,
        group_size=group_size,
        residual_length=residual_length,
        fp_prefix_layers=fp_prefix_layers,
        readout_backend=readout_backend,
    )
    install_native_qwen_adapter(model, adapter)
    return adapter


__all__ = ["NativeKVAdapter", "install_kivi_om"]
