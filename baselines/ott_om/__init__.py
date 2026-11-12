"""OTT-OM: OTT's released cache transitions connected to Qwen2.5-Omni."""

from pathlib import Path
from typing import Any

from baselines.common.qwen_packed_adapter import (
    NativeKVAdapter,
    install_native_qwen_adapter,
)


def install_ott_om(
    model: Any,
    *,
    source_root: str | Path,
    extension_root: str | Path | None = None,
    k_bits: int = 2,
    v_bits: int = 2,
    group_size: int = 128,
    residual_length: int = 32,
    fp_prefix_layers: int = 0,
    readout_backend: str = "native_packed",
    sink_num: int = 3,
    sink_max_size: int = 32,
) -> NativeKVAdapter:
    """Install the model-side OTT adapter without changing OTT's core."""
    adapter = NativeKVAdapter(
        method="ott",
        baseline_root=source_root,
        native_extension_root=extension_root,
        kivi_implementation="official",
        k_bits=k_bits,
        v_bits=v_bits,
        group_size=group_size,
        residual_length=residual_length,
        fp_prefix_layers=fp_prefix_layers,
        readout_backend=readout_backend,
        sink_num=sink_num,
        sink_max_size=sink_max_size,
    )
    install_native_qwen_adapter(model, adapter)
    return adapter


__all__ = ["NativeKVAdapter", "install_ott_om"]
