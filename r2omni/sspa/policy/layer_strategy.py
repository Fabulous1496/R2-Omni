from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LayerStrategyConfig:
    """Layer grouping used by the released SSPA path."""

    mode: str
    num_layers: int
    block_start_layer: int = 0
    block_group_size: int = 4
    block_end_layer: int | None = None


def _baseline_entry(layer_id: int) -> dict[str, Any]:
    return {
        "layer_id": layer_id,
        "enabled": False,
        "mode": "baseline",
        "score_at_this_layer": False,
        "assignment_source_layer": None,
        "group_start_layer": None,
        "group_end_layer": None,
    }


def build_layer_strategy(config: LayerStrategyConfig) -> dict[int, dict[str, Any]]:
    """Build the baseline or four-layer-group strategy used in the paper."""

    strategy = {
        layer_id: _baseline_entry(layer_id)
        for layer_id in range(config.num_layers)
    }
    if config.mode in {"baseline", "none"}:
        return strategy
    if config.mode != "block":
        raise ValueError("release supports only baseline and block strategies")
    if not 0 <= config.block_start_layer < config.num_layers:
        raise ValueError("block_start_layer is outside the model")
    if config.block_group_size <= 0:
        raise ValueError("block_group_size must be positive")

    active_end = min(
        config.num_layers - 1,
        config.block_end_layer
        if config.block_end_layer is not None
        else config.num_layers - 1,
    )
    if active_end < config.block_start_layer:
        raise ValueError("block_end_layer precedes block_start_layer")

    for group_start in range(
        config.block_start_layer,
        active_end + 1,
        config.block_group_size,
    ):
        group_end = min(group_start + config.block_group_size - 1, active_end)
        for layer_id in range(group_start, group_end + 1):
            strategy[layer_id] = {
                "layer_id": layer_id,
                "enabled": True,
                "mode": "block",
                "score_at_this_layer": layer_id == group_start,
                "assignment_source_layer": group_start,
                "group_start_layer": group_start,
                "group_end_layer": group_end,
            }
    return strategy


def summarize_layer_strategy(
    strategy: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    enabled = [
        layer_id
        for layer_id, item in sorted(strategy.items())
        if item["enabled"]
    ]
    groups = [
        {"start": layer_id, "end": int(item["group_end_layer"])}
        for layer_id, item in sorted(strategy.items())
        if item["score_at_this_layer"]
    ]
    return {
        "mode": "block" if enabled else "baseline",
        "enabled_layers": enabled,
        "active_scope": ",".join(map(str, enabled)) if enabled else "none",
        "groups": groups,
    }
