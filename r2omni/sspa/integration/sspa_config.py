from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class SSPAConfig:
    """Configuration of the released INT4-K / tiered-V path."""

    generated_cache_policy: ClassVar[str] = "chunked_recompress"
    generated_value_precision: ClassVar[str] = "int2"
    recompress_chunk_size: ClassVar[int] = 128
    score_normalization: ClassVar[str] = "rank"
    residual_query_objective: ClassVar[str] = "cvar"
    precision_allocator: ClassVar[str] = "residual_rate_distortion"
    value_distortion_metric: ClassVar[str] = "k_conditioned_readout"

    fp_append_capacity: int = 260
    page_size: int = 32
    query_anchor_count: int = 8
    query_anchor_reduction: str = "ema"
    query_anchor_ema_decay: float = 0.8
    attention_weight: float = 0.6
    av_alignment_topk: int = 4
    temporal_num_chunks: int = 16

    value_int2_group_size: int = 32
    residual_allocation_rounds: int = 4
    k_conditioned_iterations: int = 2
    residual_query_cvar_fraction: float = 0.25
    residual_transition_weight: float = 0.5
    output_priority_weight: float = 0.25
    semantic_priority_weight: float = 0.75
    video_precision_weight: float = 0.6
    audio_precision_weight: float = 1.8

    precision_sigmoid_steepness: float = 8.0
    shallow_v_payload_bit_ceiling: float = 3.6
    deep_v_payload_bit_ceiling: float = 3.0
    recent_fp_window: int = 128

    def __post_init__(self) -> None:
        if self.fp_append_capacity < 1:
            raise ValueError("fp_append_capacity must be positive")
        if self.page_size not in {16, 32, 64}:
            raise ValueError("page_size must be 16, 32, or 64")
        if self.query_anchor_count < 1:
            raise ValueError("query_anchor_count must be positive")
        if self.query_anchor_reduction not in {"mean", "max", "ema"}:
            raise ValueError("query_anchor_reduction must be mean, max, or ema")
        if not 0.0 < self.query_anchor_ema_decay <= 1.0:
            raise ValueError("query_anchor_ema_decay must lie in (0, 1]")
        if not 0.0 <= self.attention_weight <= 1.0:
            raise ValueError("attention_weight must lie in [0, 1]")
        if self.av_alignment_topk < 1 or self.temporal_num_chunks < 1:
            raise ValueError("alignment top-k and temporal chunks must be positive")
        if self.value_int2_group_size not in {32, 64, 128}:
            raise ValueError("value_int2_group_size must be 32, 64, or 128")
        if not 1 <= self.residual_allocation_rounds <= 32:
            raise ValueError("residual_allocation_rounds must lie in [1, 32]")
        if not 1 <= self.k_conditioned_iterations <= 4:
            raise ValueError("k_conditioned_iterations must lie in [1, 4]")
        if not 0.0 < self.residual_query_cvar_fraction <= 1.0:
            raise ValueError("residual_query_cvar_fraction must lie in (0, 1]")
        if not 0.0 <= self.semantic_priority_weight <= 1.0:
            raise ValueError("semantic_priority_weight must lie in [0, 1]")
        if not 0.0 <= self.output_priority_weight <= 1.0:
            raise ValueError("output_priority_weight must lie in [0, 1]")
        if not 0.0 <= self.residual_transition_weight <= 1.0:
            raise ValueError("residual_transition_weight must lie in [0, 1]")
        if any(
            not math.isfinite(value) or value < 0.0
            for value in (self.video_precision_weight, self.audio_precision_weight)
        ) or self.video_precision_weight + self.audio_precision_weight <= 0.0:
            raise ValueError("modality precision weights must be finite and non-negative")
        if (
            not math.isfinite(self.precision_sigmoid_steepness)
            or self.precision_sigmoid_steepness <= 0.0
        ):
            raise ValueError("precision_sigmoid_steepness must be positive")
        if not (
            2.0
            <= self.deep_v_payload_bit_ceiling
            <= self.shallow_v_payload_bit_ceiling
            <= 16.0
        ):
            raise ValueError("V payload ceilings must decrease within [2, 16]")
        if self.recent_fp_window < 1:
            raise ValueError("recent_fp_window must be positive")
