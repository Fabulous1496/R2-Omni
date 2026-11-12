from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .modality_metadata import AUDIO, VIDEO


def _grouped_gqa_logits(q: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """Compute QK logits without materializing KV heads repeated for GQA.

    ``index_select``-expanding ``[B,Hkv,S,D]`` to ``[B,Hq,S,D]`` is especially
    expensive during long-video calibration.  Qwen lays out query heads in
    contiguous groups belonging to each KV head, so the same contraction can
    be expressed as ``[B,Hkv,G,Q,D] x [B,Hkv,S,D]`` and flattened afterwards.
    """
    if q.ndim != 4 or key.ndim != 4:
        raise ValueError("q and key must have shape [B,Hq,Q,D] and [B,Hkv,S,D]")
    if int(q.shape[0]) != int(key.shape[0]) or int(q.shape[-1]) != int(key.shape[-1]):
        raise ValueError("q and key batch/head dimensions are incompatible")
    batch, query_heads, queries, head_dim = map(int, q.shape)
    kv_heads = int(key.shape[1])
    if query_heads % kv_heads:
        raise ValueError("GQA requires query heads divisible by KV heads")
    groups = query_heads // kv_heads
    grouped_q = q.float().reshape(batch, kv_heads, groups, queries, head_dim)
    logits = torch.einsum("bhgqd,bhsd->bhgqs", grouped_q, key.float())
    return logits.reshape(batch, query_heads, queries, int(key.shape[2])) / math.sqrt(
        head_dim
    )


def _grouped_gqa_readout(
    attention: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    """Compute attention-times-V without repeating KV heads across GQA groups."""
    if attention.ndim != 4 or value.ndim != 3:
        raise ValueError("attention/value must be [B,Hq,Q,S] and [Hkv,S,D]")
    batch, query_heads, queries, tokens = map(int, attention.shape)
    kv_heads, value_tokens, head_dim = map(int, value.shape)
    if tokens != value_tokens or query_heads % kv_heads:
        raise ValueError("attention and value have incompatible GQA dimensions")
    groups = query_heads // kv_heads
    grouped_attention = attention.reshape(
        batch, kv_heads, groups, queries, tokens
    )
    output = torch.einsum(
        "bhgqs,hsd->bhgqd", grouped_attention, value.float()
    )
    return output.reshape(batch, query_heads, queries, head_dim)




def reduce_query_attention_probabilities(
    values: torch.Tensor,
    *,
    reduce_heads: str = "mean",
    reduce_queries: str = "mean",
    query_ema_decay: float = 0.8,
) -> torch.Tensor:
    """Reduce cached ``[B,H,Q,S]`` attention without recomputing QK logits."""
    if values.ndim != 4 or not torch.is_floating_point(values):
        raise ValueError("attention probabilities must be floating [B,H,Q,S]")
    if reduce_queries == "mean":
        values = values.mean(dim=2)
    elif reduce_queries == "max":
        values = values.amax(dim=2)
    elif reduce_queries == "ema":
        if not 0.0 < query_ema_decay <= 1.0:
            raise ValueError("query_ema_decay must be in (0, 1]")
        query_count = int(values.shape[2])
        # Queries are ordered oldest -> newest.  A decay below one emphasizes
        # the instruction-final anchors while still requiring local agreement.
        age = torch.arange(
            query_count - 1, -1, -1, device=values.device, dtype=values.dtype
        )
        weights = float(query_ema_decay) ** age
        weights = weights / weights.sum()
        values = (values * weights.view(1, 1, -1, 1)).sum(dim=2)
    else:
        raise ValueError("reduce_queries must be mean, max, or ema")
    if reduce_heads == "mean":
        return values.mean(dim=(0, 1))
    if reduce_heads == "max":
        return values.amax(dim=(0, 1))
    raise ValueError("reduce_heads must be mean or max")


def compute_query_attention_probabilities(
    q: torch.Tensor,
    k: torch.Tensor,
    surviving_indices: torch.Tensor,
    *,
    apply_softmax: bool = True,
) -> torch.Tensor:
    """Return GQA-expanded per-query attention with shape ``[B,Hq,Q,S]``."""
    if q.ndim == 3:
        q = q.unsqueeze(2)
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError("q and k must have shape [B,H,Q,D] and [B,Hkv,S,D]")
    if q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k batch/head dimensions are incompatible")
    hq, hkv = int(q.shape[1]), int(k.shape[1])
    if hq % hkv:
        raise ValueError("GQA requires query heads divisible by KV heads")
    indices = surviving_indices.to(device=k.device, dtype=torch.long).flatten()
    selected = k.index_select(2, indices)
    logits = _grouped_gqa_logits(q, selected)
    return (
        torch.softmax(logits, dim=-1, dtype=torch.float32)
        if apply_softmax
        else logits
    )


def infer_temporal_chunk_ids(modality: torch.Tensor, num_chunks: int = 16) -> torch.Tensor:
    """Map audio and video sequences independently onto a shared relative-time grid."""
    if num_chunks <= 0:
        raise ValueError("num_chunks must be positive")
    vector = modality.flatten()
    result = torch.full_like(vector, -1, dtype=torch.long)
    for value in (VIDEO, AUDIO):
        positions = (vector == value).nonzero(as_tuple=False).flatten()
        count = int(positions.numel())
        if count:
            chunks = torch.div(torch.arange(count, device=vector.device) * num_chunks, count, rounding_mode="floor")
            result[positions] = chunks.clamp_max(num_chunks - 1)
    return result


def compute_av_alignment_score(
    token_features: torch.Tensor,
    modality: torch.Tensor,
    temporal_chunk_ids: torch.Tensor,
    surviving_indices: torch.Tensor,
    topk: int = 4,
    num_temporal_chunks: int = 16,
) -> torch.Tensor:
    """Return chunk-local cross-modal alignment without a full audio-by-video matrix."""
    if token_features.ndim == 3:
        if int(token_features.shape[0]) != 1:
            raise ValueError("token_features batch size must be 1")
        token_features = token_features[0]
    if token_features.ndim != 2:
        raise ValueError("token_features must have shape [S,D] or [1,S,D]")
    if topk <= 0:
        raise ValueError("topk must be positive")
    if num_temporal_chunks <= 0:
        raise ValueError("num_temporal_chunks must be positive")
    indices = surviving_indices.to(token_features.device, torch.long).flatten()
    current_features = token_features.index_select(0, indices).float()
    current_modality = modality.to(token_features.device).flatten().index_select(0, indices)
    current_chunks = temporal_chunk_ids.to(token_features.device).flatten().index_select(0, indices)
    normalized = F.normalize(current_features, dim=-1)
    scores = torch.zeros(indices.numel(), device=token_features.device, dtype=torch.float32)

    for source_modality, target_modality in ((AUDIO, VIDEO), (VIDEO, AUDIO)):
        source_all = (current_modality == source_modality).nonzero(as_tuple=False).flatten()
        target_all = (current_modality == target_modality).nonzero(as_tuple=False).flatten()
        if not source_all.numel() or not target_all.numel():
            continue
        global_centroid = F.normalize(normalized.index_select(0, target_all).mean(dim=0, keepdim=True), dim=-1)
        source_chunk_values = current_chunks.index_select(0, source_all)
        for chunk in range(num_temporal_chunks):
            source = source_all[source_chunk_values == chunk]
            target = target_all[current_chunks.index_select(0, target_all) == chunk]
            if not target.numel():
                scores[source] = (normalized.index_select(0, source) * global_centroid).sum(dim=-1)
                continue
            similarity = normalized.index_select(0, source) @ normalized.index_select(0, target).T
            k_value = min(topk, int(target.numel()))
            scores[source] = similarity.topk(k_value, dim=-1).values.mean(dim=-1)
    return scores


def normalize_scores(scores: torch.Tensor, method: str = "rank") -> torch.Tensor:
    """Normalize a one-dimensional score vector into [0, 1]."""
    values = scores.float().flatten()
    if not values.numel():
        return values
    method = method.lower()
    if method == "rank":
        if values.numel() == 1:
            return torch.ones_like(values)
        order = torch.argsort(values, stable=True)
        sorted_values = values.index_select(0, order)
        # Equal scores must remain equal.  Assigning ordinal ranks to a tied
        # vector fabricates a position-dependent signal; this is especially
        # harmful for video-only inputs where AV alignment is unavailable and
        # every raw alignment score is zero.
        _, inverse, counts = torch.unique_consecutive(
            sorted_values,
            return_inverse=True,
            return_counts=True,
        )
        ends = counts.cumsum(0) - 1
        starts = ends - counts + 1
        average_ranks = (starts + ends).to(values.dtype) * 0.5
        sorted_ranks = average_ranks.index_select(0, inverse)
        ranks = torch.empty_like(values)
        ranks[order] = sorted_ranks
        return ranks / (values.numel() - 1)
    if method == "minmax":
        low, high = values.amin(), values.amax()
        return torch.where(high > low, (values - low) / (high - low), torch.full_like(values, 0.5))
    if method == "zsigmoid":
        std = values.std(unbiased=False)
        return torch.sigmoid((values - values.mean()) / std.clamp_min(1e-6))
    raise ValueError("normalization method must be rank, minmax, or zsigmoid")


def resolve_cross_modal_alignment(
    attention: torch.Tensor,
    alignment: torch.Tensor,
    modality: torch.Tensor,
) -> tuple[torch.Tensor, bool]:
    """Use attention as the explicit fallback when AV alignment is undefined.

    ``compute_av_alignment_score`` has no cross-modal target for audio-only or
    video-only inputs and therefore returns zeros.  Ranking those zeros must not
    create a synthetic ordering.  Falling back to the already available query
    attention preserves a meaningful retention signal without pretending that
    cross-modal evidence exists.
    """

    values = modality.flatten()
    has_video = bool((values == VIDEO).any())
    has_audio = bool((values == AUDIO).any())
    if has_video and has_audio:
        return alignment, False
    return attention, True


def normalize_scores_by_modality(
    scores: torch.Tensor,
    current_modality: torch.Tensor,
    method: str = "rank",
) -> torch.Tensor:
    result = torch.zeros_like(scores, dtype=torch.float32)
    for value in (VIDEO, AUDIO):
        positions = (current_modality == value).nonzero(as_tuple=False).flatten()
        if positions.numel():
            result[positions] = normalize_scores(scores.index_select(0, positions), method)
    return result




def fuse_semantic_value_priority(
    semantic_score: torch.Tensor,
    value_action_score: torch.Tensor,
    semantic_weight: float,
) -> torch.Tensor:
    """Blend semantic importance into an action-specific V-risk priority.

    The returned score is only a bounded priority modifier for residual
    rate-distortion allocation. It does not alter the measured action gain,
    so a semantically important page with non-positive residual gain cannot be
    promoted by this fusion.
    """
    if not 0.0 <= semantic_weight <= 1.0:
        raise ValueError("semantic priority weight must be in [0, 1]")
    if semantic_score.shape != value_action_score.shape:
        raise ValueError(
            "semantic and value action scores must have identical shapes"
        )
    return (
        semantic_weight * semantic_score
        + (1.0 - semantic_weight) * value_action_score
    )




@dataclass(frozen=True)
class ResidualReadoutActions:
    """Model-output residual and the two production V upgrade directions.

    The correction tensors use ``[B,Hq,Q,P,D]``.  Keeping their direction,
    rather than reducing them immediately to page-local MSE, lets the allocator
    account for cancellation with K error and with upgrades selected in earlier
    rounds.
    """

    residual: torch.Tensor
    int2_to_int4: torch.Tensor
    int4_to_fp16: torch.Tensor
    key_reconstruction_mse: torch.Tensor


@dataclass(frozen=True)
class KConditionedValueContext:
    """K-invariant V calibration state shared by candidate K layouts."""

    reference_output: torch.Tensor
    int2_value: torch.Tensor
    int2_to_int4_delta: torch.Tensor
    int4_to_fp16_delta: torch.Tensor
    kv_head: torch.Tensor
    page_indices: torch.Tensor
    page_valid: torch.Tensor
    q_shape: tuple[int, ...]
    key_shape: tuple[int, ...]


def _production_value_reconstructions(
    value: torch.Tensor,
    *,
    int2_group_size: int,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulate the deployed INT2/INT4 V formats without persistent packing."""
    heads, tokens, head_dim = map(int, value.shape)
    work = value.float()
    grouped = work.reshape(
        heads, tokens, head_dim // int2_group_size, int2_group_size
    )
    minimum = torch.minimum(
        grouped.amin(dim=-1, keepdim=True),
        torch.zeros((), device=work.device),
    )
    maximum = torch.maximum(
        grouped.amax(dim=-1, keepdim=True),
        torch.zeros((), device=work.device),
    )
    scale2 = ((maximum - minimum) / 3.0).clamp_min(float(epsilon))
    zero2 = torch.round(-minimum / scale2).clamp(0.0, 3.0)
    restored2 = (
        torch.round(grouped / scale2 + zero2).clamp(0.0, 3.0) - zero2
    ) * scale2

    scale4 = (work.abs().amax(dim=-1, keepdim=True) / 7.0).clamp_min(
        float(epsilon)
    )
    restored4 = torch.round(work / scale4).clamp(-7.0, 7.0) * scale4
    return restored2.reshape_as(work), restored4


def _validate_page_geometry(
    page_offsets: torch.Tensor,
    page_lengths: torch.Tensor,
    *,
    tokens: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    offsets = page_offsets.flatten().to(device=device, dtype=torch.long)
    lengths = page_lengths.flatten().to(device=device, dtype=torch.long)
    if offsets.numel() == 0 or offsets.numel() != lengths.numel():
        raise ValueError("page offsets/lengths must have equal non-zero size")
    if bool((lengths <= 0).any()) or bool((offsets < 0).any()):
        raise ValueError("page offsets and lengths must describe positive pages")
    if bool((offsets + lengths > tokens).any()):
        raise ValueError("page geometry exceeds the value-cache length")
    width = int(lengths.max())
    lanes = torch.arange(width, device=device)
    indices = offsets[:, None] + lanes[None, :]
    valid = lanes[None, :] < lengths[:, None]
    return offsets, lengths, indices.clamp_max(tokens - 1), valid


def _tail_mean(values: torch.Tensor, dim: int, fraction: float) -> torch.Tensor:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("tail fraction must lie in (0, 1]")
    count = int(values.shape[dim])
    tail = max(1, math.ceil(count * fraction))
    return values.topk(tail, dim=dim, largest=True, sorted=False).values.mean(dim=dim)




def _normalize_av_page_risk(
    values: torch.Tensor,
    page_modality: torch.Tensor,
    method: str,
) -> torch.Tensor:
    modality = page_modality.flatten().to(device=values.device, dtype=torch.long)
    if int(modality.numel()) != int(values.numel()):
        raise ValueError("page modality and risk must have identical length")
    # Text/other pages are outside the requested A/V comparison and receive a
    # neutral prior. Audio and video are normalized independently so their
    # very different token counts and raw activation scales remain comparable.
    result = torch.full_like(values.float(), 0.5)
    for value in (VIDEO, AUDIO):
        positions = (modality == value).nonzero(as_tuple=False).flatten()
        if positions.numel():
            result[positions] = normalize_scores(
                values.index_select(0, positions), method
            )
    return result


def compute_output_action_sensitivity(
    actions: ResidualReadoutActions,
    page_modality: torch.Tensor,
    *,
    query_tail_fraction: float = 0.25,
    transition_weight: float = 0.0,
    normalization: str = "rank",
) -> torch.Tensor:
    """Return modality-normalized output energy for both V upgrades.

    The result has shape ``[2, pages]`` for INT2->INT4 and INT4->BF16.
    It is the model-visible energy of each real output correction, with the
    same difficult-query and transition emphasis as the residual objective.
    """
    if not 0.0 <= transition_weight <= 1.0:
        raise ValueError("transition_weight must lie in [0, 1]")

    def action_risk(correction: torch.Tensor) -> torch.Tensor:
        per_query = correction.float().square().mean(dim=(0, 1, 4))
        if transition_weight > 0.0 and int(correction.shape[2]) > 1:
            delta = correction[:, :, 1:, :, :] - correction[:, :, :-1, :, :]
            transition = torch.zeros_like(per_query)
            transition[1:] = delta.float().square().mean(dim=(0, 1, 4)) * (
                int(correction.shape[2]) / (int(correction.shape[2]) - 1)
            )
            per_query = (
                (1.0 - transition_weight) * per_query
                + transition_weight * transition
            )
        return _tail_mean(per_query, dim=0, fraction=query_tail_fraction)

    risk24 = _normalize_av_page_risk(
        action_risk(actions.int2_to_int4), page_modality, normalization
    )
    risk4f = _normalize_av_page_risk(
        action_risk(actions.int4_to_fp16), page_modality, normalization
    )
    return torch.stack((risk24, risk4f))


def simulate_production_key(
    key: torch.Tensor,
    token_order: torch.Tensor,
    *,
    key_bits: int = 4,
    superblock_size: int = 32,
) -> torch.Tensor:
    """Return logical-order K after the deployed physical-order roundtrip.

    Production K calibration blocks are formed *after* pages are reordered by
    V tier.  ``token_order`` is therefore part of the quantizer state even
    though it adds no persistent metadata.  This helper uses the canonical
    runtime quantizer so calibration and cache packing cannot silently drift.
    """
    if key.ndim == 4:
        if int(key.shape[0]) != 1:
            raise ValueError("production K simulation supports batch size 1")
        logical_key = key[0]
    elif key.ndim == 3:
        logical_key = key
    else:
        raise ValueError("key must be [Hkv,S,D] or [1,Hkv,S,D]")
    if not torch.is_floating_point(logical_key):
        raise ValueError("key must be floating point")
    tokens = int(logical_key.shape[1])
    order = token_order.flatten().to(device=logical_key.device, dtype=torch.long)
    if int(order.numel()) != tokens:
        raise ValueError("token_order must contain every K token")
    if bool((order < 0).any()) or bool((order >= tokens).any()):
        raise ValueError("token_order contains an out-of-range index")
    if int(torch.unique(order).numel()) != tokens:
        raise ValueError("token_order must be a permutation")

    # Local import preserves the policy/runtime dependency boundary during
    # package initialization while still sharing the exact deployed affine
    # quantizer implementation.
    if key_bits == 2:
        from ..runtime.int2_reference import quantize_int2_key_per_channel

        ordered = logical_key.index_select(1, order).contiguous()
        restored_ordered = quantize_int2_key_per_channel(
            ordered, superblock_size=superblock_size
        ).dequantize()
    elif key_bits == 4 and logical_key.is_cuda:
        # Reuse the fused production gather/pack kernel on GPU. This avoids the
        # reference quantizer's Python loop over superblocks and calibrates the
        # exact FP16-scale/uint8-zero representation consumed by decode.
        from ..runtime.compact_int4_key import build_compact_int4_key_arena

        empty = torch.empty(0, device=logical_key.device, dtype=torch.long)
        arena = build_compact_int4_key_arena(
            logical_key,
            (order, empty, empty),
            superblock_size=superblock_size,
        )
        restored_ordered = arena.dequantize()
    elif key_bits == 4:
        from ..runtime.compact_int4_key import quantize_int4_key_per_channel

        ordered = logical_key.index_select(1, order).contiguous()
        packed, scale, zero = quantize_int4_key_per_channel(
            ordered, superblock_size=superblock_size
        )
        codes = torch.stack((packed & 15, (packed >> 4) & 15), dim=-1).reshape_as(
            ordered
        )
        blocks = torch.arange(
            int(ordered.shape[1]), device=ordered.device
        ) // int(superblock_size)
        restored_ordered = (
            codes.float() - zero.index_select(1, blocks).float()
        ) * scale.index_select(1, blocks).float()
    else:
        raise ValueError("production key simulation supports INT2 or INT4")
    restored = torch.empty_like(restored_ordered)
    restored.index_copy_(1, order, restored_ordered)
    return restored

def prepare_k_conditioned_value_context(
    q: torch.Tensor,
    k: torch.Tensor,
    value: torch.Tensor,
    page_offsets: torch.Tensor,
    page_lengths: torch.Tensor,
    *,
    int2_group_size: int = 32,
    epsilon: float = 1e-8,
    reference_attention_probabilities: torch.Tensor | None = None,
) -> KConditionedValueContext:
    """Precompute V reconstructions and the FP target once per scoring layer."""
    if q.ndim == 3:
        q = q.unsqueeze(2)
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError("q and k must be [B,Hq,Q,D] and [B,Hkv,S,D]")
    if int(q.shape[0]) != 1 or int(k.shape[0]) != 1:
        raise ValueError("residual calibration currently supports batch size 1")
    if value.ndim == 4:
        if int(value.shape[0]) != 1:
            raise ValueError("value batch size must be 1")
        value = value[0]
    if value.ndim != 3 or not torch.is_floating_point(value):
        raise ValueError("value must be floating [Hkv,S,D] or [1,Hkv,S,D]")
    heads, tokens, head_dim = map(int, value.shape)
    if tuple(k.shape[1:]) != (heads, tokens, head_dim):
        raise ValueError("K and V cache shapes are incompatible")
    if int(q.shape[-1]) != head_dim or int(q.shape[1]) % heads:
        raise ValueError("query shape is incompatible with GQA K/V heads")
    if int2_group_size <= 0 or head_dim % int2_group_size:
        raise ValueError("int2_group_size must divide head_dim")
    _, _, page_indices, valid = _validate_page_geometry(
        page_offsets, page_lengths, tokens=tokens, device=value.device
    )

    if reference_attention_probabilities is None:
        all_tokens = torch.arange(tokens, device=value.device, dtype=torch.long)
        reference_attention = compute_query_attention_probabilities(q, k, all_tokens)
    else:
        reference_attention = reference_attention_probabilities
        expected = (int(q.shape[0]), int(q.shape[1]), int(q.shape[2]), tokens)
        if tuple(reference_attention.shape) != expected:
            raise ValueError("cached reference attention has incompatible shape")
    restored2, restored4 = _production_value_reconstructions(
        value, int2_group_size=int2_group_size, epsilon=epsilon
    )
    hq = int(q.shape[1])
    kv_head = torch.arange(hq, device=value.device) // (hq // heads)
    reference_output = _grouped_gqa_readout(reference_attention, value)
    return KConditionedValueContext(
        reference_output=reference_output,
        int2_value=restored2,
        int2_to_int4_delta=restored4 - restored2,
        int4_to_fp16_delta=value.float() - restored4,
        kv_head=kv_head,
        page_indices=page_indices,
        page_valid=valid,
        q_shape=tuple(int(size) for size in q.shape),
        key_shape=tuple(int(size) for size in k.shape),
    )


def compute_k_conditioned_value_residual_actions(
    q: torch.Tensor,
    k: torch.Tensor,
    value: torch.Tensor,
    page_offsets: torch.Tensor,
    page_lengths: torch.Tensor,
    key_token_order: torch.Tensor,
    *,
    key_bits: int = 4,
    key_superblock_size: int = 32,
    int2_group_size: int = 32,
    epsilon: float = 1e-8,
    reference_attention_probabilities: torch.Tensor | None = None,
    value_context: KConditionedValueContext | None = None,
) -> ResidualReadoutActions:
    """Build residual-directed V actions conditional on deployed INT4 K.

    The FP target is ``softmax(Q K_fp^T) V_fp``.  The base state is the actual
    ``softmax(Q K_int4^T) V_int2`` readout, so the residual includes K-induced
    attention drift. Each page action is a signed output-space correction,
    allowing a later allocator to reward helpful K/V error cancellation and
    reject locally accurate but globally harmful upgrades.
    """
    if q.ndim == 3:
        q = q.unsqueeze(2)
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError("q and k must be [B,Hq,Q,D] and [B,Hkv,S,D]")
    context = value_context or prepare_k_conditioned_value_context(
        q,
        k,
        value,
        page_offsets,
        page_lengths,
        int2_group_size=int2_group_size,
        epsilon=epsilon,
        reference_attention_probabilities=reference_attention_probabilities,
    )
    if tuple(int(size) for size in q.shape) != context.q_shape or tuple(
        int(size) for size in k.shape
    ) != context.key_shape:
        raise ValueError("value_context is incompatible with q/k shapes")
    heads, tokens, head_dim = context.key_shape[1:]
    all_tokens = torch.arange(tokens, device=k.device, dtype=torch.long)
    quantized_key = simulate_production_key(
        k,
        key_token_order,
        key_bits=key_bits,
        superblock_size=key_superblock_size,
    )
    quantized_attention = compute_query_attention_probabilities(
        q, quantized_key.unsqueeze(0), all_tokens
    )
    base_output = _grouped_gqa_readout(
        quantized_attention, context.int2_value
    )
    residual = context.reference_output - base_output

    page_attention = quantized_attention[..., context.page_indices]
    page_attention = page_attention.masked_fill(
        ~context.page_valid.view(1, 1, 1, *context.page_valid.shape), 0.0
    )

    def page_corrections(delta: torch.Tensor) -> torch.Tensor:
        gathered = delta[:, context.page_indices, :]
        gathered = gathered.masked_fill(
            ~context.page_valid[None, :, :, None], 0.0
        )
        batch, query_heads, queries, pages, width = map(
            int, page_attention.shape
        )
        kv_heads = int(gathered.shape[0])
        groups = query_heads // kv_heads
        grouped_attention = page_attention.reshape(
            batch, kv_heads, groups, queries, pages, width
        )
        correction = torch.einsum(
            "bhgqpw,hpwd->bhgqpd", grouped_attention, gathered
        )
        return correction.reshape(
            batch, query_heads, queries, pages, int(gathered.shape[-1])
        )

    return ResidualReadoutActions(
        residual=residual,
        int2_to_int4=page_corrections(context.int2_to_int4_delta),
        int4_to_fp16=page_corrections(context.int4_to_fp16_delta),
        key_reconstruction_mse=(k[0].float() - quantized_key).square().mean(),
    )
