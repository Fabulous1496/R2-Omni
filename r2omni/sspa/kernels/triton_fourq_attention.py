from __future__ import annotations

from dataclasses import dataclass

import torch

from ..runtime.precision_pool import PrecisionPool, ThreePrecisionCache


@dataclass(frozen=True)
class _PartialAttentionState:
    max_logit: torch.Tensor
    exp_sum: torch.Tensor
    weighted_value_sum: torch.Tensor


def _pool_kv(
    pool: PrecisionPool,
    *,
    key_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if pool.token_count == 0:
        return None
    key = key_override
    if key is None:
        key = (
            pool.fp_k[:, : pool.token_count].float()
            if pool.fp_k is not None
            else None if pool.packed_k is None else pool.packed_k.dequantize().float()
        )
    value = (
        pool.fp_v[:, : pool.token_count].float()
        if pool.fp_v is not None
        else None if pool.packed_v is None else pool.packed_v.dequantize().float()
    )
    if key is None or value is None:
        raise ValueError(f"non-empty {pool.name} pool has incomplete payload")
    return key, value


def _partial_state(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, sm_scale: float) -> _PartialAttentionState:
    hq, hkv = int(query.shape[1]), int(key.shape[0])
    if hq % hkv:
        raise ValueError("GQA requires Hq divisible by Hkv")
    kv_head = torch.arange(hq, device=query.device) // (hq // hkv)
    expanded_k = key.index_select(0, kv_head)
    expanded_v = value.index_select(0, kv_head)
    logits = torch.einsum("bhd,hsd->bhs", query.float(), expanded_k) * sm_scale
    maximum = logits.amax(dim=-1)
    weights = torch.exp(logits - maximum.unsqueeze(-1))
    return _PartialAttentionState(
        max_logit=maximum,
        exp_sum=weights.sum(dim=-1),
        weighted_value_sum=torch.einsum("bhs,hsd->bhd", weights, expanded_v),
    )


def _merge_state(first: _PartialAttentionState, second: _PartialAttentionState) -> _PartialAttentionState:
    maximum = torch.maximum(first.max_logit, second.max_logit)
    first_weight = torch.exp(first.max_logit - maximum)
    second_weight = torch.exp(second.max_logit - maximum)
    return _PartialAttentionState(
        max_logit=maximum,
        exp_sum=first.exp_sum * first_weight + second.exp_sum * second_weight,
        weighted_value_sum=(
            first.weighted_value_sum * first_weight.unsqueeze(-1)
            + second.weighted_value_sum * second_weight.unsqueeze(-1)
        ),
    )


def triton_mixed_reference(
    query: torch.Tensor,
    cache: ThreePrecisionCache,
    *,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """T4 oracle for FP/INT4/INT2/Recent-FP pools and evicted pages.

    This intentionally dequantizes for correctness. The production split-K
    path must never call this function.
    """
    if query.ndim == 4:
        if int(query.shape[2]) != 1:
            raise ValueError("query must have one decode token")
        query = query[:, :, 0]
    if query.ndim != 3 or int(query.shape[0]) != 1:
        raise ValueError("query must be [1,Hq,D]")
    scale = float(sm_scale) if sm_scale is not None else int(query.shape[-1]) ** -0.5
    state = None
    pools = (cache.fp_pool, cache.int4_pool, cache.int2_pool, cache.recent_fp_pool)
    for pool_index, pool in enumerate(pools):
        key_override = None
        if cache.compact_key is not None and pool_index < 3 and pool.token_count:
            if pool_index == 1 and cache.generated_value_precision == "int4":
                # Static INT4-V keys keep their original compact-K segment,
                # while generated INT4-V keys are appended after all three
                # static V tiers.  Reconstruct the logical INT4 pool without
                # accidentally reading the intervening static INT2-K segment.
                static_tokens = min(cache.static_int4_tokens, pool.token_count)
                generated_tokens = pool.token_count - static_tokens
                pieces = [
                    cache.compact_key.dequantize_range(
                        cache.compact_key.segment_offsets[1], static_tokens
                    )
                ]
                if generated_tokens:
                    pieces.append(
                        cache.compact_key.dequantize_range(
                            cache.generated_k_offset, generated_tokens
                        )
                    )
                key_override = torch.cat(pieces, dim=1)
            else:
                key_override = cache.compact_key.dequantize_range(
                    cache.compact_key.segment_offsets[pool_index], pool.token_count
                )
        payload = _pool_kv(pool, key_override=key_override)
        if payload is None:
            continue
        part = _partial_state(query, payload[0], payload[1], scale)
        state = part if state is None else _merge_state(state, part)
    if state is None:
        raise ValueError("four-quadrant cache has no surviving tokens")
    return (state.weighted_value_sum / state.exp_sum.unsqueeze(-1)).to(query.dtype)
