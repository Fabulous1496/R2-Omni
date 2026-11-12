import torch

from r2omni.sspa.policy.four_quadrant import PageClass
from r2omni.sspa.runtime.cache_plan import LayerCompressionPlan
from r2omni.sspa.runtime.dynamic_fourq_cache import (
    DynamicCacheStats,
    append_generated_token,
)
from r2omni.sspa.runtime.precision_pool import (
    build_three_precision_cache,
    precision_cache_active_storage_breakdown,
    precision_cache_storage_breakdown,
)


def _plan(*, recent_capacity: int = 128) -> LayerCompressionPlan:
    page_size = 32
    classes = torch.tensor(
        [
            PageClass.FP_PAGE,
            PageClass.INT4_PAGE,
            PageClass.INT2_PAGE,
            PageClass.EVICTED_PAGE,
        ],
        dtype=torch.long,
    )
    positions = torch.arange(page_size * len(classes), dtype=torch.long)
    return LayerCompressionPlan(
        layer_id=0,
        fp_pages=torch.tensor([0]),
        int4_pages=torch.tensor([1]),
        int2_pages=torch.tensor([2]),
        evicted_pages=torch.tensor([3]),
        protected_pages=torch.empty(0, dtype=torch.long),
        recent_fp_capacity=recent_capacity,
        logical_positions=positions,
        global_token_indices=positions.clone(),
        page_offsets=torch.arange(len(classes), dtype=torch.long) * page_size,
        page_lengths=torch.full((len(classes),), page_size, dtype=torch.long),
        page_classes=classes,
    )


def _cache(
    *,
    matched_fp: bool = False,
    cache_variant: str = "v_fourq_k_int4_compact",
):
    key = torch.randn(2, 128, 128, dtype=torch.float16)
    value = torch.randn_like(key)
    return build_three_precision_cache(
        key,
        value,
        _plan(),
        matched_fp=matched_fp,
        cache_variant=cache_variant,
        compact_key_append_capacity=1024,
    )


def test_active_storage_excludes_reserved_decode_capacity() -> None:
    cache = _cache()
    active = precision_cache_active_storage_breakdown(cache)
    allocated = precision_cache_storage_breakdown(cache)

    assert active["persistent_cache_bytes"] < allocated["persistent_cache_bytes"]
    assert active["kv_payload_bytes"] < allocated["kv_payload_bytes"]
    assert cache.compact_key is not None
    assert cache.compact_key.active_nbytes < cache.compact_key.nbytes


def test_active_storage_tracks_append_and_flush_without_reallocation() -> None:
    cache = _cache()
    initial_active = precision_cache_active_storage_breakdown(cache)
    initial_allocated = precision_cache_storage_breakdown(cache)
    stats = DynamicCacheStats()
    token = torch.randn(2, 1, 128, dtype=torch.float16)

    for index in range(129):
        append_generated_token(
            cache,
            token,
            token,
            global_token_index=128 + index,
            logical_position=128 + index,
            stats=stats,
        )

    final_active = precision_cache_active_storage_breakdown(cache)
    final_allocated = precision_cache_storage_breakdown(cache)
    assert stats.flushed_tokens == 128
    assert cache.recent_fp_pool.active_length == 1
    assert final_active["persistent_cache_bytes"] > initial_active["persistent_cache_bytes"]
    assert final_allocated["persistent_cache_bytes"] == initial_allocated["persistent_cache_bytes"]


def test_matched_fp_recent_pool_counts_only_active_tokens() -> None:
    cache = _cache(matched_fp=True)
    active = precision_cache_active_storage_breakdown(cache)
    allocated = precision_cache_storage_breakdown(cache)

    assert cache.recent_fp_pool.active_length == 0
    assert cache.recent_fp_pool.active_payload_bytes == 0
    assert active["persistent_cache_bytes"] < allocated["persistent_cache_bytes"]


def test_active_storage_supports_compact_int2_keys() -> None:
    cache = _cache(cache_variant="v_fourq_k_int2_compact")
    active = precision_cache_active_storage_breakdown(cache)
    allocated = precision_cache_storage_breakdown(cache)

    assert cache.compact_key is not None
    assert cache.compact_key.bit_width == 2
    assert cache.compact_key.active_nbytes < cache.compact_key.nbytes
    assert active["persistent_cache_bytes"] < allocated["persistent_cache_bytes"]
