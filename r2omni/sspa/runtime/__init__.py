"""Cache lifecycle, layout, allocation, and static execution plans."""

from .packed_kv_layout import PackedKVCache, PackedKVLayer, PackedTensor
from .int2_reference import AffineInt2Tensor, Int2KVReference, build_int2_kv_reference
from .compact_int4_key import CompactInt4KeyArena, build_compact_int4_key_arena
from .precision_pool import (
    ThreePrecisionCache,
    build_three_precision_cache,
    precision_cache_active_storage_breakdown,
    precision_cache_storage_breakdown,
)
from .dynamic_fourq_cache import DynamicCacheStats, append_generated_token

__all__ = [
    "AffineInt2Tensor",
    "Int2KVReference",
    "PackedKVCache",
    "PackedKVLayer",
    "PackedTensor",
    "build_int2_kv_reference",
    "CompactInt4KeyArena",
    "build_compact_int4_key_arena",
    "ThreePrecisionCache",
    "build_three_precision_cache",
    "precision_cache_active_storage_breakdown",
    "precision_cache_storage_breakdown",
    "DynamicCacheStats",
    "append_generated_token",
]
