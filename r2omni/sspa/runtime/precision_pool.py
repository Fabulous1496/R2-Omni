from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from ..policy.four_quadrant import PageClass
from .cache_plan import LayerCompressionPlan
from .compact_int2_key import CompactInt2KeyArena, build_compact_int2_key_arena
from .compact_int4_key import CompactInt4KeyArena, build_compact_int4_key_arena
from .fused_value_pack import (
    CompactInt2ValueArena,
    CompactInt4ValueArena,
    build_compact_int4_value_arena,
    build_compact_int2_value_arena,
    fused_gather_pack_int4_value,
)
from .int2_reference import (
    AffineInt2Tensor,
    quantize_int2_key,
    quantize_int2_key_per_channel,
    quantize_int2_value,
)
from .packed_kv_layout import PackedTensor, make_packed_tensor


CacheVariant = Literal[
    "v_fourq_k_int2_compact",
    "v_fourq_k_int4_compact",
    "v_fourq_k_bf16",
]


@dataclass(frozen=True)
class PhysicalPageIndex:
    original_page_ids: torch.Tensor
    physical_offsets: torch.Tensor
    lengths: torch.Tensor
    global_token_indices: torch.Tensor
    logical_positions: torch.Tensor


@dataclass
class CompactMetadataArena:
    """One backing allocation for compact-cache page and token metadata."""

    storage: torch.Tensor
    int2_original_page_ids_buffer: torch.Tensor
    int2_physical_offsets_buffer: torch.Tensor
    int2_lengths_buffer: torch.Tensor
    int2_global_index_buffer: torch.Tensor
    int2_logical_position_buffer: torch.Tensor
    int2_page_count: int
    int2_token_count: int

    int4_original_page_ids_buffer: torch.Tensor
    int4_physical_offsets_buffer: torch.Tensor
    int4_lengths_buffer: torch.Tensor
    int4_global_index_buffer: torch.Tensor
    int4_logical_position_buffer: torch.Tensor
    int4_page_count: int
    int4_token_count: int

    def _append_pages(
        self,
        pool: "PrecisionPool",
        global_indices: torch.Tensor,
        logical_positions: torch.Tensor,
        page_size: int,
        *,
        precision: Literal["int2", "int4"],
    ) -> None:
        token_count = int(global_indices.numel())
        if (
            token_count <= 0
            or token_count != int(logical_positions.numel())
            or token_count % int(page_size)
        ):
            raise ValueError(
                f"compact {precision.upper()} metadata append requires matching "
                "complete pages"
            )
        if (
            global_indices.dtype != torch.long
            or logical_positions.dtype != torch.long
            or global_indices.device != self.storage.device
            or logical_positions.device != self.storage.device
        ):
            raise ValueError(
                f"compact {precision.upper()} metadata append requires same-device "
                "int64 indices"
            )
        page_count = getattr(self, f"{precision}_page_count")
        current_tokens = getattr(self, f"{precision}_token_count")
        original = getattr(self, f"{precision}_original_page_ids_buffer")
        offsets = getattr(self, f"{precision}_physical_offsets_buffer")
        lengths = getattr(self, f"{precision}_lengths_buffer")
        global_buffer = getattr(self, f"{precision}_global_index_buffer")
        logical_buffer = getattr(self, f"{precision}_logical_position_buffer")
        pages = token_count // int(page_size)
        page_end = page_count + pages
        token_end = current_tokens + token_count
        if page_end > lengths.numel():
            raise RuntimeError(f"compact {precision.upper()} page metadata capacity exhausted")
        if token_end > global_buffer.numel():
            raise RuntimeError(f"compact {precision.upper()} token metadata capacity exhausted")
        token_slice = slice(current_tokens, token_end)
        global_buffer[token_slice].copy_(global_indices)
        logical_buffer[token_slice].copy_(logical_positions)
        setattr(self, f"{precision}_page_count", page_end)
        setattr(self, f"{precision}_token_count", token_end)
        pool.page_index = PhysicalPageIndex(
            original_page_ids=original[:page_end],
            physical_offsets=offsets[:page_end],
            lengths=lengths[:page_end],
            global_token_indices=global_buffer[:token_end],
            logical_positions=logical_buffer[:token_end],
        )

    def append_int2_pages(
        self,
        pool: "PrecisionPool",
        global_indices: torch.Tensor,
        logical_positions: torch.Tensor,
        page_size: int,
    ) -> None:
        self._append_pages(
            pool,
            global_indices,
            logical_positions,
            page_size,
            precision="int2",
        )

    def append_int4_pages(
        self,
        pool: "PrecisionPool",
        global_indices: torch.Tensor,
        logical_positions: torch.Tensor,
        page_size: int,
    ) -> None:
        self._append_pages(
            pool,
            global_indices,
            logical_positions,
            page_size,
            precision="int4",
        )


@dataclass
class PrecisionPool:
    name: str
    page_index: PhysicalPageIndex
    fp_k: torch.Tensor | None = None
    fp_v: torch.Tensor | None = None
    packed_k: PackedTensor | AffineInt2Tensor | None = None
    packed_v: (
        PackedTensor | AffineInt2Tensor | CompactInt2ValueArena
        | CompactInt4ValueArena | None
    ) = None
    active_length: int | None = None
    capacity: int = 0
    length_device: torch.Tensor | None = None
    global_index_buffer: torch.Tensor | None = None
    logical_position_buffer: torch.Tensor | None = None

    @property
    def token_count(self) -> int:
        if self.active_length is not None:
            return int(self.active_length)
        return int(self.page_index.global_token_indices.numel())

    @property
    def payload_bytes(self) -> int:
        fp_k_bytes = 0 if self.fp_k is None else self.fp_k.numel() * self.fp_k.element_size()
        fp_v_bytes = 0 if self.fp_v is None else self.fp_v.numel() * self.fp_v.element_size()
        packed_k_bytes = 0 if self.packed_k is None else self.packed_k.nbytes
        packed_v_bytes = 0 if self.packed_v is None else self.packed_v.nbytes
        return int(fp_k_bytes + fp_v_bytes + packed_k_bytes + packed_v_bytes)

    @property
    def active_payload_bytes(self) -> int:
        """Payload bytes for tokens currently stored in this pool."""
        active = self.token_count

        def fp_bytes(tensor: torch.Tensor | None) -> int:
            if tensor is None:
                return 0
            if self.active_length is None:
                return int(tensor.numel() * tensor.element_size())
            per_token = int(tensor.shape[0]) * int(tensor.shape[2])
            return int(per_token * active * tensor.element_size())

        def packed_bytes(packed: object | None) -> int:
            if packed is None:
                return 0
            active_nbytes = getattr(packed, "active_nbytes", None)
            return int(active_nbytes if active_nbytes is not None else packed.nbytes)

        return int(
            fp_bytes(self.fp_k)
            + fp_bytes(self.fp_v)
            + packed_bytes(self.packed_k)
            + packed_bytes(self.packed_v)
        )


@dataclass
class ThreePrecisionCache:
    fp_pool: PrecisionPool
    int4_pool: PrecisionPool
    int2_pool: PrecisionPool
    recent_fp_pool: PrecisionPool
    evicted_global_indices: torch.Tensor
    page_size: int
    quantization_superblock: int
    matched_fp: bool
    cache_variant: CacheVariant
    compact_key: CompactInt2KeyArena | CompactInt4KeyArena | None = None
    compact_value: CompactInt2ValueArena | None = None
    compact_int4_value: CompactInt4ValueArena | None = None
    metadata_arena: CompactMetadataArena | None = None
    recent_flush_size: int = 128
    generated_value_precision: Literal["int2", "int4"] = "int2"
    static_int4_tokens: int = 0
    generated_k_offset: int = 0

    @property
    def payload_bytes(self) -> int:
        pool_bytes = sum(
            pool.payload_bytes
            for pool in (self.fp_pool, self.int4_pool, self.int2_pool, self.recent_fp_pool)
        )
        return pool_bytes + (0 if self.compact_key is None else self.compact_key.nbytes)

    @property
    def survivor_count(self) -> int:
        return sum(pool.token_count for pool in (self.fp_pool, self.int4_pool, self.int2_pool, self.recent_fp_pool))

    @property
    def uniform_int4_key(self) -> bool:
        return (
            self.cache_variant == "v_fourq_k_int4_compact"
            and not self.matched_fp
        )

    @property
    def uniform_int2_key(self) -> bool:
        return (
            self.cache_variant == "v_fourq_k_int2_compact"
            and not self.matched_fp
        )

    @property
    def full_precision_key(self) -> bool:
        return self.cache_variant == "v_fourq_k_bf16" or self.matched_fp

    @property
    def compact_int4_key(self) -> bool:
        return self.compact_key is not None and self.compact_key.bit_width == 4

    @property
    def compact_int2_key(self) -> bool:
        return self.compact_key is not None and self.compact_key.bit_width == 2

    @property
    def compact_int2_value(self) -> bool:
        return self.compact_value is not None

    @property
    def storage_bytes(self) -> int:
        return precision_cache_storage_breakdown(self)["persistent_cache_bytes"]


def _storage_key(tensor: torch.Tensor) -> tuple[str, int | None, int]:
    storage = tensor.untyped_storage()
    return tensor.device.type, tensor.device.index, int(storage.data_ptr())


def _unique_storage_bytes(tensors: list[torch.Tensor | None]) -> tuple[int, set[tuple[str, int | None, int]]]:
    total = 0
    seen: set[tuple[str, int | None, int]] = set()
    for tensor in tensors:
        if tensor is None:
            continue
        key = _storage_key(tensor)
        if key in seen:
            continue
        seen.add(key)
        total += int(tensor.untyped_storage().nbytes())
    return total, seen


def precision_cache_storage_breakdown(cache: ThreePrecisionCache) -> dict[str, int]:
    """Count allocated tensor capacity, including unused reserved space."""
    payload_tensors: list[torch.Tensor | None] = []
    metadata_tensors: list[torch.Tensor | None] = [
        cache.evicted_global_indices,
        None if cache.metadata_arena is None else cache.metadata_arena.storage,
    ]
    if cache.compact_key is not None:
        payload_tensors.extend(cache.compact_key.allocation_tensors())
    for pool in (cache.fp_pool, cache.int4_pool, cache.int2_pool, cache.recent_fp_pool):
        payload_tensors.extend((pool.fp_k, pool.fp_v))
        for packed in (pool.packed_k, pool.packed_v):
            if packed is not None:
                payload_tensors.extend((packed.packed, packed.scale, packed.zero_point))
        metadata_tensors.extend((
            pool.page_index.original_page_ids,
            pool.page_index.physical_offsets,
            pool.page_index.lengths,
            pool.page_index.global_token_indices,
            pool.page_index.logical_positions,
            pool.length_device,
            pool.global_index_buffer,
            pool.logical_position_buffer,
        ))
    payload_bytes, payload_storages = _unique_storage_bytes(payload_tensors)
    metadata_bytes = 0
    metadata_storages: set[tuple[str, int | None, int]] = set()
    for tensor in metadata_tensors:
        if tensor is None:
            continue
        key = _storage_key(tensor)
        if key in payload_storages or key in metadata_storages:
            continue
        metadata_storages.add(key)
        metadata_bytes += int(tensor.untyped_storage().nbytes())
    return {
        "kv_payload_bytes": payload_bytes,
        "kv_payload_allocation_count": len(payload_storages),
        "metadata_bytes": metadata_bytes,
        "metadata_allocation_count": len(metadata_storages),
        "persistent_cache_bytes": payload_bytes + metadata_bytes,
        "persistent_allocation_count": len(payload_storages) + len(metadata_storages),
    }


def precision_cache_active_storage_breakdown(
    cache: ThreePrecisionCache,
) -> dict[str, int]:
    """Count populated KV payload and the metadata needed to address it.

    Unlike :func:`precision_cache_storage_breakdown`, this excludes unused
    append capacity reserved for future decode tokens.  It is the appropriate
    numerator for end-of-generation KV compression; allocated capacity remains
    available separately for memory auditing.
    """
    pools = (cache.fp_pool, cache.int4_pool, cache.int2_pool, cache.recent_fp_pool)
    payload_bytes = sum(pool.active_payload_bytes for pool in pools)
    if cache.compact_key is not None:
        payload_bytes += cache.compact_key.active_nbytes

    metadata_bytes = int(
        cache.evicted_global_indices.numel()
        * cache.evicted_global_indices.element_size()
    )
    metadata_fields = 0
    for pool in pools:
        token_count = pool.token_count
        if pool.active_length is None:
            page_count = int(pool.page_index.lengths.numel())
        else:
            page_count = (
                token_count + cache.page_size - 1
            ) // cache.page_size
        page_tensors = (
            pool.page_index.original_page_ids,
            pool.page_index.physical_offsets,
            pool.page_index.lengths,
        )
        token_tensors = (
            pool.page_index.global_token_indices,
            pool.page_index.logical_positions,
        )
        metadata_bytes += sum(
            page_count * tensor.element_size() for tensor in page_tensors
        )
        metadata_bytes += sum(
            token_count * tensor.element_size() for tensor in token_tensors
        )
        metadata_fields += sum(page_count > 0 for _ in page_tensors)
        metadata_fields += sum(token_count > 0 for _ in token_tensors)
        if pool.length_device is not None:
            metadata_bytes += pool.length_device.element_size()
            metadata_fields += 1

    return {
        "kv_payload_bytes": int(payload_bytes),
        "kv_payload_allocation_count": 0,
        "metadata_bytes": int(metadata_bytes),
        "metadata_allocation_count": int(metadata_fields),
        "persistent_cache_bytes": int(payload_bytes + metadata_bytes),
        "persistent_allocation_count": int(metadata_fields),
    }


def _coalesce_compact_metadata(cache: ThreePrecisionCache) -> CompactMetadataArena:
    """Copy metadata into one arena with in-place generated low-bit tails."""
    if cache.compact_value is None:
        raise ValueError("metadata coalescing requires a compact INT2 value cache")
    key_append_tokens = (
        0
        if cache.compact_key is None
        else cache.compact_key.capacity - cache.compact_key.active_length
    )
    int2_append_tokens = cache.compact_value.capacity - cache.compact_value.active_length
    int2_append_tokens = min(int2_append_tokens, key_append_tokens)
    int4_append_tokens = (
        0
        if cache.compact_int4_value is None
        else min(
            cache.compact_int4_value.capacity
            - cache.compact_int4_value.active_length,
            key_append_tokens,
        )
    )
    int2_append_tokens = max(
        0, int(int2_append_tokens) // cache.page_size * cache.page_size
    )
    int4_append_tokens = max(
        0, int(int4_append_tokens) // cache.page_size * cache.page_size
    )
    int2 = cache.int2_pool
    int2_pages = int(int2.page_index.lengths.numel())
    int2_tokens = int(int2.token_count)
    int2_page_capacity = int2_pages + int2_append_tokens // cache.page_size
    int2_token_capacity = int2_tokens + int2_append_tokens
    int4 = cache.int4_pool
    int4_pages = int(int4.page_index.lengths.numel())
    int4_tokens = int(int4.token_count)
    int4_page_capacity = int4_pages + int4_append_tokens // cache.page_size
    int4_token_capacity = int4_tokens + int4_append_tokens

    specs: list[tuple[str, torch.Tensor, int, int]] = []

    def add(name: str, source: torch.Tensor, capacity: int | None = None, fill: int = 0) -> None:
        if source.dtype != torch.long or source.ndim != 1:
            raise ValueError(f"compact metadata field {name} must be rank-1 int64")
        size = int(source.numel()) if capacity is None else int(capacity)
        if size < source.numel():
            raise ValueError(f"compact metadata capacity for {name} is too small")
        specs.append((name, source, size, int(fill)))

    add("evicted", cache.evicted_global_indices)
    pools = (cache.fp_pool, cache.int4_pool, cache.int2_pool, cache.recent_fp_pool)
    for pool in pools:
        page_capacity = (
            int2_page_capacity
            if pool is int2
            else int4_page_capacity
            if pool is int4
            else None
        )
        token_capacity = (
            int2_token_capacity
            if pool is int2
            else int4_token_capacity
            if pool is int4
            else None
        )
        add(
            f"{pool.name}.original_page_ids",
            pool.page_index.original_page_ids,
            page_capacity,
            -1,
        )
        add(
            f"{pool.name}.physical_offsets",
            pool.page_index.physical_offsets,
            page_capacity,
        )
        add(f"{pool.name}.lengths", pool.page_index.lengths, page_capacity)
        global_source = (
            pool.global_index_buffer
            if pool.global_index_buffer is not None
            else pool.page_index.global_token_indices
        )
        logical_source = (
            pool.logical_position_buffer
            if pool.logical_position_buffer is not None
            else pool.page_index.logical_positions
        )
        global_capacity = (
            int(global_source.numel()) if pool.global_index_buffer is not None else token_capacity
        )
        logical_capacity = (
            int(logical_source.numel())
            if pool.logical_position_buffer is not None
            else token_capacity
        )
        add(f"{pool.name}.global", global_source, global_capacity, -1)
        add(f"{pool.name}.logical", logical_source, logical_capacity, -1)

    length_pools = [pool for pool in pools if pool.length_device is not None]
    total_int64 = sum(size for _, _, size, _ in specs) + len(length_pools)
    storage = torch.empty(
        (max(1, total_int64),), device=cache.evicted_global_indices.device, dtype=torch.long
    )
    views: dict[str, torch.Tensor] = {}
    cursor = 0
    for name, source, size, fill in specs:
        view = storage[cursor : cursor + size]
        if size:
            view.fill_(fill)
            if source.numel():
                view[: source.numel()].copy_(source)
        views[name] = view
        cursor += size
    for precision, page_count, page_capacity, token_count in (
        ("int2", int2_pages, int2_page_capacity, int2_tokens),
        ("int4", int4_pages, int4_page_capacity, int4_tokens),
    ):
        if page_capacity <= page_count:
            continue
        future_pages = page_capacity - page_count
        views[f"{precision}.physical_offsets"][page_count:].copy_(
            token_count
            + torch.arange(future_pages, device=storage.device, dtype=torch.long)
            * cache.page_size
        )
        views[f"{precision}.lengths"][page_count:].fill_(cache.page_size)
    length_views: dict[str, torch.Tensor] = {}
    for pool in length_pools:
        raw = storage[cursor : cursor + 1]
        length = raw.view(torch.int32)[:1]
        length.copy_(pool.length_device)
        length_views[pool.name] = length
        cursor += 1

    cache.evicted_global_indices = views["evicted"]
    for pool in pools:
        page_count = (
            int2_pages
            if pool is int2
            else int4_pages
            if pool is int4
            else int(pool.page_index.lengths.numel())
        )
        token_count = (
            int2_tokens
            if pool is int2
            else int4_tokens
            if pool is int4
            else int(pool.page_index.global_token_indices.numel())
        )
        global_view = views[f"{pool.name}.global"]
        logical_view = views[f"{pool.name}.logical"]
        pool.page_index = PhysicalPageIndex(
            original_page_ids=views[f"{pool.name}.original_page_ids"][:page_count],
            physical_offsets=views[f"{pool.name}.physical_offsets"][:page_count],
            lengths=views[f"{pool.name}.lengths"][:page_count],
            global_token_indices=(
                global_view if pool.global_index_buffer is not None else global_view[:token_count]
            ),
            logical_positions=(
                logical_view
                if pool.logical_position_buffer is not None
                else logical_view[:token_count]
            ),
        )
        if pool.global_index_buffer is not None:
            pool.global_index_buffer = global_view
        if pool.logical_position_buffer is not None:
            pool.logical_position_buffer = logical_view
        if pool.length_device is not None:
            pool.length_device = length_views[pool.name]

    return CompactMetadataArena(
        storage=storage,
        int2_original_page_ids_buffer=views["int2.original_page_ids"],
        int2_physical_offsets_buffer=views["int2.physical_offsets"],
        int2_lengths_buffer=views["int2.lengths"],
        int2_global_index_buffer=views["int2.global"],
        int2_logical_position_buffer=views["int2.logical"],
        int2_page_count=int2_pages,
        int2_token_count=int2_tokens,
        int4_original_page_ids_buffer=views["int4.original_page_ids"],
        int4_physical_offsets_buffer=views["int4.physical_offsets"],
        int4_lengths_buffer=views["int4.lengths"],
        int4_global_index_buffer=views["int4.global"],
        int4_logical_position_buffer=views["int4.logical"],
        int4_page_count=int4_pages,
        int4_token_count=int4_tokens,
    )


def _page_tokens(plan: LayerCompressionPlan, page_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    token_parts = []
    offsets = []
    lengths = []
    cursor = 0
    for page_id in page_ids.tolist():
        start = int(plan.page_offsets[page_id])
        length = int(plan.page_lengths[page_id])
        token_parts.append(torch.arange(start, start + length, device=plan.page_offsets.device))
        offsets.append(cursor)
        lengths.append(length)
        cursor += length
    tokens = torch.cat(token_parts) if token_parts else torch.empty(0, dtype=torch.long, device=plan.page_offsets.device)
    return (
        tokens,
        torch.tensor(offsets, dtype=torch.long, device=tokens.device),
        torch.tensor(lengths, dtype=torch.long, device=tokens.device),
        page_ids.to(torch.long),
    )


def _compact_page_layouts(
    plan: LayerCompressionPlan,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """Build every compact pool layout with one small device-to-host count copy."""
    classes = plan.page_classes
    category = torch.full_like(classes, 3, dtype=torch.long)
    fp = (
        (classes == int(PageClass.FP_PAGE))
        | (classes == int(PageClass.PROTECTED_FP_PAGE))
        | (classes == int(PageClass.RECENT_FP_PAGE))
    )
    category[fp] = 0
    category[classes == int(PageClass.INT4_PAGE)] = 1
    category[classes == int(PageClass.INT2_PAGE)] = 2

    page_order = torch.argsort(category, stable=True)
    ordered_lengths = plan.page_lengths.index_select(0, page_order)
    ordered_starts = plan.page_offsets.index_select(0, page_order)
    ordered_offsets = ordered_lengths.cumsum(0) - ordered_lengths
    sequence_length = int(plan.logical_positions.numel())
    repeated_starts = torch.repeat_interleave(
        ordered_starts, ordered_lengths, output_size=sequence_length
    )
    repeated_offsets = torch.repeat_interleave(
        ordered_offsets, ordered_lengths, output_size=sequence_length
    )
    token_order = repeated_starts + torch.arange(
        sequence_length, device=classes.device, dtype=torch.long
    ) - repeated_offsets

    page_counts = torch.bincount(category, minlength=4)
    token_counts = torch.zeros(4, device=classes.device, dtype=torch.long)
    token_counts.scatter_add_(0, category, plan.page_lengths)
    counts = torch.cat((page_counts, token_counts)).tolist()
    page_counts_cpu = [int(value) for value in counts[:4]]
    token_counts_cpu = [int(value) for value in counts[4:]]

    layouts = []
    page_cursor = 0
    token_cursor = 0
    for page_count, token_count in zip(page_counts_cpu, token_counts_cpu):
        pages = page_order[page_cursor : page_cursor + page_count]
        lengths = ordered_lengths[page_cursor : page_cursor + page_count]
        offsets = (
            ordered_offsets[page_cursor : page_cursor + page_count] - token_cursor
        )
        tokens = token_order[token_cursor : token_cursor + token_count]
        layouts.append((tokens, offsets, lengths, pages))
        page_cursor += page_count
        token_cursor += token_count
    empty = plan.page_offsets[:0]
    recent_layout = (empty, empty, empty, empty)
    return layouts[0], layouts[1], layouts[2], recent_layout, layouts[3]


def precision_ordered_token_indices(plan: LayerCompressionPlan) -> torch.Tensor:
    """Return the exact FP16-V/INT4-V/INT2-V physical K calibration order."""
    fp_layout, i4_layout, i2_layout, _, evicted_layout = _compact_page_layouts(plan)
    return torch.cat(
        (fp_layout[0], i4_layout[0], i2_layout[0], evicted_layout[0])
    )


def _make_pool(
    name: str,
    key: torch.Tensor,
    value: torch.Tensor,
    plan: LayerCompressionPlan,
    page_ids: torch.Tensor,
    *,
    value_bits: int,
    key_bits: int,
    matched_fp: bool,
    int2_group_size: int,
    strict_per_channel_key: bool = False,
    skip_key: bool = False,
    skip_value: bool = False,
    page_layout: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> PrecisionPool:
    tokens, offsets, lengths, original_pages = (
        _page_tokens(plan, page_ids) if page_layout is None else page_layout
    )
    page_index = PhysicalPageIndex(
        original_page_ids=original_pages,
        physical_offsets=offsets,
        lengths=lengths,
        global_token_indices=plan.global_token_indices.index_select(0, tokens),
        logical_positions=plan.logical_positions.index_select(0, tokens),
    )
    if tokens.numel() == 0:
        return PrecisionPool(name=name, page_index=page_index)
    selected_v = None if skip_value else value.index_select(1, tokens).contiguous()
    selected_k = None if skip_key else key.index_select(1, tokens).contiguous()
    if matched_fp:
        assert selected_k is not None and selected_v is not None
        return PrecisionPool(name=name, page_index=page_index, fp_k=selected_k, fp_v=selected_v)
    pool = PrecisionPool(name=name, page_index=page_index)
    if skip_key:
        pass
    elif key_bits == 16:
        assert selected_k is not None
        pool.fp_k = selected_k
    elif key_bits == 4:
        assert selected_k is not None
        pool.packed_k = make_packed_tensor(selected_k, 4, "per_channel", 1)
    elif key_bits == 2:
        assert selected_k is not None
        pool.packed_k = (
            quantize_int2_key_per_channel(selected_k)
            if strict_per_channel_key
            else quantize_int2_key(selected_k, group_size=int2_group_size)
        )
    else:
        raise ValueError("key_bits must be 16, 4, or 2")
    if skip_value:
        pass
    elif value_bits == 16:
        assert selected_v is not None
        pool.fp_v = selected_v
    elif value_bits == 4:
        assert selected_v is not None
        pool.packed_v = make_packed_tensor(selected_v, 4, "per_token", 2)
    elif value_bits == 2:
        assert selected_v is not None
        pool.packed_v = quantize_int2_value(selected_v, group_size=int2_group_size)
    else:
        raise ValueError("value_bits must be 16, 4, or 2")
    return pool


def _preallocate_recent_pool(pool: PrecisionPool, key: torch.Tensor, *, page_size: int, capacity: int = 256) -> PrecisionPool:
    active = pool.token_count
    if capacity < page_size or capacity % page_size:
        raise ValueError("Recent-FP capacity must be positive and page aligned")
    if active >= capacity:
        raise ValueError("initial page-rounded recent FP region exceeds dynamic capacity")
    h, d = int(key.shape[0]), int(key.shape[2])
    k_buffer = torch.zeros((h, capacity, d), device=key.device, dtype=key.dtype)
    v_buffer = torch.zeros_like(k_buffer)
    if active:
        assert pool.fp_k is not None and pool.fp_v is not None
        k_buffer[:, :active].copy_(pool.fp_k)
        v_buffer[:, :active].copy_(pool.fp_v)
    global_buffer = torch.full((capacity,), -1, device=key.device, dtype=torch.long)
    logical_buffer = torch.full((capacity,), -1, device=key.device, dtype=torch.long)
    if active:
        global_buffer[:active].copy_(pool.page_index.global_token_indices[:active])
        logical_buffer[:active].copy_(pool.page_index.logical_positions[:active])
    num_pages = capacity // page_size
    offsets = torch.arange(num_pages, device=key.device, dtype=torch.long) * page_size
    lengths = torch.zeros(num_pages, device=key.device, dtype=torch.long)
    for page in range(num_pages):
        lengths[page] = max(0, min(page_size, active - page * page_size))
    pool.fp_k = k_buffer
    pool.fp_v = v_buffer
    pool.active_length = active
    pool.capacity = capacity
    pool.length_device = torch.tensor(active, device=key.device, dtype=torch.int32)
    pool.global_index_buffer = global_buffer
    pool.logical_position_buffer = logical_buffer
    pool.page_index = PhysicalPageIndex(
        original_page_ids=torch.full((num_pages,), -1, device=key.device, dtype=torch.long),
        physical_offsets=offsets,
        lengths=lengths,
        global_token_indices=global_buffer,
        logical_positions=logical_buffer,
    )
    return pool


def build_three_precision_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    plan: LayerCompressionPlan,
    *,
    page_size: int = 32,
    quantization_superblock: int = 128,
    key_quantization_superblock: int = 32,
    int2_group_size: int = 32,
    generated_value_precision: Literal["int2", "int4"] = "int2",
    matched_fp: bool = False,
    preallocate_recent: bool = True,
    cache_variant: CacheVariant = "v_fourq_k_int4_compact",
    compact_key_append_capacity: int | None = None,
    recent_flush_size: int = 128,
) -> ThreePrecisionCache:
    """Stable page reorder into continuous FP/INT4/INT2/Recent-FP pools."""
    if key.shape != value.shape or key.ndim != 3:
        raise ValueError("key/value must be logical [Hkv,S,D]")
    if int(key.shape[1]) != int(plan.logical_positions.numel()):
        raise ValueError("K/V sequence length must match the static plan")
    if page_size not in {16, 32, 64}:
        raise ValueError("page_size must be 16, 32, or 64")
    if quantization_superblock != 128:
        raise ValueError("formal precision pools use 128-token quantization superblocks")
    if key_quantization_superblock not in {32, 64, 128}:
        raise ValueError("K quantization superblock must be 32, 64, or 128")
    if (
        recent_flush_size not in {32, 64, 128}
        or recent_flush_size % page_size
        or recent_flush_size % key_quantization_superblock
    ):
        raise ValueError(
            "recent_flush_size must be 32/64/128 and align to page/K blocks"
        )
    if preallocate_recent and plan.recent_fp_capacity < recent_flush_size:
        raise ValueError("Recent-FP capacity must cover one configured flush")
    if cache_variant not in {
        "v_fourq_k_int2_compact",
        "v_fourq_k_int4_compact",
        "v_fourq_k_bf16",
    }:
        raise ValueError(f"unsupported cache_variant: {cache_variant}")
    if generated_value_precision not in {"int2", "int4"}:
        raise ValueError("generated_value_precision must be int2 or int4")
    empty_pages = torch.empty(0, device=key.device, dtype=torch.long)
    fp_layout, i4_layout, i2_layout, recent_layout, evicted_layout = _compact_page_layouts(plan)
    if matched_fp:
        survivor_pages = (
            plan.page_classes != int(PageClass.EVICTED_PAGE)
        ).nonzero(as_tuple=False).flatten()
        cache = ThreePrecisionCache(
            fp_pool=_make_pool(
                "fp", key, value, plan, survivor_pages,
                value_bits=16, key_bits=16, matched_fp=True,
                int2_group_size=int2_group_size,
            ),
            int4_pool=_make_pool(
                "int4", key, value, plan, empty_pages,
                value_bits=16, key_bits=16, matched_fp=True,
                int2_group_size=int2_group_size,
            ),
            int2_pool=_make_pool(
                "int2", key, value, plan, empty_pages,
                value_bits=16, key_bits=16, matched_fp=True,
                int2_group_size=int2_group_size,
            ),
            recent_fp_pool=_make_pool(
                "recent_fp", key, value, plan, empty_pages,
                value_bits=16, key_bits=16, matched_fp=True,
                int2_group_size=int2_group_size,
            ),
            evicted_global_indices=plan.global_token_indices.index_select(
                0, evicted_layout[0]
            ),
            page_size=int(page_size),
            quantization_superblock=int(quantization_superblock),
            matched_fp=True,
            cache_variant=cache_variant,
            recent_flush_size=int(recent_flush_size),
            generated_value_precision=str(generated_value_precision),
        )
        if preallocate_recent:
            cache.recent_fp_pool = _preallocate_recent_pool(
                cache.recent_fp_pool,
                key,
                page_size=page_size,
                capacity=plan.recent_fp_capacity,
            )
        return cache
    matched_history = fp_layout[3]
    full_precision_key = cache_variant == "v_fourq_k_bf16"
    recent_pool_pages = empty_pages
    cache = ThreePrecisionCache(
        fp_pool=_make_pool(
            "fp", key, value, plan, matched_history,
            value_bits=16, key_bits=16 if (matched_fp or full_precision_key) else 2,
            matched_fp=matched_fp, int2_group_size=int2_group_size,
            strict_per_channel_key=True,
            skip_key=not (matched_fp or full_precision_key),
            skip_value=False,
            page_layout=fp_layout,
        ),
        int4_pool=_make_pool(
            "int4", key, value, plan,
            i4_layout[3],
            value_bits=16 if matched_fp else 4,
            key_bits=16 if (matched_fp or full_precision_key) else 2,
            matched_fp=matched_fp, int2_group_size=int2_group_size,
            strict_per_channel_key=True,
            skip_key=not (matched_fp or full_precision_key),
            skip_value=not matched_fp,
            page_layout=i4_layout,
        ),
        int2_pool=_make_pool(
            "int2", key, value, plan,
            i2_layout[3],
            value_bits=16 if matched_fp else 2,
            key_bits=16 if (matched_fp or full_precision_key) else 2,
            matched_fp=matched_fp, int2_group_size=int2_group_size,
            strict_per_channel_key=True,
            skip_key=not (matched_fp or full_precision_key),
            skip_value=not matched_fp,
            page_layout=i2_layout,
        ),
        recent_fp_pool=_make_pool(
            "recent_fp", key, value, plan, recent_pool_pages,
            value_bits=16, key_bits=16, matched_fp=matched_fp,
            int2_group_size=int2_group_size,
            page_layout=recent_layout,
        ),
        evicted_global_indices=plan.global_token_indices.index_select(0, evicted_layout[0]),
        page_size=int(page_size),
        quantization_superblock=int(quantization_superblock),
        matched_fp=bool(matched_fp),
        cache_variant=cache_variant,
        compact_key=None,
        compact_value=None,
        metadata_arena=None,
        recent_flush_size=int(recent_flush_size),
        generated_value_precision=str(generated_value_precision),
        static_int4_tokens=int(i4_layout[0].numel()),
    )
    append_capacity = (
        plan.recent_fp_capacity
        if compact_key_append_capacity is None
        else int(compact_key_append_capacity)
    )
    if not matched_fp:
        if generated_value_precision == "int4":
            cache.compact_int4_value = build_compact_int4_value_arena(
                value,
                i4_layout[0],
                append_capacity=append_capacity,
            )
            cache.int4_pool.packed_v = cache.compact_int4_value
        elif i4_layout[0].numel():
            cache.int4_pool.packed_v = fused_gather_pack_int4_value(
                value, i4_layout[0]
            )
        cache.compact_value = build_compact_int2_value_arena(
            value,
            i2_layout[0],
            append_capacity=(
                0 if generated_value_precision == "int4" else append_capacity
            ),
            group_size=int2_group_size,
        )
        cache.int2_pool.packed_v = cache.compact_value
        if not full_precision_key:
            segment_tokens = (fp_layout[0], i4_layout[0], i2_layout[0])
            key_builder_kwargs = {
                "append_capacity": append_capacity,
                "superblock_size": key_quantization_superblock,
            }
            cache.compact_key = (
                build_compact_int2_key_arena(
                    key, segment_tokens, **key_builder_kwargs
                )
                if cache_variant == "v_fourq_k_int2_compact"
                else build_compact_int4_key_arena(
                    key, segment_tokens, **key_builder_kwargs
                )
            )
            cache.generated_k_offset = int(cache.compact_key.active_length)
    if preallocate_recent:
        cache.recent_fp_pool = _preallocate_recent_pool(
            cache.recent_fp_pool, key, page_size=page_size, capacity=plan.recent_fp_capacity
        )
    partition_count = sum(
        int(layout[0].numel())
        for layout in (fp_layout, i4_layout, i2_layout, recent_layout, evicted_layout)
    )
    if partition_count != key.shape[1]:
        raise ValueError("compact physical pools must cover every logical token")
    if not matched_fp:
        cache.metadata_arena = _coalesce_compact_metadata(cache)
    return cache
