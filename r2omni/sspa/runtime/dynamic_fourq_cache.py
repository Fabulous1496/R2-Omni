from __future__ import annotations

from dataclasses import dataclass

import torch

from .int2_reference import (
    AffineInt2Tensor,
    build_int2_kv_reference,
    quantize_int2_key_per_channel,
    quantize_int2_value,
)
from .precision_pool import PhysicalPageIndex, PrecisionPool, ThreePrecisionCache


@dataclass
class DynamicCacheStats:
    appended_tokens: int = 0
    flushed_superblocks: int = 0
    flushed_tokens: int = 0
    flushed_int4_value_tokens: int = 0
    recent_reallocation_count: int = 0
    local_tail_dequant_tokens: int = 0
    full_dequant_tensor_count: int = 0


def _tail_tensor(tensor: AffineInt2Tensor, start: int) -> torch.Tensor:
    h, s, d = tensor.original_shape
    if start >= s:
        return torch.empty((h, 0, d), device=tensor.packed.device, dtype=torch.float32)
    d4 = d // 4
    packed = tensor.packed.reshape(h, s, d4)[:, start:].contiguous().flatten()
    if tensor.axis in {"per_channel", "per_channel_group"}:
        scale = tensor.scale[:, start // 128 :].contiguous()
        zero = tensor.zero_point[:, start // 128 :].contiguous()
    else:
        scale = tensor.scale[:, start:].contiguous()
        zero = tensor.zero_point[:, start:].contiguous()
    tail = AffineInt2Tensor(
        packed=packed,
        scale=scale,
        zero_point=zero,
        original_shape=(h, s - start, d),
        group_size=tensor.group_size,
        axis=tensor.axis,
        superblock_size=tensor.superblock_size,
    )
    return tail.dequantize()


def _append_int2_superblock(
    pool: PrecisionPool,
    key_block: torch.Tensor,
    value_block: torch.Tensor,
    *,
    stats: DynamicCacheStats,
    strict_per_channel_key: bool,
) -> None:
    append_length = int(key_block.shape[1])
    if append_length <= 0 or value_block.shape != key_block.shape:
        raise ValueError("INT2 flush must contain positive matching K/V tokens")
    if pool.token_count == 0:
        if strict_per_channel_key:
            pool.packed_k = quantize_int2_key_per_channel(key_block)
            pool.packed_v = quantize_int2_value(value_block)
        else:
            packed = build_int2_kv_reference(key_block, value_block)
            pool.packed_k, pool.packed_v = packed.key, packed.value
        return
    if not isinstance(pool.packed_k, AffineInt2Tensor) or not isinstance(pool.packed_v, AffineInt2Tensor):
        raise TypeError("INT2 pool payload is invalid")
    old_k, old_v = pool.packed_k, pool.packed_v
    h, old_s, d = old_k.original_shape
    tail_count = old_s % 128
    prefix = old_s - tail_count
    tail_k = _tail_tensor(old_k, prefix)
    tail_v = _tail_tensor(old_v, prefix)
    stats.local_tail_dequant_tokens += tail_count
    combined_k = torch.cat((tail_k.to(key_block.dtype), key_block), dim=1)
    combined_v = torch.cat((tail_v.to(value_block.dtype), value_block), dim=1)
    if strict_per_channel_key:
        rebuilt_key = quantize_int2_key_per_channel(combined_k)
        rebuilt_value = quantize_int2_value(combined_v)
    else:
        rebuilt = build_int2_kv_reference(combined_k, combined_v)
        rebuilt_key, rebuilt_value = rebuilt.key, rebuilt.value
    d4 = d // 4
    prefix_k_bytes = old_k.packed.reshape(h, old_s, d4)[:, :prefix]
    prefix_v_bytes = old_v.packed.reshape(h, old_s, d4)[:, :prefix]
    new_k_bytes = rebuilt_key.packed.reshape(h, tail_count + append_length, d4)
    new_v_bytes = rebuilt_value.packed.reshape(h, tail_count + append_length, d4)
    pool.packed_k = AffineInt2Tensor(
        packed=torch.cat((prefix_k_bytes, new_k_bytes), dim=1).contiguous().flatten(),
        scale=torch.cat((old_k.scale[:, : prefix // 128], rebuilt_key.scale), dim=1),
        zero_point=torch.cat((old_k.zero_point[:, : prefix // 128], rebuilt_key.zero_point), dim=1),
        original_shape=(h, old_s + append_length, d), group_size=old_k.group_size,
        axis=old_k.axis, superblock_size=128,
    )
    pool.packed_v = AffineInt2Tensor(
        packed=torch.cat((prefix_v_bytes, new_v_bytes), dim=1).contiguous().flatten(),
        scale=torch.cat((old_v.scale[:, :prefix], rebuilt_value.scale), dim=1),
        zero_point=torch.cat((old_v.zero_point[:, :prefix], rebuilt_value.zero_point), dim=1),
        original_shape=(h, old_s + append_length, d), group_size=old_v.group_size,
        axis=old_v.axis, superblock_size=128,
    )


def _refresh_recent_pages(pool: PrecisionPool, page_size: int) -> None:
    assert pool.active_length is not None and pool.length_device is not None
    pool.length_device.fill_(pool.active_length)
    offsets = pool.page_index.physical_offsets
    lengths = pool.page_index.lengths
    if offsets.shape != lengths.shape:
        raise ValueError("Recent-FP page offsets and lengths must have matching shapes")
    lengths.copy_(
        (int(pool.active_length) - offsets).clamp(min=0, max=int(page_size))
    )


def _append_lowbit_page_metadata(
    pool: PrecisionPool,
    global_indices: torch.Tensor,
    logical_positions: torch.Tensor,
    page_size: int,
) -> None:
    old_tokens = pool.token_count
    append_length = int(global_indices.numel())
    if append_length <= 0 or append_length != int(logical_positions.numel()):
        raise ValueError("low-bit page metadata requires matching token indices")
    if append_length % page_size:
        raise ValueError("low-bit metadata append must contain complete pages")
    pages = append_length // page_size
    offsets = old_tokens + torch.arange(pages, device=global_indices.device, dtype=torch.long) * page_size
    lengths = torch.full((pages,), page_size, device=global_indices.device, dtype=torch.long)
    old = pool.page_index
    pool.page_index = PhysicalPageIndex(
        original_page_ids=torch.cat((old.original_page_ids, torch.full((pages,), -1, device=global_indices.device, dtype=torch.long))),
        physical_offsets=torch.cat((old.physical_offsets, offsets)),
        lengths=torch.cat((old.lengths, lengths)),
        global_token_indices=torch.cat((old.global_token_indices[:old_tokens], global_indices.to(torch.long))),
        logical_positions=torch.cat((old.logical_positions[:old_tokens], logical_positions.to(torch.long))),
    )


def _append_matched_fp_block(
    pool: PrecisionPool,
    key_block: torch.Tensor,
    value_block: torch.Tensor,
    global_indices: torch.Tensor,
    logical_positions: torch.Tensor,
    page_size: int,
) -> None:
    old_tokens = pool.token_count
    pool.fp_k = key_block.contiguous() if pool.fp_k is None else torch.cat((pool.fp_k, key_block), dim=1)
    pool.fp_v = value_block.contiguous() if pool.fp_v is None else torch.cat((pool.fp_v, value_block), dim=1)
    append_length = int(key_block.shape[1])
    if append_length <= 0 or append_length % page_size:
        raise ValueError("matched-FP append must contain complete pages")
    pages = append_length // page_size
    offsets = old_tokens + torch.arange(pages, device=key_block.device, dtype=torch.long) * page_size
    lengths = torch.full((pages,), page_size, device=key_block.device, dtype=torch.long)
    old = pool.page_index
    pool.page_index = PhysicalPageIndex(
        original_page_ids=torch.cat((old.original_page_ids, torch.full((pages,), -1, device=key_block.device, dtype=torch.long))),
        physical_offsets=torch.cat((old.physical_offsets, offsets)),
        lengths=torch.cat((old.lengths, lengths)),
        global_token_indices=torch.cat((old.global_token_indices[:old_tokens], global_indices.to(torch.long))),
        logical_positions=torch.cat((old.logical_positions[:old_tokens], logical_positions.to(torch.long))),
    )


def append_generated_token(
    cache: ThreePrecisionCache,
    new_key: torch.Tensor,
    new_value: torch.Tensor,
    *,
    global_token_index: int,
    logical_position: int,
    stats: DynamicCacheStats,
) -> bool:
    """Append to fixed Recent-FP storage; flush one configured old-token chunk."""
    pool = cache.recent_fp_pool
    flush_size = int(cache.recent_flush_size)
    if (
        pool.active_length is None
        or pool.capacity < flush_size
        or pool.length_device is None
    ):
        raise RuntimeError("Recent-FP pool was not preallocated for dynamic generation")
    if new_key.ndim == 4:
        new_key, new_value = new_key[0], new_value[0]
    if new_key.shape != new_value.shape or new_key.ndim != 3 or int(new_key.shape[1]) != 1:
        raise ValueError("new K/V must be [Hkv,1,D] or [1,Hkv,1,D]")
    flushed = False
    if pool.active_length == pool.capacity:
        assert pool.fp_k is not None and pool.fp_v is not None
        assert pool.global_index_buffer is not None and pool.logical_position_buffer is not None
        key_block = pool.fp_k[:, :flush_size].clone()
        value_block = pool.fp_v[:, :flush_size].clone()
        global_block = pool.global_index_buffer[:flush_size].clone()
        logical_block = pool.logical_position_buffer[:flush_size].clone()
        if cache.matched_fp:
            _append_matched_fp_block(
                cache.fp_pool, key_block, value_block,
                global_block, logical_block, cache.page_size,
            )
        else:
            metadata_pool = cache.int2_pool
            if cache.compact_key is not None:
                key_tail = cache.compact_key.append_superblock(key_block)
                stats.local_tail_dequant_tokens += key_tail
                if cache.generated_value_precision == "int4":
                    if cache.compact_int4_value is None:
                        raise RuntimeError(
                            "generated INT4 V policy is missing its compact arena"
                        )
                    cache.compact_int4_value.append_block(value_block)
                    metadata_pool = cache.int4_pool
                    stats.flushed_int4_value_tokens += flush_size
                else:
                    if cache.compact_value is None:
                        raise RuntimeError(
                            "compact INT2 K cache is missing its compact INT2 V arena"
                        )
                    cache.compact_value.append_superblock(value_block)
            elif cache.full_precision_key:
                raise RuntimeError(
                    "K-BF16/V-FourQ requires an append_fp_only Recent-FP capacity "
                    "covering the requested generation horizon"
                )
            else:
                _append_int2_superblock(
                    cache.int2_pool, key_block, value_block, stats=stats,
                    strict_per_channel_key=False,
                )
            if cache.metadata_arena is not None and metadata_pool is cache.int2_pool:
                cache.metadata_arena.append_int2_pages(
                    cache.int2_pool, global_block, logical_block, cache.page_size
                )
            elif cache.metadata_arena is not None and metadata_pool is cache.int4_pool:
                cache.metadata_arena.append_int4_pages(
                    cache.int4_pool, global_block, logical_block, cache.page_size
                )
            else:
                _append_lowbit_page_metadata(
                    metadata_pool, global_block, logical_block, cache.page_size
                )
        remaining = pool.active_length - flush_size
        pool.fp_k[:, :remaining].copy_(pool.fp_k[:, flush_size : pool.active_length].clone())
        pool.fp_v[:, :remaining].copy_(pool.fp_v[:, flush_size : pool.active_length].clone())
        pool.global_index_buffer[:remaining].copy_(pool.global_index_buffer[flush_size : pool.active_length].clone())
        pool.logical_position_buffer[:remaining].copy_(pool.logical_position_buffer[flush_size : pool.active_length].clone())
        pool.active_length = remaining
        stats.flushed_superblocks += 1
        stats.flushed_tokens += flush_size
        flushed = True
    position = pool.active_length
    assert pool.fp_k is not None and pool.fp_v is not None
    assert pool.global_index_buffer is not None and pool.logical_position_buffer is not None
    pool.fp_k[:, position : position + 1].copy_(new_key)
    pool.fp_v[:, position : position + 1].copy_(new_value)
    pool.global_index_buffer[position] = int(global_token_index)
    pool.logical_position_buffer[position] = int(logical_position)
    pool.active_length += 1
    _refresh_recent_pages(pool, cache.page_size)
    stats.appended_tokens += 1
    return flushed
