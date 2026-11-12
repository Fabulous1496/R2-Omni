from __future__ import annotations

from dataclasses import dataclass

import torch

from .int2_reference import quantize_int2_key_per_channel


def _round_up(value: int, multiple: int) -> int:
    return ((int(value) + int(multiple) - 1) // int(multiple)) * int(multiple)


@dataclass
class CompactInt2KeyArena:
    """One per-channel affine INT2 K allocation shared by all V pools."""

    packed: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor
    active_length: int
    capacity: int
    segment_offsets: tuple[int, int, int]
    original_dtype: torch.dtype
    superblock_size: int = 128
    bit_width: int = 2

    @property
    def head_count(self) -> int:
        return int(self.packed.shape[0])

    @property
    def head_dim(self) -> int:
        return int(self.packed.shape[2]) * 4

    @property
    def superblock_count(self) -> int:
        return int(self.scale.shape[1])

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.allocation_tensors())

    @property
    def active_nbytes(self) -> int:
        """Bytes occupied by the populated prefix, excluding reserved capacity."""
        if self.active_length <= 0:
            return 0
        blocks = (self.active_length + self.superblock_size - 1) // self.superblock_size
        packed = self.head_count * self.active_length * (self.head_dim // 4)
        scales = self.head_count * blocks * self.head_dim * self.scale.element_size()
        offsets = (
            self.head_count
            * blocks
            * self.head_dim
            * self.zero_point.element_size()
        )
        return int(packed + scales + offsets)

    def allocation_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.packed, self.scale, self.zero_point

    def dequantize_range(self, start: int, length: int) -> torch.Tensor:
        start, length = int(start), int(length)
        if start < 0 or length < 0 or start + length > self.active_length:
            raise ValueError("compact INT2 K range is outside the active payload")
        if not length:
            return torch.empty(
                (self.head_count, 0, self.head_dim),
                device=self.packed.device,
                dtype=torch.float32,
            )
        payload = self.packed[:, start : start + length]
        codes = torch.stack(
            (
                payload & 3,
                (payload >> 2) & 3,
                (payload >> 4) & 3,
                (payload >> 6) & 3,
            ),
            dim=-1,
        ).reshape(self.head_count, length, self.head_dim)
        blocks = (
            torch.arange(start, start + length, device=payload.device)
            // self.superblock_size
        )
        return (
            codes.float() * self.scale.index_select(1, blocks).float()
            + self.zero_point.index_select(1, blocks).float()
        )

    def dequantize(self) -> torch.Tensor:
        return self.dequantize_range(0, self.active_length)

    def _write(self, start: int, key: torch.Tensor) -> None:
        quantized = quantize_int2_key_per_channel(
            key, superblock_size=self.superblock_size
        )
        length = int(key.shape[1])
        packed = quantized.packed.reshape(
            self.head_count, length, self.head_dim // 4
        )
        block = start // self.superblock_size
        self.packed[:, start : start + length].copy_(packed)
        self.scale[:, block : block + quantized.scale.shape[1]].copy_(
            quantized.scale
        )
        self.zero_point[:, block : block + quantized.zero_point.shape[1]].copy_(
            quantized.zero_point
        )

    def append_superblock(self, key_block: torch.Tensor) -> int:
        if key_block.shape != (self.head_count, 128, self.head_dim):
            raise ValueError("compact INT2 K flush requires 128 matching tokens")
        if self.active_length + 128 > self.capacity:
            raise RuntimeError("compact INT2 K capacity exhausted")
        prefix = self.active_length // self.superblock_size * self.superblock_size
        tail_count = self.active_length - prefix
        tail = self.dequantize_range(prefix, tail_count).to(key_block.dtype)
        self._write(prefix, torch.cat((tail, key_block), dim=1))
        self.active_length += 128
        return tail_count


def build_compact_int2_key_arena(
    key: torch.Tensor,
    segment_token_indices: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    append_capacity: int = 0,
    superblock_size: int = 128,
) -> CompactInt2KeyArena:
    if key.ndim != 3 or not torch.is_floating_point(key) or int(key.shape[2]) % 4:
        raise ValueError("key must be floating [H,S,D] with D divisible by four")
    lengths = tuple(int(indices.numel()) for indices in segment_token_indices)
    offsets = (0, lengths[0], lengths[0] + lengths[1])
    active = sum(lengths)
    capacity = max(128, _round_up(active + int(append_capacity), 128))
    h, _, d = map(int, key.shape)
    blocks = capacity // superblock_size
    arena = CompactInt2KeyArena(
        packed=torch.zeros((h, capacity, d // 4), device=key.device, dtype=torch.uint8),
        scale=torch.zeros((h, blocks, d), device=key.device, dtype=torch.float16),
        zero_point=torch.zeros((h, blocks, d), device=key.device, dtype=torch.float16),
        active_length=active,
        capacity=capacity,
        segment_offsets=offsets,
        original_dtype=key.dtype,
        superblock_size=superblock_size,
    )
    if active:
        order = torch.cat(tuple(indices.to(torch.long) for indices in segment_token_indices))
        arena._write(0, key.index_select(1, order).contiguous())
    return arena
