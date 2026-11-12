from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice as cuda_libdevice

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - CUDA-only fast path
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:

    @triton.jit(do_not_specialize=["ACTIVE", "SOURCE_S", "CAPACITY", "SUPERBLOCKS"])
    def _gather_pack_int4_key_kernel(
        key_ptr,
        token_indices_ptr,
        packed_ptr,
        scale_ptr,
        zero_ptr,
        ACTIVE,
        SOURCE_S,
        CAPACITY,
        SUPERBLOCKS,
        D: tl.constexpr,
        D2: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_BYTES: tl.constexpr,
    ):
        """Gather and affine-pack per-channel INT4 K in 128-token blocks."""
        head = tl.program_id(0)
        block = tl.program_id(1)
        channel_tile = tl.program_id(2)
        token_local = tl.arange(0, BLOCK_T)
        position = block * BLOCK_T + token_local
        valid = position < ACTIVE
        source = tl.load(token_indices_ptr + position, mask=valid, other=0).to(tl.int64)
        byte = channel_tile * BLOCK_BYTES + tl.arange(0, BLOCK_BYTES)
        d0, d1 = byte * 2, byte * 2 + 1
        base = head * SOURCE_S * D + source[:, None] * D
        mask = valid[:, None]
        x0 = tl.load(key_ptr + base + d0[None, :], mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(key_ptr + base + d1[None, :], mask=mask, other=0.0).to(tl.float32)

        minimum0 = tl.minimum(tl.min(tl.where(mask, x0, float("inf")), axis=0), 0.0)
        minimum1 = tl.minimum(tl.min(tl.where(mask, x1, float("inf")), axis=0), 0.0)
        maximum0 = tl.maximum(tl.max(tl.where(mask, x0, -float("inf")), axis=0), 0.0)
        maximum1 = tl.maximum(tl.max(tl.where(mask, x1, -float("inf")), axis=0), 0.0)
        scale0 = tl.maximum((maximum0 - minimum0) / 15.0, 1.0e-6)
        scale1 = tl.maximum((maximum1 - minimum1) / 15.0, 1.0e-6)
        zero0 = tl.minimum(tl.maximum(cuda_libdevice.rint(-minimum0 / scale0), 0.0), 15.0)
        zero1 = tl.minimum(tl.maximum(cuda_libdevice.rint(-minimum1 / scale1), 0.0), 15.0)
        q0 = tl.minimum(
            tl.maximum(cuda_libdevice.rint(x0 / scale0[None, :] + zero0[None, :]), 0.0),
            15.0,
        ).to(tl.uint8)
        q1 = tl.minimum(
            tl.maximum(cuda_libdevice.rint(x1 / scale1[None, :] + zero1[None, :]), 0.0),
            15.0,
        ).to(tl.uint8)
        packed = q0 | (q1 << 4)
        packed_offset = head * CAPACITY * D2 + position[:, None] * D2 + byte[None, :]
        tl.store(packed_ptr + packed_offset, packed, mask=mask)

        meta = head * SUPERBLOCKS * D + block * D
        tl.store(scale_ptr + meta + d0, scale0)
        tl.store(scale_ptr + meta + d1, scale1)
        tl.store(zero_ptr + meta + d0, zero0.to(tl.uint8))
        tl.store(zero_ptr + meta + d1, zero1.to(tl.uint8))


def _round_up(value: int, multiple: int) -> int:
    return ((int(value) + int(multiple) - 1) // int(multiple)) * int(multiple)


def quantize_int4_key_per_channel(
    key: torch.Tensor,
    *,
    superblock_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference affine INT4 quantizer over token blocks, independently per channel."""
    if key.ndim != 3 or int(key.shape[2]) % 2:
        raise ValueError("INT4 K must be [H,S,D] with even D")
    if superblock_size not in {32, 64, 128}:
        raise ValueError("per-channel INT4 K superblock must be 32, 64, or 128 tokens")
    h, length, d = map(int, key.shape)
    blocks = max(1, _round_up(length, superblock_size) // superblock_size)
    scale = torch.zeros((h, blocks, d), device=key.device, dtype=torch.float16)
    zero = torch.zeros((h, blocks, d), device=key.device, dtype=torch.uint8)
    codes = torch.zeros((h, length, d), device=key.device, dtype=torch.uint8)
    for block, start in enumerate(range(0, length, superblock_size)):
        source = key[:, start : start + superblock_size].float()
        minimum = torch.minimum(source.amin(dim=1), torch.zeros_like(source[:, 0]))
        maximum = torch.maximum(source.amax(dim=1), torch.zeros_like(source[:, 0]))
        step = ((maximum - minimum) / 15.0).clamp_min(1.0e-6)
        zp = (-minimum / step).round().clamp(0, 15)
        codes[:, start : start + source.shape[1]] = (
            source / step[:, None, :] + zp[:, None, :]
        ).round().clamp(0, 15).to(torch.uint8)
        scale[:, block] = step.to(torch.float16)
        zero[:, block] = zp.to(torch.uint8)
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous()
    return packed, scale, zero


@dataclass
class CompactInt4KeyArena:
    """One per-channel affine INT4 K allocation shared by all FourQ V pools."""

    packed: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor
    active_length: int
    capacity: int
    segment_offsets: tuple[int, int, int]
    original_dtype: torch.dtype
    superblock_size: int = 128
    bit_width: int = 4

    def __post_init__(self) -> None:
        if self.packed.ndim != 3 or self.packed.dtype != torch.uint8:
            raise ValueError("compact INT4 K storage must be uint8 [H,capacity,D/2]")
        h, capacity, d2 = map(int, self.packed.shape)
        if self.superblock_size not in {32, 64, 128}:
            raise ValueError("compact INT4 K superblock must be 32, 64, or 128 tokens")
        blocks = (capacity + self.superblock_size - 1) // self.superblock_size
        if capacity != self.capacity or not 0 <= self.active_length <= capacity:
            raise ValueError("compact INT4 K active length/capacity is invalid")
        if self.scale.shape != (h, blocks, d2 * 2):
            raise ValueError("compact INT4 K scale shape is invalid")
        if self.zero_point.shape != self.scale.shape:
            raise ValueError("compact INT4 K zero-point shape is invalid")
        if self.scale.dtype != torch.float16 or self.zero_point.dtype != torch.uint8:
            raise ValueError("compact INT4 K metadata dtypes are invalid")
        if tuple(sorted(self.segment_offsets)) != self.segment_offsets:
            raise ValueError("compact INT4 K segment offsets must be monotonic")

    @property
    def head_count(self) -> int:
        return int(self.packed.shape[0])

    @property
    def head_dim(self) -> int:
        return int(self.packed.shape[2]) * 2

    @property
    def superblock_count(self) -> int:
        return int(self.scale.shape[1])

    @property
    def nbytes(self) -> int:
        return int(sum(t.numel() * t.element_size() for t in self.allocation_tensors()))

    @property
    def active_nbytes(self) -> int:
        """Bytes occupied by the populated prefix, excluding reserved capacity."""
        if self.active_length <= 0:
            return 0
        blocks = (self.active_length + self.superblock_size - 1) // self.superblock_size
        packed = self.head_count * self.active_length * (self.head_dim // 2)
        scales = self.head_count * blocks * self.head_dim * self.scale.element_size()
        zeros = self.head_count * blocks * self.head_dim * self.zero_point.element_size()
        return int(packed + scales + zeros)

    def allocation_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.packed, self.scale, self.zero_point

    def dequantize_range(self, start: int, length: int) -> torch.Tensor:
        start, length = int(start), int(length)
        if start < 0 or length < 0 or start + length > self.active_length:
            raise ValueError("compact INT4 K range is outside the active payload")
        if length == 0:
            return torch.empty(
                (self.head_count, 0, self.head_dim),
                device=self.packed.device,
                dtype=torch.float32,
            )
        payload = self.packed[:, start : start + length]
        codes = torch.stack((payload & 15, (payload >> 4) & 15), dim=-1).reshape(
            self.head_count, length, self.head_dim
        )
        blocks = (
            torch.arange(start, start + length, device=payload.device)
            // self.superblock_size
        )
        scale = self.scale.index_select(1, blocks).float()
        zero = self.zero_point.index_select(1, blocks).float()
        return (codes.float() - zero) * scale

    def dequantize(self) -> torch.Tensor:
        return self.dequantize_range(0, self.active_length)

    def _write(self, start: int, key: torch.Tensor) -> None:
        packed, scale, zero = quantize_int4_key_per_channel(
            key, superblock_size=self.superblock_size
        )
        length = int(key.shape[1])
        self.packed[:, start : start + length].copy_(packed)
        block = start // self.superblock_size
        self.scale[:, block : block + scale.shape[1]].copy_(scale)
        self.zero_point[:, block : block + zero.shape[1]].copy_(zero)

    def append_superblock(self, key_block: torch.Tensor) -> int:
        if key_block.shape != (self.head_count, 128, self.head_dim):
            raise ValueError("compact INT4 K flush requires 128 matching tokens")
        if self.active_length + 128 > self.capacity:
            raise RuntimeError("compact INT4 K capacity exhausted")
        prefix = self.active_length // self.superblock_size * self.superblock_size
        tail_count = self.active_length - prefix
        tail = self.dequantize_range(prefix, tail_count).to(key_block.dtype)
        self._write(prefix, torch.cat((tail, key_block), dim=1))
        self.active_length += 128
        return tail_count


def build_compact_int4_key_arena(
    key: torch.Tensor,
    segment_token_indices: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    append_capacity: int = 0,
    superblock_size: int = 128,
) -> CompactInt4KeyArena:
    if key.ndim != 3 or not torch.is_floating_point(key) or int(key.shape[2]) % 2:
        raise ValueError("key must be floating [H,S,D] with even D")
    if superblock_size not in {32, 64, 128}:
        raise ValueError("compact per-channel INT4 K superblock must be 32, 64, or 128 tokens")
    lengths = tuple(int(indices.numel()) for indices in segment_token_indices)
    offsets = (0, lengths[0], lengths[0] + lengths[1])
    active = sum(lengths)
    capacity = max(128, _round_up(active + int(append_capacity), 128))
    h, _, d = map(int, key.shape)
    blocks = capacity // superblock_size
    arena = CompactInt4KeyArena(
        packed=torch.zeros((h, capacity, d // 2), device=key.device, dtype=torch.uint8),
        scale=torch.zeros((h, blocks, d), device=key.device, dtype=torch.float16),
        zero_point=torch.zeros((h, blocks, d), device=key.device, dtype=torch.uint8),
        active_length=active,
        capacity=capacity,
        segment_offsets=offsets,
        original_dtype=key.dtype,
        superblock_size=superblock_size,
    )
    if not active:
        return arena
    ordered = torch.cat(tuple(indices.to(torch.long) for indices in segment_token_indices))
    if key.is_cuda and _TRITON_AVAILABLE:
        block_bytes = 8
        grid = (
            h,
            (active + superblock_size - 1) // superblock_size,
            (d // 2 + block_bytes - 1) // block_bytes,
        )
        _gather_pack_int4_key_kernel[grid](
            key,
            ordered,
            arena.packed,
            arena.scale,
            arena.zero_point,
            active,
            int(key.shape[1]),
            capacity,
            blocks,
            D=d,
            D2=d // 2,
            BLOCK_T=superblock_size,
            BLOCK_BYTES=block_bytes,
            num_warps=4,
            num_stages=2,
        )
    else:
        for start in range(0, active, superblock_size):
            tokens = ordered[start : start + superblock_size]
            arena._write(start, key.index_select(1, tokens).contiguous())
    return arena
