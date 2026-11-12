from __future__ import annotations

from dataclasses import dataclass

import torch

from .int2_reference import AffineInt2Tensor, quantize_int2_value
from .packed_kv_layout import PackedTensor, make_packed_tensor

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice as cuda_libdevice

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - environments without Triton
    triton = None
    tl = None
    cuda_libdevice = None
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:

    @triton.jit(do_not_specialize=[
        "ACTIVE", "SOURCE_S", "CAPACITY", "DESTINATION_START"
    ])
    def _gather_pack_int4_value_kernel(
        value_ptr,
        token_indices_ptr,
        packed_ptr,
        scale_ptr,
        ACTIVE,
        SOURCE_S,
        CAPACITY,
        DESTINATION_START,
        D: tl.constexpr,
        D2: tl.constexpr,
        USE_INDICES: tl.constexpr,
    ):
        head_token = tl.program_id(0)
        head = head_token // ACTIVE
        token = head_token - head * ACTIVE
        if USE_INDICES:
            source = tl.load(token_indices_ptr + token).to(tl.int64)
        else:
            source = token
        byte = tl.arange(0, D2)
        base = value_ptr + head * SOURCE_S * D + source * D
        value0 = tl.load(base + byte * 2).to(tl.float32)
        value1 = tl.load(base + byte * 2 + 1).to(tl.float32)
        maximum = tl.maximum(
            tl.max(tl.abs(value0), axis=0), tl.max(tl.abs(value1), axis=0)
        )
        scale = tl.maximum(maximum / 7.0, 1.0e-6)
        q0 = tl.minimum(
            tl.maximum(cuda_libdevice.rint(tl.math.div_rn(value0, scale)), -7.0),
            7.0,
        ).to(tl.int16)
        q1 = tl.minimum(
            tl.maximum(cuda_libdevice.rint(tl.math.div_rn(value1, scale)), -7.0),
            7.0,
        ).to(tl.int16)
        packed = (q0 & 15).to(tl.uint8) | ((q1 & 15).to(tl.uint8) << 4)
        destination = DESTINATION_START + token
        tl.store(
            packed_ptr + head * CAPACITY * D2 + destination * D2 + byte,
            packed,
        )
        tl.store(scale_ptr + head * CAPACITY + destination, scale)

    @triton.jit(do_not_specialize=[
        "ACTIVE", "SOURCE_S", "CAPACITY", "DESTINATION_START"
    ])
    def _gather_pack_int2_value_kernel(
        value_ptr,
        token_indices_ptr,
        packed_ptr,
        scale_ptr,
        zero_ptr,
        ACTIVE,
        SOURCE_S,
        CAPACITY,
        DESTINATION_START,
        D: tl.constexpr,
        D4: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        GROUPS: tl.constexpr,
        GROUP_BYTES: tl.constexpr,
        USE_INDICES: tl.constexpr,
    ):
        head_token = tl.program_id(0)
        head = head_token // ACTIVE
        token = head_token - head * ACTIVE
        group = tl.program_id(1)
        if USE_INDICES:
            source = tl.load(token_indices_ptr + token).to(tl.int64)
        else:
            source = token
        byte = tl.arange(0, GROUP_BYTES)
        dim0 = group * GROUP_SIZE + byte * 4
        base = value_ptr + head * SOURCE_S * D + source * D
        value0 = tl.load(base + dim0).to(tl.float32)
        value1 = tl.load(base + dim0 + 1).to(tl.float32)
        value2 = tl.load(base + dim0 + 2).to(tl.float32)
        value3 = tl.load(base + dim0 + 3).to(tl.float32)
        minimum = tl.minimum(
            tl.minimum(tl.min(value0, axis=0), tl.min(value1, axis=0)),
            tl.minimum(tl.min(value2, axis=0), tl.min(value3, axis=0)),
        )
        maximum = tl.maximum(
            tl.maximum(tl.max(value0, axis=0), tl.max(value1, axis=0)),
            tl.maximum(tl.max(value2, axis=0), tl.max(value3, axis=0)),
        )
        minimum = tl.minimum(minimum, 0.0)
        maximum = tl.maximum(maximum, 0.0)
        scale = tl.maximum((maximum - minimum) / 3.0, 1.0e-6)
        zero = tl.minimum(
            tl.maximum(cuda_libdevice.rint(tl.math.div_rn(-minimum, scale)), 0.0),
            3.0,
        )
        q0 = tl.minimum(
            tl.maximum(cuda_libdevice.rint(tl.math.div_rn(value0, scale) + zero), 0.0),
            3.0,
        ).to(tl.uint8)
        q1 = tl.minimum(
            tl.maximum(cuda_libdevice.rint(tl.math.div_rn(value1, scale) + zero), 0.0),
            3.0,
        ).to(tl.uint8)
        q2 = tl.minimum(
            tl.maximum(cuda_libdevice.rint(tl.math.div_rn(value2, scale) + zero), 0.0),
            3.0,
        ).to(tl.uint8)
        q3 = tl.minimum(
            tl.maximum(cuda_libdevice.rint(tl.math.div_rn(value3, scale) + zero), 0.0),
            3.0,
        ).to(tl.uint8)
        packed = q0 | (q1 << 2) | (q2 << 4) | (q3 << 6)
        destination = DESTINATION_START + token
        packed_offset = (
            head * CAPACITY * D4
            + destination * D4
            + group * GROUP_BYTES
            + byte
        )
        tl.store(packed_ptr + packed_offset, packed)
        meta = head * CAPACITY * GROUPS + destination * GROUPS + group
        tl.store(scale_ptr + meta, scale)
        tl.store(zero_ptr + meta, zero.to(tl.uint8))


def fused_gather_pack_int4_value(
    value: torch.Tensor,
    token_indices: torch.Tensor,
) -> PackedTensor:
    """Gather per-token INT4 V directly into its persistent packed tensors."""
    if value.ndim != 3 or int(value.shape[2]) != 128:
        raise ValueError("INT4 V source must be [Hkv,S,128]")
    if token_indices.ndim != 1 or token_indices.device != value.device:
        raise ValueError("INT4 V token indices must be rank-1 on the V device")
    active = int(token_indices.numel())
    if not value.is_cuda or not _TRITON_AVAILABLE:
        selected = value.index_select(1, token_indices.to(torch.long)).contiguous()
        return make_packed_tensor(selected, 4, "per_token", 2)
    h, source_s, d = map(int, value.shape)
    packed_storage = torch.empty((h, active, d // 2), device=value.device, dtype=torch.uint8)
    scale = torch.empty((h, active, 1), device=value.device, dtype=torch.float32)
    if active:
        _gather_pack_int4_value_kernel[(h * active,)](
            value,
            token_indices,
            packed_storage,
            scale,
            active,
            source_s,
            active,
            0,
            D=d,
            D2=d // 2,
            USE_INDICES=True,
            num_warps=4,
        )
    return PackedTensor(
        packed=packed_storage.flatten(),
        scale=scale,
        zero_point=None,
        num_bits=4,
        original_shape=(h, active, d),
        dtype=value.dtype,
        quant_scheme="per_token",
        pack_metadata={"pack_axis": "flatten_c_order", "fused_gather": True},
        quant_metadata={"dim": 2, "fused_gather": True},
    )


@dataclass
class CompactInt4ValueArena:
    """Preallocated per-token INT4 V storage for generated-token flushes."""

    packed: torch.Tensor
    scale: torch.Tensor
    active_length: int
    capacity: int
    original_dtype: torch.dtype
    num_bits: int = 4
    quant_scheme: str = "per_token"
    zero_point: None = None

    def __post_init__(self) -> None:
        if self.packed.ndim != 3 or self.packed.dtype != torch.uint8:
            raise ValueError("compact INT4 V packed storage must be uint8 [H,capacity,D/2]")
        h, capacity, _ = map(int, self.packed.shape)
        if capacity != self.capacity or not 0 <= self.active_length <= capacity:
            raise ValueError("compact INT4 V active length/capacity is invalid")
        if self.scale.shape != (h, capacity, 1) or self.scale.dtype != torch.float32:
            raise ValueError("compact INT4 V scale storage must be float32 [H,capacity,1]")

    @property
    def original_shape(self) -> tuple[int, int, int]:
        return (
            int(self.packed.shape[0]),
            int(self.active_length),
            int(self.packed.shape[2]) * 2,
        )

    @property
    def nbytes(self) -> int:
        return int(sum(t.numel() * t.element_size() for t in self.allocation_tensors()))

    @property
    def active_nbytes(self) -> int:
        """Bytes occupied by the populated prefix, excluding reserved capacity."""
        h, _, d2 = map(int, self.packed.shape)
        packed = h * self.active_length * d2 * self.packed.element_size()
        scales = h * self.active_length * self.scale.element_size()
        return int(packed + scales)

    def allocation_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.packed, self.scale

    def dequantize_range(self, start: int, length: int) -> torch.Tensor:
        start, length = int(start), int(length)
        if start < 0 or length < 0 or start + length > self.active_length:
            raise ValueError("compact INT4 V dequantization range is outside active storage")
        h, _, d2 = map(int, self.packed.shape)
        if length == 0:
            return torch.empty(
                (h, 0, d2 * 2), device=self.packed.device, dtype=torch.float32
            )
        packed = self.packed[:, start : start + length]
        low = (packed & 15).to(torch.int8)
        high = ((packed >> 4) & 15).to(torch.int8)
        low = torch.where(low >= 8, low - 16, low)
        high = torch.where(high >= 8, high - 16, high)
        codes = torch.stack((low, high), dim=-1).reshape(h, length, d2 * 2)
        return codes.float() * self.scale[:, start : start + length].float()

    def dequantize(self) -> torch.Tensor:
        return self.dequantize_range(0, self.active_length)

    def append_block(self, value_block: torch.Tensor) -> None:
        h, _, d = self.original_shape
        if (
            value_block.ndim != 3
            or int(value_block.shape[0]) != h
            or int(value_block.shape[2]) != d
            or int(value_block.shape[1]) <= 0
        ):
            raise ValueError("compact INT4 V append must contain matching tokens")
        length = int(value_block.shape[1])
        destination = self.active_length
        if destination + length > self.capacity:
            raise RuntimeError(
                f"compact INT4 V capacity exhausted: need {destination + length}, "
                f"have {self.capacity}"
            )
        if value_block.is_cuda and _TRITON_AVAILABLE:
            _gather_pack_int4_value_kernel[(h * length,)](
                value_block,
                self.packed,
                self.packed,
                self.scale,
                length,
                length,
                self.capacity,
                destination,
                D=d,
                D2=d // 2,
                USE_INDICES=False,
                num_warps=4,
            )
        else:
            indices = torch.arange(length, device=value_block.device)
            packed = fused_gather_pack_int4_value(value_block, indices)
            self.packed[:, destination : destination + length].copy_(
                packed.packed.reshape(h, length, d // 2)
            )
            self.scale[:, destination : destination + length].copy_(packed.scale)
        self.active_length += length


def build_compact_int4_value_arena(
    value: torch.Tensor,
    token_indices: torch.Tensor,
    *,
    append_capacity: int = 0,
) -> CompactInt4ValueArena:
    if value.ndim != 3 or int(value.shape[2]) != 128:
        raise ValueError("compact INT4 V source must be [Hkv,S,128]")
    if append_capacity < 0:
        raise ValueError("append_capacity must be non-negative")
    active = int(token_indices.numel())
    capacity = max(1, active + int(append_capacity))
    h, _, d = map(int, value.shape)
    arena = CompactInt4ValueArena(
        packed=torch.empty(
            (h, capacity, d // 2), device=value.device, dtype=torch.uint8
        ),
        scale=torch.empty(
            (h, capacity, 1), device=value.device, dtype=torch.float32
        ),
        active_length=active,
        capacity=capacity,
        original_dtype=value.dtype,
    )
    if active:
        if value.is_cuda and _TRITON_AVAILABLE:
            _gather_pack_int4_value_kernel[(h * active,)](
                value,
                token_indices,
                arena.packed,
                arena.scale,
                active,
                int(value.shape[1]),
                capacity,
                0,
                D=d,
                D2=d // 2,
                USE_INDICES=True,
                num_warps=4,
            )
        else:
            packed = fused_gather_pack_int4_value(value, token_indices)
            arena.packed[:, :active].copy_(packed.packed.reshape(h, active, d // 2))
            arena.scale[:, :active].copy_(packed.scale)
    return arena


def fused_gather_pack_int2_value(
    value: torch.Tensor,
    token_indices: torch.Tensor,
    *,
    group_size: int = 32,
) -> AffineInt2Tensor:
    """Gather per-token-group INT2 V directly into persistent packed tensors."""
    if value.ndim != 3 or int(value.shape[2]) != 128:
        raise ValueError("INT2 V source must be [Hkv,S,128]")
    if group_size not in {32, 64, 128}:
        raise ValueError("fused compact INT2 V requires group_size 32, 64, or 128")
    if token_indices.ndim != 1 or token_indices.device != value.device:
        raise ValueError("INT2 V token indices must be rank-1 on the V device")
    active = int(token_indices.numel())
    if not value.is_cuda or not _TRITON_AVAILABLE:
        selected = value.index_select(1, token_indices.to(torch.long)).contiguous()
        return quantize_int2_value(selected, group_size=group_size)
    h, source_s, d = map(int, value.shape)
    groups = d // group_size
    packed_storage = torch.empty((h, active, d // 4), device=value.device, dtype=torch.uint8)
    scale = torch.empty((h, active, groups, 1), device=value.device, dtype=torch.float16)
    zero = torch.empty((h, active, groups, 1), device=value.device, dtype=torch.uint8)
    if active:
        _gather_pack_int2_value_kernel[(h * active, groups)](
            value,
            token_indices,
            packed_storage,
            scale,
            zero,
            active,
            source_s,
            active,
            0,
            D=d,
            D4=d // 4,
            GROUP_SIZE=group_size,
            GROUPS=groups,
            GROUP_BYTES=group_size // 4,
            USE_INDICES=True,
            num_warps=2,
        )
    return AffineInt2Tensor(
        packed=packed_storage.flatten(),
        scale=scale,
        zero_point=zero,
        original_shape=(h, active, d),
        group_size=group_size,
        axis="per_token_group",
        superblock_size=128,
    )


@dataclass
class CompactInt2ValueArena:
    """Preallocated per-token INT2 V storage for in-place generated flushes."""

    packed: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor
    active_length: int
    capacity: int
    original_dtype: torch.dtype
    group_size: int = 32
    axis: str = "per_token_group"
    superblock_size: int = 128

    def __post_init__(self) -> None:
        if self.packed.ndim != 3 or self.packed.dtype != torch.uint8:
            raise ValueError("compact INT2 V packed storage must be uint8 [H,capacity,D/4]")
        h, capacity, d4 = map(int, self.packed.shape)
        groups = d4 * 4 // self.group_size
        if capacity != self.capacity or not 0 <= self.active_length <= capacity:
            raise ValueError("compact INT2 V active length/capacity is invalid")
        if self.scale.shape != (h, capacity, groups, 1):
            raise ValueError("compact INT2 V scale shape is invalid")
        if self.zero_point.shape != self.scale.shape:
            raise ValueError("compact INT2 V zero-point shape is invalid")
        if self.scale.dtype != torch.float16 or self.zero_point.dtype != torch.uint8:
            raise ValueError("compact INT2 V metadata dtypes are invalid")

    @property
    def original_shape(self) -> tuple[int, int, int]:
        return int(self.packed.shape[0]), int(self.active_length), int(self.packed.shape[2]) * 4

    @property
    def nbytes(self) -> int:
        return int(sum(t.numel() * t.element_size() for t in self.allocation_tensors()))

    @property
    def active_nbytes(self) -> int:
        """Bytes occupied by the populated prefix, excluding reserved capacity."""
        h, _, d4 = map(int, self.packed.shape)
        groups = d4 * 4 // self.group_size
        packed = h * self.active_length * d4 * self.packed.element_size()
        scales = h * self.active_length * groups * self.scale.element_size()
        zeros = h * self.active_length * groups * self.zero_point.element_size()
        return int(packed + scales + zeros)

    def allocation_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.packed, self.scale, self.zero_point

    def dequantize_range(self, start: int, length: int) -> torch.Tensor:
        start, length = int(start), int(length)
        if start < 0 or length < 0 or start + length > self.active_length:
            raise ValueError("compact INT2 V dequantization range is outside active storage")
        h, _, d4 = map(int, self.packed.shape)
        d = d4 * 4
        if length == 0:
            return torch.empty((h, 0, d), device=self.packed.device, dtype=torch.float32)
        packed = self.packed[:, start : start + length]
        codes = torch.stack((
            packed & 3,
            (packed >> 2) & 3,
            (packed >> 4) & 3,
            (packed >> 6) & 3,
        ), dim=-1).reshape(h, length, d)
        groups = d // self.group_size
        grouped = codes.reshape(h, length, groups, self.group_size).float()
        scale = self.scale[:, start : start + length].float()
        zero = self.zero_point[:, start : start + length].float()
        return ((grouped - zero) * scale).reshape(h, length, d)

    def dequantize(self) -> torch.Tensor:
        return self.dequantize_range(0, self.active_length)

    def append_superblock(self, value_block: torch.Tensor) -> None:
        h, _, d = self.original_shape
        if (
            value_block.ndim != 3
            or int(value_block.shape[0]) != h
            or int(value_block.shape[2]) != d
            or int(value_block.shape[1]) <= 0
        ):
            raise ValueError("compact INT2 V flush must contain matching tokens")
        append_length = int(value_block.shape[1])
        destination = self.active_length
        if destination + append_length > self.capacity:
            raise RuntimeError(
                f"compact INT2 V capacity exhausted: need {destination + append_length}, have {self.capacity}"
            )
        groups = d // self.group_size
        if value_block.is_cuda and _TRITON_AVAILABLE:
            _gather_pack_int2_value_kernel[(h * append_length, groups)](
                value_block,
                self.packed,
                self.packed,
                self.scale,
                self.zero_point,
                append_length,
                append_length,
                self.capacity,
                destination,
                D=d,
                D4=d // 4,
                GROUP_SIZE=self.group_size,
                GROUPS=groups,
                GROUP_BYTES=self.group_size // 4,
                USE_INDICES=False,
                num_warps=2,
            )
        else:
            quantized = quantize_int2_value(value_block, group_size=self.group_size)
            self.packed[:, destination : destination + append_length].copy_(
                quantized.packed.reshape(h, append_length, d // 4)
            )
            self.scale[:, destination : destination + append_length].copy_(quantized.scale)
            self.zero_point[:, destination : destination + append_length].copy_(quantized.zero_point)
        self.active_length += append_length


def build_compact_int2_value_arena(
    value: torch.Tensor,
    token_indices: torch.Tensor,
    *,
    append_capacity: int = 0,
    group_size: int = 32,
) -> CompactInt2ValueArena:
    if value.ndim != 3 or int(value.shape[2]) != 128:
        raise ValueError("compact INT2 V source must be [Hkv,S,128]")
    if group_size not in {32, 64, 128}:
        raise ValueError("compact INT2 V requires group_size 32, 64, or 128")
    if append_capacity < 0:
        raise ValueError("append_capacity must be non-negative")
    active = int(token_indices.numel())
    capacity = max(128, ((active + int(append_capacity) + 127) // 128) * 128)
    h, source_s, d = map(int, value.shape)
    groups = d // group_size
    packed = torch.empty((h, capacity, d // 4), device=value.device, dtype=torch.uint8)
    scale = torch.empty((h, capacity, groups, 1), device=value.device, dtype=torch.float16)
    zero = torch.empty((h, capacity, groups, 1), device=value.device, dtype=torch.uint8)
    arena = CompactInt2ValueArena(
        packed=packed,
        scale=scale,
        zero_point=zero,
        active_length=active,
        capacity=capacity,
        original_dtype=value.dtype,
        group_size=group_size,
    )
    if active:
        if value.is_cuda and _TRITON_AVAILABLE:
            _gather_pack_int2_value_kernel[(h * active, groups)](
                value,
                token_indices,
                arena.packed,
                arena.scale,
                arena.zero_point,
                active,
                source_s,
                capacity,
                0,
                D=d,
                D4=d // 4,
                GROUP_SIZE=group_size,
                GROUPS=groups,
                GROUP_BYTES=group_size // 4,
                USE_INDICES=True,
                num_warps=2,
            )
        else:
            quantized = fused_gather_pack_int2_value(
                value, token_indices, group_size=group_size
            )
            arena.packed[:, :active].copy_(quantized.packed.reshape(h, active, d // 4))
            arena.scale[:, :active].copy_(quantized.scale)
            arena.zero_point[:, :active].copy_(quantized.zero_point)
    return arena
