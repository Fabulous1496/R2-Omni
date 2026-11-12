from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from .quantization import unpack_int2

Int2Mode = Literal["plain_int2", "hadamard_int2", "scaled_hadamard_int2"]


def _check_group_size(head_dim: int, group_size: int) -> None:
    if group_size <= 0 or head_dim % group_size:
        raise ValueError("group_size must be positive and divide head_dim")


def normalized_fwht(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal Walsh-Hadamard transform over the last dimension."""
    width = int(x.shape[-1])
    if width < 1 or width & (width - 1):
        raise ValueError("Hadamard head dimension must be a positive power of two")
    out = x.float().contiguous()
    stride = 1
    while stride < width:
        shape = (*out.shape[:-1], -1, 2, stride)
        pair = out.reshape(shape)
        left, right = pair[..., 0, :], pair[..., 1, :]
        out = torch.stack((left + right, left - right), dim=-2).reshape_as(out)
        stride *= 2
    return out * (width ** -0.5)


@dataclass
class AffineInt2Tensor:
    packed: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor
    original_shape: tuple[int, ...]
    group_size: int
    axis: Literal["per_channel", "per_channel_group", "per_token_group"]
    superblock_size: int = 128

    @property
    def nbytes(self) -> int:
        return int(
            self.packed.numel() * self.packed.element_size()
            + self.scale.numel() * self.scale.element_size()
            + self.zero_point.numel() * self.zero_point.element_size()
        )

    def codes(self) -> torch.Tensor:
        # The generic signed unpack maps bit patterns 2/3 to -2/-1. Masking
        # recovers the affine unsigned codes 0..3 without another unpack path.
        return (unpack_int2(self.packed, self.original_shape).to(torch.int16) & 3).to(torch.uint8)

    def dequantize(self) -> torch.Tensor:
        codes = self.codes().float()
        h, s, d = self.original_shape
        if self.axis == "per_channel":
            block = torch.arange(s, device=codes.device) // self.superblock_size
            scale = self.scale.index_select(1, block)
            # OScaR stores the real-domain dequantization offset rather than a
            # clipped integer zero point: dequant(code) = code * scale + offset.
            offset = self.zero_point.index_select(1, block)
            return codes * scale.float() + offset.float()
        grouped = codes.reshape(h, s, d // self.group_size, self.group_size)
        if self.axis == "per_channel_group":
            block = torch.arange(s, device=codes.device) // self.superblock_size
            scale = self.scale.index_select(1, block)
            zero = self.zero_point.index_select(1, block)
        else:
            scale, zero = self.scale, self.zero_point
        return ((grouped - zero.float()) * scale.float()).reshape(self.original_shape)


@dataclass
class Int2KVReference:
    key: AffineInt2Tensor
    value: AffineInt2Tensor
    mode: Int2Mode
    channel_scale: torch.Tensor | None
    residual_k: torch.Tensor
    residual_v: torch.Tensor
    residual_capacity: int = 128
    flush_count: int = 0
    full_dequant_tensor_count: int = 0
    fallback_count: int = 0

    @property
    def storage_bytes(self) -> int:
        scale_bytes = 0 if self.channel_scale is None else self.channel_scale.numel() * self.channel_scale.element_size()
        return int(self.key.nbytes + self.value.nbytes + self.residual_k.numel() * self.residual_k.element_size() + self.residual_v.numel() * self.residual_v.element_size() + scale_bytes)


def _affine_quantize(x: torch.Tensor, *, group_size: int, axis: Literal["per_channel_group", "per_token_group"]) -> AffineInt2Tensor:
    if x.ndim != 3 or not torch.is_floating_point(x):
        raise ValueError("x must be floating [H,S,D]")
    h, s, d = (int(v) for v in x.shape)
    _check_group_size(d, group_size)
    groups = x.float().reshape(h, s, d // group_size, group_size)
    if axis == "per_channel_group":
        superblock = 128
        num_blocks = (s + superblock - 1) // superblock
        minimum_parts = []
        maximum_parts = []
        for block in range(num_blocks):
            part = groups[:, block * superblock : min((block + 1) * superblock, s)]
            minimum_parts.append(part.amin(dim=(1, 3), keepdim=True))
            maximum_parts.append(part.amax(dim=(1, 3), keepdim=True))
        minimum = torch.cat(minimum_parts, dim=1)
        maximum = torch.cat(maximum_parts, dim=1)
    elif axis == "per_token_group":
        minimum = groups.amin(dim=3, keepdim=True)
        maximum = groups.amax(dim=3, keepdim=True)
    else:
        raise ValueError("unsupported affine INT2 axis")

    # A clipped affine zero-point can represent a one-sided interval only when
    # the calibrated real range includes zero.  Without this expansion, e.g.
    # [10, 11] produces scale=(11-10)/3 and zero_point=0, so every value
    # saturates to a dequantized value near 1.  Including zero preserves the
    # existing uint8 zero-point layout while making positive/negative-only and
    # non-zero constant groups well-defined.
    minimum = torch.minimum(minimum, torch.zeros_like(minimum))
    maximum = torch.maximum(maximum, torch.zeros_like(maximum))
    if axis == "per_channel_group":
        token_blocks = torch.arange(s, device=x.device) // superblock
        quant_minimum = minimum.index_select(1, token_blocks)
        quant_maximum = maximum.index_select(1, token_blocks)
    else:
        quant_minimum, quant_maximum = minimum, maximum
    scale = ((maximum - minimum) / 3.0).clamp_min(1e-6)
    zero_point = torch.round(-minimum / scale).clamp(0, 3).to(torch.uint8)
    quant_scale = ((quant_maximum - quant_minimum) / 3.0).clamp_min(1e-6)
    quant_zero = torch.round(-quant_minimum / quant_scale).clamp(0, 3)
    codes = torch.round(groups / quant_scale + quant_zero).clamp(0, 3).to(torch.int16).reshape(h, s, d)
    flat = codes.to(torch.uint8).contiguous().flatten()
    padding = (-flat.numel()) % 4
    if padding:
        flat = torch.nn.functional.pad(flat, (0, padding))
    grouped_codes = flat.reshape(-1, 4)
    packed = grouped_codes[:, 0] | (grouped_codes[:, 1] << 2) | (grouped_codes[:, 2] << 4) | (grouped_codes[:, 3] << 6)
    return AffineInt2Tensor(
        packed=packed,
        scale=scale.to(torch.float16),
        zero_point=zero_point,
        original_shape=(h, s, d),
        group_size=int(group_size),
        axis=axis,
    )


def quantize_int2_key(key: torch.Tensor, *, group_size: int = 32) -> AffineInt2Tensor:
    """Quantize logical ``[Hkv,S,D]`` keys with per-channel-group metadata."""
    return _affine_quantize(key.float(), group_size=group_size, axis="per_channel_group")


def quantize_int2_key_per_channel(
    key: torch.Tensor, *, superblock_size: int = 128, clip_ratio: float = 1.0
) -> AffineInt2Tensor:
    """Exact OScaR affine INT2 with one scale/offset per channel and token block."""
    if key.ndim != 3 or not torch.is_floating_point(key):
        raise ValueError("key must be floating [H,S,D]")
    h, s, d = map(int, key.shape)
    if superblock_size not in {32, 64, 128}:
        raise ValueError("per-channel INT2 K block size must be 32, 64, or 128")
    if not 0.0 < float(clip_ratio) <= 1.0:
        raise ValueError("clip_ratio must lie in (0, 1]")
    work = key.float()
    minima = []
    maxima = []
    for start in range(0, s, superblock_size):
        block = work[:, start : start + superblock_size]
        minima.append(block.amin(dim=1, keepdim=True))
        maxima.append(block.amax(dim=1, keepdim=True))
    minimum = torch.cat(minima, dim=1)
    maximum = torch.cat(maxima, dim=1)
    if float(clip_ratio) < 1.0:
        center = (minimum + maximum) * 0.5
        minimum = center + float(clip_ratio) * (minimum - center)
        maximum = center + float(clip_ratio) * (maximum - center)
    value_range = maximum - minimum
    nonconstant = value_range > 0
    scale = torch.where(nonconstant, value_range / 3.0, torch.zeros_like(value_range))
    scale_inv = torch.where(nonconstant, 3.0 / value_range.clamp_min(1e-30), torch.zeros_like(value_range))
    zero = torch.where(nonconstant, torch.round(-minimum * scale_inv), torch.zeros_like(minimum))
    offset = torch.where(nonconstant, -zero * scale, minimum)
    token_blocks = torch.arange(s, device=key.device) // superblock_size
    token_zero = zero.index_select(1, token_blocks)
    token_inv = scale_inv.index_select(1, token_blocks)
    codes = torch.round(work * token_inv + token_zero).clamp(0, 3).to(torch.uint8)
    flat = codes.contiguous().flatten()
    padding = (-flat.numel()) % 4
    if padding:
        flat = torch.nn.functional.pad(flat, (0, padding))
    grouped_codes = flat.reshape(-1, 4)
    packed = (
        grouped_codes[:, 0]
        | (grouped_codes[:, 1] << 2)
        | (grouped_codes[:, 2] << 4)
        | (grouped_codes[:, 3] << 6)
    )
    return AffineInt2Tensor(
        packed=packed,
        scale=scale.to(torch.float16),
        # Keep the public field for layout compatibility. For per-channel K it
        # contains OScaR's floating dequantization offset, not an integer zero.
        zero_point=offset.to(torch.float16),
        original_shape=(h, s, d),
        group_size=1,
        axis="per_channel",
        superblock_size=superblock_size,
    )


def quantize_int2_value(value: torch.Tensor, *, group_size: int = 32) -> AffineInt2Tensor:
    """Quantize logical ``[Hkv,S,D]`` values with per-token-group metadata."""
    return _affine_quantize(value.float(), group_size=group_size, axis="per_token_group")


def _channel_preconditioner(key: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    rms = key.float().square().mean(dim=1, keepdim=True).sqrt().clamp_min(eps)
    geometric_mean = rms.log().mean(dim=-1, keepdim=True).exp()
    return (rms / geometric_mean).sqrt().clamp(0.25, 4.0)


def build_int2_kv_reference(
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    group_size: int = 32,
    mode: Int2Mode = "plain_int2",
    residual_capacity: int = 128,
) -> Int2KVReference:
    """Build the T1 affine INT2 cache for logical [Hkv,S,D] K/V tensors."""
    if key.shape != value.shape or key.ndim != 3:
        raise ValueError("key/value must have the same [Hkv,S,D] shape")
    if mode not in {"plain_int2", "hadamard_int2", "scaled_hadamard_int2"}:
        raise ValueError("unsupported INT2 mode")
    if residual_capacity != 128:
        raise ValueError("the formal INT2 path uses a 128-token residual superblock")
    channel_scale = None
    transformed_k = key.float()
    transformed_v = value.float()
    if mode == "hadamard_int2":
        transformed_k = normalized_fwht(transformed_k)
        transformed_v = normalized_fwht(transformed_v)
    elif mode == "scaled_hadamard_int2":
        channel_scale = _channel_preconditioner(key)
        transformed_k = normalized_fwht(transformed_k * channel_scale)
        transformed_v = normalized_fwht(transformed_v * channel_scale)
    h, _, d = key.shape
    empty = torch.empty((h, 0, d), dtype=key.dtype, device=key.device)
    return Int2KVReference(
        key=quantize_int2_key(transformed_k, group_size=group_size),
        value=quantize_int2_value(transformed_v, group_size=group_size),
        mode=mode,
        channel_scale=channel_scale,
        residual_k=empty,
        residual_v=empty.clone(),
    )
