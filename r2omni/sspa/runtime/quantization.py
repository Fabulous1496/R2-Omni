from __future__ import annotations

from typing import Any

import torch


def signed_qrange(num_bits: int) -> tuple[int, int]:
    if num_bits == 2:
        # Symmetric INT2 uses {-1, 0, 1}; code -2 is left unused.
        return -1, 1
    if num_bits == 4:
        return -7, 7
    if num_bits == 8:
        return -127, 127
    raise ValueError("num_bits must be one of 2, 4, or 8")


def quantize_tensor_symmetric(
    x: torch.Tensor,
    num_bits: int,
    dim: int = -1,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Symmetric uniform quantization.

    The returned q tensor is signed int16 before packing. Scale is computed
    along ``dim`` with keepdim=True so it broadcasts back to x.
    """
    if not torch.is_tensor(x) or not torch.is_floating_point(x):
        raise TypeError("x must be a floating-point torch.Tensor")
    qmin, qmax = signed_qrange(int(num_bits))
    work = x.float()
    dim = int(dim)
    amax = work.abs().amax(dim=dim, keepdim=True)
    scale = (amax / float(qmax)).clamp_min(float(eps))
    q = torch.round(work / scale).clamp(qmin, qmax).to(torch.int16)
    deq = dequantize_tensor_symmetric(q, scale, num_bits).to(work.dtype)
    err = quant_error_stats(work, deq)
    metadata = {
        "num_bits": int(num_bits),
        "qmin": int(qmin),
        "qmax": int(qmax),
        "dim": dim,
        "eps": float(eps),
        "input_dtype": str(x.dtype),
        "scale_dtype": str(scale.dtype),
        **err,
    }
    return q, scale, metadata


def dequantize_tensor_symmetric(q: torch.Tensor, scale: torch.Tensor, num_bits: int) -> torch.Tensor:
    signed_qrange(int(num_bits))
    return q.float() * scale.float()


def quant_error_stats(reference: torch.Tensor, approx: torch.Tensor, eps: float = 1e-12) -> dict[str, float]:
    ref = reference.float().flatten()
    app = approx.float().flatten()
    diff = (ref - app).abs()
    denom = ref.abs().mean().clamp_min(float(eps))
    if ref.numel() == 0:
        cos = 1.0
    elif float(ref.norm().item()) == 0.0 and float(app.norm().item()) == 0.0:
        cos = 1.0
    else:
        cos = torch.nn.functional.cosine_similarity(ref, app, dim=0, eps=eps).item()
    return {
        "max_abs_error": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_error": float(diff.mean().item()) if diff.numel() else 0.0,
        "relative_error": float((diff.mean() / denom).item()) if diff.numel() else 0.0,
        "cosine_similarity": float(cos),
    }


def _pack_signed(q: torch.Tensor, num_bits: int) -> tuple[torch.Tensor, dict[str, Any]]:
    qmin, qmax = signed_qrange(num_bits)
    flat = q.to(torch.int16).contiguous().flatten()
    if bool(((flat < qmin) | (flat > qmax)).any()):
        raise ValueError(f"q contains values outside signed INT{num_bits} range [{qmin}, {qmax}]")
    per_byte = 8 // int(num_bits)
    padding = (-flat.numel()) % per_byte
    if padding:
        flat = torch.nn.functional.pad(flat, (0, padding))
    mask = (1 << int(num_bits)) - 1
    codes = (flat.to(torch.int16) & mask).to(torch.uint8)
    grouped = codes.view(-1, per_byte)
    packed = torch.zeros(grouped.shape[0], dtype=torch.uint8, device=q.device)
    for slot in range(per_byte):
        packed |= grouped[:, slot] << (slot * int(num_bits))
    meta = {
        "num_bits": int(num_bits),
        "signed": True,
        "original_shape": tuple(int(v) for v in q.shape),
        "original_numel": int(q.numel()),
        "padded_length": int(flat.numel()),
        "padding": int(padding),
        "packed_shape": tuple(int(v) for v in packed.shape),
        "pack_axis": "flatten_c_order",
    }
    return packed, meta


def _unpack_signed(packed: torch.Tensor, original_shape: tuple[int, ...], num_bits: int) -> torch.Tensor:
    if packed.dtype != torch.uint8:
        raise TypeError("packed tensor must have dtype torch.uint8")
    per_byte = 8 // int(num_bits)
    mask = (1 << int(num_bits)) - 1
    parts = [((packed >> (slot * int(num_bits))) & mask) for slot in range(per_byte)]
    codes = torch.stack(parts, dim=1).flatten()
    count = int(torch.tensor(original_shape).prod().item())
    codes = codes[:count].to(torch.int16)
    sign_bit = 1 << (int(num_bits) - 1)
    signed = torch.where(codes >= sign_bit, codes - (1 << int(num_bits)), codes)
    return signed.reshape(tuple(int(v) for v in original_shape))


def pack_int4(q: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    """Pack signed 4-bit integers into uint8; two values per byte."""
    return _pack_signed(q, 4)


def unpack_int4(packed: torch.Tensor, original_shape: tuple[int, ...], signed: bool = True) -> torch.Tensor:
    if not signed:
        raise NotImplementedError("unsigned INT4 unpack is not used by this symmetric KV path")
    return _unpack_signed(packed, tuple(original_shape), 4)


def pack_int2(q: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    """Pack signed 2-bit integers into uint8; four values per byte."""
    return _pack_signed(q, 2)


def unpack_int2(packed: torch.Tensor, original_shape: tuple[int, ...], signed: bool = True) -> torch.Tensor:
    if not signed:
        raise NotImplementedError("unsigned INT2 unpack is not used by this symmetric KV path")
    return _unpack_signed(packed, tuple(original_shape), 2)


def pack_lowbit(q: torch.Tensor, num_bits: int) -> tuple[torch.Tensor, dict[str, Any]]:
    if int(num_bits) == 4:
        return pack_int4(q)
    if int(num_bits) == 2:
        return pack_int2(q)
    if int(num_bits) == 8:
        qmin, qmax = signed_qrange(8)
        flat = q.to(torch.int16).contiguous().flatten()
        if bool(((flat < qmin) | (flat > qmax)).any()):
            raise ValueError(f"q contains values outside signed INT8 range [{qmin}, {qmax}]")
        packed = flat.to(torch.int8).view(torch.uint8)
        return packed, {
            "num_bits": 8,
            "signed": True,
            "original_shape": tuple(int(v) for v in q.shape),
            "original_numel": int(q.numel()),
            "padded_length": int(q.numel()),
            "padding": 0,
            "packed_shape": tuple(int(v) for v in packed.shape),
            "pack_axis": "flatten_c_order",
        }
    raise ValueError("num_bits must be 2, 4, or 8")


def unpack_lowbit(packed: torch.Tensor, original_shape: tuple[int, ...], num_bits: int, signed: bool = True) -> torch.Tensor:
    if int(num_bits) == 4:
        return unpack_int4(packed, original_shape, signed=signed)
    if int(num_bits) == 2:
        return unpack_int2(packed, original_shape, signed=signed)
    if int(num_bits) == 8:
        return packed.view(torch.int8).to(torch.int16).reshape(tuple(original_shape))
    raise ValueError("num_bits must be 2, 4, or 8")
