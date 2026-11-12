#!/usr/bin/env python3
"""Compare packed GEMV with explicit dequantization for Omni GQA layouts."""

from __future__ import annotations

import argparse

import torch

from baselines.common.qwen_packed_adapter import NativeKVAdapter


def _repeat_heads(tensor: torch.Tensor, ratio: int) -> torch.Tensor:
    if ratio == 1:
        return tensor
    batch, heads, x, y = tensor.shape
    return tensor[:, :, None].expand(batch, heads, ratio, x, y).reshape(
        batch, heads * ratio, x, y
    )


def _dequant_last(
    packed: torch.Tensor,
    scale: torch.Tensor,
    minimum: torch.Tensor,
    bits: int,
    group_size: int,
) -> torch.Tensor:
    factor = 32 // bits
    shifts = torch.arange(factor, device=packed.device, dtype=torch.int32) * bits
    values = ((packed[..., None] >> shifts) & (2**bits - 1)).flatten(-2)
    groups = values.shape[-1] // group_size
    shaped = values.view(*values.shape[:-1], groups, group_size).to(scale.dtype)
    return (shaped * scale.unsqueeze(-1) + minimum.unsqueeze(-1)).flatten(-2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("kivi", "ott"), required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--extension-root", required=True)
    parser.add_argument("--query-heads", type=int, default=28)
    parser.add_argument("--kv-heads", type=int, default=4)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    ratio = args.query_heads // args.kv_heads
    assert args.query_heads % args.kv_heads == 0
    adapter = NativeKVAdapter(
        method=args.method,
        baseline_root=args.source_root,
        native_extension_root=args.extension_root,
        kivi_implementation="official",
        k_bits=2,
        v_bits=2,
        group_size=128,
        residual_length=128,
    )
    torch.manual_seed(7)
    query = torch.randn(1, args.query_heads, 1, 128, device="cuda", dtype=torch.float16)
    keys = torch.randn(1, args.kv_heads, 256, 128, device="cuda", dtype=torch.float16)
    weights = torch.softmax(
        torch.randn(1, args.query_heads, 1, 256, device="cuda", dtype=torch.float16),
        dim=-1,
    )
    values = torch.randn_like(keys)

    def packed_bmm(left, packed, scale, minimum, bits):
        if args.method == "ott":
            packed = _repeat_heads(packed, ratio)
            scale = _repeat_heads(scale, ratio)
            minimum = _repeat_heads(minimum, ratio)
        return adapter.packed_bmm(128, left, packed, scale, minimum, bits)

    for bits in (2, 4):
        key_q, key_s, key_m = adapter.quantize(
            keys.transpose(2, 3).contiguous(), 128, bits
        )
        value_q, value_s, value_m = adapter.quantize(values, 128, bits)
        key_deq = _dequant_last(key_q, key_s, key_m, bits, 128)
        value_deq = _dequant_last(value_q, value_s, value_m, bits, 128)
        key_ref = query @ _repeat_heads(key_deq, ratio)
        value_ref = weights @ _repeat_heads(value_deq, ratio)
        key_out = packed_bmm(query, key_q, key_s, key_m, bits)
        value_out = packed_bmm(weights, value_q, value_s, value_m, bits)
        key_error = float((key_out - key_ref).abs().max())
        value_error = float((value_out - value_ref).abs().max())
        assert torch.isfinite(key_out).all() and torch.isfinite(value_out).all()
        assert key_error <= 0.0625 and value_error <= 0.0625
        print(
            f"{args.method} QH={args.query_heads} KVH={args.kv_heads} "
            f"INT{bits}: max|K|={key_error:.6f}, max|V|={value_error:.6f}"
        )


if __name__ == "__main__":
    main()
