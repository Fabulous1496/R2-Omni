from __future__ import annotations

from collections.abc import Sequence

import torch

from .triton_runtime import triton_available

if triton_available():
    import triton
    import triton.language as tl

    @triton.jit
    def _fwht_stage(value, GROUPS: tl.constexpr, STRIDE: tl.constexpr):
        pair = tl.permute(
            tl.reshape(value, (GROUPS, 2, STRIDE)), (0, 2, 1)
        )
        left, right = tl.split(pair)
        joined = tl.join(left + right, left - right)
        return tl.reshape(tl.permute(joined, (0, 2, 1)), (128,))

    @triton.jit
    def _fwht_128(value):
        """Register-only normalized FWHT for one 128-wide attention head."""
        value = _fwht_stage(value, 64, 1)
        value = _fwht_stage(value, 32, 2)
        value = _fwht_stage(value, 16, 4)
        value = _fwht_stage(value, 8, 8)
        value = _fwht_stage(value, 4, 16)
        value = _fwht_stage(value, 2, 32)
        value = _fwht_stage(value, 1, 64)
        return value * 0.08838834764831845

    @triton.jit
    def _fwht_128_kernel(input_ptr, output_ptr, ROWS, D: tl.constexpr):
        row = tl.program_id(0)
        d = tl.arange(0, D)
        value = tl.load(input_ptr + row * D + d, mask=row < ROWS, other=0.0).to(
            tl.float32
        )
        value = _fwht_128(value)
        tl.store(output_ptr + row * D + d, value, mask=row < ROWS)

    @triton.jit
    def _qwen_qk_mrope_kernel(
        q_ptr,
        k_ptr,
        q_out_ptr,
        k_out_ptr,
        cos_ptr,
        sin_ptr,
        COS_AXIS_STRIDE: tl.constexpr,
        COS_TOKEN_STRIDE: tl.constexpr,
        COS_D_STRIDE: tl.constexpr,
        SIN_AXIS_STRIDE: tl.constexpr,
        SIN_TOKEN_STRIDE: tl.constexpr,
        SIN_D_STRIDE: tl.constexpr,
        HQ: tl.constexpr,
        HKV: tl.constexpr,
        TOKENS: tl.constexpr,
        D: tl.constexpr,
        SECTION_0: tl.constexpr,
        SECTION_1: tl.constexpr,
        APPLY_HADAMARD: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)
        head = row // TOKENS
        token = row - head * TOKENS
        is_query = head < HQ
        is_key = (head >= HQ) & (head < HQ + HKV)
        local_head = tl.where(is_query, head, head - HQ)
        d = tl.arange(0, BLOCK_D)
        valid = d < D
        other_d = tl.where(d < D // 2, d + D // 2, d - D // 2)
        row_offset = (local_head * TOKENS + token) * D
        offset = row_offset + d
        other_offset = row_offset + other_d
        value = tl.load(q_ptr + offset, mask=is_query & valid, other=0.0).to(tl.float32)
        value += tl.load(
            k_ptr + offset,
            mask=is_key & valid,
            other=0.0,
        ).to(tl.float32)
        other = tl.load(
            q_ptr + other_offset, mask=is_query & valid, other=0.0
        ).to(tl.float32)
        other += tl.load(
            k_ptr + other_offset,
            mask=is_key & valid,
            other=0.0,
        ).to(tl.float32)
        rotated = tl.where(d < D // 2, -other, other)
        section_d = d % (D // 2)
        axis = tl.where(
            section_d < SECTION_0,
            0,
            tl.where(section_d < SECTION_0 + SECTION_1, 1, 2),
        )
        cosine = tl.load(
            cos_ptr
            + axis * COS_AXIS_STRIDE
            + token * COS_TOKEN_STRIDE
            + d * COS_D_STRIDE,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        sine = tl.load(
            sin_ptr
            + axis * SIN_AXIS_STRIDE
            + token * SIN_TOKEN_STRIDE
            + d * SIN_D_STRIDE,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        result = value * cosine + rotated * sine
        if APPLY_HADAMARD:
            result = _fwht_128(result)
        tl.store(q_out_ptr + offset, result, mask=is_query & valid)
        tl.store(
            k_out_ptr + offset,
            result,
            mask=is_key & valid,
        )

    @triton.jit
    def _qwen_batched_qk_mrope_kernel(
        q_ptr,
        k_ptr,
        q_out_ptr,
        k_out_ptr,
        cos_ptr,
        sin_ptr,
        COS_AXIS_STRIDE: tl.constexpr,
        COS_BATCH_STRIDE: tl.constexpr,
        COS_TOKEN_STRIDE: tl.constexpr,
        COS_D_STRIDE: tl.constexpr,
        SIN_AXIS_STRIDE: tl.constexpr,
        SIN_BATCH_STRIDE: tl.constexpr,
        SIN_TOKEN_STRIDE: tl.constexpr,
        SIN_D_STRIDE: tl.constexpr,
        HQ: tl.constexpr,
        HKV: tl.constexpr,
        TOKENS: tl.constexpr,
        D: tl.constexpr,
        SECTION_0: tl.constexpr,
        SECTION_1: tl.constexpr,
        APPLY_HADAMARD: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)
        rows_per_request = (HQ + HKV) * TOKENS
        request = row // rows_per_request
        local_row = row - request * rows_per_request
        head = local_row // TOKENS
        token = local_row - head * TOKENS
        is_query = head < HQ
        local_head = tl.where(is_query, head, head - HQ)
        d = tl.arange(0, BLOCK_D)
        valid = d < D
        other_d = tl.where(d < D // 2, d + D // 2, d - D // 2)
        q_row = (request * HQ + local_head) * TOKENS + token
        k_row = (request * HKV + local_head) * TOKENS + token
        q_offset = q_row * D + d
        q_other_offset = q_row * D + other_d
        k_offset = k_row * D + d
        k_other_offset = k_row * D + other_d
        value = tl.load(q_ptr + q_offset, mask=is_query & valid, other=0.0).to(
            tl.float32
        )
        value += tl.load(
            k_ptr + k_offset, mask=(~is_query) & valid, other=0.0
        ).to(tl.float32)
        other = tl.load(
            q_ptr + q_other_offset, mask=is_query & valid, other=0.0
        ).to(tl.float32)
        other += tl.load(
            k_ptr + k_other_offset,
            mask=(~is_query) & valid,
            other=0.0,
        ).to(tl.float32)
        rotated = tl.where(d < D // 2, -other, other)
        section_d = d % (D // 2)
        axis = tl.where(
            section_d < SECTION_0,
            0,
            tl.where(section_d < SECTION_0 + SECTION_1, 1, 2),
        )
        cosine = tl.load(
            cos_ptr
            + axis * COS_AXIS_STRIDE
            + request * COS_BATCH_STRIDE
            + token * COS_TOKEN_STRIDE
            + d * COS_D_STRIDE,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        sine = tl.load(
            sin_ptr
            + axis * SIN_AXIS_STRIDE
            + request * SIN_BATCH_STRIDE
            + token * SIN_TOKEN_STRIDE
            + d * SIN_D_STRIDE,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        result = value * cosine + rotated * sine
        if APPLY_HADAMARD:
            result = _fwht_128(result)
        tl.store(q_out_ptr + q_offset, result, mask=is_query & valid)
        tl.store(k_out_ptr + k_offset, result, mask=(~is_query) & valid)


def qwen_qk_mrope(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: Sequence[int],
    *,
    output: tuple[torch.Tensor, torch.Tensor] | None = None,
    apply_hadamard: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Qwen multimodal RoPE to one or more tokens in one Triton launch."""
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("Qwen fused mRoPE requires rank-4 Q/K")
    if query.shape[0] != key.shape[0]:
        raise ValueError("Qwen fused mRoPE requires equal Q/K batch sizes")
    if query.shape[2] <= 0 or key.shape[2] != query.shape[2]:
        raise ValueError("Qwen fused mRoPE requires equal positive Q/K token counts")
    if query.shape[-1] != 128 or key.shape[-1] != 128:
        raise ValueError("Qwen fused mRoPE currently requires head_dim=128")
    if query.device != key.device or query.dtype != key.dtype:
        raise ValueError("Q/K device and dtype must match")
    if cos.device != query.device or sin.device != query.device:
        raise ValueError("Q/K and rotary tensors must share a device")
    if cos.shape != sin.shape or cos.ndim != 4 or cos.shape[0] != 3:
        raise ValueError("Qwen fused mRoPE requires cos/sin [3,B,S,D]")
    tokens = int(query.shape[2])
    batch_size = int(query.shape[0])
    if cos.shape[1] != batch_size or cos.shape[-2:] != (tokens, 128):
        raise ValueError("Qwen fused mRoPE rotary positions must match Q/K tokens")
    sections = tuple(int(value) for value in mrope_section)
    if len(sections) != 3 or sum(sections) != 64:
        raise ValueError("Qwen mRoPE sections must contain three values summing to 64")
    if not query.is_cuda or not triton_available():
        raise RuntimeError("fused Qwen mRoPE requires Triton CUDA")
    query = query.contiguous()
    key = key.contiguous()
    if output is None:
        query_out, key_out = torch.empty_like(query), torch.empty_like(key)
    else:
        query_out, key_out = output
        if query_out.shape != query.shape or key_out.shape != key.shape:
            raise ValueError("fused Qwen mRoPE output shapes must match Q/K")
        if (
            query_out.device != query.device
            or key_out.device != key.device
            or query_out.dtype != query.dtype
            or key_out.dtype != key.dtype
        ):
            raise ValueError("fused Qwen mRoPE outputs must match Q/K device and dtype")
    if batch_size == 1:
        # Keep the quality-critical production launch byte-for-byte unchanged.
        _qwen_qk_mrope_kernel[
            ((int(query.shape[1]) + int(key.shape[1])) * tokens,)
        ](
            query,
            key,
            query_out,
            key_out,
            cos,
            sin,
            COS_AXIS_STRIDE=int(cos.stride(0)),
            COS_TOKEN_STRIDE=int(cos.stride(-2)),
            COS_D_STRIDE=int(cos.stride(-1)),
            SIN_AXIS_STRIDE=int(sin.stride(0)),
            SIN_TOKEN_STRIDE=int(sin.stride(-2)),
            SIN_D_STRIDE=int(sin.stride(-1)),
            HQ=int(query.shape[1]),
            HKV=int(key.shape[1]),
            TOKENS=tokens,
            D=128,
            SECTION_0=sections[0],
            SECTION_1=sections[1],
            APPLY_HADAMARD=bool(apply_hadamard),
            BLOCK_D=128,
            num_warps=4,
        )
    else:
        _qwen_batched_qk_mrope_kernel[
            (
                batch_size
                * (int(query.shape[1]) + int(key.shape[1]))
                * tokens,
            )
        ](
            query,
            key,
            query_out,
            key_out,
            cos,
            sin,
            COS_AXIS_STRIDE=int(cos.stride(0)),
            COS_BATCH_STRIDE=int(cos.stride(1)),
            COS_TOKEN_STRIDE=int(cos.stride(-2)),
            COS_D_STRIDE=int(cos.stride(-1)),
            SIN_AXIS_STRIDE=int(sin.stride(0)),
            SIN_BATCH_STRIDE=int(sin.stride(1)),
            SIN_TOKEN_STRIDE=int(sin.stride(-2)),
            SIN_D_STRIDE=int(sin.stride(-1)),
            HQ=int(query.shape[1]),
            HKV=int(key.shape[1]),
            TOKENS=tokens,
            D=128,
            SECTION_0=sections[0],
            SECTION_1=sections[1],
            APPLY_HADAMARD=bool(apply_hadamard),
            BLOCK_D=128,
            num_warps=4,
        )
    return query_out, key_out


def triton_fwht_128(
    tensor: torch.Tensor, *, output: torch.Tensor | None = None
) -> torch.Tensor:
    """Apply a normalized 128-wide FWHT in one Triton launch."""
    if tensor.shape[-1] != 128 or not torch.is_floating_point(tensor):
        raise ValueError("fused FWHT requires a floating tensor with last dimension 128")
    if not tensor.is_cuda or not triton_available():
        raise RuntimeError("fused FWHT requires Triton CUDA")
    tensor = tensor.contiguous()
    if output is None:
        output = torch.empty_like(tensor)
    elif output.shape != tensor.shape or output.device != tensor.device or output.dtype != tensor.dtype:
        raise ValueError("fused FWHT output must match input shape, device, and dtype")
    rows = tensor.numel() // 128
    _fwht_128_kernel[(rows,)](
        tensor,
        output,
        rows,
        D=128,
        num_warps=4,
    )
    return output
