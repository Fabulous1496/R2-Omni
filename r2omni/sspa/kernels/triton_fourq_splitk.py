from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import torch

from .triton_runtime import triton_available
from ..runtime.precision_pool import ThreePrecisionCache

if triton_available():
    import triton
    import triton.language as tl
    from .triton_qwen_mrope import _fwht_128

    @triton.jit
    def _fourq_join_channels(x0, x1, x2, x3):
        """Restore four interleaved channel lanes as one contiguous tensor."""
        return tl.interleave(
            tl.interleave(x0, x2),
            tl.interleave(x1, x3),
        )

    @triton.jit
    def _fourq_qk(
        q0, q1, q2, q3, k0, k1, k2, k3,
        USE_TENSOR_CORES: tl.constexpr,
        FUSE_TENSOR_CORE_DOTS: tl.constexpr,
    ):
        if USE_TENSOR_CORES:
            if FUSE_TENSOR_CORE_DOTS:
                query = _fourq_join_channels(q0, q1, q2, q3)
                key = _fourq_join_channels(k0, k1, k2, k3)
                return tl.dot(
                    query.to(tl.float16),
                    tl.trans(key.to(tl.float16)),
                    out_dtype=tl.float32,
                )
            return (
                tl.dot(q0.to(tl.float16), tl.trans(k0.to(tl.float16)), out_dtype=tl.float32)
                + tl.dot(q1.to(tl.float16), tl.trans(k1.to(tl.float16)), out_dtype=tl.float32)
                + tl.dot(q2.to(tl.float16), tl.trans(k2.to(tl.float16)), out_dtype=tl.float32)
                + tl.dot(q3.to(tl.float16), tl.trans(k3.to(tl.float16)), out_dtype=tl.float32)
            )
        return tl.sum(
            q0[:, None, :] * k0[None, :, :]
            + q1[:, None, :] * k1[None, :, :]
            + q2[:, None, :] * k2[None, :, :]
            + q3[:, None, :] * k3[None, :, :],
            axis=2,
        )

    @triton.jit
    def _fourq_pv(p, value, USE_TENSOR_CORES: tl.constexpr):
        if USE_TENSOR_CORES:
            return tl.dot(
                p.to(tl.float16), value.to(tl.float16), out_dtype=tl.float32
            )
        return tl.sum(p[:, :, None] * value[None, :, :], axis=1)

    @triton.jit
    def _fourq_compact_k_metadata(
        scale_ptr, zero_ptr, kv_h, superblocks, token, d0, d1, d2, d3,
        SUPERBLOCK_SIZE: tl.constexpr,
        METADATA_TILE_N: tl.constexpr,
    ):
        """Load the per-channel affine row selected by each token."""
        token_block = token // SUPERBLOCK_SIZE
        # A 128-token attention tile spans at most 5/3/2 metadata rows for
        # 32/64/128-token calibration. Load every row once and broadcast it to
        # matching tokens instead of issuing one metadata load per token.
        first_block = tl.min(token_block, axis=0)
        first = kv_h * superblocks * 128 + first_block * 128
        s0 = tl.load(scale_ptr + first + d0).to(tl.float32)[None, :]
        s1 = tl.load(scale_ptr + first + d1).to(tl.float32)[None, :]
        s2 = tl.load(scale_ptr + first + d2).to(tl.float32)[None, :]
        s3 = tl.load(scale_ptr + first + d3).to(tl.float32)[None, :]
        z0 = tl.load(zero_ptr + first + d0).to(tl.float32)[None, :]
        z1 = tl.load(zero_ptr + first + d1).to(tl.float32)[None, :]
        z2 = tl.load(zero_ptr + first + d2).to(tl.float32)[None, :]
        z3 = tl.load(zero_ptr + first + d3).to(tl.float32)[None, :]
        for row_offset in tl.static_range(
            1, METADATA_TILE_N // SUPERBLOCK_SIZE + 1
        ):
            row = first_block + row_offset
            row_valid = row < superblocks
            base = kv_h * superblocks * 128 + row * 128
            ns0 = tl.load(scale_ptr + base + d0, mask=row_valid, other=0.0).to(tl.float32)
            ns1 = tl.load(scale_ptr + base + d1, mask=row_valid, other=0.0).to(tl.float32)
            ns2 = tl.load(scale_ptr + base + d2, mask=row_valid, other=0.0).to(tl.float32)
            ns3 = tl.load(scale_ptr + base + d3, mask=row_valid, other=0.0).to(tl.float32)
            nz0 = tl.load(zero_ptr + base + d0, mask=row_valid, other=0).to(tl.float32)
            nz1 = tl.load(zero_ptr + base + d1, mask=row_valid, other=0).to(tl.float32)
            nz2 = tl.load(zero_ptr + base + d2, mask=row_valid, other=0).to(tl.float32)
            nz3 = tl.load(zero_ptr + base + d3, mask=row_valid, other=0).to(tl.float32)
            use_row = token_block[:, None] == row
            s0, s1 = tl.where(use_row, ns0[None, :], s0), tl.where(use_row, ns1[None, :], s1)
            s2, s3 = tl.where(use_row, ns2[None, :], s2), tl.where(use_row, ns3[None, :], s3)
            z0, z1 = tl.where(use_row, nz0[None, :], z0), tl.where(use_row, nz1[None, :], z1)
            z2, z3 = tl.where(use_row, nz2[None, :], z2), tl.where(use_row, nz3[None, :], z3)
        return s0, s1, s2, s3, z0, z1, z2, z3

    @triton.jit
    def _full_head_compact_k_metadata(
        scale_ptr,
        zero_ptr,
        kv_h,
        superblocks,
        token,
        d,
        SUPERBLOCK_SIZE: tl.constexpr,
        METADATA_TILE_N: tl.constexpr,
        D: tl.constexpr,
    ):
        """Load compact INT4-K metadata directly in contiguous head order."""
        token_block = token // SUPERBLOCK_SIZE
        first_block = tl.min(token_block, axis=0)
        first = kv_h * superblocks * D + first_block * D
        scale = tl.load(scale_ptr + first + d).to(tl.float32)[None, :]
        zero = tl.load(zero_ptr + first + d).to(tl.float32)[None, :]
        for row_offset in tl.static_range(
            1, METADATA_TILE_N // SUPERBLOCK_SIZE + 1
        ):
            row = first_block + row_offset
            row_valid = row < superblocks
            base = kv_h * superblocks * D + row * D
            next_scale = tl.load(
                scale_ptr + base + d, mask=row_valid, other=0.0
            ).to(tl.float32)
            next_zero = tl.load(
                zero_ptr + base + d, mask=row_valid, other=0
            ).to(tl.float32)
            use_row = token_block[:, None] == row
            scale = tl.where(use_row, next_scale[None, :], scale)
            zero = tl.where(use_row, next_zero[None, :], zero)
        return scale, zero

    @triton.jit(do_not_specialize=[
        "S_FP", "S_RECENT", "S_I4", "S_I2",
        "N_FP", "N_RECENT", "N_I4", "K_CAPACITY", "K_SUPERBLOCKS",
        "K_FP_OFFSET", "K_I4_OFFSET", "K_I2_OFFSET", "V_I4_CAPACITY",
        "V_I2_CAPACITY", "DO_APPEND", "TOTAL", "STRIDE",
    ])
    def _full_head_page_partial_kernel(
        q_ptr, fp_k_ptr, fp_ks_ptr, fp_kz_ptr, fp_v_ptr,
        recent_k_ptr, recent_v_ptr, recent_len_ptr,
        append_k_ptr, append_v_ptr, DO_APPEND,
        i4_k_ptr, i4_ks_ptr, i4_kz_ptr, i4_v_ptr, i4_vs_ptr,
        i2_k_ptr, i2_ks_ptr, i2_kz_ptr, i2_v_ptr, i2_vs_ptr, i2_vz_ptr,
        pm_ptr, pl_ptr, po_ptr, batch_strides_ptr,
        S_FP, S_RECENT, S_I4, S_I2,
        N_FP, N_RECENT, N_I4, TOTAL, STRIDE,
        D: tl.constexpr, D2: tl.constexpr, D4: tl.constexpr,
        GQA: tl.constexpr, Q_BLOCK: tl.constexpr,
        GROUP_SIZE: tl.constexpr, GROUPS: tl.constexpr,
        K_CAPACITY, K_SUPERBLOCKS, K_FP_OFFSET, K_I4_OFFSET, K_I2_OFFSET,
        K_SUPERBLOCK_SIZE: tl.constexpr,
        V_I4_CAPACITY, V_I2_CAPACITY,
        V_COMPACT_INT4: tl.constexpr, V_COMPACT_INT2: tl.constexpr,
        BLOCK_N: tl.constexpr, SM_SCALE: tl.constexpr,
    ):
        """Production INT4-K mixed-V path in native contiguous head order.

        This throughput specialization avoids splitting every 128-D head into
        four 32-D lanes only to interleave them again before the Tensor-Core
        QK/PV operations.  It is selected only for homogeneous batch>1
        executors; the quality-critical batch=1 kernel remains unchanged.
        """
        request = tl.program_id(0)
        kv_h = tl.program_id(1)
        block = tl.program_id(2)
        q_ptr += request * tl.load(batch_strides_ptr + 0)
        fp_k_ptr += request * tl.load(batch_strides_ptr + 1)
        fp_ks_ptr += request * tl.load(batch_strides_ptr + 2)
        fp_kz_ptr += request * tl.load(batch_strides_ptr + 3)
        fp_v_ptr += request * tl.load(batch_strides_ptr + 4)
        recent_k_ptr += request * tl.load(batch_strides_ptr + 5)
        recent_v_ptr += request * tl.load(batch_strides_ptr + 6)
        recent_len_ptr += request * tl.load(batch_strides_ptr + 7)
        append_k_ptr += request * tl.load(batch_strides_ptr + 8)
        append_v_ptr += request * tl.load(batch_strides_ptr + 9)
        i4_k_ptr += request * tl.load(batch_strides_ptr + 10)
        i4_ks_ptr += request * tl.load(batch_strides_ptr + 11)
        i4_kz_ptr += request * tl.load(batch_strides_ptr + 12)
        i4_v_ptr += request * tl.load(batch_strides_ptr + 13)
        i4_vs_ptr += request * tl.load(batch_strides_ptr + 14)
        i2_k_ptr += request * tl.load(batch_strides_ptr + 15)
        i2_ks_ptr += request * tl.load(batch_strides_ptr + 16)
        i2_kz_ptr += request * tl.load(batch_strides_ptr + 17)
        i2_v_ptr += request * tl.load(batch_strides_ptr + 18)
        i2_vs_ptr += request * tl.load(batch_strides_ptr + 19)
        i2_vz_ptr += request * tl.load(batch_strides_ptr + 20)
        pm_ptr += request * tl.load(batch_strides_ptr + 23)
        pl_ptr += request * tl.load(batch_strides_ptr + 24)
        po_ptr += request * tl.load(batch_strides_ptr + 25)

        q_local = tl.arange(0, Q_BLOCK)
        q_valid = q_local < GQA
        q_head = kv_h * GQA + q_local
        d = tl.arange(0, D)
        query = tl.load(
            q_ptr + q_head[:, None] * D + d[None, :],
            mask=q_valid[:, None],
            other=0.0,
        ).to(tl.float16)
        n = tl.arange(0, BLOCK_N)
        db2 = tl.arange(0, D2)
        db4 = tl.arange(0, D4)

        if block < N_FP:
            token = block * BLOCK_N + n
            valid = token < S_FP
            k_token = K_FP_OFFSET + token
            kbyte = (
                kv_h * K_CAPACITY * D2
                + k_token[:, None] * D2
                + db2[None, :]
            )
            packed_k = tl.load(kbyte + fp_k_ptr, mask=valid[:, None], other=0)
            kscale, kzero = _full_head_compact_k_metadata(
                fp_ks_ptr, fp_kz_ptr, kv_h, K_SUPERBLOCKS, k_token, d,
                K_SUPERBLOCK_SIZE, BLOCK_N, D,
            )
            kcode = tl.interleave(packed_k & 15, (packed_k >> 4) & 15)
            key = (kcode.to(tl.float32) - kzero) * kscale
            base = kv_h * S_FP * D + token[:, None] * D
            value = tl.load(
                fp_v_ptr + base + d[None, :],
                mask=valid[:, None],
                other=0.0,
            ).to(tl.float32)
        elif block < N_FP + N_RECENT:
            local = block - N_FP
            token = local * BLOCK_N + n
            recent_length = tl.load(recent_len_ptr)
            staged = token < recent_length
            appended = (DO_APPEND != 0) & (token == recent_length)
            valid = staged | appended
            base = kv_h * S_RECENT * D + token[:, None] * D
            append_base = kv_h * D + d
            key = tl.load(
                recent_k_ptr + base + d[None, :],
                mask=staged[:, None],
                other=0.0,
            ).to(tl.float32)
            appended_key = tl.load(
                append_k_ptr + append_base,
                mask=DO_APPEND != 0,
                other=0.0,
            ).to(tl.float32)
            key += tl.where(appended[:, None], appended_key[None, :], 0.0)
            value = tl.load(
                recent_v_ptr + base + d[None, :],
                mask=staged[:, None],
                other=0.0,
            ).to(tl.float32)
            appended_value = tl.load(
                append_v_ptr + append_base,
                mask=DO_APPEND != 0,
                other=0.0,
            ).to(tl.float32)
            value += tl.where(
                appended[:, None], appended_value[None, :], 0.0
            )
        elif block < N_FP + N_RECENT + N_I4:
            local = block - N_FP - N_RECENT
            token = local * BLOCK_N + n
            valid = token < S_I4
            k_token = K_I4_OFFSET + token
            kbyte = (
                kv_h * K_CAPACITY * D2
                + k_token[:, None] * D2
                + db2[None, :]
            )
            packed_k = tl.load(kbyte + i4_k_ptr, mask=valid[:, None], other=0)
            kscale, kzero = _full_head_compact_k_metadata(
                i4_ks_ptr, i4_kz_ptr, kv_h, K_SUPERBLOCKS, k_token, d,
                K_SUPERBLOCK_SIZE, BLOCK_N, D,
            )
            kcode = tl.interleave(packed_k & 15, (packed_k >> 4) & 15)
            key = (kcode.to(tl.float32) - kzero) * kscale
            i4_v_stride = V_I4_CAPACITY if V_COMPACT_INT4 else S_I4
            i4_vbyte = (
                kv_h * i4_v_stride * D2 + token[:, None] * D2
                + db2[None, :]
            )
            i4_packed_v = tl.load(
                i4_v_ptr + i4_vbyte, mask=valid[:, None], other=0
            )
            i4_vcode = tl.interleave(
                i4_packed_v & 15, (i4_packed_v >> 4) & 15
            )
            signed_v = tl.where(
                i4_vcode >= 8,
                i4_vcode.to(tl.int16) - 16,
                i4_vcode.to(tl.int16),
            ).to(tl.float32)
            i4_vscale = tl.load(
                i4_vs_ptr + kv_h * i4_v_stride + token,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            value = signed_v * i4_vscale[:, None]
        else:
            local = block - N_FP - N_RECENT - N_I4
            token = local * BLOCK_N + n
            valid = token < S_I2
            k_token = K_I2_OFFSET + token
            kbyte = (
                kv_h * K_CAPACITY * D2
                + k_token[:, None] * D2
                + db2[None, :]
            )
            packed_k = tl.load(kbyte + i2_k_ptr, mask=valid[:, None], other=0)
            kscale, kzero = _full_head_compact_k_metadata(
                i2_ks_ptr, i2_kz_ptr, kv_h, K_SUPERBLOCKS, k_token, d,
                K_SUPERBLOCK_SIZE, BLOCK_N, D,
            )
            kcode = tl.interleave(packed_k & 15, (packed_k >> 4) & 15)
            key = (kcode.to(tl.float32) - kzero) * kscale
            i2_v_stride = V_I2_CAPACITY if V_COMPACT_INT2 else S_I2
            i2_vbyte = (
                kv_h * i2_v_stride * D4 + token[:, None] * D4
                + db4[None, :]
            )
            i2_packed_v = tl.load(
                i2_v_ptr + i2_vbyte, mask=valid[:, None], other=0
            )
            group = (db4 * 4) // GROUP_SIZE
            vmeta = (
                kv_h * i2_v_stride * GROUPS
                + token[:, None] * GROUPS
                + group[None, :]
            )
            i2_vscale = tl.load(
                i2_vs_ptr + vmeta, mask=valid[:, None], other=0.0
            ).to(tl.float32)
            i2_vzero = tl.load(
                i2_vz_ptr + vmeta, mask=valid[:, None], other=0
            ).to(tl.float32)
            i2_v0 = (
                ((i2_packed_v >> 0) & 3).to(tl.float32) - i2_vzero
            ) * i2_vscale
            i2_v1 = (
                ((i2_packed_v >> 2) & 3).to(tl.float32) - i2_vzero
            ) * i2_vscale
            i2_v2 = (
                ((i2_packed_v >> 4) & 3).to(tl.float32) - i2_vzero
            ) * i2_vscale
            i2_v3 = (
                ((i2_packed_v >> 6) & 3).to(tl.float32) - i2_vzero
            ) * i2_vscale
            value = _fourq_join_channels(i2_v0, i2_v1, i2_v2, i2_v3)

        score = tl.dot(query, tl.trans(key.to(tl.float16)), out_dtype=tl.float32)
        score *= SM_SCALE
        score = tl.where(valid[None, :], score, -float("inf"))
        maximum = tl.max(score, axis=1)
        probability = tl.where(
            valid[None, :], tl.exp(score - maximum[:, None]), 0.0
        )
        denominator = tl.sum(probability, axis=1)
        output = tl.dot(
            probability.to(tl.float16),
            value.to(tl.float16),
            out_dtype=tl.float32,
        )
        state = q_head[:, None] * STRIDE + block
        tl.store(pm_ptr + q_head * STRIDE + block, maximum, mask=q_valid)
        tl.store(pl_ptr + q_head * STRIDE + block, denominator, mask=q_valid)
        tl.store(
            po_ptr + state * D + d[None, :],
            output,
            mask=q_valid[:, None],
        )

    @triton.jit
    def _fourq_apply_sparse_k_residual(
        k0,
        k1,
        k2,
        k3,
        residual_ptr,
        indices_ptr,
        kv_h,
        k_token,
        token_valid,
        dp,
        K_CAPACITY: tl.constexpr,
        R: tl.constexpr,
    ):
        """Scatter a few exact channel residuals into the existing K registers."""
        for r in tl.static_range(0, R):
            channel = tl.load(indices_ptr + kv_h * R + r).to(tl.int32)
            group = channel // 4
            lane = channel % 4
            residual = tl.load(
                residual_ptr
                + kv_h * K_CAPACITY * R
                + k_token * R
                + r,
                mask=token_valid,
                other=0.0,
            ).to(tl.float32)
            selected_group = dp[None, :] == group
            k0 += tl.where(selected_group & (lane == 0), residual[:, None], 0.0)
            k1 += tl.where(selected_group & (lane == 1), residual[:, None], 0.0)
            k2 += tl.where(selected_group & (lane == 2), residual[:, None], 0.0)
            k3 += tl.where(selected_group & (lane == 3), residual[:, None], 0.0)
        return k0, k1, k2, k3

    @triton.jit(do_not_specialize=["POSITION", "GLOBAL_INDEX", "LOGICAL_POSITION"])
    def _append_recent_fp_kernel(
        recent_k,
        recent_v,
        global_indices,
        logical_positions,
        length_device,
        page_lengths,
        new_k,
        new_v,
        POSITION,
        GLOBAL_INDEX,
        LOGICAL_POSITION,
        H_KV: tl.constexpr,
        D: tl.constexpr,
        CAPACITY: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offset = tl.arange(0, BLOCK)
        payload = H_KV * D
        valid = offset < payload
        head = offset // D
        dim = offset % D
        destination = head * CAPACITY * D + POSITION * D + dim
        tl.store(recent_k + destination, tl.load(new_k + offset, mask=valid), mask=valid)
        tl.store(recent_v + destination, tl.load(new_v + offset, mask=valid), mask=valid)
        leader = offset == 0
        tl.store(global_indices + POSITION + offset, GLOBAL_INDEX, mask=leader)
        tl.store(logical_positions + POSITION + offset, LOGICAL_POSITION, mask=leader)
        tl.store(length_device + offset, POSITION + 1, mask=leader)
        tl.store(
            page_lengths + POSITION // PAGE_SIZE + offset,
            POSITION % PAGE_SIZE + 1,
            mask=leader,
        )

    @triton.jit(do_not_specialize=[
        "S_FP", "S_RECENT", "S_I4", "S_I2",
        "N_FP", "N_RECENT", "N_I4", "FP_I2_SUPERBLOCKS", "I4_I2_SUPERBLOCKS",
        "I2_SUPERBLOCKS", "K_CAPACITY", "K_SUPERBLOCKS", "K_FP_OFFSET",
        "K_I4_OFFSET", "K_I2_OFFSET", "STATIC_I4_TOKENS", "K_GENERATED_OFFSET",
        "V_I4_CAPACITY", "V_I2_CAPACITY", "DO_APPEND", "TOTAL", "STRIDE",
    ])
    def _fourq_page_partial_kernel(
        q_ptr, fp_k_ptr, fp_ks_ptr, fp_kz_ptr, fp_v_ptr,
        recent_k_ptr, recent_v_ptr, recent_len_ptr,
        append_k_ptr, append_v_ptr, DO_APPEND,
        i4_k_ptr, i4_ks_ptr, i4_kz_ptr, i4_v_ptr, i4_vs_ptr,
        i2_k_ptr, i2_ks_ptr, i2_kz_ptr, i2_v_ptr, i2_vs_ptr, i2_vz_ptr,
        k_residual_ptr, k_residual_indices_ptr,
        pm_ptr, pl_ptr, po_ptr, batch_strides_ptr,
        S_FP, S_RECENT, S_I4, S_I2,
        N_FP, N_RECENT, N_I4,
        TOTAL, STRIDE, D: tl.constexpr, D2: tl.constexpr, D4: tl.constexpr,
        GQA: tl.constexpr, Q_BLOCK: tl.constexpr,
        GROUP_SIZE: tl.constexpr, GROUPS: tl.constexpr,
        FP_I2_SUPERBLOCKS, I4_I2_SUPERBLOCKS, I2_SUPERBLOCKS,
        K_CAPACITY, K_SUPERBLOCKS, K_FP_OFFSET, K_I4_OFFSET, K_I2_OFFSET,
        STATIC_I4_TOKENS, K_GENERATED_OFFSET,
        K_SUPERBLOCK_SIZE: tl.constexpr,
        V_I4_CAPACITY, V_I2_CAPACITY,
        K_UNIFORM_INT2: tl.constexpr, K_COMPACT_INT2: tl.constexpr,
        K_UNIFORM_INT4: tl.constexpr, K_COMPACT_INT4: tl.constexpr,
        K_FULL_PRECISION: tl.constexpr,
        K_RESIDUAL_CHANNELS: tl.constexpr,
        V_COMPACT_INT4: tl.constexpr, V_COMPACT_INT2: tl.constexpr,
        USE_TENSOR_CORES_QK: tl.constexpr,
        USE_TENSOR_CORES_PV: tl.constexpr,
        FUSE_TENSOR_CORE_DOTS: tl.constexpr,
        BROADCAST_K_METADATA: tl.constexpr,
        SINGLE_REQUEST: tl.constexpr,
        TIGHT_METADATA_ROWS: tl.constexpr,
        BLOCK_N: tl.constexpr, SM_SCALE: tl.constexpr,
    ):
        request = tl.program_id(0)
        kv_h = tl.program_id(1)
        block = tl.program_id(2)
        # The production executor currently accepts exactly one request.  Keep
        # the generic strided layout available for future batched callers, but
        # compile its metadata loads and zero-offset pointer arithmetic out of
        # the single-request specialization used today.
        if not SINGLE_REQUEST:
            q_ptr += request * tl.load(batch_strides_ptr + 0)
            fp_k_ptr += request * tl.load(batch_strides_ptr + 1)
            fp_ks_ptr += request * tl.load(batch_strides_ptr + 2)
            fp_kz_ptr += request * tl.load(batch_strides_ptr + 3)
            fp_v_ptr += request * tl.load(batch_strides_ptr + 4)
            recent_k_ptr += request * tl.load(batch_strides_ptr + 5)
            recent_v_ptr += request * tl.load(batch_strides_ptr + 6)
            recent_len_ptr += request * tl.load(batch_strides_ptr + 7)
            append_k_ptr += request * tl.load(batch_strides_ptr + 8)
            append_v_ptr += request * tl.load(batch_strides_ptr + 9)
            i4_k_ptr += request * tl.load(batch_strides_ptr + 10)
            i4_ks_ptr += request * tl.load(batch_strides_ptr + 11)
            i4_kz_ptr += request * tl.load(batch_strides_ptr + 12)
            i4_v_ptr += request * tl.load(batch_strides_ptr + 13)
            i4_vs_ptr += request * tl.load(batch_strides_ptr + 14)
            i2_k_ptr += request * tl.load(batch_strides_ptr + 15)
            i2_ks_ptr += request * tl.load(batch_strides_ptr + 16)
            i2_kz_ptr += request * tl.load(batch_strides_ptr + 17)
            i2_v_ptr += request * tl.load(batch_strides_ptr + 18)
            i2_vs_ptr += request * tl.load(batch_strides_ptr + 19)
            i2_vz_ptr += request * tl.load(batch_strides_ptr + 20)
            k_residual_ptr += request * tl.load(batch_strides_ptr + 21)
            k_residual_indices_ptr += request * tl.load(batch_strides_ptr + 22)
            pm_ptr += request * tl.load(batch_strides_ptr + 23)
            pl_ptr += request * tl.load(batch_strides_ptr + 24)
            po_ptr += request * tl.load(batch_strides_ptr + 25)
        q_local = tl.arange(0, Q_BLOCK)
        q_valid = q_local < GQA
        q_head = kv_h * GQA + q_local
        dp = tl.arange(0, D4)
        d0, d1, d2, d3 = dp * 4, dp * 4 + 1, dp * 4 + 2, dp * 4 + 3
        q0 = tl.load(q_ptr + q_head[:, None] * D + d0[None, :], mask=q_valid[:, None], other=0.0).to(tl.float32)
        q1 = tl.load(q_ptr + q_head[:, None] * D + d1[None, :], mask=q_valid[:, None], other=0.0).to(tl.float32)
        q2 = tl.load(q_ptr + q_head[:, None] * D + d2[None, :], mask=q_valid[:, None], other=0.0).to(tl.float32)
        q3 = tl.load(q_ptr + q_head[:, None] * D + d3[None, :], mask=q_valid[:, None], other=0.0).to(tl.float32)
        n = tl.arange(0, BLOCK_N)
        a0 = tl.zeros((Q_BLOCK, D4), tl.float32)
        a1 = tl.zeros((Q_BLOCK, D4), tl.float32)
        a2 = tl.zeros((Q_BLOCK, D4), tl.float32)
        a3 = tl.zeros((Q_BLOCK, D4), tl.float32)

        if block < N_FP:
            token = block * BLOCK_N + n
            valid = token < S_FP
            base = kv_h * S_FP * D + token[:, None] * D
            if K_FULL_PRECISION:
                k0 = tl.load(fp_k_ptr + base + d0[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
                k1 = tl.load(fp_k_ptr + base + d1[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
                k2 = tl.load(fp_k_ptr + base + d2[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
                k3 = tl.load(fp_k_ptr + base + d3[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
            elif K_UNIFORM_INT4:
                k_token = K_FP_OFFSET + token if K_COMPACT_INT4 else token
                k_stride = K_CAPACITY if K_COMPACT_INT4 else S_FP
                k_superblocks = K_SUPERBLOCKS if K_COMPACT_INT4 else FP_I2_SUPERBLOCKS
                byte0 = kv_h * k_stride * D2 + k_token[:, None] * D2 + (d0 // 2)[None, :]
                byte1 = byte0 + 1
                kb0 = tl.load(fp_k_ptr + byte0, mask=valid[:, None], other=0)
                kb1 = tl.load(fp_k_ptr + byte1, mask=valid[:, None], other=0)
                ks0, ks1, ks2, ks3, kz0, kz1, kz2, kz3 = _fourq_compact_k_metadata(
                    fp_ks_ptr, fp_kz_ptr, kv_h, k_superblocks,
                    k_token, d0, d1, d2, d3, K_SUPERBLOCK_SIZE,
                    BLOCK_N if TIGHT_METADATA_ROWS else 128,
                )
                k0 = ((kb0 & 15).to(tl.float32) - kz0) * ks0
                k1 = (((kb0 >> 4) & 15).to(tl.float32) - kz1) * ks1
                k2 = ((kb1 & 15).to(tl.float32) - kz2) * ks2
                k3 = (((kb1 >> 4) & 15).to(tl.float32) - kz3) * ks3
            elif K_UNIFORM_INT2:
                if K_COMPACT_INT2:
                    k_token = K_FP_OFFSET + token
                    k_stride = K_CAPACITY
                    k_superblocks = K_SUPERBLOCKS
                else:
                    k_token = token
                    k_stride = S_FP
                    k_superblocks = FP_I2_SUPERBLOCKS
                byte = kv_h * k_stride * D4 + k_token[:, None] * D4 + dp[None, :]
                kb = tl.load(fp_k_ptr + byte, mask=valid[:, None], other=0)
                if K_COMPACT_INT2 and BROADCAST_K_METADATA:
                    ks0, ks1, ks2, ks3, kz0, kz1, kz2, kz3 = _fourq_compact_k_metadata(
                        fp_ks_ptr, fp_kz_ptr, kv_h, k_superblocks,
                        k_token, d0, d1, d2, d3, K_SUPERBLOCK_SIZE,
                        BLOCK_N if TIGHT_METADATA_ROWS else 128,
                    )
                else:
                    meta = kv_h * k_superblocks * D + (k_token // K_SUPERBLOCK_SIZE)[:, None] * D
                    ks0 = tl.load(fp_ks_ptr + meta + d0[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
                    ks1 = tl.load(fp_ks_ptr + meta + d1[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
                    ks2 = tl.load(fp_ks_ptr + meta + d2[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
                    ks3 = tl.load(fp_ks_ptr + meta + d3[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
                    kz0 = tl.load(fp_kz_ptr + meta + d0[None, :], mask=valid[:, None], other=0).to(tl.float32)
                    kz1 = tl.load(fp_kz_ptr + meta + d1[None, :], mask=valid[:, None], other=0).to(tl.float32)
                    kz2 = tl.load(fp_kz_ptr + meta + d2[None, :], mask=valid[:, None], other=0).to(tl.float32)
                    kz3 = tl.load(fp_kz_ptr + meta + d3[None, :], mask=valid[:, None], other=0).to(tl.float32)
                if K_COMPACT_INT2:
                    k0 = ((kb >> 0) & 3).to(tl.float32) * ks0 + kz0
                    k1 = ((kb >> 2) & 3).to(tl.float32) * ks1 + kz1
                    k2 = ((kb >> 4) & 3).to(tl.float32) * ks2 + kz2
                    k3 = ((kb >> 6) & 3).to(tl.float32) * ks3 + kz3
                else:
                    k0 = (((kb >> 0) & 3).to(tl.float32) - kz0) * ks0
                    k1 = (((kb >> 2) & 3).to(tl.float32) - kz1) * ks1
                    k2 = (((kb >> 4) & 3).to(tl.float32) - kz2) * ks2
                    k3 = (((kb >> 6) & 3).to(tl.float32) - kz3) * ks3
            else:
                k0 = tl.load(fp_k_ptr + base + d0[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
                k1 = tl.load(fp_k_ptr + base + d1[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
                k2 = tl.load(fp_k_ptr + base + d2[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
                k3 = tl.load(fp_k_ptr + base + d3[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
            if K_COMPACT_INT2 and K_RESIDUAL_CHANNELS > 0:
                k0, k1, k2, k3 = _fourq_apply_sparse_k_residual(
                    k0,
                    k1,
                    k2,
                    k3,
                    k_residual_ptr,
                    k_residual_indices_ptr,
                    kv_h,
                    K_FP_OFFSET + token,
                    valid,
                    dp,
                    K_CAPACITY,
                    K_RESIDUAL_CHANNELS,
                )
            score = _fourq_qk(
                q0, q1, q2, q3, k0, k1, k2, k3,
                USE_TENSOR_CORES_QK, FUSE_TENSOR_CORE_DOTS,
            ) * SM_SCALE
            score = tl.where(valid[None, :], score, -float("inf"))
            maximum = tl.max(score, axis=1)
            p = tl.where(valid[None, :], tl.exp(score - maximum[:, None]), 0.0)
            v0 = tl.load(fp_v_ptr + base + d0[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
            v1 = tl.load(fp_v_ptr + base + d1[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
            v2 = tl.load(fp_v_ptr + base + d2[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
            v3 = tl.load(fp_v_ptr + base + d3[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
        elif block < N_FP + N_RECENT:
            local = block - N_FP
            token = local * BLOCK_N + n
            recent_length = tl.load(recent_len_ptr)
            staged = token < recent_length
            appended = (DO_APPEND != 0) & (token == recent_length)
            valid = staged | appended
            base = kv_h * S_RECENT * D + token[:, None] * D
            append_base = kv_h * D
            k0 = tl.load(recent_k_ptr + base + d0[None, :], mask=staged[:, None], other=0.0).to(tl.float32)
            k1 = tl.load(recent_k_ptr + base + d1[None, :], mask=staged[:, None], other=0.0).to(tl.float32)
            k2 = tl.load(recent_k_ptr + base + d2[None, :], mask=staged[:, None], other=0.0).to(tl.float32)
            k3 = tl.load(recent_k_ptr + base + d3[None, :], mask=staged[:, None], other=0.0).to(tl.float32)
            append_enabled = DO_APPEND != 0
            ak0 = tl.load(append_k_ptr + append_base + d0, mask=append_enabled, other=0.0).to(tl.float32)
            ak1 = tl.load(append_k_ptr + append_base + d1, mask=append_enabled, other=0.0).to(tl.float32)
            ak2 = tl.load(append_k_ptr + append_base + d2, mask=append_enabled, other=0.0).to(tl.float32)
            ak3 = tl.load(append_k_ptr + append_base + d3, mask=append_enabled, other=0.0).to(tl.float32)
            k0 += tl.where(appended[:, None], ak0[None, :], 0.0)
            k1 += tl.where(appended[:, None], ak1[None, :], 0.0)
            k2 += tl.where(appended[:, None], ak2[None, :], 0.0)
            k3 += tl.where(appended[:, None], ak3[None, :], 0.0)
            score = _fourq_qk(
                q0, q1, q2, q3, k0, k1, k2, k3,
                USE_TENSOR_CORES_QK, FUSE_TENSOR_CORE_DOTS,
            ) * SM_SCALE
            score = tl.where(valid[None, :], score, -float("inf"))
            maximum = tl.max(score, axis=1)
            p = tl.where(valid[None, :], tl.exp(score - maximum[:, None]), 0.0)
            v0 = tl.load(recent_v_ptr + base + d0[None, :], mask=staged[:, None], other=0.0).to(tl.float32)
            v1 = tl.load(recent_v_ptr + base + d1[None, :], mask=staged[:, None], other=0.0).to(tl.float32)
            v2 = tl.load(recent_v_ptr + base + d2[None, :], mask=staged[:, None], other=0.0).to(tl.float32)
            v3 = tl.load(recent_v_ptr + base + d3[None, :], mask=staged[:, None], other=0.0).to(tl.float32)
            av0 = tl.load(append_v_ptr + append_base + d0, mask=append_enabled, other=0.0).to(tl.float32)
            av1 = tl.load(append_v_ptr + append_base + d1, mask=append_enabled, other=0.0).to(tl.float32)
            av2 = tl.load(append_v_ptr + append_base + d2, mask=append_enabled, other=0.0).to(tl.float32)
            av3 = tl.load(append_v_ptr + append_base + d3, mask=append_enabled, other=0.0).to(tl.float32)
            v0 += tl.where(appended[:, None], av0[None, :], 0.0)
            v1 += tl.where(appended[:, None], av1[None, :], 0.0)
            v2 += tl.where(appended[:, None], av2[None, :], 0.0)
            v3 += tl.where(appended[:, None], av3[None, :], 0.0)
        elif block < N_FP + N_RECENT + N_I4:
            local = block - N_FP - N_RECENT
            token = local * BLOCK_N + n
            valid = token < S_I4
            if K_FULL_PRECISION:
                kbase = kv_h * S_I4 * D + token[:, None] * D
                k0 = tl.load(i2_k_ptr + kbase + d0[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
                k1 = tl.load(i2_k_ptr + kbase + d1[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
                k2 = tl.load(i2_k_ptr + kbase + d2[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
                k3 = tl.load(i2_k_ptr + kbase + d3[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
            elif K_UNIFORM_INT4:
                k_token = K_I4_OFFSET + token if K_COMPACT_INT4 else token
                k_stride = K_CAPACITY if K_COMPACT_INT4 else S_I4
                k_superblocks = K_SUPERBLOCKS if K_COMPACT_INT4 else I4_I2_SUPERBLOCKS
                kbyte0 = kv_h * k_stride * D2 + k_token[:, None] * D2 + (d0 // 2)[None, :]
                kbyte1 = kbyte0 + 1
                kb0 = tl.load(i4_k_ptr + kbyte0, mask=valid[:, None], other=0)
                kb1 = tl.load(i4_k_ptr + kbyte1, mask=valid[:, None], other=0)
                ks0, ks1, ks2, ks3, kz0, kz1, kz2, kz3 = _fourq_compact_k_metadata(
                    i4_ks_ptr, i4_kz_ptr, kv_h, k_superblocks,
                    k_token, d0, d1, d2, d3, K_SUPERBLOCK_SIZE,
                    BLOCK_N if TIGHT_METADATA_ROWS else 128,
                )
                k0 = ((kb0 & 15).to(tl.float32) - kz0) * ks0
                k1 = (((kb0 >> 4) & 15).to(tl.float32) - kz1) * ks1
                k2 = ((kb1 & 15).to(tl.float32) - kz2) * ks2
                k3 = (((kb1 >> 4) & 15).to(tl.float32) - kz3) * ks3
            elif K_UNIFORM_INT2:
                if K_COMPACT_INT2:
                    # Static INT4-V tokens occupy the middle compact-K
                    # segment. Generated INT4-V tokens are appended after all
                    # static tiers, so their physical K addresses have a
                    # separate tail origin.
                    k_token = tl.where(
                        token < STATIC_I4_TOKENS,
                        K_I4_OFFSET + token,
                        K_GENERATED_OFFSET + token - STATIC_I4_TOKENS,
                    )
                    k_stride = K_CAPACITY
                    k_superblocks = K_SUPERBLOCKS
                else:
                    k_token = token
                    k_stride = S_I4
                    k_superblocks = I4_I2_SUPERBLOCKS
                kbyte = kv_h * k_stride * D4 + k_token[:, None] * D4 + dp[None, :]
                kb = tl.load(i4_k_ptr + kbyte, mask=valid[:, None], other=0)
                if K_COMPACT_INT2 and BROADCAST_K_METADATA:
                    ks0, ks1, ks2, ks3, kz0, kz1, kz2, kz3 = _fourq_compact_k_metadata(
                        i4_ks_ptr, i4_kz_ptr, kv_h, k_superblocks,
                        k_token, d0, d1, d2, d3, K_SUPERBLOCK_SIZE,
                        BLOCK_N if TIGHT_METADATA_ROWS else 128,
                    )
                else:
                    meta = kv_h * k_superblocks * D + (k_token // K_SUPERBLOCK_SIZE)[:, None] * D
                    ks0 = tl.load(i4_ks_ptr + meta + d0[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
                    ks1 = tl.load(i4_ks_ptr + meta + d1[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
                    ks2 = tl.load(i4_ks_ptr + meta + d2[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
                    ks3 = tl.load(i4_ks_ptr + meta + d3[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
                    kz0 = tl.load(i4_kz_ptr + meta + d0[None, :], mask=valid[:, None], other=0).to(tl.float32)
                    kz1 = tl.load(i4_kz_ptr + meta + d1[None, :], mask=valid[:, None], other=0).to(tl.float32)
                    kz2 = tl.load(i4_kz_ptr + meta + d2[None, :], mask=valid[:, None], other=0).to(tl.float32)
                    kz3 = tl.load(i4_kz_ptr + meta + d3[None, :], mask=valid[:, None], other=0).to(tl.float32)
                if K_COMPACT_INT2:
                    k0 = ((kb >> 0) & 3).to(tl.float32) * ks0 + kz0
                    k1 = ((kb >> 2) & 3).to(tl.float32) * ks1 + kz1
                    k2 = ((kb >> 4) & 3).to(tl.float32) * ks2 + kz2
                    k3 = ((kb >> 6) & 3).to(tl.float32) * ks3 + kz3
                else:
                    k0 = (((kb >> 0) & 3).to(tl.float32) - kz0) * ks0
                    k1 = (((kb >> 2) & 3).to(tl.float32) - kz1) * ks1
                    k2 = (((kb >> 4) & 3).to(tl.float32) - kz2) * ks2
                    k3 = (((kb >> 6) & 3).to(tl.float32) - kz3) * ks3
            else:
                kbyte0 = kv_h * S_I4 * (D // 2) + token[:, None] * (D // 2) + (d0 // 2)[None, :]
                kbyte1 = kbyte0 + 1
                kb0 = tl.load(i4_k_ptr + kbyte0, mask=valid[:, None], other=0)
                kb1 = tl.load(i4_k_ptr + kbyte1, mask=valid[:, None], other=0)
                c0, c1, c2, c3 = kb0 & 15, (kb0 >> 4) & 15, kb1 & 15, (kb1 >> 4) & 15
                s0 = tl.load(i4_ks_ptr + kv_h * D + d0).to(tl.float32)
                s1 = tl.load(i4_ks_ptr + kv_h * D + d1).to(tl.float32)
                s2 = tl.load(i4_ks_ptr + kv_h * D + d2).to(tl.float32)
                s3 = tl.load(i4_ks_ptr + kv_h * D + d3).to(tl.float32)
                k0 = tl.where(c0 >= 8, c0.to(tl.int16) - 16, c0.to(tl.int16)).to(tl.float32) * s0[None, :]
                k1 = tl.where(c1 >= 8, c1.to(tl.int16) - 16, c1.to(tl.int16)).to(tl.float32) * s1[None, :]
                k2 = tl.where(c2 >= 8, c2.to(tl.int16) - 16, c2.to(tl.int16)).to(tl.float32) * s2[None, :]
                k3 = tl.where(c3 >= 8, c3.to(tl.int16) - 16, c3.to(tl.int16)).to(tl.float32) * s3[None, :]
            if K_COMPACT_INT2 and K_RESIDUAL_CHANNELS > 0:
                residual_token = tl.where(
                    token < STATIC_I4_TOKENS,
                    K_I4_OFFSET + token,
                    K_GENERATED_OFFSET + token - STATIC_I4_TOKENS,
                )
                k0, k1, k2, k3 = _fourq_apply_sparse_k_residual(
                    k0,
                    k1,
                    k2,
                    k3,
                    k_residual_ptr,
                    k_residual_indices_ptr,
                    kv_h,
                    residual_token,
                    valid,
                    dp,
                    K_CAPACITY,
                    K_RESIDUAL_CHANNELS,
                )
            score = _fourq_qk(
                q0, q1, q2, q3, k0, k1, k2, k3,
                USE_TENSOR_CORES_QK, FUSE_TENSOR_CORE_DOTS,
            ) * SM_SCALE
            score = tl.where(valid[None, :], score, -float("inf"))
            maximum = tl.max(score, axis=1)
            p = tl.where(valid[None, :], tl.exp(score - maximum[:, None]), 0.0)
            v_stride = V_I4_CAPACITY if V_COMPACT_INT4 else S_I4
            vbyte0 = kv_h * v_stride * (D // 2) + token[:, None] * (D // 2) + (d0 // 2)[None, :]
            vbyte1 = vbyte0 + 1
            vb0 = tl.load(i4_v_ptr + vbyte0, mask=valid[:, None], other=0)
            vb1 = tl.load(i4_v_ptr + vbyte1, mask=valid[:, None], other=0)
            i4_vscale = tl.load(i4_vs_ptr + kv_h * v_stride + token, mask=valid, other=0.0).to(tl.float32)
            c0, c1, c2, c3 = vb0 & 15, (vb0 >> 4) & 15, vb1 & 15, (vb1 >> 4) & 15
            v0 = tl.where(c0 >= 8, c0.to(tl.int16) - 16, c0.to(tl.int16)).to(tl.float32) * i4_vscale[:, None]
            v1 = tl.where(c1 >= 8, c1.to(tl.int16) - 16, c1.to(tl.int16)).to(tl.float32) * i4_vscale[:, None]
            v2 = tl.where(c2 >= 8, c2.to(tl.int16) - 16, c2.to(tl.int16)).to(tl.float32) * i4_vscale[:, None]
            v3 = tl.where(c3 >= 8, c3.to(tl.int16) - 16, c3.to(tl.int16)).to(tl.float32) * i4_vscale[:, None]
        else:
            local = block - N_FP - N_RECENT - N_I4
            token = local * BLOCK_N + n
            valid = token < S_I2
            group = d0 // GROUP_SIZE
            if K_FULL_PRECISION:
                kbase = kv_h * S_I2 * D + token[:, None] * D
                k0 = tl.load(i2_k_ptr + kbase + d0[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
                k1 = tl.load(i2_k_ptr + kbase + d1[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
                k2 = tl.load(i2_k_ptr + kbase + d2[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
                k3 = tl.load(i2_k_ptr + kbase + d3[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
            elif K_UNIFORM_INT4:
                k_token = K_I2_OFFSET + token if K_COMPACT_INT4 else token
                k_stride = K_CAPACITY if K_COMPACT_INT4 else S_I2
                k_superblocks = K_SUPERBLOCKS if K_COMPACT_INT4 else I2_SUPERBLOCKS
                kbyte0 = kv_h * k_stride * D2 + k_token[:, None] * D2 + (d0 // 2)[None, :]
                kbyte1 = kbyte0 + 1
                kb0 = tl.load(i2_k_ptr + kbyte0, mask=valid[:, None], other=0)
                kb1 = tl.load(i2_k_ptr + kbyte1, mask=valid[:, None], other=0)
                ks0, ks1, ks2, ks3, kz0, kz1, kz2, kz3 = _fourq_compact_k_metadata(
                    i2_ks_ptr, i2_kz_ptr, kv_h, k_superblocks,
                    k_token, d0, d1, d2, d3, K_SUPERBLOCK_SIZE,
                    BLOCK_N if TIGHT_METADATA_ROWS else 128,
                )
                k0 = ((kb0 & 15).to(tl.float32) - kz0) * ks0
                k1 = (((kb0 >> 4) & 15).to(tl.float32) - kz1) * ks1
                k2 = ((kb1 & 15).to(tl.float32) - kz2) * ks2
                k3 = (((kb1 >> 4) & 15).to(tl.float32) - kz3) * ks3
            elif K_UNIFORM_INT2:
                k_token = K_I2_OFFSET + token if K_COMPACT_INT2 else token
                k_stride = K_CAPACITY if K_COMPACT_INT2 else S_I2
                byte = kv_h * k_stride * D4 + k_token[:, None] * D4 + dp[None, :]
                kb = tl.load(i2_k_ptr + byte, mask=valid[:, None], other=0)
                if K_COMPACT_INT2:
                    k_superblocks = K_SUPERBLOCKS
                else:
                    k_superblocks = I2_SUPERBLOCKS
                if K_COMPACT_INT2 and BROADCAST_K_METADATA:
                    ks0, ks1, ks2, ks3, kz0, kz1, kz2, kz3 = _fourq_compact_k_metadata(
                        i2_ks_ptr, i2_kz_ptr, kv_h, k_superblocks,
                        k_token, d0, d1, d2, d3, K_SUPERBLOCK_SIZE,
                        BLOCK_N if TIGHT_METADATA_ROWS else 128,
                    )
                else:
                    meta = kv_h * k_superblocks * D + (k_token // K_SUPERBLOCK_SIZE)[:, None] * D
                    ks0 = tl.load(i2_ks_ptr + meta + d0[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
                    ks1 = tl.load(i2_ks_ptr + meta + d1[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
                    ks2 = tl.load(i2_ks_ptr + meta + d2[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
                    ks3 = tl.load(i2_ks_ptr + meta + d3[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
                    kz0 = tl.load(i2_kz_ptr + meta + d0[None, :], mask=valid[:, None], other=0).to(tl.float32)
                    kz1 = tl.load(i2_kz_ptr + meta + d1[None, :], mask=valid[:, None], other=0).to(tl.float32)
                    kz2 = tl.load(i2_kz_ptr + meta + d2[None, :], mask=valid[:, None], other=0).to(tl.float32)
                    kz3 = tl.load(i2_kz_ptr + meta + d3[None, :], mask=valid[:, None], other=0).to(tl.float32)
                if K_COMPACT_INT2:
                    k0 = ((kb >> 0) & 3).to(tl.float32) * ks0 + kz0
                    k1 = ((kb >> 2) & 3).to(tl.float32) * ks1 + kz1
                    k2 = ((kb >> 4) & 3).to(tl.float32) * ks2 + kz2
                    k3 = ((kb >> 6) & 3).to(tl.float32) * ks3 + kz3
                else:
                    k0 = (((kb >> 0) & 3).to(tl.float32) - kz0) * ks0
                    k1 = (((kb >> 2) & 3).to(tl.float32) - kz1) * ks1
                    k2 = (((kb >> 4) & 3).to(tl.float32) - kz2) * ks2
                    k3 = (((kb >> 6) & 3).to(tl.float32) - kz3) * ks3
            else:
                k_token = token
                k_stride = S_I2
                byte = kv_h * k_stride * D4 + k_token[:, None] * D4 + dp[None, :]
                kb = tl.load(i2_k_ptr + byte, mask=valid[:, None], other=0)
                meta = kv_h * I2_SUPERBLOCKS * GROUPS + (token // 128)[:, None] * GROUPS + group[None, :]
                ks = tl.load(i2_ks_ptr + meta, mask=valid[:, None], other=0.0).to(tl.float32)
                kz = tl.load(i2_kz_ptr + meta, mask=valid[:, None], other=0).to(tl.float32)
                k0 = (((kb >> 0) & 3).to(tl.float32) - kz) * ks
                k1 = (((kb >> 2) & 3).to(tl.float32) - kz) * ks
                k2 = (((kb >> 4) & 3).to(tl.float32) - kz) * ks
                k3 = (((kb >> 6) & 3).to(tl.float32) - kz) * ks
            if K_COMPACT_INT2 and K_RESIDUAL_CHANNELS > 0:
                k0, k1, k2, k3 = _fourq_apply_sparse_k_residual(
                    k0,
                    k1,
                    k2,
                    k3,
                    k_residual_ptr,
                    k_residual_indices_ptr,
                    kv_h,
                    K_I2_OFFSET + token,
                    valid,
                    dp,
                    K_CAPACITY,
                    K_RESIDUAL_CHANNELS,
                )
            score = _fourq_qk(
                q0, q1, q2, q3, k0, k1, k2, k3,
                USE_TENSOR_CORES_QK, FUSE_TENSOR_CORE_DOTS,
            ) * SM_SCALE
            score = tl.where(valid[None, :], score, -float("inf"))
            maximum = tl.max(score, axis=1)
            p = tl.where(valid[None, :], tl.exp(score - maximum[:, None]), 0.0)
            if V_COMPACT_INT2:
                v_stride = V_I2_CAPACITY
            else:
                v_stride = S_I2
            vbyte = kv_h * v_stride * D4 + token[:, None] * D4 + dp[None, :]
            vb = tl.load(i2_v_ptr + vbyte, mask=valid[:, None], other=0)
            vmeta = kv_h * v_stride * GROUPS + token[:, None] * GROUPS + group[None, :]
            i2_vscale = tl.load(i2_vs_ptr + vmeta, mask=valid[:, None], other=0.0).to(tl.float32)
            i2_vzero = tl.load(i2_vz_ptr + vmeta, mask=valid[:, None], other=0).to(tl.float32)
            v0 = (((vb >> 0) & 3).to(tl.float32) - i2_vzero) * i2_vscale
            v1 = (((vb >> 2) & 3).to(tl.float32) - i2_vzero) * i2_vscale
            v2 = (((vb >> 4) & 3).to(tl.float32) - i2_vzero) * i2_vscale
            v3 = (((vb >> 6) & 3).to(tl.float32) - i2_vzero) * i2_vscale

        denom = tl.sum(p, axis=1)
        state = q_head[:, None] * STRIDE + block
        tl.store(pm_ptr + q_head * STRIDE + block, maximum, mask=q_valid)
        tl.store(pl_ptr + q_head * STRIDE + block, denom, mask=q_valid)
        if FUSE_TENSOR_CORE_DOTS:
            value = _fourq_join_channels(v0, v1, v2, v3)
            acc = tl.dot(
                p.to(tl.float16),
                value.to(tl.float16),
                out_dtype=tl.float32,
            )
            d = tl.arange(0, D)
            tl.store(
                po_ptr + state * D + d[None, :],
                acc,
                mask=q_valid[:, None],
            )
        else:
            a0 = _fourq_pv(p, v0, USE_TENSOR_CORES_PV)
            a1 = _fourq_pv(p, v1, USE_TENSOR_CORES_PV)
            a2 = _fourq_pv(p, v2, USE_TENSOR_CORES_PV)
            a3 = _fourq_pv(p, v3, USE_TENSOR_CORES_PV)
            tl.store(po_ptr + state * D + d0[None, :], a0, mask=q_valid[:, None])
            tl.store(po_ptr + state * D + d1[None, :], a1, mask=q_valid[:, None])
            tl.store(po_ptr + state * D + d2[None, :], a2, mask=q_valid[:, None])
            tl.store(po_ptr + state * D + d3[None, :], a3, mask=q_valid[:, None])

    @triton.jit(do_not_specialize=[
        "POSITION", "GLOBAL_INDEX", "LOGICAL_POSITION", "DO_APPEND", "TOTAL", "STRIDE",
    ])
    def _fourq_partial_reduce_kernel(
        pm, pl, po, out,
        recent_k, recent_v, global_indices, logical_positions, length_device,
        page_lengths,
        append_k, append_v, POSITION, GLOBAL_INDEX, LOGICAL_POSITION, DO_APPEND,
        batch_strides_ptr, batch_append_metadata,
        TOTAL, STRIDE, BLOCK_B: tl.constexpr, ACTIVE_BLOCKS: tl.constexpr,
        D: tl.constexpr,
        H_KV: tl.constexpr, RECENT_CAPACITY: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        APPLY_OUTPUT_HADAMARD: tl.constexpr,
        SINGLE_REQUEST: tl.constexpr,
        BATCH_APPEND_SHARED: tl.constexpr,
    ):
        request = tl.program_id(0)
        h = tl.program_id(1)
        if not SINGLE_REQUEST:
            pm += request * tl.load(batch_strides_ptr + 23)
            pl += request * tl.load(batch_strides_ptr + 24)
            po += request * tl.load(batch_strides_ptr + 25)
            out += request * tl.load(batch_strides_ptr + 26)
            recent_k += request * tl.load(batch_strides_ptr + 5)
            recent_v += request * tl.load(batch_strides_ptr + 6)
            length_device += request * tl.load(batch_strides_ptr + 7)
            append_k += request * tl.load(batch_strides_ptr + 8)
            append_v += request * tl.load(batch_strides_ptr + 9)
            global_indices += request * tl.load(batch_strides_ptr + 27)
            logical_positions += request * tl.load(batch_strides_ptr + 28)
            page_lengths += request * tl.load(batch_strides_ptr + 29)
            # Requests in a homogeneous batch may have different active
            # Recent-FP lengths and logical/global positions.  Read the append
            # destination from the request-local device length and the two
            # caller-provided indices from compact batch metadata.
            POSITION = tl.load(length_device)
            if not BATCH_APPEND_SHARED:
                GLOBAL_INDEX = tl.load(batch_append_metadata + request * 2)
                LOGICAL_POSITION = tl.load(
                    batch_append_metadata + request * 2 + 1
                )
        blocks = tl.arange(0, BLOCK_B)
        valid = blocks < TOTAL
        maxima = tl.load(pm + h * STRIDE + blocks, mask=valid, other=-float("inf")).to(tl.float32)
        maximum = tl.max(maxima, axis=0)
        weight = tl.where(valid, tl.exp(maxima - maximum), 0.0)
        sums = tl.load(pl + h * STRIDE + blocks, mask=valid, other=0.0).to(tl.float32)
        denominator = tl.sum(weight * sums, axis=0)
        d = tl.arange(0, D)
        acc = tl.zeros((D,), tl.float32)
        # BLOCK_B must remain a power of two for the vector max/sum above.
        # Iterating that padded extent for the 128-D output accumulator caused
        # a sharp latency cliff whenever TOTAL crossed a power-of-two boundary
        # (notably at 32K). Stop at the exact prepared split count instead.
        # Active blocks retain the same order and arithmetic as before; only
        # iterations whose mask was always false are omitted.
        for block in range(0, ACTIVE_BLOCKS):
            active = block < TOTAL
            values = tl.load(
                po + (h * STRIDE + block) * D + d,
                mask=active,
                other=0.0,
            ).to(tl.float32)
            block_max = tl.load(pm + h * STRIDE + block, mask=active, other=-float("inf"))
            acc += tl.where(active, tl.exp(block_max - maximum) * values, 0.0)
        result = acc / denominator
        if APPLY_OUTPUT_HADAMARD:
            # Preserve the previous two-launch semantics: the attention result
            # was first rounded into the BF16 output buffer, then reloaded by
            # the standalone FWHT kernel. Fusing after this explicit roundtrip
            # removes one launch without silently changing model numerics.
            result = _fwht_128(result.to(tl.bfloat16).to(tl.float32))
        tl.store(out + h * D + d, result)
        append_head = (DO_APPEND != 0) & (h < H_KV)
        append_offset = h * D + d
        destination = h * RECENT_CAPACITY * D + POSITION * D + d
        tl.store(recent_k + destination, tl.load(append_k + append_offset, mask=append_head), mask=append_head)
        tl.store(recent_v + destination, tl.load(append_v + append_offset, mask=append_head), mask=append_head)
        leader = (DO_APPEND != 0) & (h == 0) & (d == 0)
        tl.store(global_indices + POSITION + d, GLOBAL_INDEX, mask=leader)
        tl.store(logical_positions + POSITION + d, LOGICAL_POSITION, mask=leader)
        tl.store(length_device + d, POSITION + 1, mask=leader)
        tl.store(
            page_lengths + POSITION // PAGE_SIZE + d,
            POSITION % PAGE_SIZE + 1,
            mask=leader,
        )

    @triton.jit
    def _fourq_hierarchical_group_reduce_kernel(
        pm,
        pl,
        po,
        group_pm,
        group_pl,
        group_po,
        TOTAL: tl.constexpr,
        STRIDE: tl.constexpr,
        GROUP_STRIDE: tl.constexpr,
        GROUP_BLOCKS: tl.constexpr,
        D: tl.constexpr,
    ):
        """Collapse a bounded set of split-K states into one exact softmax state.

        This adds no cache approximation.  It only reassociates the FP32 sum so
        that long-context output reduction exposes enough independent programs
        to occupy the GPU instead of making one program per query head walk all
        split states serially.
        """
        h = tl.program_id(0)
        group = tl.program_id(1)
        first = group * GROUP_BLOCKS
        blocks = first + tl.arange(0, GROUP_BLOCKS)
        valid = blocks < TOTAL
        maxima = tl.load(
            pm + h * STRIDE + blocks,
            mask=valid,
            other=-float("inf"),
        ).to(tl.float32)
        maximum = tl.max(maxima, axis=0)
        weights = tl.where(valid, tl.exp(maxima - maximum), 0.0)
        sums = tl.load(
            pl + h * STRIDE + blocks,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        denominator = tl.sum(weights * sums, axis=0)

        d = tl.arange(0, D)
        acc = tl.zeros((D,), tl.float32)
        for local_block in range(0, GROUP_BLOCKS):
            block = first + local_block
            active = block < TOTAL
            values = tl.load(
                po + (h * STRIDE + block) * D + d,
                mask=active,
                other=0.0,
            ).to(tl.float32)
            block_max = tl.load(
                pm + h * STRIDE + block,
                mask=active,
                other=-float("inf"),
            ).to(tl.float32)
            acc += tl.where(
                active,
                tl.exp(block_max - maximum) * values,
                0.0,
            )

        destination = h * GROUP_STRIDE + group
        tl.store(group_pm + destination, maximum)
        tl.store(group_pl + destination, denominator)
        tl.store(group_po + destination * D + d, acc)

    @triton.jit
    def _fourq_batched_hierarchical_group_reduce_kernel(
        pm,
        pl,
        po,
        group_pm,
        group_pl,
        group_po,
        TOTAL: tl.constexpr,
        STRIDE: tl.constexpr,
        GROUP_STRIDE: tl.constexpr,
        GROUP_BLOCKS: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
    ):
        """Request-strided variant of the validated batch=1 group reduction.

        The batch=1 production path continues to launch
        ``_fourq_hierarchical_group_reduce_kernel``.  Keeping a separate kernel
        makes the quality-critical launch geometry and arithmetic immutable
        while allowing homogeneous throughput batches to share one launch.
        """
        request = tl.program_id(0)
        h = tl.program_id(1)
        group = tl.program_id(2)
        source_head = request * H + h
        first = group * GROUP_BLOCKS
        blocks = first + tl.arange(0, GROUP_BLOCKS)
        valid = blocks < TOTAL
        maxima = tl.load(
            pm + source_head * STRIDE + blocks,
            mask=valid,
            other=-float("inf"),
        ).to(tl.float32)
        maximum = tl.max(maxima, axis=0)
        weights = tl.where(valid, tl.exp(maxima - maximum), 0.0)
        sums = tl.load(
            pl + source_head * STRIDE + blocks,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        denominator = tl.sum(weights * sums, axis=0)

        d = tl.arange(0, D)
        acc = tl.zeros((D,), tl.float32)
        for local_block in range(0, GROUP_BLOCKS):
            block = first + local_block
            active = block < TOTAL
            values = tl.load(
                po + (source_head * STRIDE + block) * D + d,
                mask=active,
                other=0.0,
            ).to(tl.float32)
            block_max = tl.load(
                pm + source_head * STRIDE + block,
                mask=active,
                other=-float("inf"),
            ).to(tl.float32)
            acc += tl.where(
                active,
                tl.exp(block_max - maximum) * values,
                0.0,
            )

        destination = source_head * GROUP_STRIDE + group
        tl.store(group_pm + destination, maximum)
        tl.store(group_pl + destination, denominator)
        tl.store(group_po + destination * D + d, acc)

@dataclass
class SplitKWorkspace:
    partial_max: torch.Tensor
    partial_sum: torch.Tensor
    partial_output: torch.Tensor
    output: torch.Tensor
    empty_bf16: torch.Tensor
    empty_fp16: torch.Tensor
    empty_u8: torch.Tensor
    batch_strides: torch.Tensor
    group_batch_strides: torch.Tensor
    group_partial_max: torch.Tensor
    group_partial_sum: torch.Tensor
    group_partial_output: torch.Tensor
    hierarchical_group_blocks: int
    hierarchical_group_capacity: int
    block_n: int
    total_blocks: int


@dataclass
class SplitKExecutor:
    cache: ThreePrecisionCache
    workspace: SplitKWorkspace
    args: tuple[torch.Tensor, ...]
    s_fp: int
    s_recent: int
    s_i4: int
    s_i2: int
    n_fp: int
    n_recent: int
    n_i4: int
    total: int
    fp_i2_superblocks: int
    i4_i2_superblocks: int
    i2_superblocks: int
    uniform_int2_key: bool
    compact_int2_key: bool
    uniform_int4_key: bool
    compact_int4_key: bool
    full_precision_key: bool
    key_residual_channels: int
    k_capacity: int
    k_superblocks: int
    k_superblock_size: int
    k_fp_offset: int
    k_i4_offset: int
    k_i2_offset: int
    compact_int2_value: bool
    v_i2_capacity: int
    v_int2_group_size: int
    v_int2_groups: int
    reduce_block: int
    block_n: int
    num_warps: int
    num_stages: int
    reduce_num_warps: int
    use_tensor_cores_qk: bool
    use_tensor_cores_pv: bool
    fuse_tensor_core_dots: bool
    broadcast_k_metadata: bool
    single_request_fastpath: bool
    tight_metadata_rows: bool
    hierarchical_reduce: bool
    hierarchical_group_blocks: int
    hierarchical_group_count: int
    output_hadamard: bool = False
    native_full_head_layout: bool = False
    num_query_heads: int = 16
    num_key_value_heads: int = 2
    head_dim: int = 128
    static_int4_tokens: int = 0
    generated_k_offset: int = 0
    compact_int4_value: bool = False
    v_i4_capacity: int = 1
    batched_recent: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ] | None = None
    batch_append_metadata: torch.Tensor | None = None
    batched_recent_lengths: list[int] | None = None

    def __call__(
        self,
        query: torch.Tensor,
        append: tuple[
            torch.Tensor,
            torch.Tensor,
            int | Sequence[int] | torch.Tensor,
            int | Sequence[int] | torch.Tensor,
        ] | None = None,
    ) -> torch.Tensor:
        if query.ndim == 4:
            query = query[:, :, 0]
        batch_size = int(self.workspace.output.shape[0])
        if (
            query.shape != (batch_size, self.num_query_heads, self.head_dim)
            or query.device != self.workspace.output.device
            or not query.is_contiguous()
        ):
            raise ValueError("query does not match prepared split-K executor")
        pm, pl, po, out = (
            self.workspace.partial_max,
            self.workspace.partial_sum,
            self.workspace.partial_output,
            self.workspace.output,
        )
        recent = self.cache.recent_fp_pool
        if self.batched_recent is None:
            recent_k = recent.fp_k
            recent_v = recent.fp_v
            recent_global_indices = recent.global_index_buffer
            recent_logical_positions = recent.logical_position_buffer
            recent_length_device = recent.length_device
            recent_page_lengths = recent.page_index.lengths
        else:
            (
                recent_k,
                recent_v,
                recent_global_indices,
                recent_logical_positions,
                recent_length_device,
                recent_page_lengths,
            ) = self.batched_recent
        if append is None:
            append_k = append_v = self.workspace.empty_bf16
            position = global_index = logical_position = 0
            do_append = 0
            batch_append_shared = batch_size > 1
        else:
            append_k, append_v, global_index, logical_position = append
            if append_k.ndim == 4:
                append_k, append_v = append_k[:, :, 0], append_v[:, :, 0]
            expected_append_shape = (
                batch_size,
                self.num_key_value_heads,
                self.head_dim,
            )
            if (
                append_k.numel()
                != batch_size * self.num_key_value_heads * self.head_dim
                or append_v.numel()
                != batch_size * self.num_key_value_heads * self.head_dim
                or append_k.device != query.device
                or append_v.device != query.device
                or not append_k.is_contiguous()
                or not append_v.is_contiguous()
            ):
                raise ValueError(
                    "append K/V must be contiguous [B,Hkv,1,D] or [B,Hkv,D]"
                )
            append_k = append_k.view(expected_append_shape)
            append_v = append_v.view(expected_append_shape)
            if batch_size == 1:
                batch_append_shared = False
                if recent.active_length is None or recent.active_length >= recent.capacity:
                    raise RuntimeError(
                        "prepared executor cannot fuse an append into a full Recent-FP pool"
                    )
                position = int(recent.active_length)
                global_index = int(global_index)
                logical_position = int(logical_position)
            else:
                if self.batched_recent_lengths is None or self.batch_append_metadata is None:
                    raise RuntimeError("batched executor append metadata is unavailable")
                if any(length >= recent.capacity for length in self.batched_recent_lengths):
                    raise RuntimeError(
                        "prepared batched executor cannot fuse an append into a full Recent-FP pool"
                    )
                # The native throughput protocol advances every request at
                # one shared position. Pass those two scalar indices directly
                # to the fused reduction, eliminating two device fill kernels
                # per compressed layer and decode step. Explicit per-request
                # indices retain the metadata-table path for heterogeneous
                # schedulers.
                batch_append_shared = isinstance(global_index, int) and isinstance(
                    logical_position, int
                )
                if not batch_append_shared:
                    for column, value in enumerate(
                        (global_index, logical_position)
                    ):
                        target = self.batch_append_metadata[:, column]
                        values = torch.as_tensor(
                            value, device=query.device, dtype=torch.int64
                        ).reshape(-1)
                        if values.numel() == 1:
                            target.copy_(values.expand(batch_size))
                        elif values.numel() == batch_size:
                            target.copy_(values)
                        else:
                            raise ValueError(
                                "batched append indices must contain one value per request"
                            )
                # The reduce kernel reads each request's position directly
                # from its device-side Recent-FP length.
                position = 0
                if batch_append_shared:
                    global_index = int(global_index)
                    logical_position = int(logical_position)
                else:
                    global_index = logical_position = 0
            do_append = 1
        if self.native_full_head_layout:
            _full_head_page_partial_kernel[
                (batch_size, self.num_key_value_heads, self.total)
            ](
                query,
                *self.args[:7],
                append_k,
                append_v,
                do_append,
                *self.args[7:18],
                pm,
                pl,
                po,
                self.workspace.batch_strides,
                S_FP=self.s_fp,
                S_RECENT=self.s_recent,
                S_I4=self.s_i4,
                S_I2=self.s_i2,
                N_FP=self.n_fp,
                N_RECENT=self.n_recent,
                N_I4=self.n_i4,
                TOTAL=self.total,
                STRIDE=self.workspace.total_blocks,
                D=self.head_dim,
                D2=self.head_dim // 2,
                D4=self.head_dim // 4,
                GQA=self.num_query_heads // self.num_key_value_heads,
                Q_BLOCK=16,
                GROUP_SIZE=self.v_int2_group_size,
                GROUPS=self.v_int2_groups,
                K_CAPACITY=self.k_capacity,
                K_SUPERBLOCKS=self.k_superblocks,
                K_SUPERBLOCK_SIZE=self.k_superblock_size,
                K_FP_OFFSET=self.k_fp_offset,
                K_I4_OFFSET=self.k_i4_offset,
                K_I2_OFFSET=self.k_i2_offset,
                V_I4_CAPACITY=self.v_i4_capacity,
                V_I2_CAPACITY=self.v_i2_capacity,
                V_COMPACT_INT4=self.compact_int4_value,
                V_COMPACT_INT2=self.compact_int2_value,
                BLOCK_N=self.block_n,
                SM_SCALE=self.head_dim ** -0.5,
                num_warps=self.num_warps,
                num_stages=self.num_stages,
            )
        else:
            _fourq_page_partial_kernel[
                (batch_size, self.num_key_value_heads, self.total)
            ](
                query, *self.args[:7], append_k, append_v, do_append, *self.args[7:], pm, pl, po,
                self.workspace.batch_strides,
                S_FP=self.s_fp, S_RECENT=self.s_recent, S_I4=self.s_i4, S_I2=self.s_i2,
                N_FP=self.n_fp, N_RECENT=self.n_recent, N_I4=self.n_i4, TOTAL=self.total,
                STRIDE=self.workspace.total_blocks,
                D=self.head_dim,
                D2=self.head_dim // 2,
                D4=self.head_dim // 4,
                GQA=self.num_query_heads // self.num_key_value_heads,
                Q_BLOCK=(
                    16
                    if self.use_tensor_cores_qk or self.use_tensor_cores_pv
                    else 1
                    << (
                        self.num_query_heads // self.num_key_value_heads - 1
                    ).bit_length()
                ),
                GROUP_SIZE=self.v_int2_group_size, GROUPS=self.v_int2_groups,
                FP_I2_SUPERBLOCKS=self.fp_i2_superblocks,
                I4_I2_SUPERBLOCKS=self.i4_i2_superblocks,
                I2_SUPERBLOCKS=self.i2_superblocks,
                K_CAPACITY=self.k_capacity,
                K_SUPERBLOCKS=self.k_superblocks,
                K_SUPERBLOCK_SIZE=self.k_superblock_size,
                K_FP_OFFSET=self.k_fp_offset,
                K_I4_OFFSET=self.k_i4_offset,
                K_I2_OFFSET=self.k_i2_offset,
                STATIC_I4_TOKENS=self.static_int4_tokens,
                K_GENERATED_OFFSET=self.generated_k_offset,
                V_I4_CAPACITY=self.v_i4_capacity,
                V_I2_CAPACITY=self.v_i2_capacity,
                K_UNIFORM_INT2=self.uniform_int2_key,
                K_COMPACT_INT2=self.compact_int2_key,
                K_UNIFORM_INT4=self.uniform_int4_key,
                K_COMPACT_INT4=self.compact_int4_key,
                K_FULL_PRECISION=self.full_precision_key,
                K_RESIDUAL_CHANNELS=self.key_residual_channels,
                V_COMPACT_INT4=self.compact_int4_value,
                V_COMPACT_INT2=self.compact_int2_value,
                USE_TENSOR_CORES_QK=self.use_tensor_cores_qk,
                USE_TENSOR_CORES_PV=self.use_tensor_cores_pv,
                FUSE_TENSOR_CORE_DOTS=(
                    self.fuse_tensor_core_dots
                ),
                BROADCAST_K_METADATA=self.broadcast_k_metadata,
                SINGLE_REQUEST=self.single_request_fastpath,
                TIGHT_METADATA_ROWS=self.tight_metadata_rows,
                BLOCK_N=self.block_n, SM_SCALE=self.head_dim ** -0.5,
                num_warps=self.num_warps, num_stages=self.num_stages,
            )
        reduce_pm, reduce_pl, reduce_po = pm, pl, po
        reduce_total = self.total
        reduce_stride = self.workspace.total_blocks
        reduce_block = self.reduce_block
        if self.hierarchical_reduce:
            if batch_size == 1:
                _fourq_hierarchical_group_reduce_kernel[
                    (self.num_query_heads, self.hierarchical_group_count)
                ](
                    pm,
                    pl,
                    po,
                    self.workspace.group_partial_max,
                    self.workspace.group_partial_sum,
                    self.workspace.group_partial_output,
                    TOTAL=self.total,
                    STRIDE=self.workspace.total_blocks,
                    GROUP_STRIDE=self.workspace.hierarchical_group_capacity,
                    GROUP_BLOCKS=self.hierarchical_group_blocks,
                    D=self.head_dim,
                    num_warps=self.reduce_num_warps,
                )
            else:
                _fourq_batched_hierarchical_group_reduce_kernel[
                    (
                        batch_size,
                        self.num_query_heads,
                        self.hierarchical_group_count,
                    )
                ](
                    pm,
                    pl,
                    po,
                    self.workspace.group_partial_max,
                    self.workspace.group_partial_sum,
                    self.workspace.group_partial_output,
                    TOTAL=self.total,
                    STRIDE=self.workspace.total_blocks,
                    GROUP_STRIDE=self.workspace.hierarchical_group_capacity,
                    GROUP_BLOCKS=self.hierarchical_group_blocks,
                    H=self.num_query_heads,
                    D=self.head_dim,
                    num_warps=self.reduce_num_warps,
                )
            reduce_pm = self.workspace.group_partial_max
            reduce_pl = self.workspace.group_partial_sum
            reduce_po = self.workspace.group_partial_output
            reduce_total = self.hierarchical_group_count
            reduce_stride = self.workspace.hierarchical_group_capacity
            reduce_block = 1 << (reduce_total - 1).bit_length()
        _fourq_partial_reduce_kernel[(batch_size, self.num_query_heads)](
            reduce_pm, reduce_pl, reduce_po, out,
            recent_k, recent_v, recent_global_indices,
            recent_logical_positions, recent_length_device,
            recent_page_lengths,
            append_k, append_v, position, global_index, logical_position, do_append,
            (
                self.workspace.group_batch_strides
                if self.hierarchical_reduce and batch_size > 1
                else self.workspace.batch_strides
            ),
            (
                self.workspace.batch_strides
                if self.batch_append_metadata is None
                else self.batch_append_metadata
            ),
            TOTAL=reduce_total,
            STRIDE=reduce_stride,
            BLOCK_B=reduce_block,
            ACTIVE_BLOCKS=reduce_total,
            D=self.head_dim,
            H_KV=self.num_key_value_heads,
            RECENT_CAPACITY=recent.capacity,
            PAGE_SIZE=self.cache.page_size,
            APPLY_OUTPUT_HADAMARD=self.output_hadamard,
            SINGLE_REQUEST=self.single_request_fastpath,
            BATCH_APPEND_SHARED=batch_append_shared,
            num_warps=self.reduce_num_warps,
        )
        if do_append and batch_size > 1:
            assert self.batched_recent_lengths is not None
            self.batched_recent_lengths[:] = [
                length + 1 for length in self.batched_recent_lengths
            ]
        return out


def _launch_block_count(active_blocks: int, capacity_blocks: int) -> int:
    """Keep one padding split for occupancy; skip larger future-growth reserves."""
    if active_blocks <= 0 or capacity_blocks < active_blocks:
        raise ValueError("invalid active/capacity block counts")
    return capacity_blocks if capacity_blocks - active_blocks <= 1 else active_blocks


def append_recent_fp_triton(
    cache: ThreePrecisionCache,
    new_key: torch.Tensor,
    new_value: torch.Tensor,
    *,
    global_token_index: int,
    logical_position: int,
) -> None:
    """Append one decode token to Recent-FP with one CUDA launch."""
    if not triton_available() or not new_key.is_cuda:
        raise RuntimeError("fused Recent-FP append requires Triton CUDA")
    pool = cache.recent_fp_pool
    if new_key.ndim == 4:
        new_key, new_value = new_key[0], new_value[0]
    if new_key.shape != new_value.shape or new_key.ndim != 3 or int(new_key.shape[1]) != 1:
        raise ValueError("new K/V must be [Hkv,1,D] or [1,Hkv,1,D]")
    if pool.active_length is None or pool.active_length >= pool.capacity:
        raise RuntimeError("Recent-FP staging buffer has no append capacity")
    if any(value is None for value in (
        pool.fp_k,
        pool.fp_v,
        pool.global_index_buffer,
        pool.logical_position_buffer,
        pool.length_device,
    )):
        raise RuntimeError("Recent-FP staging buffers are incomplete")
    h_kv, _, head_dim = map(int, new_key.shape)
    block = 1 << (h_kv * head_dim - 1).bit_length()
    _append_recent_fp_kernel[(1,)](
        pool.fp_k,
        pool.fp_v,
        pool.global_index_buffer,
        pool.logical_position_buffer,
        pool.length_device,
        pool.page_index.lengths,
        new_key,
        new_value,
        pool.active_length,
        int(global_token_index),
        int(logical_position),
        H_KV=h_kv,
        D=head_dim,
        CAPACITY=pool.capacity,
        PAGE_SIZE=cache.page_size,
        BLOCK=block,
        num_warps=4,
    )
    pool.active_length += 1


def allocate_splitk_workspace(
    cache: ThreePrecisionCache,
    *,
    num_query_heads: int = 16,
    block_n: int = 128,
    reserve_tokens: int = 0,
    device: torch.device | None = None,
    partial_output_dtype: torch.dtype = torch.float32,
    hierarchical_group_blocks: int = 64,
) -> SplitKWorkspace:
    if partial_output_dtype not in {torch.float16, torch.float32}:
        raise ValueError("partial_output_dtype must be float16 or float32")
    if hierarchical_group_blocks not in {16, 32, 64}:
        raise ValueError("hierarchical_group_blocks must be 16, 32, or 64")
    pools = (cache.fp_pool, cache.recent_fp_pool, cache.int4_pool, cache.int2_pool)
    required = required_splitk_blocks(cache, block_n=block_n, reserve_tokens=reserve_tokens)
    if required <= 0:
        raise ValueError("cannot allocate workspace for an empty cache")
    total = required
    num_query_heads = int(num_query_heads)
    if num_query_heads < 1:
        raise ValueError("num_query_heads must be positive")
    if device is None:
        payload = cache.compact_key.packed if cache.compact_key is not None else next(
            (
                pool.fp_k if pool.fp_k is not None else pool.packed_k.packed
                for pool in pools
                if pool.token_count
            ),
            None,
        )
        assert payload is not None
        device = payload.device
    group_capacity = (total + hierarchical_group_blocks - 1) // hierarchical_group_blocks
    return SplitKWorkspace(
        partial_max=torch.empty(
            (num_query_heads, total), device=device, dtype=torch.float32
        ),
        partial_sum=torch.empty(
            (num_query_heads, total), device=device, dtype=torch.float32
        ),
        partial_output=torch.empty(
            (num_query_heads, total, 128),
            device=device,
            dtype=partial_output_dtype,
        ),
        output=torch.empty(
            (1, num_query_heads, 128), device=device, dtype=torch.bfloat16
        ),
        empty_bf16=torch.empty(1, device=device, dtype=torch.bfloat16),
        empty_fp16=torch.empty(1, device=device, dtype=torch.float16),
        empty_u8=torch.empty(1, device=device, dtype=torch.uint8),
        batch_strides=torch.zeros(30, device=device, dtype=torch.int64),
        group_batch_strides=torch.zeros(30, device=device, dtype=torch.int64),
        group_partial_max=torch.empty(
            (num_query_heads, group_capacity), device=device, dtype=torch.float32
        ),
        group_partial_sum=torch.empty(
            (num_query_heads, group_capacity), device=device, dtype=torch.float32
        ),
        group_partial_output=torch.empty(
            (num_query_heads, group_capacity, 128),
            device=device,
            dtype=torch.float32,
        ),
        hierarchical_group_blocks=int(hierarchical_group_blocks),
        hierarchical_group_capacity=int(group_capacity),
        block_n=int(block_n),
        total_blocks=int(total),
    )


def required_splitk_blocks(
    cache: ThreePrecisionCache,
    *,
    block_n: int = 128,
    reserve_tokens: int = 0,
) -> int:
    if block_n < 1:
        raise ValueError("block_n must be positive")
    pools = (cache.fp_pool, cache.recent_fp_pool, cache.int4_pool, cache.int2_pool)
    return sum(
        ((pool.capacity if pool.active_length is not None else pool.token_count) + block_n - 1)
        // block_n
        for pool in pools
    ) + (max(0, int(reserve_tokens)) + block_n - 1) // block_n


def prepare_splitk_executor(
    cache: ThreePrecisionCache,
    workspace: SplitKWorkspace,
    *,
    block_n: int = 128,
    num_warps: int = 8,
    num_stages: int = 2,
    reduce_num_warps: int = 8,
    use_tensor_cores_qk: bool = True,
    use_tensor_cores_pv: bool = False,
    broadcast_k_metadata: bool = False,
    single_request_fastpath: bool | None = None,
    tight_metadata_rows: bool | None = None,
    output_hadamard: bool = False,
    hierarchical_reduce: bool = True,
) -> SplitKExecutor:
    if workspace.block_n != block_n:
        raise ValueError("cache/workspace is incompatible with a mixed split-K executor")
    fp, i4, i2, recent = cache.fp_pool, cache.int4_pool, cache.int2_pool, cache.recent_fp_pool
    if recent.fp_k is None or recent.fp_k.ndim != 3:
        raise ValueError("Recent-FP pool must expose [Hkv,capacity,D] K storage")
    num_key_value_heads = int(recent.fp_k.shape[0])
    head_dim = int(recent.fp_k.shape[2])
    num_query_heads = int(workspace.output.shape[1])
    if head_dim != 128:
        raise ValueError("split-K executor currently requires head_dim=128")
    if num_query_heads % num_key_value_heads:
        raise ValueError("query heads must be divisible by key/value heads")
    if num_query_heads // num_key_value_heads > 16:
        raise ValueError("split-K executor supports at most 16 GQA heads per KV head")
    expected_workspace_shapes = (
        (num_query_heads, workspace.total_blocks),
        (num_query_heads, workspace.total_blocks),
        (num_query_heads, workspace.total_blocks, head_dim),
        (1, num_query_heads, head_dim),
    )
    actual_workspace_shapes = tuple(
        tuple(tensor.shape)
        for tensor in (
            workspace.partial_max,
            workspace.partial_sum,
            workspace.partial_output,
            workspace.output,
        )
    )
    if actual_workspace_shapes != expected_workspace_shapes:
        raise ValueError("split-K workspace geometry is inconsistent")
    def item(pool, side, attr, dtype):
        payload = getattr(pool, f"packed_{side}")
        if pool.token_count:
            return getattr(payload, attr)
        return workspace.empty_u8 if dtype == torch.uint8 else workspace.empty_fp16
    uniform_int2_key = cache.uniform_int2_key
    uniform_int4_key = cache.uniform_int4_key
    full_precision_key = cache.full_precision_key
    compact = cache.compact_key
    fp_k = (
        compact.packed
        if compact is not None else item(fp, "k", "packed", torch.uint8)
        if uniform_int2_key else fp.fp_k if fp.token_count else workspace.empty_bf16
    )
    fp_ks = (
        compact.scale if compact is not None
        else item(fp, "k", "scale", torch.float16) if uniform_int2_key else workspace.empty_fp16
    )
    fp_kz = (
        compact.zero_point if compact is not None
        else item(fp, "k", "zero_point", torch.uint8) if uniform_int2_key else workspace.empty_u8
    )
    fp_v = fp.fp_v if fp.token_count else workspace.empty_bf16
    recent_k, recent_v = (
        (recent.fp_k, recent.fp_v)
        if recent.capacity > 0
        else (workspace.empty_bf16, workspace.empty_bf16)
    )
    if recent.length_device is None:
        raise ValueError("Recent-FP pool must expose a device-side active length")
    args = (
        fp_k, fp_ks, fp_kz, fp_v, recent_k, recent_v, recent.length_device,
        compact.packed if compact is not None else (
            i4.fp_k if full_precision_key and i4.token_count else
            item(i4, "k", "packed", torch.uint8)
        ),
        compact.scale if compact is not None else (
            workspace.empty_fp16 if full_precision_key else
            item(i4, "k", "scale", torch.float16)
        ),
        compact.zero_point if compact is not None else (
            item(i4, "k", "zero_point", torch.uint8) if uniform_int2_key else workspace.empty_u8
        ),
        item(i4, "v", "packed", torch.uint8), item(i4, "v", "scale", torch.float16),
        compact.packed if compact is not None else (
            i2.fp_k if full_precision_key and i2.token_count else
            item(i2, "k", "packed", torch.uint8)
        ),
        compact.scale if compact is not None else (
            workspace.empty_fp16 if full_precision_key else
            item(i2, "k", "scale", torch.float16)
        ),
        compact.zero_point if compact is not None else (
            workspace.empty_u8 if full_precision_key else
            item(i2, "k", "zero_point", torch.uint8)
        ),
        item(i2, "v", "packed", torch.uint8), item(i2, "v", "scale", torch.float16), item(i2, "v", "zero_point", torch.uint8),
        (
            getattr(compact, "residual_values", None)
            if compact is not None
            and getattr(compact, "residual_values", None) is not None
            else workspace.empty_bf16
        ),
        (
            getattr(compact, "residual_indices", None)
            if compact is not None
            and getattr(compact, "residual_indices", None) is not None
            else workspace.empty_u8
        ),
    )
    nfp = (fp.token_count + block_n - 1) // block_n
    recent_extent = recent.capacity if recent.active_length is not None else recent.token_count
    nr = (recent_extent + block_n - 1) // block_n
    ni4 = (i4.token_count + block_n - 1) // block_n
    ni2 = (i2.token_count + block_n - 1) // block_n
    total = nfp + nr + ni4 + ni2
    if total > workspace.total_blocks:
        raise ValueError("workspace block capacity is smaller than the active cache")
    launch_total = _launch_block_count(total, workspace.total_blocks)
    hierarchical_group_count = (
        launch_total + workspace.hierarchical_group_blocks - 1
    ) // workspace.hierarchical_group_blocks
    value_group_size = (
        int(cache.compact_value.group_size)
        if cache.compact_value is not None
        else int(getattr(i2.packed_v, "group_size", 32))
        if i2.token_count
        else 32
    )
    if value_group_size not in {32, 64, 128} or 128 % value_group_size:
        raise ValueError("split-K INT2 V group size must be 32, 64, or 128")
    return SplitKExecutor(
        cache=cache, workspace=workspace, args=args,
        s_fp=fp.token_count, s_recent=recent_extent, s_i4=i4.token_count, s_i2=i2.token_count,
        n_fp=nfp, n_recent=nr, n_i4=ni4, total=launch_total,
        fp_i2_superblocks=max(1, (fp.token_count + 127) // 128),
        i4_i2_superblocks=max(1, (i4.token_count + 127) // 128),
        i2_superblocks=max(1, (i2.token_count + 127) // 128),
        uniform_int2_key=uniform_int2_key,
        compact_int2_key=cache.compact_int2_key,
        uniform_int4_key=uniform_int4_key,
        compact_int4_key=cache.compact_int4_key,
        full_precision_key=full_precision_key,
        key_residual_channels=(
            0 if compact is None else int(getattr(compact, "residual_channels", 0))
        ),
        k_capacity=1 if compact is None else compact.capacity,
        k_superblocks=1 if compact is None else compact.superblock_count,
        k_superblock_size=128 if compact is None else compact.superblock_size,
        k_fp_offset=0 if compact is None else compact.segment_offsets[0],
        k_i4_offset=0 if compact is None else compact.segment_offsets[1],
        k_i2_offset=0 if compact is None else compact.segment_offsets[2],
        static_int4_tokens=cache.static_int4_tokens,
        generated_k_offset=cache.generated_k_offset,
        compact_int4_value=cache.compact_int4_value is not None,
        v_i4_capacity=(
            1
            if cache.compact_int4_value is None
            else cache.compact_int4_value.capacity
        ),
        compact_int2_value=cache.compact_value is not None,
        v_i2_capacity=1 if cache.compact_value is None else cache.compact_value.capacity,
        v_int2_group_size=value_group_size,
        v_int2_groups=128 // value_group_size,
        reduce_block=1 << (launch_total - 1).bit_length(),
        block_n=block_n,
        num_warps=num_warps, num_stages=num_stages,
        reduce_num_warps=reduce_num_warps,
        use_tensor_cores_qk=bool(use_tensor_cores_qk),
        use_tensor_cores_pv=bool(use_tensor_cores_pv),
        fuse_tensor_core_dots=False,
        # A tile that straddles static and generated INT4-V tokens maps to two
        # disjoint compact-K ranges. The pointwise metadata path supports this
        # mapping; the contiguous-row broadcast shortcut intentionally does not.
        broadcast_k_metadata=(
            broadcast_k_metadata
            and not (
                cache.compact_int4_value is not None
                and i4.token_count > cache.static_int4_tokens
            )
        ),
        # The metadata-free specialization is consistently beneficial for the
        # occupancy-oriented 64-token tile.  The 128-token tile is already
        # launch-bound at short lengths and keeps the generic path by default.
        single_request_fastpath=(
            block_n == 64
            if single_request_fastpath is None
            else bool(single_request_fastpath)
        ),
        # Compact-K metadata is laid out in 128-token superblocks, but a
        # 64-token attention tile can touch at most the rows covered by that
        # tile.  The tighter bound is bit-exact and consistently faster for
        # the occupancy-oriented long-context schedule.
        tight_metadata_rows=(
            block_n == 64
            if tight_metadata_rows is None
            else bool(tight_metadata_rows)
        ),
        hierarchical_reduce=bool(hierarchical_reduce and launch_total > 1),
        hierarchical_group_blocks=workspace.hierarchical_group_blocks,
        hierarchical_group_count=hierarchical_group_count,
        output_hadamard=bool(output_hadamard),
        num_query_heads=num_query_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
    )


def allocate_batched_splitk_workspace_like(
    workspace: SplitKWorkspace,
    batch_size: int,
) -> SplitKWorkspace:
    """Allocate a homogeneous throughput workspace without touching B=1.

    The existing allocator deliberately retains its validated two/three-
    dimensional partial buffers.  Batched callers use leading request storage
    and a second stride table for the hierarchical intermediate buffers.
    """
    batch_size = int(batch_size)
    if batch_size < 2:
        raise ValueError("batched split-K workspace requires batch_size >= 2")
    if workspace.output.ndim != 3 or int(workspace.output.shape[0]) != 1:
        raise ValueError("source workspace must be a validated batch=1 workspace")
    if workspace.partial_max.ndim != 2 or workspace.partial_output.ndim != 3:
        raise ValueError("source partial workspace must use batch=1 geometry")

    device = workspace.output.device
    h = int(workspace.output.shape[1])
    d = int(workspace.output.shape[2])
    total = int(workspace.total_blocks)
    group_capacity = int(workspace.hierarchical_group_capacity)
    return SplitKWorkspace(
        partial_max=torch.empty(
            (batch_size, h, total), device=device, dtype=torch.float32
        ),
        partial_sum=torch.empty(
            (batch_size, h, total), device=device, dtype=torch.float32
        ),
        partial_output=torch.empty(
            (batch_size, h, total, d),
            device=device,
            dtype=workspace.partial_output.dtype,
        ),
        output=torch.empty(
            (batch_size, h, d), device=device, dtype=workspace.output.dtype
        ),
        empty_bf16=torch.empty(1, device=device, dtype=torch.bfloat16),
        empty_fp16=torch.empty(1, device=device, dtype=torch.float16),
        empty_u8=torch.empty(1, device=device, dtype=torch.uint8),
        batch_strides=torch.zeros(30, device=device, dtype=torch.int64),
        group_batch_strides=torch.zeros(30, device=device, dtype=torch.int64),
        group_partial_max=torch.empty(
            (batch_size, h, group_capacity),
            device=device,
            dtype=torch.float32,
        ),
        group_partial_sum=torch.empty(
            (batch_size, h, group_capacity),
            device=device,
            dtype=torch.float32,
        ),
        group_partial_output=torch.empty(
            (batch_size, h, group_capacity, d),
            device=device,
            dtype=torch.float32,
        ),
        hierarchical_group_blocks=int(workspace.hierarchical_group_blocks),
        hierarchical_group_capacity=group_capacity,
        block_n=int(workspace.block_n),
        total_blocks=total,
    )


def _stack_homogeneous_tensors(
    tensors: Sequence[torch.Tensor],
    name: str,
) -> torch.Tensor:
    first = tensors[0]
    for request, tensor in enumerate(tensors[1:], start=1):
        if (
            tensor.shape != first.shape
            or tensor.dtype != first.dtype
            or tensor.device != first.device
        ):
            raise ValueError(
                f"batched split-K {name} differs for request {request}: "
                f"expected shape/dtype/device={tuple(first.shape)}/{first.dtype}/"
                f"{first.device}, got {tuple(tensor.shape)}/{tensor.dtype}/"
                f"{tensor.device}"
            )
    return torch.stack(tuple(tensors), dim=0).contiguous()


def stack_homogeneous_splitk_executors(
    executors: Sequence[SplitKExecutor],
    *,
    workspace: SplitKWorkspace | None = None,
    throughput_use_tensor_cores_pv: bool = False,
    throughput_direct_reduce_max_splits: int = 0,
) -> SplitKExecutor:
    """Combine independent B=1 cache snapshots into one throughput executor.

    This is intentionally an opt-in batch>1 path.  It requires identical cache
    geometry and kernel policy, copies each persistent payload once into a
    request-major arena, and then keeps generated Recent-FP state in that arena.
    The original executors remain unchanged and continue to be the quality
    oracle.  Recompression is a scheduler boundary: callers must unstack or
    rebuild a homogeneous executor when any request fills its Recent-FP pool.

    Throughput-only arithmetic changes are explicit here instead of mutating
    the source B=1 executors. PV Tensor Cores may be enabled for B>1, and the
    redundant hierarchical group pass may be skipped when the full split set
    already fits in a small direct reduction. Both defaults preserve the
    previously validated batched path.
    """
    executors = tuple(executors)
    if len(executors) < 2:
        raise ValueError("stacking split-K executors requires at least two requests")
    first = executors[0]
    if first.workspace.output.ndim != 3 or int(first.workspace.output.shape[0]) != 1:
        raise ValueError("source split-K executors must use batch=1 workspaces")
    throughput_direct_reduce_max_splits = int(
        throughput_direct_reduce_max_splits
    )
    if throughput_direct_reduce_max_splits not in {0, 16, 32, 64, 128}:
        raise ValueError(
            "throughput_direct_reduce_max_splits must be 0/16/32/64/128"
        )

    scalar_fields = (
        "s_fp", "s_recent", "s_i4", "s_i2", "n_fp", "n_recent", "n_i4",
        "total", "fp_i2_superblocks", "i4_i2_superblocks", "i2_superblocks",
        "uniform_int2_key", "compact_int2_key", "uniform_int4_key",
        "compact_int4_key", "full_precision_key", "key_residual_channels",
        "k_capacity", "k_superblocks", "k_superblock_size", "k_fp_offset",
        "k_i4_offset", "k_i2_offset", "static_int4_tokens",
        "generated_k_offset", "compact_int4_value", "v_i4_capacity",
        "compact_int2_value", "v_i2_capacity", "v_int2_group_size",
        "v_int2_groups", "reduce_block", "block_n", "num_warps",
        "num_stages", "reduce_num_warps", "use_tensor_cores_qk",
        "use_tensor_cores_pv", "fuse_tensor_core_dots",
        "broadcast_k_metadata", "tight_metadata_rows",
        "hierarchical_reduce", "hierarchical_group_blocks",
        "hierarchical_group_count", "output_hadamard", "num_query_heads",
        "native_full_head_layout", "num_key_value_heads", "head_dim",
    )
    workspace_fields = (
        "total_blocks", "block_n", "hierarchical_group_blocks",
        "hierarchical_group_capacity",
    )
    for request, executor in enumerate(executors[1:], start=1):
        differences = [
            field
            for field in scalar_fields
            if getattr(executor, field) != getattr(first, field)
        ]
        differences.extend(
            f"workspace.{field}"
            for field in workspace_fields
            if getattr(executor.workspace, field) != getattr(first.workspace, field)
        )
        if executor.cache.page_size != first.cache.page_size:
            differences.append("cache.page_size")
        if executor.cache.recent_fp_pool.capacity != first.cache.recent_fp_pool.capacity:
            differences.append("recent_fp.capacity")
        if differences:
            raise ValueError(
                f"split-K request {request} is outside the homogeneous bucket: "
                + ", ".join(differences)
            )
        if len(executor.args) != len(first.args):
            raise ValueError("split-K executor argument layouts differ")

    batch_size = len(executors)
    if workspace is None:
        workspace = allocate_batched_splitk_workspace_like(
            first.workspace, batch_size
        )
    else:
        expected_output = (
            batch_size,
            first.num_query_heads,
            first.head_dim,
        )
        if tuple(workspace.output.shape) != expected_output:
            raise ValueError(
                "reused batched workspace output geometry differs: "
                f"expected {expected_output}, got {tuple(workspace.output.shape)}"
            )
        compatible = (
            workspace.total_blocks >= first.workspace.total_blocks
            and workspace.block_n == first.workspace.block_n
            and workspace.hierarchical_group_blocks
            == first.workspace.hierarchical_group_blocks
            and workspace.hierarchical_group_capacity
            >= first.workspace.hierarchical_group_capacity
        )
        if not compatible:
            workspace = allocate_batched_splitk_workspace_like(
                first.workspace, batch_size
            )
        else:
            # Layers can have different FP/INT4/INT2 pool extents. Share the
            # large capacity buffers, but keep the tiny stride tables local so
            # one layer cannot overwrite another layer's cache layout.
            workspace = replace(
                workspace,
                batch_strides=torch.zeros_like(workspace.batch_strides),
                group_batch_strides=torch.zeros_like(
                    workspace.group_batch_strides
                ),
            )
    # Compact K and empty placeholders intentionally appear at multiple ABI
    # positions. Preserve those aliases in the request-major arena instead of
    # allocating duplicate copies that would erase the cache-memory benefit.
    stacked_by_source: dict[tuple[int, ...], torch.Tensor] = {}
    batched_arg_list: list[torch.Tensor] = []
    for index in range(len(first.args)):
        sources = tuple(executor.args[index] for executor in executors)
        source_key = tuple(id(tensor) for tensor in sources)
        stacked = stacked_by_source.get(source_key)
        if stacked is None:
            stacked = _stack_homogeneous_tensors(
                sources, f"argument[{index}]"
            )
            stacked_by_source[source_key] = stacked
        batched_arg_list.append(stacked)
    batched_args = tuple(batched_arg_list)
    if len(batched_args) != 20:
        raise RuntimeError("unexpected split-K argument layout")

    recent_pools = tuple(executor.cache.recent_fp_pool for executor in executors)
    if any(
        pool.global_index_buffer is None
        or pool.logical_position_buffer is None
        or pool.length_device is None
        for pool in recent_pools
    ):
        raise ValueError("batched Recent-FP metadata buffers are incomplete")
    recent_global_indices = _stack_homogeneous_tensors(
        tuple(pool.global_index_buffer for pool in recent_pools),
        "Recent-FP global indices",
    )
    recent_logical_positions = _stack_homogeneous_tensors(
        tuple(pool.logical_position_buffer for pool in recent_pools),
        "Recent-FP logical positions",
    )
    recent_page_lengths = _stack_homogeneous_tensors(
        tuple(pool.page_index.lengths for pool in recent_pools),
        "Recent-FP page lengths",
    )
    recent_lengths = [
        int(pool.active_length)
        for pool in recent_pools
        if pool.active_length is not None
    ]
    if len(recent_lengths) != batch_size:
        raise ValueError("batched Recent-FP pools require active lengths")

    append_stride = first.num_key_value_heads * first.head_dim
    strides = [
        first.num_query_heads * first.head_dim,
        *(int(tensor.stride(0)) for tensor in batched_args[:7]),
        append_stride,
        append_stride,
        *(int(tensor.stride(0)) for tensor in batched_args[7:]),
        int(workspace.partial_max.stride(0)),
        int(workspace.partial_sum.stride(0)),
        int(workspace.partial_output.stride(0)),
        int(workspace.output.stride(0)),
        int(recent_global_indices.stride(0)),
        int(recent_logical_positions.stride(0)),
        int(recent_page_lengths.stride(0)),
    ]
    if len(strides) != 30:
        raise RuntimeError("unexpected split-K stride table layout")
    stride_tensor = torch.tensor(strides, device=workspace.output.device, dtype=torch.int64)
    workspace.batch_strides.copy_(stride_tensor)
    group_strides = list(strides)
    group_strides[23] = int(workspace.group_partial_max.stride(0))
    group_strides[24] = int(workspace.group_partial_sum.stride(0))
    group_strides[25] = int(workspace.group_partial_output.stride(0))
    workspace.group_batch_strides.copy_(
        torch.tensor(group_strides, device=workspace.output.device, dtype=torch.int64)
    )

    throughput_hierarchical_reduce = bool(first.hierarchical_reduce)
    if (
        throughput_direct_reduce_max_splits > 0
        and int(first.total) <= throughput_direct_reduce_max_splits
    ):
        throughput_hierarchical_reduce = False

    # Four narrow Tensor-Core dots minimize register pressure at low batch,
    # whereas one full-head dot removes redundant MMA setup once independent
    # request/KV-head groups can occupy at least half of the GPU SMs.  This is
    # one hardware-relative occupancy rule, not a batch-size lookup table.
    sm_count = int(
        torch.cuda.get_device_properties(workspace.output.device).multi_processor_count
    )
    fuse_tensor_core_dots = bool(
        first.use_tensor_cores_qk
        and (first.use_tensor_cores_pv or throughput_use_tensor_cores_pv)
        and batch_size * first.num_key_value_heads * 2 >= sm_count
    )
    native_full_head_layout = bool(
        fuse_tensor_core_dots
        and first.uniform_int4_key
        and first.compact_int4_key
        and not first.full_precision_key
        and first.key_residual_channels == 0
        and first.block_n == 128
        and first.head_dim == 128
    )

    return replace(
        first,
        workspace=workspace,
        args=batched_args,
        single_request_fastpath=False,
        batched_recent=(
            batched_args[4],
            batched_args[5],
            recent_global_indices,
            recent_logical_positions,
            batched_args[6],
            recent_page_lengths,
        ),
        batch_append_metadata=torch.zeros(
            (batch_size, 2), device=workspace.output.device, dtype=torch.int64
        ),
        batched_recent_lengths=recent_lengths,
        use_tensor_cores_pv=(
            bool(first.use_tensor_cores_pv)
            or bool(throughput_use_tensor_cores_pv)
        ),
        fuse_tensor_core_dots=fuse_tensor_core_dots,
        native_full_head_layout=native_full_head_layout,
        hierarchical_reduce=throughput_hierarchical_reduce,
    )


def allocated_workspace_bytes(workspace: SplitKWorkspace) -> int:
    return sum(
        int(tensor.numel() * tensor.element_size())
        for tensor in (
            workspace.partial_max,
            workspace.partial_sum,
            workspace.partial_output,
            workspace.group_partial_max,
            workspace.group_partial_sum,
            workspace.group_partial_output,
            workspace.output,
        )
    )
