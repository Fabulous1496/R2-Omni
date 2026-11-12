from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .quantization import dequantize_tensor_symmetric, pack_lowbit, quantize_tensor_symmetric, unpack_lowbit


@dataclass
class PackedTensor:
    packed: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor | None
    num_bits: int
    original_shape: tuple[int, ...]
    dtype: torch.dtype
    quant_scheme: str
    pack_metadata: dict[str, Any]
    quant_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def nbytes(self) -> int:
        total = self.packed.numel() * self.packed.element_size() + self.scale.numel() * self.scale.element_size()
        if self.zero_point is not None:
            total += self.zero_point.numel() * self.zero_point.element_size()
        return int(total)

    def dequantize(self) -> torch.Tensor:
        q = unpack_lowbit(self.packed, self.original_shape, self.num_bits, signed=True)
        return dequantize_tensor_symmetric(q, self.scale, self.num_bits).to(self.dtype)


@dataclass
class PackedKVLayer:
    fp_k: torch.Tensor | None
    fp_v: torch.Tensor | None
    int4_k: PackedTensor | None
    int4_v: PackedTensor | None
    int2_k: PackedTensor | None
    int2_v: PackedTensor | None
    keep_indices_fp: torch.Tensor | None
    keep_indices_int4: torch.Tensor | None
    keep_indices_int2: torch.Tensor | None
    evicted_indices: torch.Tensor | None
    original_seq_len: int
    compressed_seq_len: int
    original_layout: str = "BHSD"
    dtype: torch.dtype | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    fp_k_buffer: torch.Tensor | None = None
    fp_v_buffer: torch.Tensor | None = None
    fp_index_buffer: torch.Tensor | None = None
    fp_length: int = 0
    fp_capacity: int = 0
    fp_reallocation_count: int = 0

    @property
    def packed_bytes(self) -> int:
        total = 0
        fp_k_storage = self.fp_k_buffer if self.fp_k_buffer is not None else self.fp_k
        fp_v_storage = self.fp_v_buffer if self.fp_v_buffer is not None else self.fp_v
        fp_index_storage = self.fp_index_buffer if self.fp_index_buffer is not None else self.keep_indices_fp
        for tensor in (fp_k_storage, fp_v_storage, fp_index_storage, self.keep_indices_int4, self.keep_indices_int2, self.evicted_indices):
            if tensor is not None:
                total += tensor.numel() * tensor.element_size()
        for tensor in (self.int4_k, self.int4_v, self.int2_k, self.int2_v):
            if tensor is not None:
                total += tensor.nbytes
        return int(total)

    def dequantize_sorted(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pieces: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        if self.keep_indices_fp is not None and self.keep_indices_fp.numel():
            assert self.fp_k is not None and self.fp_v is not None
            pieces.append((self.keep_indices_fp.long(), self.fp_k, self.fp_v))
        if self.keep_indices_int4 is not None and self.keep_indices_int4.numel():
            assert self.int4_k is not None and self.int4_v is not None
            pieces.append((self.keep_indices_int4.long(), self.int4_k.dequantize(), self.int4_v.dequantize()))
        if self.keep_indices_int2 is not None and self.keep_indices_int2.numel():
            assert self.int2_k is not None and self.int2_v is not None
            pieces.append((self.keep_indices_int2.long(), self.int2_k.dequantize(), self.int2_v.dequantize()))
        if not pieces:
            raise ValueError("PackedKVLayer contains no kept tokens")
        indices = torch.cat([item[0] for item in pieces], dim=0)
        order = torch.argsort(indices, stable=True)
        sorted_indices = indices.index_select(0, order)
        key = torch.cat([item[1] for item in pieces], dim=1).index_select(1, order)
        value = torch.cat([item[2] for item in pieces], dim=1).index_select(1, order)
        return key.unsqueeze(0).contiguous(), value.unsqueeze(0).contiguous(), sorted_indices


@dataclass
class PackedKVCache:
    layers: list[PackedKVLayer]
    metadata: dict[str, Any]


def make_packed_tensor(tensor: torch.Tensor, num_bits: int, quant_scheme: str, dim: int) -> PackedTensor:
    q, scale, qmeta = quantize_tensor_symmetric(tensor, num_bits, dim=dim)
    packed, pmeta = pack_lowbit(q, num_bits)
    return PackedTensor(
        packed=packed,
        scale=scale,
        zero_point=None,
        num_bits=int(num_bits),
        original_shape=tuple(int(v) for v in tensor.shape),
        dtype=tensor.dtype,
        quant_scheme=quant_scheme,
        pack_metadata=pmeta,
        quant_metadata=qmeta,
    )


def tensor_nbytes(tensor: torch.Tensor | None) -> int:
    return 0 if tensor is None else int(tensor.numel() * tensor.element_size())
