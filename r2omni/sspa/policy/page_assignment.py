from __future__ import annotations

from dataclasses import dataclass

import torch

from .four_quadrant import PageClass
from .modality_metadata import OTHER, TEXT


@dataclass(frozen=True)
class PageTable:
    offsets: torch.Tensor
    lengths: torch.Tensor
    modality: torch.Tensor
    token_indices: torch.Tensor
    logical_positions: torch.Tensor
    global_token_indices: torch.Tensor
    protected: torch.Tensor
    recent: torch.Tensor

    @property
    def num_pages(self) -> int:
        return int(self.offsets.numel())


@dataclass(frozen=True)
class PageAssignment:
    page_table: PageTable
    page_classes: torch.Tensor
    attention_scores: torch.Tensor
    alignment_scores: torch.Tensor
    fused_scores: torch.Tensor


def project_token_bits_to_pages(page_table: PageTable, token_bits: torch.Tensor) -> PageAssignment:
    """Conservatively project token precision to non-interleaved physical pages."""
    bits = token_bits.flatten().to(device=page_table.offsets.device, dtype=torch.int8)
    if bits.numel() != page_table.token_indices.numel():
        raise ValueError("token_bits must have one value per page-table token")
    if not bool(torch.all((bits == 0) | (bits == 2) | (bits == 4) | (bits == 16))):
        raise ValueError("token_bits may contain only 0, 2, 4, or 16")
    classes = torch.empty(page_table.num_pages, device=bits.device, dtype=torch.int8)
    for page, (offset, length) in enumerate(zip(page_table.offsets.tolist(), page_table.lengths.tolist())):
        local = bits[offset : offset + length]
        if page_table.protected[page]:
            value = PageClass.PROTECTED_FP_PAGE
        elif page_table.recent[page]:
            value = PageClass.RECENT_FP_PAGE
        elif bool((local == 16).any()):
            value = PageClass.FP_PAGE
        elif bool((local == 4).any()):
            value = PageClass.INT4_PAGE
        elif bool((local == 2).any()):
            value = PageClass.INT2_PAGE
        else:
            value = PageClass.EVICTED_PAGE
        classes[page] = int(value)
    zeros = torch.zeros(page_table.num_pages, device=bits.device, dtype=torch.float32)
    return PageAssignment(page_table, classes, zeros, zeros, zeros)


def build_page_table(
    modality: torch.Tensor,
    *,
    page_size: int = 32,
    logical_positions: torch.Tensor | None = None,
    global_token_indices: torch.Tensor | None = None,
    protected_mask: torch.Tensor | None = None,
    recent_mask: torch.Tensor | None = None,
) -> PageTable:
    """Build contiguous pages that never cross a modality boundary."""
    if page_size not in {16, 32, 64}:
        raise ValueError("page_size must be one of 16, 32, or 64")
    modality = modality.flatten().to(torch.long)
    token_count = int(modality.numel())
    device = modality.device
    logical = (
        torch.arange(token_count, device=device)
        if logical_positions is None
        else logical_positions.flatten().to(device=device, dtype=torch.long)
    )
    global_indices = (
        torch.arange(token_count, device=device)
        if global_token_indices is None
        else global_token_indices.flatten().to(device=device, dtype=torch.long)
    )
    # Preserve the public helper's conservative default when callers do not
    # provide a policy mask.  When a mask is supplied, however, it is the
    # complete source of truth: managers use an explicit all-false mask for
    # ``protect_text=False`` and must not have text silently re-protected here.
    protected = (
        (modality == TEXT) | (modality == OTHER)
        if protected_mask is None
        else protected_mask.flatten().to(device=device, dtype=torch.bool)
    )
    recent = (
        torch.zeros(token_count, dtype=torch.bool, device=device)
        if recent_mask is None
        else recent_mask.flatten().to(device=device, dtype=torch.bool)
    )
    for name, value in (("logical_positions", logical), ("global_token_indices", global_indices), ("protected_mask", protected), ("recent_mask", recent)):
        if value.numel() != token_count:
            raise ValueError(f"{name} must have one value per token")

    # Convert tiny metadata vectors once. Scalar indexing a CUDA tensor in the
    # page-building loop would otherwise synchronize once per token/page.
    modality_values = modality.tolist()
    protected_values = protected.tolist()
    recent_values = recent.tolist()
    offsets: list[int] = []
    lengths: list[int] = []
    page_modalities: list[int] = []
    page_protected: list[bool] = []
    page_recent: list[bool] = []
    start = 0
    while start < token_count:
        value = int(modality_values[start])
        boundary = start + 1
        while boundary < token_count and boundary - start < page_size and modality_values[boundary] == value:
            boundary += 1
        offsets.append(start)
        lengths.append(boundary - start)
        page_modalities.append(value)
        page_protected.append(any(protected_values[start:boundary]))
        page_recent.append(any(recent_values[start:boundary]))
        start = boundary
    return PageTable(
        offsets=torch.tensor(offsets, dtype=torch.long, device=device),
        lengths=torch.tensor(lengths, dtype=torch.long, device=device),
        modality=torch.tensor(page_modalities, dtype=torch.long, device=device),
        token_indices=torch.arange(token_count, dtype=torch.long, device=device),
        logical_positions=logical.clone(),
        global_token_indices=global_indices.clone(),
        protected=torch.tensor(page_protected, dtype=torch.bool, device=device),
        recent=torch.tensor(page_recent, dtype=torch.bool, device=device),
    )


def aggregate_page_scores(
    token_scores: torch.Tensor,
    page_table: PageTable,
    *,
    reduction: str = "topk_mean",
    topk: int = 4,
) -> torch.Tensor:
    if reduction not in {"mean", "max", "topk_mean"}:
        raise ValueError("reduction must be mean, max, or topk_mean")
    if reduction == "topk_mean" and topk <= 0:
        raise ValueError("topk must be positive")
    values = token_scores.flatten().float()
    if values.numel() != page_table.token_indices.numel():
        raise ValueError("token_scores must have one value per page-table token")
    if page_table.num_pages == 0:
        return torch.empty(0, device=values.device)
    width = 64
    lanes = torch.arange(width, device=values.device)
    indices = page_table.offsets[:, None] + lanes[None, :]
    valid = lanes[None, :] < page_table.lengths[:, None]
    pages = values.index_select(0, indices.clamp_max(values.numel() - 1).flatten()).view(page_table.num_pages, width)
    if reduction == "mean":
        return pages.masked_fill(~valid, 0.0).sum(dim=1) / page_table.lengths
    if reduction == "max":
        return pages.masked_fill(~valid, -torch.inf).amax(dim=1)
    k = min(int(topk), width)
    selected = pages.masked_fill(~valid, -torch.inf).topk(k, dim=1).values
    selected = selected.masked_fill(~torch.isfinite(selected), 0.0)
    return selected.sum(dim=1) / page_table.lengths.clamp_max(k)
