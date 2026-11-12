from __future__ import annotations

from dataclasses import dataclass

import torch

from ..policy.four_quadrant import PageClass
from ..policy.page_assignment import PageAssignment


@dataclass(frozen=True)
class LayerCompressionPlan:
    layer_id: int
    fp_pages: torch.Tensor
    int4_pages: torch.Tensor
    int2_pages: torch.Tensor
    evicted_pages: torch.Tensor
    protected_pages: torch.Tensor
    recent_fp_capacity: int
    logical_positions: torch.Tensor
    global_token_indices: torch.Tensor
    page_offsets: torch.Tensor
    page_lengths: torch.Tensor
    page_classes: torch.Tensor


def build_layer_compression_plan(
    layer_id: int,
    assignment: PageAssignment,
    *,
    recent_fp_capacity: int = 256,
) -> LayerCompressionPlan:
    if layer_id < 0:
        raise ValueError("layer_id must be non-negative")
    if recent_fp_capacity <= 0:
        raise ValueError("recent_fp_capacity must be positive")
    classes = assignment.page_classes
    fp = ((classes == int(PageClass.FP_PAGE)) | (classes == int(PageClass.RECENT_FP_PAGE)) | (classes == int(PageClass.PROTECTED_FP_PAGE))).nonzero(as_tuple=False).flatten()
    return LayerCompressionPlan(
        layer_id=int(layer_id),
        fp_pages=fp,
        int4_pages=(classes == int(PageClass.INT4_PAGE)).nonzero(as_tuple=False).flatten(),
        int2_pages=(classes == int(PageClass.INT2_PAGE)).nonzero(as_tuple=False).flatten(),
        evicted_pages=(classes == int(PageClass.EVICTED_PAGE)).nonzero(as_tuple=False).flatten(),
        protected_pages=(classes == int(PageClass.PROTECTED_FP_PAGE)).nonzero(as_tuple=False).flatten(),
        recent_fp_capacity=int(recent_fp_capacity),
        logical_positions=assignment.page_table.logical_positions.clone(),
        global_token_indices=assignment.page_table.global_token_indices.clone(),
        page_offsets=assignment.page_table.offsets.clone(),
        page_lengths=assignment.page_table.lengths.clone(),
        page_classes=classes.clone(),
    )
