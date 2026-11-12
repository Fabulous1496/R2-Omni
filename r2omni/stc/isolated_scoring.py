from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class IsolatedSelection:
    kept_indices: list[int]
    score_source: str
    used_encoder_attention: bool


def select_isolated_tokens(
    features: torch.Tensor,
    indices: list[int],
    *,
    keep_ratio: float,
    policy: str,
    encoder_attention_scores: torch.Tensor | None = None,
    reference_mean: torch.Tensor | None = None,
) -> IsolatedSelection:
    """Select isolated tokens, preferring encoder attention with explicit fallbacks."""
    if not indices or keep_ratio <= 0:
        return IsolatedSelection([], "none", False)
    # A full keep ratio is an explicit no-pruning policy for tokens outside
    # aggregation regions.  Avoid scoring/top-k work and preserve input order.
    if keep_ratio >= 1.0:
        return IsolatedSelection(list(indices), "keep_all", False)
    count = min(len(indices), max(1, int(len(indices) * min(1.0, keep_ratio))))
    index = torch.as_tensor(indices, dtype=torch.long, device=features.device)
    if policy == "encoder_attention_topk" and encoder_attention_scores is not None:
        scores = encoder_attention_scores.to(features.device).index_select(0, index).float()
        source, used_attention = "encoder_attention", True
    elif policy in {"encoder_attention_topk", "mean_cosine_distance_topk"} and features.numel():
        selected = features.index_select(0, index)
        mean = features.mean(dim=0, keepdim=True) if reference_mean is None else reference_mean.reshape(1, -1)
        scores = 1.0 - F.cosine_similarity(selected, mean, dim=-1)
        source = "mean_cosine_distance" if policy == "mean_cosine_distance_topk" else "cosine_distance_fallback"
        used_attention = False
    else:
        return IsolatedSelection(indices[:count], "legacy_order_fallback", False)
    top = torch.topk(scores, k=count, largest=True, sorted=False).indices.detach().cpu().tolist()
    kept = sorted(indices[local] for local in top)
    return IsolatedSelection(kept, source, used_attention)
