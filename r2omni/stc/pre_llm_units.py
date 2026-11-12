from __future__ import annotations

import math
import time
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from r2omni.stc.isolated_scoring import select_isolated_tokens


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load the STC configuration") from exc
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or not isinstance(config.get("stc"), dict):
        raise ValueError(f"{path} must contain an stc mapping")
    return config


def _cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1, eps=eps).squeeze(0)


def _to_int_list(values: torch.Tensor | list[int] | tuple[int, ...]) -> list[int]:
    if isinstance(values, torch.Tensor):
        return [int(v) for v in values.detach().cpu().flatten().tolist()]
    return [int(v) for v in values]


def _split_large_region(region: list[int], max_region_size: int) -> list[list[int]]:
    if max_region_size <= 0 or len(region) <= max_region_size:
        return [region]
    return [region[i : i + max_region_size] for i in range(0, len(region), max_region_size)]


@lru_cache(maxsize=64)
def _grid_edges(height: int, width: int, neighbor_size: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    offsets = [(0, 1), (1, 0)] if neighbor_size == 4 else [
        (0, 1), (1, -1), (1, 0), (1, 1)
    ]
    sources: list[int] = []
    targets: list[int] = []
    for y in range(height):
        for x in range(width):
            index = y * width + x
            for dy, dx in offsets:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    sources.append(index)
                    targets.append(ny * width + nx)
    return tuple(sources), tuple(targets)


def _regions_from_components(
    components: dict[int, list[int]],
    min_region_size: int,
    max_region_size: int,
) -> tuple[list[list[int]], list[int]]:
    regions: list[list[int]] = []
    isolated: list[int] = []
    for component in components.values():
        if len(component) < min_region_size:
            isolated.extend(component)
            continue
        for part in _split_large_region(sorted(component), max_region_size):
            if len(part) < min_region_size:
                isolated.extend(part)
            else:
                regions.append(part)
    return regions, sorted(isolated)


def build_spatial_regions(
    frame_features: torch.Tensor,
    height: int,
    width: int,
    threshold: float,
    neighbor_size: int = 8,
    min_region_size: int = 2,
    max_region_size: int = 128,
) -> tuple[list[list[int]], list[int]]:
    """Build local spatial regions on one frame using only local neighbors."""

    if neighbor_size not in {4, 8}:
        raise ValueError("neighbor_size must be 4 or 8")
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if min_region_size <= 0:
        raise ValueError("min_region_size must be positive")
    if 0 < max_region_size < min_region_size:
        raise ValueError("max_region_size must be non-positive or at least min_region_size")
    num_tokens = int(frame_features.shape[0])
    if num_tokens != height * width:
        raise ValueError(f"frame token count {num_tokens} does not match height*width {height * width}")

    edge_sources, edge_targets = _grid_edges(height, width, neighbor_size)
    if edge_sources:
        source_index = torch.tensor(edge_sources, dtype=torch.long, device=frame_features.device)
        target_index = torch.tensor(edge_targets, dtype=torch.long, device=frame_features.device)
        similarities = F.cosine_similarity(
            frame_features.index_select(0, source_index),
            frame_features.index_select(0, target_index),
            dim=-1,
        )
    else:
        similarities = frame_features.new_empty((0,))

    parent = list(range(num_tokens))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Keep cosine work on the feature device and synchronize only once per
    # frame.  The scalar .item() version serialized every small CUDA kernel.
    if edge_sources:
        connected = (similarities > threshold).detach().cpu().tolist()
        for idx, nidx, should_union in zip(edge_sources, edge_targets, connected):
            if should_union:
                union(idx, nidx)

    components: dict[int, list[int]] = {}
    for idx in range(num_tokens):
        components.setdefault(find(idx), []).append(idx)
    return _regions_from_components(components, min_region_size, max_region_size)


def _select_batched_region_anchors(
    features: torch.Tensor,
    regions: list[list[int]],
    min_region_size: int = 2,
    residual_anchors_per_region: int = 0,
) -> tuple[list[list[int]], int, int]:
    """Batched fast path for the canonical one-prototype region policy."""

    if not regions:
        return [], 0, 0

    lengths = [len(region) for region in regions]
    max_len = max(lengths)
    num_regions = len(regions)
    mean_sims = features.new_full((num_regions, max_len), float("-inf"))
    rows_by_length: dict[int, list[int]] = {}
    for row_idx, region in enumerate(regions):
        length = len(region)
        rows_by_length.setdefault(length, []).append(row_idx)

    # Group equal-length regions so the BF16 reduction shape is identical to
    # the scalar path, while each group still uses one gather/mean/cosine batch.
    for length, row_indices in rows_by_length.items():
        if len(row_indices) == 1:
            row_idx = row_indices[0]
            grouped_indices = torch.as_tensor(regions[row_idx], dtype=torch.long, device=features.device)
            grouped_features = features.index_select(0, grouped_indices)
            grouped_mean = grouped_features.mean(dim=0)
            mean_sims[row_idx, :length] = F.cosine_similarity(
                grouped_features, grouped_mean.unsqueeze(0), dim=-1
            )
            continue
        grouped_indices = torch.as_tensor(
            [regions[row_idx] for row_idx in row_indices],
            dtype=torch.long,
            device=features.device,
        )
        grouped_features = features[grouped_indices]
        grouped_means = grouped_features.mean(dim=1)
        grouped_sims = F.cosine_similarity(grouped_features, grouped_means.unsqueeze(1), dim=-1)
        row_tensor = torch.as_tensor(row_indices, dtype=torch.long, device=features.device)
        mean_sims[row_tensor, :length] = grouped_sims

    # Sort only valid entries. Sorting padded rows changes BF16 tie ordering as
    # max_len changes. Pad the already-sorted orders and transfer once.
    proto_rows = []
    for row_idx, length in enumerate(lengths):
        proto = torch.argsort(mean_sims[row_idx, :length], descending=True)
        proto_rows.append(F.pad(proto, (0, max_len - length), value=-1))
    proto_padded = torch.stack(proto_rows).detach().cpu().tolist()
    proto_orders_cpu = [row[:length] for row, length in zip(proto_padded, lengths)]

    selections: list[list[int]] = []
    prototype_count = 0
    residual_count = 0
    for region, proto_order in zip(regions, proto_orders_cpu):
        if len(region) < min_region_size:
            selections.append(list(region))
            residual_count += len(region)
            continue
        prototype_count += 1
        residual_count_for_region = min(
            max(0, int(residual_anchors_per_region)),
            max(0, len(region) - 1),
        )
        residual_local = list(reversed(proto_order))[:residual_count_for_region]
        selections.append(
            [region[proto_order[0]], *(region[index] for index in residual_local)]
        )
        residual_count += len(residual_local)

    return selections, prototype_count, residual_count


def compute_frame_similarity(f1_features: torch.Tensor, f2_features: torch.Tensor) -> float:
    """Compute cosine(mean(F1), mean(F2))."""

    if f1_features.numel() == 0 or f2_features.numel() == 0:
        return 0.0
    return float(_cosine(f1_features.mean(dim=0), f2_features.mean(dim=0)).item())


def _infer_video_shape(video_features: torch.Tensor, video_grid_thw: Any, video_merge_size: int = 1) -> tuple[int, int, int]:
    total = int(video_features.shape[0])
    if video_grid_thw is not None:
        values = _to_int_list(video_grid_thw)
        if len(values) >= 3:
            t, h, w = values[:3]
            grid_total = t * h * w
            if grid_total == total:
                return t, h, w

            # Qwen2.5-Omni's processor returns the pre-merge video grid.
            # Its placeholder length is T * (H / merge_size) * (W / merge_size).
            merge_size = max(1, int(video_merge_size))
            if h % merge_size == 0 and w % merge_size == 0:
                mh, mw = h // merge_size, w // merge_size
                if t * mh * mw == total:
                    return t, mh, mw

            if h * w > 0 and total % (h * w) == 0:
                return total // (h * w), h, w

        warnings.warn(
            f"video_grid_thw={values[:3] if len(values) >= 3 else values} does not match "
            f"{total} video tokens; falling back to inferred pseudo-frames.",
            stacklevel=2,
        )
    side = int(math.sqrt(total))
    if side > 0 and side * side == total:
        warnings.warn("video_grid_thw is unavailable; treating video tokens as one pseudo-frame.", stacklevel=2)
        return 1, side, side
    frame_tokens = max(1, side * side)
    if frame_tokens > 1 and total % frame_tokens == 0:
        warnings.warn("video_grid_thw is unavailable; using inferred pseudo-frames.", stacklevel=2)
        return total // frame_tokens, side, side
    warnings.warn("video_grid_thw is unavailable; treating video tokens as a 1D temporal sequence.", stacklevel=2)
    return total, 1, 1


def _mark_isolated(
    keep_mask: torch.Tensor,
    indices: list[int],
) -> None:
    if not indices:
        return
    index_tensor = torch.tensor(indices, dtype=torch.long, device=keep_mask.device)
    keep_mask[index_tensor] = True


def consolidate_video_tokens(
    video_features: torch.Tensor,
    video_grid_thw: Any,
    config: dict[str, Any],
    video_merge_size: int = 1,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Keep one spatial prototype per redundant region and rank isolated tokens."""

    started_total = time.perf_counter()
    isolated_scoring_time = 0.0
    cfg = config["stc"]["video"]
    keep_mask = torch.zeros(video_features.shape[0], dtype=torch.bool, device=video_features.device)
    if not cfg.get("enabled", True):
        keep_mask.fill_(True)
        return keep_mask, {"num_regions": 0, "num_prototypes": 0, "num_residuals": 0, "num_isolated": 0}
    if video_features.shape[0] == 0:
        return keep_mask, {"num_regions": 0, "num_prototypes": 0, "num_residuals": 0, "num_isolated": 0}

    t, h, w = _infer_video_shape(video_features, video_grid_thw, video_merge_size)
    frame_tokens = h * w
    prototypes = residuals = isolated_count = isolated_total = num_regions = 0
    isolated_sources: set[str] = set()
    def process_frame(frame_id: int, frame_sim: float | None = None) -> list[list[int]]:
        nonlocal prototypes, residuals, isolated_count, isolated_total, num_regions, isolated_scoring_time
        offset = frame_id * frame_tokens
        frame = video_features[offset : offset + frame_tokens]
        regions, isolated = build_spatial_regions(
            frame,
            h,
            w,
            float(cfg["region_cos_threshold"]),
            int(cfg["neighborhood"]),
            int(cfg["min_region_size"]),
            int(cfg["max_region_size"]),
        )
        global_regions = [[offset + idx for idx in region] for region in regions]
        selections, frame_prototypes, frame_residuals = _select_batched_region_anchors(
            video_features,
            global_regions,
            min_region_size=int(cfg["min_region_size"]),
            residual_anchors_per_region=int(cfg.get("residual_anchors_per_region", 0)),
        )
        kept = [idx for selection in selections for idx in selection]
        if kept:
            keep_mask[torch.as_tensor(kept, dtype=torch.long, device=keep_mask.device)] = True
        prototypes += frame_prototypes
        residuals += frame_residuals
        num_regions += len(global_regions)
        global_isolated = [offset + idx for idx in isolated]
        isolated_total += len(global_isolated)
        isolated_started = time.perf_counter()
        selection = select_isolated_tokens(
            video_features,
            global_isolated,
            keep_ratio=float(cfg["isolated_keep_ratio"]),
            policy=str(cfg["isolated_policy"]),
            encoder_attention_scores=None,
            reference_mean=frame.mean(dim=0),
        )
        isolated_scoring_time += time.perf_counter() - isolated_started
        isolated_sources.add(selection.score_source)
        _mark_isolated(keep_mask, selection.kept_indices)
        isolated_count += len(selection.kept_indices)
        return global_regions

    frame = 0
    while frame < t:
        f1_regions = process_frame(frame)
        if frame + 1 >= t:
            break
        f1 = video_features[frame * frame_tokens : (frame + 1) * frame_tokens]
        f2 = video_features[(frame + 1) * frame_tokens : (frame + 2) * frame_tokens]
        frame_sim = compute_frame_similarity(f1, f2)
        frame_threshold = float(cfg["frame_sim_threshold"])
        inherit_frame = frame_sim >= frame_threshold
        if inherit_frame:
            local_regions = [
                [idx - frame * frame_tokens for idx in region]
                for region in f1_regions
            ]
            region_for_spatial: dict[int, int] = {}
            for ridx, local_region in enumerate(local_regions):
                for local_idx in local_region:
                    region_for_spatial[local_idx] = ridx
            isolated_candidates: list[int] = []
            for spatial_idx in range(frame_tokens):
                ridx = region_for_spatial.get(spatial_idx)
                global_idx = (frame + 1) * frame_tokens + spatial_idx
                if ridx is None:
                    isolated_candidates.append(global_idx)
            if isolated_candidates:
                isolated_total += len(isolated_candidates)
                isolated_started = time.perf_counter()
                selection = select_isolated_tokens(
                    video_features,
                    isolated_candidates,
                    keep_ratio=float(cfg["isolated_keep_ratio"]),
                    policy=str(cfg["isolated_policy"]),
                    encoder_attention_scores=None,
                    reference_mean=f2.mean(dim=0),
                )
                isolated_scoring_time += time.perf_counter() - isolated_started
                isolated_sources.add(selection.score_source)
                _mark_isolated(
                    keep_mask,
                    selection.kept_indices,
                )
                isolated_count += len(selection.kept_indices)
        else:
            process_frame(frame + 1, frame_sim=frame_sim)
        frame += 2

    total_time = time.perf_counter() - started_total
    return keep_mask, {
        "num_regions": num_regions,
        "num_prototypes": prototypes,
        "num_residuals": residuals,
        "num_isolated": isolated_count,
        "num_isolated_total": isolated_total,
        "isolated_score_source": sorted(isolated_sources),
        "used_encoder_attention": "encoder_attention" in isolated_sources,
        "region_build_time_sec": max(0.0, total_time - isolated_scoring_time),
        "isolated_scoring_time_sec": isolated_scoring_time,
        "similarity_backend": "index_select",
        "neighborhood": int(cfg["neighborhood"]),
        "shape": {"t": t, "h": h, "w": w},
    }


def consolidate_audio_tokens(
    audio_features: torch.Tensor,
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """A-ToMe-style adjacent audio pair merging before LLM prefill.

    This is a pre-LLM approximation of A-ToMe: within each audio chunk, score
    adjacent token pairs by cosine similarity, greedily choose non-overlapping
    pairs above a threshold, average the pair into the left token, and drop the
    right token during sequence compaction.
    """

    started_total = time.perf_counter()
    cfg = config["stc"]["audio"]
    keep_mask = torch.ones(audio_features.shape[0], dtype=torch.bool, device=audio_features.device)
    if audio_features.shape[0] == 0 or not cfg.get("enabled", True):
        return keep_mask, {
            "num_regions": 0,
            "num_prototypes": 0,
            "num_residuals": 0,
            "num_boundary": 0,
            "num_isolated": 0,
            "merge_pairs": [],
        }

    chunk_tokens = max(1, int(cfg["chunk_tokens"]))
    threshold = float(cfg["pair_merge_threshold"])
    max_pair_ratio = float(cfg.get("max_pair_merge_ratio", 0.5))
    max_pair_ratio = min(0.5, max(0.0, max_pair_ratio))
    min_keep_ratio = float(cfg.get("min_keep_ratio", 1.0 - max_pair_ratio))
    min_keep_ratio = min(1.0, max(0.5, min_keep_ratio))
    max_pair_ratio = min(max_pair_ratio, 1.0 - min_keep_ratio)
    merge_pairs: list[tuple[int, int]] = []
    similarity_sum = 0.0
    similarity_count = 0

    # Preserve the original chunk-local reduction/sort shapes for numerical
    # equivalence, but concatenate every GPU worklist before one CPU transfer.
    # This removes O(num_chunks) synchronizations without changing the default
    # candidate ordering inside any chunk.
    token_count = int(audio_features.shape[0])
    if token_count >= 2 and max_pair_ratio > 0.0:
        candidate_parts: list[torch.Tensor] = []
        score_parts: list[torch.Tensor] = []
        for chunk_start in range(0, token_count, chunk_tokens):
            chunk_end = min(token_count, chunk_start + chunk_tokens)
            chunk_len = chunk_end - chunk_start
            max_pairs = int(chunk_len * max_pair_ratio)
            if chunk_len < 2 or max_pairs <= 0:
                continue
            similarities = F.cosine_similarity(
                audio_features[chunk_start : chunk_end - 1],
                audio_features[chunk_start + 1 : chunk_end],
                dim=-1,
            )
            candidates = (similarities >= threshold).nonzero(
                as_tuple=False
            ).flatten()
            if not candidates.numel():
                continue
            candidate_scores = similarities.index_select(0, candidates)
            order = torch.argsort(candidate_scores, descending=True)
            ordered_candidates = candidates.index_select(0, order)
            ordered_scores = candidate_scores.index_select(0, order)
            # A selected edge in the adjacent-token path can invalidate only
            # itself and its two neighbours.  Therefore the first 3M sorted
            # candidates always contain M non-overlapping selections whenever
            # the per-chunk M-pair budget can be filled.  Later candidates can
            # never affect the accepted prefix, so avoid transferring them for
            # aggressive low-threshold configurations.
            transfer_limit = min(int(ordered_candidates.numel()), 3 * max_pairs)
            candidate_parts.append(
                ordered_candidates[:transfer_limit] + chunk_start
            )
            score_parts.append(ordered_scores[:transfer_limit])

        if candidate_parts:
            # Positions are well below float32's exact-integer range. Packing
            # positions and scores together makes this one D2H copy and one
            # synchronization rather than two independent transfers.
            packed_worklist_cpu = torch.stack(
                (
                    torch.cat(candidate_parts).float(),
                    torch.cat(score_parts).float(),
                ),
                dim=1,
            ).detach().cpu().tolist()
            used: set[int] = set()
            accepted_per_chunk: dict[int, int] = {}
            for left_value, similarity_value in packed_worklist_cpu:
                left = int(left_value)
                right = left + 1
                chunk_start = left // chunk_tokens * chunk_tokens
                chunk_end = min(token_count, chunk_start + chunk_tokens)
                chunk_len = chunk_end - chunk_start
                max_pairs = int(chunk_len * max_pair_ratio)
                if max_pairs <= 0 or accepted_per_chunk.get(chunk_start, 0) >= max_pairs:
                    continue
                if left in used or right in used:
                    continue
                keep_mask[right] = False
                merge_pairs.append((left, right))
                similarity_sum += float(similarity_value)
                similarity_count += 1
                used.add(left)
                used.add(right)
                accepted_per_chunk[chunk_start] = (
                    accepted_per_chunk.get(chunk_start, 0) + 1
                )

    total_time = time.perf_counter() - started_total
    return keep_mask, {
        "num_regions": len(merge_pairs),
        "num_prototypes": len(merge_pairs),
        "num_residuals": 0,
        "num_boundary": 0,
        "num_isolated": 0,
        "num_isolated_total": 0,
        "isolated_score_source": ["atome_pair_merge"],
        "used_encoder_attention": False,
        "region_build_time_sec": total_time,
        "isolated_scoring_time_sec": 0.0,
        "similarity_backend": "atome_pair_merge",
        "merge_pairs": merge_pairs,
        "num_merged_pairs": len(merge_pairs),
        "pair_merge_threshold": threshold,
        "min_keep_ratio": min_keep_ratio,
        "avg_pair_similarity": similarity_sum / max(similarity_count, 1),
    }


def apply_sequence_mask_to_position_ids(position_ids: torch.Tensor | None, global_keep_mask: torch.Tensor) -> torch.Tensor | None:
    """Apply a sequence mask to common position_id layouts."""

    if position_ids is None:
        return None
    keep = global_keep_mask.to(device=position_ids.device)
    if position_ids.dim() == 1:
        return position_ids[keep]
    if position_ids.dim() == 2:
        return position_ids[:, keep]
    if position_ids.dim() == 3:
        if position_ids.shape[0] == 3:
            return position_ids[:, :, keep]
        if position_ids.shape[1] == 3:
            return position_ids[:, :, keep]
    raise ValueError(f"Unsupported position_ids shape for STC: {tuple(position_ids.shape)}")


def apply_stc(
    inputs_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor | None,
    video_token_id: int,
    audio_token_id: int,
    config: dict[str, Any],
    video_grid_thw: Any = None,
    video_merge_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor, dict[str, Any]]:
    """Apply Structure-Aware Token Consolidation (STC)."""

    if inputs_embeds.shape[0] != 1 or input_ids.shape[0] != 1:
        raise NotImplementedError("STC currently supports batch size 1 only")

    seq_len = int(input_ids.shape[1])
    video_positions = (input_ids[0] == int(video_token_id)).nonzero(as_tuple=False).flatten()
    audio_positions = (input_ids[0] == int(audio_token_id)).nonzero(as_tuple=False).flatten()
    video_features = inputs_embeds[0, video_positions, :]
    audio_features = inputs_embeds[0, audio_positions, :]

    stage_cfg = config.get("stc", {})
    profile = bool(stage_cfg.get("profile", False))
    timings: dict[str, float] = {}

    def mark_time(name: str, started_at: float) -> float:
        if profile and inputs_embeds.is_cuda:
            torch.cuda.synchronize(inputs_embeds.device)
        now = time.perf_counter()
        if profile:
            timings[name] = now - started_at
        return now

    if profile and inputs_embeds.is_cuda:
        torch.cuda.synchronize(inputs_embeds.device)
    section_start = total_start = time.perf_counter()
    if stage_cfg.get("video", {}).get("enabled", True):
        video_keep_mask, video_meta = consolidate_video_tokens(video_features, video_grid_thw, config, video_merge_size=video_merge_size)
    else:
        video_keep_mask = torch.ones(video_positions.numel(), dtype=torch.bool, device=input_ids.device)
        video_meta = {"num_regions": 0, "num_prototypes": 0, "num_residuals": 0, "num_isolated": 0}
    section_start = mark_time("video_stc_sec", section_start)

    if stage_cfg.get("audio", {}).get("enabled", True):
        audio_keep_mask, audio_meta = consolidate_audio_tokens(audio_features, config)
    else:
        audio_keep_mask = torch.ones(audio_positions.numel(), dtype=torch.bool, device=input_ids.device)
        audio_meta = {"num_regions": 0, "num_prototypes": 0, "num_residuals": 0, "num_boundary": 0, "num_isolated": 0}
    section_start = mark_time("audio_stc_sec", section_start)

    global_keep_mask = torch.ones(seq_len, dtype=torch.bool, device=input_ids.device)
    global_keep_mask[video_positions] = video_keep_mask.to(input_ids.device)
    global_keep_mask[audio_positions] = audio_keep_mask.to(input_ids.device)

    compact_source_embeds = inputs_embeds
    audio_merge_pairs = audio_meta.get("merge_pairs", [])
    if audio_merge_pairs:
        compact_source_embeds = inputs_embeds.clone()
        pair_tensor = torch.as_tensor(
            audio_merge_pairs,
            dtype=torch.long,
            device=audio_positions.device,
        )
        left_positions = audio_positions.index_select(0, pair_tensor[:, 0])
        right_positions = audio_positions.index_select(0, pair_tensor[:, 1])
        merged = 0.5 * (
            inputs_embeds[0].index_select(0, left_positions)
            + inputs_embeds[0].index_select(0, right_positions)
        )
        compact_source_embeds[0].index_copy_(0, left_positions, merged)

    new_inputs_embeds = compact_source_embeds[:, global_keep_mask, :]
    new_input_ids = input_ids[:, global_keep_mask]
    new_attention_mask = attention_mask[:, global_keep_mask] if attention_mask is not None else None
    new_position_ids = apply_sequence_mask_to_position_ids(position_ids, global_keep_mask)
    mark_time("sequence_compaction_sec", section_start)
    if profile:
        timings["stc_total_sec"] = time.perf_counter() - total_start
        timings["stc_region_time_sec"] = float(video_meta.get("region_build_time_sec", 0.0)) + float(audio_meta.get("region_build_time_sec", 0.0))
        timings["stc_isolated_scoring_time_sec"] = float(video_meta.get("isolated_scoring_time_sec", 0.0)) + float(audio_meta.get("isolated_scoring_time_sec", 0.0))
        timings["stc_apply_mask_time_sec"] = float(timings.get("sequence_compaction_sec", 0.0))

    keep_counts = torch.stack(
        (
            global_keep_mask.sum(),
            video_keep_mask.sum(),
            audio_keep_mask.sum(),
        )
    ).detach().cpu().tolist()
    metadata = {
        "summary": {
            "original_seq_len": seq_len,
            "new_seq_len": int(keep_counts[0]),
            "video_original_tokens": int(video_positions.numel()),
            "video_kept_tokens": int(keep_counts[1]),
            "audio_original_tokens": int(audio_positions.numel()),
            "audio_kept_tokens": int(keep_counts[2]),
            "num_video_regions": int(video_meta.get("num_regions", 0)),
            "num_audio_regions": int(audio_meta.get("num_regions", 0)),
            "num_prototype_anchors": int(video_meta.get("num_prototypes", 0) + audio_meta.get("num_prototypes", 0)),
            "num_residual_anchors": int(video_meta.get("num_residuals", 0) + audio_meta.get("num_residuals", 0)),
            "num_boundary_anchors": int(audio_meta.get("num_boundary", 0)),
            "num_isolated_tokens": int(video_meta.get("num_isolated", 0) + audio_meta.get("num_isolated", 0)),
            "video_num_isolated": int(video_meta.get("num_isolated_total", video_meta.get("num_isolated", 0))),
            "audio_num_isolated": int(audio_meta.get("num_isolated_total", audio_meta.get("num_isolated", 0))),
            "video_isolated_kept": int(video_meta.get("num_isolated", 0)),
            "audio_isolated_kept": int(audio_meta.get("num_isolated", 0)),
            "used_video_encoder_attention": bool(video_meta.get("used_encoder_attention", False)),
            "used_audio_encoder_attention": bool(audio_meta.get("used_encoder_attention", False)),
            "video_isolated_score_source": video_meta.get("isolated_score_source", []),
            "audio_isolated_score_source": audio_meta.get("isolated_score_source", []),
            "audio_num_merged_pairs": int(audio_meta.get("num_merged_pairs", 0)),
            "audio_pair_merge_threshold": audio_meta.get("pair_merge_threshold"),
            "audio_avg_pair_similarity": audio_meta.get("avg_pair_similarity"),
        },
    }
    if profile:
        metadata["summary"].update(timings)

    return new_inputs_embeds, new_input_ids, new_attention_mask, new_position_ids, global_keep_mask, metadata


def apply_stc_identical_batch(
    inputs_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor | None,
    video_token_id: int,
    audio_token_id: int,
    config: dict[str, Any],
    video_grid_thw: Any = None,
    video_merge_size: int = 1,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor,
    dict[str, Any],
]:
    """Apply one STC decision to a verified batch of identical requests."""
    batch_size = int(inputs_embeds.shape[0])
    if batch_size < 1 or input_ids.shape[0] != batch_size:
        raise ValueError("STC identical batch requires matching non-empty embeddings/IDs")
    if batch_size == 1:
        return apply_stc(
            inputs_embeds,
            input_ids,
            attention_mask,
            position_ids,
            video_token_id,
            audio_token_id,
            config,
            video_grid_thw,
            video_merge_size,
        )
    if not torch.equal(input_ids, input_ids[:1].expand_as(input_ids)):
        raise ValueError("STC identical-batch mode received different input IDs")
    first_attention = attention_mask[:1] if attention_mask is not None else None
    if position_ids is None:
        first_positions = None
    elif position_ids.ndim == 3 and position_ids.shape[0] == 3:
        first_positions = position_ids[:, :1]
    else:
        first_positions = position_ids[:1]
    (
        _first_embeds,
        _first_ids,
        _first_attention_out,
        _first_positions_out,
        keep_mask,
        metadata,
    ) = apply_stc(
        inputs_embeds[:1],
        input_ids[:1],
        first_attention,
        first_positions,
        video_token_id,
        audio_token_id,
        config,
        video_grid_thw,
        video_merge_size,
    )
    # Reuse the STC decision while retaining each request's own embeddings.
    compact_source_embeds = inputs_embeds
    audio_positions = (input_ids[0] == int(audio_token_id)).nonzero(
        as_tuple=False
    ).flatten()
    dropped_audio = (~keep_mask.index_select(0, audio_positions)).nonzero(
        as_tuple=False
    ).flatten()
    if dropped_audio.numel():
        left_positions = audio_positions.index_select(0, dropped_audio - 1)
        right_positions = audio_positions.index_select(0, dropped_audio)
        compact_source_embeds = inputs_embeds.clone()
        compact_source_embeds[:, left_positions, :] = 0.5 * (
            inputs_embeds[:, left_positions, :]
            + inputs_embeds[:, right_positions, :]
        )
    new_embeds = compact_source_embeds[:, keep_mask, :].contiguous()
    new_ids = input_ids[:, keep_mask].contiguous()
    new_attention = (
        None if attention_mask is None else attention_mask[:, keep_mask].contiguous()
    )
    new_positions = apply_sequence_mask_to_position_ids(position_ids, keep_mask)
    metadata["summary"]["batch_size"] = batch_size
    return (
        new_embeds,
        new_ids,
        new_attention,
        new_positions,
        keep_mask,
        metadata,
    )
