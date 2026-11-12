from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .four_quadrant import PageClass
from .modality_metadata import AUDIO, VIDEO


@dataclass(frozen=True)
class PrecisionTierTarget:
    """Token-weighted V payload ceiling for one scoring layer."""

    unit_index: int
    depth_fraction: float
    v_payload_bit_ceiling: float


@dataclass(frozen=True)
class PrecisionTierAssignment:
    page_classes: torch.Tensor
    fp16_threshold: float
    int4_threshold: float
    fp16_tokens: int
    int4_tokens: int
    int2_tokens: int
    average_v_bits: float
    predicted_distortion_before: float | None = None
    predicted_distortion_after: float | None = None
    selected_positive_gain: float | None = None
    allocation_rounds: int = 0
    predicted_transition_distortion_before: float | None = None
    predicted_transition_distortion_after: float | None = None


def _normalized_sigmoid(x: float, steepness: float) -> float:
    sigmoid = lambda value: 1.0 / (1.0 + math.exp(-steepness * (value - 0.5)))
    low, high = sigmoid(0.0), sigmoid(1.0)
    return (sigmoid(x) - low) / (high - low)


def build_sigmoid_precision_targets(
    num_units: int,
    *,
    shallow_v_payload_bit_ceiling: float = 4.0,
    deep_v_payload_bit_ceiling: float = 3.0,
    steepness: float = 8.0,
) -> list[PrecisionTierTarget]:
    """Build a depth-progressive precision schedule with no token eviction."""
    if num_units <= 0:
        raise ValueError("num_units must be positive")
    if not 2.0 <= deep_v_payload_bit_ceiling <= shallow_v_payload_bit_ceiling <= 16.0:
        raise ValueError("V payload bit ceilings must decrease within [2, 16]")
    if not math.isfinite(steepness) or steepness <= 0.0:
        raise ValueError("steepness must be finite and positive")

    targets: list[PrecisionTierTarget] = []
    for index in range(num_units):
        depth = 1.0 if num_units == 1 else index / (num_units - 1)
        aggressiveness = _normalized_sigmoid(depth, steepness)
        v_bit_ceiling = shallow_v_payload_bit_ceiling + (
            deep_v_payload_bit_ceiling - shallow_v_payload_bit_ceiling
        ) * aggressiveness
        targets.append(PrecisionTierTarget(index, depth, v_bit_ceiling))
    return targets


def _production_v_bytes_per_token(
    kv_heads: int,
    head_dim: int,
    int2_group_size: int,
) -> tuple[int, int, int]:
    if kv_heads <= 0 or head_dim <= 0 or head_dim % int2_group_size:
        raise ValueError("invalid KV shape or INT2 group size")
    groups = head_dim // int2_group_size
    # INT2: uint2 payload + FP16 scale + uint8 zero for each token/head/group.
    # INT4: uint4 payload + one FP32 symmetric scale for each token/head.
    bytes_i2 = kv_heads * (head_dim // 4 + groups * (2 + 1))
    bytes_i4 = kv_heads * (head_dim // 2 + 4)
    bytes_fp = kv_heads * head_dim * 2
    if not bytes_i2 < bytes_i4 < bytes_fp:
        raise ValueError("production V tier byte costs must be strictly ordered")
    return bytes_i2, bytes_i4, bytes_fp


def _query_weights(
    query_count: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    reduction: str,
    ema_decay: float,
) -> torch.Tensor | None:
    if reduction in {"max", "cvar"}:
        return None
    if reduction == "mean":
        return torch.full(
            (query_count,), 1.0 / query_count, device=device, dtype=dtype
        )
    if reduction != "ema":
        raise ValueError("query_reduction must be mean, max, ema, or cvar")
    if not 0.0 < ema_decay <= 1.0:
        raise ValueError("query_ema_decay must lie in (0, 1]")
    age = torch.arange(
        query_count - 1, -1, -1, device=device, dtype=dtype
    )
    weights = float(ema_decay) ** age
    return weights / weights.sum()


def _reduce_query_distortion(
    values: torch.Tensor,
    weights: torch.Tensor | None,
    *,
    reduction: str,
    cvar_fraction: float,
) -> torch.Tensor:
    """Reduce per-query distortion, preserving any trailing candidate axis."""
    if reduction == "max":
        return values.amax(dim=0)
    if reduction == "cvar":
        if not 0.0 < cvar_fraction <= 1.0:
            raise ValueError("query_cvar_fraction must lie in (0, 1]")
        tail = max(1, math.ceil(int(values.shape[0]) * cvar_fraction))
        return values.topk(tail, dim=0, largest=True, sorted=False).values.mean(dim=0)
    if weights is None:
        raise RuntimeError("weighted query reduction requires explicit weights")
    shape = (int(weights.numel()),) + (1,) * (values.ndim - 1)
    return (values * weights.view(shape)).sum(dim=0)


def _per_query_stability_distortion(
    value: torch.Tensor,
    *,
    transition_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return absolute and transition-aware readout distortion per query.

    Autoregressive generation depends not only on the response to one prompt
    query, but also on how that response changes as the query state evolves.
    The transition term measures the error in consecutive-anchor readout
    differences, ``||(R_q-R_{q-1})||^2``.  Its ``Q/(Q-1)`` factor compensates
    for the first anchor having no predecessor, so ``transition_weight`` has a
    stable meaning as the anchor count changes.

    A one-anchor calibration has no observable transition and intentionally
    falls back to the established absolute residual objective.
    """
    if not math.isfinite(transition_weight) or not 0.0 <= transition_weight <= 1.0:
        raise ValueError("transition_weight must lie in [0, 1]")
    absolute = value.square().mean(dim=(0, 1, 3))
    query_count = int(value.shape[2])
    transition = torch.zeros_like(absolute)
    if transition_weight > 0.0 and query_count > 1:
        delta = value[:, :, 1:, :] - value[:, :, :-1, :]
        transition[1:] = delta.square().mean(dim=(0, 1, 3)) * (
            query_count / (query_count - 1)
        )
        combined = (
            (1.0 - transition_weight) * absolute
            + transition_weight * transition
        )
    else:
        combined = absolute
    return combined, transition


def evaluate_residual_page_assignment_components(
    base_residual: torch.Tensor,
    int2_to_int4_correction: torch.Tensor,
    int4_to_fp16_correction: torch.Tensor,
    page_classes: torch.Tensor,
    *,
    query_reduction: str = "ema",
    query_ema_decay: float = 0.8,
    query_cvar_fraction: float = 0.25,
    transition_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate a tier assignment under the K order that produced its actions.

    Returns total distortion before/after followed by transition distortion
    before/after.  This is intentionally separate from proposal generation: V
    tiers change physical K superblocks, so a proposal must be re-simulated
    under *its own* order before it can be compared fairly with another.
    """
    residual = base_residual.float()
    correction24 = int2_to_int4_correction.float()
    correction4f = int4_to_fp16_correction.float()
    if residual.ndim != 4 or correction24.ndim != 5:
        raise ValueError("residual/actions must be [B,H,Q,D]/[B,H,Q,P,D]")
    if correction4f.shape != correction24.shape or (
        correction24.shape[:3] != residual.shape[:3]
        or correction24.shape[-1] != residual.shape[-1]
    ):
        raise ValueError("residual and correction shapes are incompatible")
    classes = page_classes.flatten().to(device=residual.device, dtype=torch.int8)
    if int(classes.numel()) != int(correction24.shape[3]):
        raise ValueError("page_classes must match the correction page axis")
    allowed = (
        (classes == int(PageClass.INT2_PAGE))
        | (classes == int(PageClass.INT4_PAGE))
        | (classes == int(PageClass.FP_PAGE))
    )
    if not bool(allowed.all()):
        raise ValueError("residual assignment supports only INT2/INT4/FP pages")
    upgraded = (classes != int(PageClass.INT2_PAGE)).nonzero(
        as_tuple=False
    ).flatten()
    fp = (classes == int(PageClass.FP_PAGE)).nonzero(as_tuple=False).flatten()
    candidate_residual = residual.clone()
    if upgraded.numel():
        candidate_residual.sub_(
            correction24.index_select(3, upgraded).sum(dim=3)
        )
    if fp.numel():
        candidate_residual.sub_(correction4f.index_select(3, fp).sum(dim=3))
    weights = _query_weights(
        int(residual.shape[2]),
        device=residual.device,
        dtype=residual.dtype,
        reduction=query_reduction,
        ema_decay=query_ema_decay,
    )

    def distortion(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        per_query, transition = _per_query_stability_distortion(
            value, transition_weight=transition_weight
        )
        return (
            _reduce_query_distortion(
                per_query,
                weights,
                reduction=query_reduction,
                cvar_fraction=query_cvar_fraction,
            ),
            _reduce_query_distortion(
                transition,
                weights,
                reduction=query_reduction,
                cvar_fraction=query_cvar_fraction,
            ),
        )

    before, transition_before = distortion(residual)
    after, transition_after = distortion(candidate_residual)
    return before, after, transition_before, transition_after


def assign_residual_pages_to_precision(
    base_residual: torch.Tensor,
    int2_to_int4_correction: torch.Tensor,
    int4_to_fp16_correction: torch.Tensor,
    page_lengths: torch.Tensor,
    target: PrecisionTierTarget,
    *,
    kv_heads: int,
    head_dim: int,
    int2_group_size: int = 32,
    allocation_rounds: int = 4,
    query_reduction: str = "ema",
    query_ema_decay: float = 0.8,
    query_cvar_fraction: float = 0.25,
    transition_weight: float = 0.0,
    page_modalities: torch.Tensor | None = None,
    modality_budget_weights: tuple[float, float] | None = None,
    action_priority_scores: torch.Tensor | None = None,
    action_priority_weight: float = 0.0,
) -> PrecisionTierAssignment:
    """Stagewise V allocation that directly reduces the current model residual.

    Each action's exact local improvement is
    ``2 <residual, correction> - ||correction||^2``.  After a bounded batch of
    upgrades, the residual is updated and every remaining action is rescored.
    Consequently, two individually attractive pages are not both upgraded when
    their combined correction would overshoot the FP target.  Negative-gain
    actions are never selected, so the nominal bit target is a ceiling rather
    than a requirement to spend harmful bytes.

    ``action_priority_scores`` contains modality-normalized output sensitivity
    for INT2->INT4 and INT4->FP16 actions. It only reorders actions that are
    already eligible under the exact residual gain; it does not alter the
    predicted distortion or make a non-positive action eligible.

    ``modality_budget_weights`` optionally partitions the *same* global byte
    ceiling into token-weighted video/audio budgets. This changes where scarce
    high-precision actions are spent without increasing storage. Modalities
    outside audio/video retain unit weight, so short instruction boundaries do
    not disappear from the allocation.
    """
    residual = base_residual.float()
    correction24 = int2_to_int4_correction.float()
    correction4f = int4_to_fp16_correction.float()
    if residual.ndim != 4:
        raise ValueError("base_residual must be [B,Hq,Q,D]")
    if correction24.ndim != 5 or correction4f.shape != correction24.shape:
        raise ValueError("upgrade corrections must share [B,Hq,Q,P,D]")
    if (
        correction24.shape[:3] != residual.shape[:3]
        or correction24.shape[-1] != residual.shape[-1]
    ):
        raise ValueError("residual and correction shapes are incompatible")
    if int(residual.shape[-1]) != head_dim:
        raise ValueError("residual head dimension does not match head_dim")
    page_count = int(correction24.shape[3])
    lengths = page_lengths.flatten().to(
        device=residual.device, dtype=torch.long
    )
    if int(lengths.numel()) != page_count or page_count == 0:
        raise ValueError("page lengths must match the correction page axis")
    if bool((lengths <= 0).any()):
        raise ValueError("page lengths must be positive")
    if not bool(torch.isfinite(residual).all()) or not bool(
        torch.isfinite(correction24).all() & torch.isfinite(correction4f).all()
    ):
        raise ValueError("residual calibration tensors must be finite")
    if allocation_rounds < 1 or allocation_rounds > 32:
        raise ValueError("allocation_rounds must lie in [1, 32]")
    if not math.isfinite(action_priority_weight) or not 0.0 <= action_priority_weight <= 1.0:
        raise ValueError("action_priority_weight must lie in [0, 1]")
    if not math.isfinite(transition_weight) or not 0.0 <= transition_weight <= 1.0:
        raise ValueError("transition_weight must lie in [0, 1]")

    resolved_page_modalities = None
    if page_modalities is not None:
        resolved_page_modalities = page_modalities.flatten().to(
            device=residual.device, dtype=torch.long
        )
        if int(resolved_page_modalities.numel()) != page_count:
            raise ValueError("page_modalities must have one value per page")
    if modality_budget_weights is not None:
        if resolved_page_modalities is None:
            raise ValueError("modality budgets require page_modalities")
        if len(modality_budget_weights) != 2 or any(
            not math.isfinite(float(weight)) or float(weight) < 0.0
            for weight in modality_budget_weights
        ):
            raise ValueError(
                "modality budget weights must contain finite non-negative video/audio values"
            )
        if sum(float(weight) for weight in modality_budget_weights) <= 0.0:
            raise ValueError("at least one modality budget weight must be positive")

    priority_scores = None
    if action_priority_scores is not None:
        priority_scores = action_priority_scores.to(
            device=residual.device, dtype=torch.float32
        )
        if tuple(priority_scores.shape) != (2, page_count):
            raise ValueError("action_priority_scores must have shape [2, pages]")
        if not bool(torch.isfinite(priority_scores).all()) or bool(
            ((priority_scores < 0.0) | (priority_scores > 1.0)).any()
        ):
            raise ValueError("action priority scores must be finite in [0, 1]")
    elif action_priority_weight > 0.0:
        raise ValueError("positive action_priority_weight requires priority scores")

    bytes_i2, bytes_i4, bytes_fp = _production_v_bytes_per_token(
        kv_heads, head_dim, int2_group_size
    )
    total_tokens = int(lengths.sum())
    base_bytes = total_tokens * bytes_i2
    full_bytes = total_tokens * bytes_fp
    byte_fraction = (target.v_payload_bit_ceiling - 2.0) / 14.0
    budget = int(round(base_bytes + byte_fraction * (full_bytes - base_bytes)))
    remaining_bytes = max(0, budget - base_bytes)
    remaining_bit_tokens = max(
        0,
        int(round(0.5 * (target.v_payload_bit_ceiling - 2.0) * total_tokens)),
    )
    initial_upgrade_bytes = remaining_bytes
    initial_bit_tokens = remaining_bit_tokens
    length_values = [int(value) for value in lengths.tolist()]

    # Optional strict per-modality ceilings. Integer remainders are assigned by
    # largest fractional part so the sub-budgets sum to the unchanged global
    # ceiling. Unused modality budget is intentionally not borrowed: this is
    # what makes audio/video precision independently controllable.
    modality_values_for_budget = (
        [int(value) for value in resolved_page_modalities.tolist()]
        if resolved_page_modalities is not None
        else [0] * page_count
    )
    remaining_bytes_by_modality: dict[int, int] | None = None
    remaining_bits_by_modality: dict[int, int] | None = None
    if modality_budget_weights is not None:
        video_weight, audio_weight = map(float, modality_budget_weights)
        token_mass: dict[int, float] = {}
        for page, modality in enumerate(modality_values_for_budget):
            weight = (
                video_weight
                if modality == VIDEO
                else audio_weight
                if modality == AUDIO
                else 1.0
            )
            token_mass[modality] = token_mass.get(modality, 0.0) + (
                length_values[page] * weight
            )
        total_mass = sum(token_mass.values())
        if total_mass <= 0.0:
            raise ValueError("modality budget has zero weighted token mass")

        def split_budget(total: int) -> dict[int, int]:
            exact = {
                modality: total * mass / total_mass
                for modality, mass in token_mass.items()
            }
            result = {
                modality: int(math.floor(value))
                for modality, value in exact.items()
            }
            remainder = total - sum(result.values())
            order = sorted(
                exact,
                key=lambda modality: (
                    -(exact[modality] - result[modality]), modality
                ),
            )
            for modality in order[:remainder]:
                result[modality] += 1
            return result

        remaining_bytes_by_modality = split_budget(remaining_bytes)
        remaining_bits_by_modality = split_budget(remaining_bit_tokens)
    class_values = [int(PageClass.INT2_PAGE)] * page_count
    query_weights = _query_weights(
        int(residual.shape[2]),
        device=residual.device,
        dtype=residual.dtype,
        reduction=query_reduction,
        ema_decay=query_ema_decay,
    )

    def distortion(value: torch.Tensor) -> torch.Tensor:
        per_query, _ = _per_query_stability_distortion(
            value, transition_weight=transition_weight
        )
        return _reduce_query_distortion(
            per_query,
            query_weights,
            reduction=query_reduction,
            cvar_fraction=query_cvar_fraction,
        )

    normalization = float(
        int(residual.shape[0]) * int(residual.shape[1]) * int(residual.shape[3])
    )
    correction24_norm = torch.einsum(
        "bhqpd,bhqpd->qp", correction24, correction24
    ) / normalization
    correction4f_norm = torch.einsum(
        "bhqpd,bhqpd->qp", correction4f, correction4f
    ) / normalization
    query_count = int(residual.shape[2])
    transition24_norm = None
    transition4f_norm = None
    if transition_weight > 0.0 and query_count > 1:
        # Correction directions are fixed throughout the stagewise search.
        # Cache their transition norms once; only the residual cross term
        # changes after an allocation round.
        correction_delta = correction24[:, :, 1:, :, :] - correction24[:, :, :-1, :, :]
        transition24_norm = torch.einsum(
            "bhqpd,bhqpd->qp", correction_delta, correction_delta
        ) / normalization
        del correction_delta
        correction_delta = correction4f[:, :, 1:, :, :] - correction4f[:, :, :-1, :, :]
        transition4f_norm = torch.einsum(
            "bhqpd,bhqpd->qp", correction_delta, correction_delta
        ) / normalization
        del correction_delta

    def marginal_gain(
        correction: torch.Tensor,
        correction_norm: torch.Tensor,
        transition_norm: torch.Tensor | None,
        residual_delta: torch.Tensor | None,
    ) -> torch.Tensor:
        current_per_query, _ = _per_query_stability_distortion(
            residual, transition_weight=transition_weight
        )
        cross = torch.einsum(
            "bhqd,bhqpd->qp", residual, correction
        ) / normalization
        per_query_page = 2.0 * cross - correction_norm
        if transition_weight > 0.0 and query_count > 1:
            assert residual_delta is not None and transition_norm is not None
            correction_delta = correction[:, :, 1:, :, :] - correction[:, :, :-1, :, :]
            transition_cross = torch.einsum(
                "bhqd,bhqpd->qp", residual_delta, correction_delta
            ) / normalization
            transition_gain = torch.zeros_like(per_query_page)
            transition_gain[1:] = (
                2.0 * transition_cross - transition_norm
            ) * (query_count / (query_count - 1))
            per_query_page = (
                (1.0 - transition_weight) * per_query_page
                + transition_weight * transition_gain
            )
        if query_reduction in {"mean", "ema"}:
            # Preserve the original direct weighted-gain arithmetic for the
            # established objectives. Computing objective_before-objective_after
            # is algebraically equal but suffers cancellation and can perturb
            # near-tied page rankings.
            assert query_weights is not None
            return (per_query_page * query_weights[:, None]).sum(dim=0)
        current_objective = _reduce_query_distortion(
            current_per_query,
            query_weights,
            reduction=query_reduction,
            cvar_fraction=query_cvar_fraction,
        )
        candidate_per_query = current_per_query[:, None] - per_query_page
        candidate_objective = _reduce_query_distortion(
            candidate_per_query,
            query_weights,
            reduction=query_reduction,
            cvar_fraction=query_cvar_fraction,
        )
        return current_objective - candidate_objective

    before = distortion(residual)
    selected_gain = 0.0
    selected_rounds = 0
    selected_fp_gains: list[float] = []
    selected_i4_gains: list[float] = []
    priority_score_values = (
        priority_scores.tolist()
        if priority_scores is not None
        else [[0.5] * page_count, [0.5] * page_count]
    )

    import heapq

    for round_index in range(allocation_rounds):
        if remaining_bytes <= 0 or remaining_bit_tokens <= 0:
            break
        # Launch both gain contractions before the single device-to-host
        # transfer for this round.  This removes the second Python-visible CUDA
        # synchronization from the allocator hot path.
        residual_delta = (
            residual[:, :, 1:, :] - residual[:, :, :-1, :]
            if transition_weight > 0.0 and query_count > 1
            else None
        )
        gain_values = torch.stack(
            (
                marginal_gain(
                    correction24,
                    correction24_norm,
                    transition24_norm,
                    residual_delta,
                ),
                marginal_gain(
                    correction4f,
                    correction4f_norm,
                    transition4f_norm,
                    residual_delta,
                ),
            ),
            dim=0,
        ).tolist()
        gain24_values = [float(value) for value in gain_values[0]]
        gain4f_values = [float(value) for value in gain_values[1]]
        rounds_left = allocation_rounds - round_index
        round_byte_budget = (
            remaining_bytes
            if rounds_left == 1
            else math.ceil(remaining_bytes / rounds_left)
        )
        round_bit_budget = (
            remaining_bit_tokens
            if rounds_left == 1
            else math.ceil(remaining_bit_tokens / rounds_left)
        )

        # Heap entry: (-priority, sequence, page, destination, base ratio,
        # gain, byte cost, bit-token cost).
        heap: list[tuple[float, int, int, int, float, float, int, int]] = []
        sequence = 0
        raw_actions: list[tuple[int, int, float, float, int, int]] = []
        for page in range(page_count):
            current = class_values[page]
            if current == int(PageClass.INT2_PAGE):
                gain = gain24_values[page]
                byte_cost = length_values[page] * (bytes_i4 - bytes_i2)
                bit_cost = length_values[page]
                destination = int(PageClass.INT4_PAGE)
            elif current == int(PageClass.INT4_PAGE):
                gain = gain4f_values[page]
                byte_cost = length_values[page] * (bytes_fp - bytes_i4)
                bit_cost = length_values[page] * 6
                destination = int(PageClass.FP_PAGE)
            else:
                continue
            if gain <= 0.0:
                continue
            ratio = gain / max(byte_cost, 1)
            raw_actions.append((page, destination, ratio, gain, byte_cost, bit_cost))
        if not raw_actions:
            break

        def risk_adjusted_ratio(page: int, destination: int, ratio: float) -> float:
            tier = 0 if destination == int(PageClass.INT4_PAGE) else 1
            score = float(priority_score_values[tier][page])
            return ratio * (1.0 + action_priority_weight * (2.0 * score - 1.0))

        for page, destination, ratio, gain, byte_cost, bit_cost in raw_actions:
            priority = risk_adjusted_ratio(page, destination, ratio)
            heapq.heappush(
                heap,
                (-priority, sequence, page, destination, ratio, gain, byte_cost, bit_cost),
            )
            sequence += 1

        selected_i4_pages: list[int] = []
        selected_fp_pages: list[int] = []
        used_bytes = 0
        used_bits = 0
        while heap:
            (
                _,
                _,
                page,
                destination,
                ratio,
                gain,
                byte_cost,
                bit_cost,
            ) = heapq.heappop(heap)
            expected = (
                int(PageClass.INT2_PAGE)
                if destination == int(PageClass.INT4_PAGE)
                else int(PageClass.INT4_PAGE)
            )
            if class_values[page] != expected:
                continue
            if (
                byte_cost > round_byte_budget - used_bytes
                or bit_cost > round_bit_budget - used_bits
                or byte_cost > remaining_bytes
                or bit_cost > remaining_bit_tokens
            ):
                continue
            if remaining_bytes_by_modality is not None:
                modality = modality_values_for_budget[page]
                assert remaining_bits_by_modality is not None
                if (
                    byte_cost > remaining_bytes_by_modality.get(modality, 0)
                    or bit_cost > remaining_bits_by_modality.get(modality, 0)
                ):
                    continue
            class_values[page] = destination
            used_bytes += byte_cost
            used_bits += bit_cost
            remaining_bytes -= byte_cost
            remaining_bit_tokens -= bit_cost
            if remaining_bytes_by_modality is not None:
                modality = modality_values_for_budget[page]
                assert remaining_bits_by_modality is not None
                remaining_bytes_by_modality[modality] -= byte_cost
                remaining_bits_by_modality[modality] -= bit_cost
            selected_gain += gain
            if destination == int(PageClass.INT4_PAGE):
                selected_i4_pages.append(page)
                selected_i4_gains.append(gain)
            else:
                selected_fp_pages.append(page)
                selected_fp_gains.append(gain)

        if not selected_i4_pages and not selected_fp_pages:
            # An indivisible page may not fit an early round's proportional
            # slice. Later slices grow as ``rounds_left`` decreases.
            continue
        correction_sum = torch.zeros_like(residual)
        if selected_i4_pages:
            indices = torch.tensor(
                selected_i4_pages, device=residual.device, dtype=torch.long
            )
            correction_sum.add_(correction24.index_select(3, indices).sum(dim=3))
        if selected_fp_pages:
            indices = torch.tensor(
                selected_fp_pages, device=residual.device, dtype=torch.long
            )
            correction_sum.add_(correction4f.index_select(3, indices).sum(dim=3))
        residual = residual - correction_sum
        selected_rounds += 1

    classes = torch.tensor(
        class_values, device=residual.device, dtype=torch.int8
    )
    fp_pages = (classes == int(PageClass.FP_PAGE)).nonzero(as_tuple=False).flatten()
    int4_pages = (classes == int(PageClass.INT4_PAGE)).nonzero(as_tuple=False).flatten()
    fp_tokens = int(lengths.index_select(0, fp_pages).sum()) if fp_pages.numel() else 0
    int4_tokens = (
        int(lengths.index_select(0, int4_pages).sum()) if int4_pages.numel() else 0
    )
    int2_tokens = total_tokens - fp_tokens - int4_tokens
    average_v_bits = (
        16.0 * fp_tokens + 4.0 * int4_tokens + 2.0 * int2_tokens
    ) / total_tokens
    # Both ceilings must hold even when physical metadata makes one tighter.
    assert initial_upgrade_bytes - remaining_bytes <= max(0, budget - base_bytes)
    assert initial_bit_tokens - remaining_bit_tokens <= initial_bit_tokens
    _, transition_before_per_query = _per_query_stability_distortion(
        base_residual.float(), transition_weight=transition_weight
    )
    _, transition_after_per_query = _per_query_stability_distortion(
        residual, transition_weight=transition_weight
    )
    transition_before = _reduce_query_distortion(
        transition_before_per_query,
        query_weights,
        reduction=query_reduction,
        cvar_fraction=query_cvar_fraction,
    )
    transition_after = _reduce_query_distortion(
        transition_after_per_query,
        query_weights,
        reduction=query_reduction,
        cvar_fraction=query_cvar_fraction,
    )
    return PrecisionTierAssignment(
        page_classes=classes,
        fp16_threshold=min(selected_fp_gains) if selected_fp_gains else math.inf,
        int4_threshold=min(selected_i4_gains) if selected_i4_gains else math.inf,
        fp16_tokens=fp_tokens,
        int4_tokens=int4_tokens,
        int2_tokens=int2_tokens,
        average_v_bits=average_v_bits,
        predicted_distortion_before=float(before),
        predicted_distortion_after=float(distortion(residual)),
        selected_positive_gain=float(selected_gain),
        allocation_rounds=selected_rounds,
        predicted_transition_distortion_before=float(transition_before),
        predicted_transition_distortion_after=float(transition_after),
    )
