from __future__ import annotations

from dataclasses import dataclass, field
from types import MethodType
from typing import Any

import torch

from .debug_utils import log_stc_summary
from .pre_llm_units import (
    apply_stc,
    apply_stc_identical_batch,
    load_config,
)


@dataclass
class STCState:
    config: dict[str, Any]
    input_ids: torch.Tensor | None = None
    video_grid_thw: Any = None
    video_merge_size: int = 1
    applied: bool = False
    last_metadata: dict[str, Any] = field(default_factory=dict)
    last_keep_mask: torch.Tensor | None = None
    next_semantic_position: torch.Tensor | None = None
    identical_batch: bool = False

    def prepare(
        self,
        *,
        input_ids: torch.Tensor,
        video_grid_thw: Any = None,
        video_merge_size: int = 1,
        identical_batch: bool = False,
    ) -> None:
        self.input_ids = input_ids.detach()
        self.video_grid_thw = video_grid_thw
        self.video_merge_size = max(1, int(video_merge_size))
        self.identical_batch = bool(identical_batch)
        self.applied = False
        self.last_metadata = {}
        self.last_keep_mask = None
        self.next_semantic_position = None


def _resolve_thinker(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "thinker", model)


def _compact_cache_position(
    cache_position: torch.Tensor | None,
    keep_mask: torch.Tensor,
    *,
    kept_count: int | None = None,
) -> torch.Tensor | None:
    """Return physical cache slots for the compacted sequence.

    ``position_ids`` carry semantic/mRoPE coordinates and therefore retain the
    original coordinates of surviving tokens. ``cache_position`` instead
    indexes physical cache slots and must stay contiguous after compaction.
    Preserve the incoming start so this also works when appending a compacted
    chunk to a non-empty cache.
    """
    if cache_position is None or cache_position.dim() != 1 or cache_position.numel() != keep_mask.numel():
        return cache_position
    if kept_count is None:
        kept_count = int(keep_mask.sum().item())
    if kept_count == 0:
        return cache_position.new_empty((0,))
    return cache_position[:1] + torch.arange(
        kept_count,
        dtype=cache_position.dtype,
        device=cache_position.device,
    )


def _index_cache_position(
    cache_position: torch.Tensor | None,
    keep_mask: torch.Tensor,
) -> torch.Tensor | None:
    """Backward-compatible name for physical cache compaction."""
    return _compact_cache_position(cache_position, keep_mask)


def compact_stc_attention_mask(
    attention_mask: torch.Tensor | None,
    keep_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    """Mirror STC prefill compaction in an externally owned decode mask."""
    if attention_mask is None or keep_mask is None:
        return attention_mask
    keep = keep_mask.flatten().to(device=attention_mask.device, dtype=torch.bool)
    source_length = int(keep.numel())
    compact_length = int(keep.sum().item())
    mask_length = int(attention_mask.shape[-1])
    if mask_length == compact_length:
        return attention_mask
    if mask_length != source_length:
        raise ValueError(
            "STC keep mask does not match decode attention mask: "
            f"mask={mask_length}, source={source_length}, compact={compact_length}"
        )
    return attention_mask.index_select(
        -1,
        keep.nonzero(as_tuple=False).flatten(),
    )


def _compact_extended_stc_attention_mask(
    attention_mask: torch.Tensor,
    keep_mask: torch.Tensor,
) -> torch.Tensor:
    """Compact the original prompt prefix while preserving decode suffixes."""
    keep = keep_mask.flatten().to(device=attention_mask.device, dtype=torch.bool)
    source_length = int(keep.numel())
    mask_length = int(attention_mask.shape[-1])
    if mask_length < source_length:
        raise ValueError(
            "STC decode attention mask is shorter than the original prompt: "
            f"mask={mask_length}, source={source_length}"
        )
    prompt = attention_mask[..., :source_length].index_select(
        -1,
        keep.nonzero(as_tuple=False).flatten(),
    )
    suffix = attention_mask[..., source_length:]
    return torch.cat((prompt, suffix), dim=-1)


def _past_sequence_length(past_key_values: Any) -> int | None:
    """Read the physical sequence length from common HF and legacy caches."""
    if past_key_values is None:
        return None
    get_seq_length = getattr(past_key_values, "get_seq_length", None)
    if callable(get_seq_length):
        try:
            return int(get_seq_length())
        except TypeError:
            return int(get_seq_length(0))
    if isinstance(past_key_values, (tuple, list)) and past_key_values:
        layer = past_key_values[0]
        if isinstance(layer, (tuple, list)) and layer:
            key = layer[0]
            if torch.is_tensor(key) and key.ndim >= 3:
                # Qwen legacy caches use [batch, heads, sequence, head_dim].
                return int(key.shape[-2])
    return None


def _normalize_stc_decode_kwargs(
    kwargs: dict[str, Any],
    state: STCState | Any,
) -> dict[str, Any]:
    """Map HF's original-length decode timeline onto the compact cache.

    ``GenerationMixin`` owns the prefill ``cache_position`` and attention mask,
    so a compaction performed inside ``thinker.model.forward`` is invisible to
    it.  On standard ``generate`` calls the first decode step therefore arrives
    with a prompt-length gap of ``L-L'``.  Custom runners in this repository
    already use the compact physical timeline and have no such gap.
    """
    if not bool(getattr(state, "applied", False)):
        return kwargs
    keep_mask = getattr(state, "last_keep_mask", None)
    cache_position = kwargs.get("cache_position", None)
    if not torch.is_tensor(keep_mask) or not torch.is_tensor(cache_position):
        return kwargs
    if cache_position.ndim != 1 or cache_position.numel() == 0:
        return kwargs

    source_length = int(keep_mask.numel())
    compact_length = int(keep_mask.sum().item())
    removed = source_length - compact_length
    if removed <= 0:
        return kwargs
    past_length = _past_sequence_length(kwargs.get("past_key_values", None))
    if past_length is None:
        return kwargs

    query_length = int(cache_position.numel())
    attention_mask = kwargs.get("attention_mask", None)
    uses_original_timeline: bool
    if torch.is_tensor(attention_mask):
        timeline_gap = int(attention_mask.shape[-1]) - past_length - query_length
        if timeline_gap == removed:
            uses_original_timeline = True
        elif timeline_gap == 0:
            uses_original_timeline = False
        else:
            raise ValueError(
                "STC decode mask/cache lengths do not describe either the "
                "original or compact timeline: "
                f"mask={attention_mask.shape[-1]}, past={past_length}, "
                f"query={query_length}, removed={removed}"
            )
    else:
        # Generation normally supplies a 2D attention mask.  For direct model
        # calls without one, fall back to the scalar physical offset.
        incoming_start = int(cache_position[0].detach().cpu().item())
        if incoming_start == past_length + removed:
            uses_original_timeline = True
        elif incoming_start == past_length:
            uses_original_timeline = False
        else:
            raise ValueError(
                "STC decode cache_position does not match the physical cache: "
                f"position={incoming_start}, past={past_length}, removed={removed}"
            )

    if not uses_original_timeline:
        return kwargs
    normalized = dict(kwargs)
    normalized["cache_position"] = cache_position - removed
    if torch.is_tensor(attention_mask):
        normalized["attention_mask"] = _compact_extended_stc_attention_mask(
            attention_mask,
            keep_mask,
        )
    return normalized


def stc_decode_position_ids(
    state: STCState | Any | None,
    *,
    step: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Build semantic mRoPE positions for custom decode after STC.

    The custom runners use compact physical ``cache_position=L'+step``.  Qwen's
    prefill-side ``rope_deltas`` were computed from the original prompt length,
    so reusing them with ``L'`` shifts continuation positions backwards.  The
    patch captures the correct next semantic position during prefill; passing it
    explicitly keeps semantic positions independent of compact cache slots.
    """
    if state is None or not bool(getattr(state, "applied", False)):
        return None
    next_position = getattr(state, "next_semantic_position", None)
    if not torch.is_tensor(next_position):
        return None
    values = next_position.detach().to(device=device, dtype=torch.long).reshape(-1)
    if int(values.numel()) == 1 and batch_size > 1:
        values = values.expand(batch_size)
    if int(values.numel()) != batch_size:
        raise ValueError(
            "STC semantic position batch does not match decode batch: "
            f"positions={values.numel()}, batch={batch_size}"
        )
    values = values + int(step)
    return values.view(1, batch_size, 1).expand(3, -1, -1)


def _next_semantic_position(
    thinker: torch.nn.Module,
    position_ids: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    cache_position: torch.Tensor | None,
    *,
    original_seq_len: int,
    batch_size: int,
) -> torch.Tensor:
    """Capture the first post-prefill semantic position before compaction."""
    rope_deltas = getattr(thinker, "rope_deltas", None)
    if torch.is_tensor(rope_deltas):
        values = rope_deltas.detach().reshape(-1)
        if int(values.numel()) == 1 and batch_size > 1:
            values = values.expand(batch_size)
        if int(values.numel()) == batch_size:
            if (
                torch.is_tensor(cache_position)
                and cache_position.dim() == 1
                and int(cache_position.numel()) == int(original_seq_len)
            ):
                physical_next = cache_position[-1:].detach() + 1
                physical_next = physical_next.to(device=values.device, dtype=values.dtype)
            else:
                physical_next = values.new_tensor([int(original_seq_len)])
            return (values + physical_next).reshape(batch_size, 1)

    if torch.is_tensor(position_ids):
        positions = position_ids.detach()
        if positions.dim() == 3 and int(positions.shape[0]) == 3:
            # [3, B, L] -> [B, 3, L]
            positions = positions.permute(1, 0, 2)
        elif positions.dim() == 3 and int(positions.shape[1]) == 3:
            # already [B, 3, L]
            pass
        elif positions.dim() == 2:
            positions = positions[:, None, :]
        else:
            positions = None
        if positions is not None and int(positions.shape[0]) == batch_size:
            if torch.is_tensor(attention_mask) and int(attention_mask.shape[-1]) == int(positions.shape[-1]):
                valid = attention_mask.to(device=positions.device, dtype=torch.bool)[:, None, :]
                minimum = torch.iinfo(positions.dtype).min
                positions = positions.masked_fill(~valid, minimum)
            return (positions.amax(dim=(1, 2)) + 1).reshape(batch_size, 1)

    device = position_ids.device if torch.is_tensor(position_ids) else next(thinker.parameters()).device
    return torch.full(
        (batch_size, 1),
        int(original_seq_len),
        dtype=torch.long,
        device=device,
    )


def _update_compacted_rope_deltas(
    thinker: torch.nn.Module,
    next_semantic_position: torch.Tensor,
    cache_position: torch.Tensor | None,
    *,
    original_seq_len: int,
    compact_seq_len: int,
) -> None:
    """Make the standard Hugging Face decode path continue the old timeline.

    Qwen decode derives semantic positions as ``cache_position + rope_deltas``.
    Once the physical cache shrinks from ``L`` to ``L'``, retaining the prefill
    delta would move continuation positions backwards.  Store the equivalent
    delta relative to the next compact physical slot so ``model.generate`` is
    correct even when a custom runner does not pass explicit ``position_ids``.
    """
    if compact_seq_len < 0 or compact_seq_len > original_seq_len:
        raise ValueError("invalid STC compact sequence length")
    if (
        torch.is_tensor(cache_position)
        and cache_position.dim() == 1
        and int(cache_position.numel()) == int(original_seq_len)
        and cache_position.numel()
    ):
        compact_next = cache_position[:1].detach() + int(compact_seq_len)
        compact_next = compact_next.to(
            device=next_semantic_position.device,
            dtype=next_semantic_position.dtype,
        )
    else:
        compact_next = next_semantic_position.new_tensor([int(compact_seq_len)])
    thinker.rope_deltas = (
        next_semantic_position.detach().reshape(-1, 1) - compact_next.reshape(1, 1)
    )


def _resolve_token_id(config: Any, name: str, explicit: int | None) -> int:
    if explicit is not None:
        return int(explicit)
    value = getattr(config, name, None)
    if value is None:
        raise RuntimeError(f"Could not resolve required Qwen2.5-Omni token id: config.{name}")
    return int(value)


def install_stc(
    model: torch.nn.Module,
    *,
    config: dict[str, Any] | str,
    video_token_id: int | None = None,
    audio_token_id: int | None = None,
    verbose: bool = True,
) -> STCState:
    """Install Stage I Structure-Aware Token Consolidation for Qwen2.5-Omni.

    The patch is runtime-only. Call `state.prepare(input_ids=..., video_grid_thw=...)`
    before generation when the model.forward path passes only fused `inputs_embeds`
    into the text backbone.
    """

    cfg = load_config(config) if isinstance(config, str) else config
    state = STCState(config=cfg)
    thinker = _resolve_thinker(model)
    target = getattr(thinker, "model", None)
    if target is None or not hasattr(target, "forward"):
        raise RuntimeError("Could not find Qwen2.5-Omni thinker.model.forward to patch.")

    original_forward = target.forward
    original_thinker_forward = thinker.forward
    thinker_config = getattr(thinker, "config", getattr(model, "config", None))
    resolved_video_token_id = _resolve_token_id(thinker_config, "video_token_index", video_token_id)
    resolved_audio_token_id = _resolve_token_id(thinker_config, "audio_token_index", audio_token_id)

    def patched_forward(self, *args, __original_forward=original_forward, **kwargs):
        inputs_embeds = kwargs.get("inputs_embeds", None)
        runtime_input_ids = kwargs.get("input_ids", None)
        source_input_ids = runtime_input_ids if runtime_input_ids is not None else state.input_ids
        is_prefill = source_input_ids is not None and source_input_ids.shape[1] > 1

        stage_cfg = state.config.get("stc", {})
        if (
            stage_cfg.get("enabled", True)
            and is_prefill
            and inputs_embeds is not None
            and int(inputs_embeds.shape[0]) > 1
            and not state.identical_batch
        ):
            raise NotImplementedError(
                "STC batch>1 requires explicit identical_batch=True"
            )
        should_apply = (
            stage_cfg.get("enabled", True)
            and (is_prefill or not stage_cfg.get("prefill_only", True))
            and inputs_embeds is not None
            and source_input_ids is not None
            and not state.applied
        )
        if not should_apply:
            return __original_forward(*args, **kwargs)

        next_semantic_position = _next_semantic_position(
            thinker,
            kwargs.get("position_ids", None),
            kwargs.get("attention_mask", None),
            kwargs.get("cache_position", None),
            original_seq_len=int(source_input_ids.shape[1]),
            batch_size=int(inputs_embeds.shape[0]),
        )

        (
            new_inputs_embeds,
            new_input_ids,
            new_attention_mask,
            new_position_ids,
            global_keep_mask,
            metadata,
        ) = (
            apply_stc_identical_batch
            if int(inputs_embeds.shape[0]) > 1
            else apply_stc
        )(
            inputs_embeds=inputs_embeds,
            input_ids=source_input_ids.to(inputs_embeds.device),
            attention_mask=kwargs.get("attention_mask", None),
            position_ids=kwargs.get("position_ids", None),
            video_token_id=resolved_video_token_id,
            audio_token_id=resolved_audio_token_id,
            video_grid_thw=kwargs.get("video_grid_thw", state.video_grid_thw),
            config=state.config,
            video_merge_size=state.video_merge_size,
        )

        original_cache_position = kwargs.get("cache_position", None)
        kwargs["inputs_embeds"] = new_inputs_embeds
        if runtime_input_ids is not None:
            kwargs["input_ids"] = new_input_ids
        if new_attention_mask is not None:
            kwargs["attention_mask"] = new_attention_mask
        if new_position_ids is not None:
            kwargs["position_ids"] = new_position_ids
        compact_seq_len = int(metadata["summary"]["new_seq_len"])
        if "cache_position" in kwargs:
            kwargs["cache_position"] = _compact_cache_position(
                kwargs.get("cache_position"),
                global_keep_mask,
                kept_count=compact_seq_len,
            )

        _update_compacted_rope_deltas(
            thinker,
            next_semantic_position,
            original_cache_position,
            original_seq_len=int(source_input_ids.shape[1]),
            compact_seq_len=compact_seq_len,
        )

        # Publish compaction state before entering the decoder. SSPA hooks run
        # inside the original forward and need the compacted modality map.
        state.applied = True
        state.last_metadata = metadata
        state.last_keep_mask = global_keep_mask.detach()
        state.next_semantic_position = next_semantic_position
        if verbose:
            log_stc_summary(metadata, state.config)
        return __original_forward(*args, **kwargs)

    def patched_thinker_forward(
        self,
        *args,
        __original_forward=original_thinker_forward,
        **kwargs,
    ):
        normalized_kwargs = _normalize_stc_decode_kwargs(kwargs, state)
        return __original_forward(*args, **normalized_kwargs)

    target.forward = MethodType(patched_forward, target)
    thinker.forward = MethodType(patched_thinker_forward, thinker)
    return state
