from __future__ import annotations

from contextlib import contextmanager
from types import MethodType
from typing import Any

import torch

from ..policy.modality_metadata import AUDIO, OTHER, TEXT, VIDEO


def get_thinker_model(model: Any) -> Any:
    return getattr(model, "thinker", model)


def get_output_value(outputs: Any, name: str) -> Any:
    if hasattr(outputs, name):
        return getattr(outputs, name)
    if isinstance(outputs, dict) and name in outputs:
        return outputs[name]
    raise AttributeError(f"model output does not contain {name!r}")


def cache_seq_len(past_key_values: Any) -> int:
    if hasattr(past_key_values, "get_seq_length"):
        return int(past_key_values.get_seq_length())
    if not isinstance(past_key_values, (tuple, list)) or not past_key_values:
        raise TypeError("cannot infer cache sequence length from this past_key_values object")
    key = past_key_values[0][0]
    if key.ndim != 4:
        raise ValueError(f"expected rank-4 key cache, got {tuple(key.shape)}")
    return int(max(key.shape[1], key.shape[2]))


def clone_inputs_for_model(inputs: dict[str, Any]) -> dict[str, Any]:
    # Shallow copy is enough: tensors are never modified in-place by this module.
    return dict(inputs)


def move_inputs_to_device(
    inputs: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype | None = None,
    *,
    fused_dtype_transfer: bool = False,
    non_blocking: bool = False,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            target_dtype = (
                dtype
                if fused_dtype_transfer
                and dtype is not None
                and torch.is_floating_point(value)
                else value.dtype
            )
            moved = value.to(
                device=device,
                dtype=target_dtype,
                non_blocking=non_blocking,
            )
            if (
                not fused_dtype_transfer
                and dtype is not None
                and torch.is_floating_point(moved)
            ):
                moved = moved.to(dtype=dtype)
            result[key] = moved
        else:
            result[key] = value
    return result


@contextmanager
def prefill_last_token_logits(thinker: Any, *, enabled: bool = True):
    """Backport GenerationMixin's logits-to-keep fastpath for older Qwen Omni.

    The installed Qwen2.5-Omni forward projects every prefill hidden state even
    though greedy generation consumes only the final logit. Newer Transformers
    models expose ``logits_to_keep=1``; this runtime-only wrapper gives the
    current copied implementation the same behavior without patching model
    weights or the official baseline generate path.
    """
    lm_head = getattr(thinker, "lm_head", None)
    if not enabled or lm_head is None or not hasattr(lm_head, "forward"):
        yield
        return
    original_forward = lm_head.forward

    def last_token_forward(
        _self: Any,
        hidden_states: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if torch.is_tensor(hidden_states) and hidden_states.ndim >= 3:
            hidden_states = hidden_states[:, -1:, :]
        return original_forward(hidden_states, *args, **kwargs)

    lm_head.forward = MethodType(last_token_forward, lm_head)
    try:
        yield
    finally:
        lm_head.forward = original_forward


def assignment_stats(bits: torch.Tensor, modality: torch.Tensor) -> dict[str, Any]:
    modality = modality.to(device=bits.device).flatten()
    bits = bits.flatten()
    return {
        "num_text_tokens": int((modality == TEXT).sum().item()),
        "num_video_tokens": int((modality == VIDEO).sum().item()),
        "num_audio_tokens": int((modality == AUDIO).sum().item()),
        "num_other_tokens": int((modality == OTHER).sum().item()),
        "num_kept_tokens": int((bits != 0).sum().item()),
        "num_evicted_tokens": int((bits == 0).sum().item()),
        "keep_ratio_actual": float((bits != 0).float().mean().item()) if bits.numel() else 0.0,
    }


def build_fallback_assignment(
    seq_len: int,
    *,
    keep_ratio: float,
    modality: torch.Tensor | None = None,
    protect_text: bool = True,
    assignment_mode: str = "uniform",
    q_importance: torch.Tensor | None = None,
    device: torch.device | None = None,
    random_seed: int = 0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Build an Evict-only fallback assignment.

    Modes:
    - uniform: keep evenly spaced non-protected candidates.
    - random: keep random non-protected candidates.
    - importance: keep highest q_importance candidates; q_importance is required.
    """
    seq_len = int(seq_len)
    keep_ratio = float(keep_ratio)
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if keep_ratio < 0.0 or keep_ratio > 1.0:
        raise ValueError("keep_ratio must be in [0, 1]")
    if device is None:
        device = modality.device if modality is not None else torch.device("cpu")
    if modality is None:
        modality = torch.full((seq_len,), OTHER, dtype=torch.long, device=device)
    else:
        modality = modality.to(device=device, dtype=torch.long).flatten()
        if int(modality.numel()) != seq_len:
            raise ValueError(f"modality length {modality.numel()} != seq_len {seq_len}")

    # OTHER includes Qwen image placeholders and other non-AV tokens for which
    # this fallback has no valid alignment signal.  Keep it conservative and
    # consistent with the active/page policies when text protection is enabled.
    protected = (
        (modality == TEXT) | (modality == OTHER)
        if protect_text
        else torch.zeros(seq_len, dtype=torch.bool, device=device)
    )
    candidates = (~protected).nonzero(as_tuple=False).flatten()
    keep_candidates = int(round(float(candidates.numel()) * keep_ratio))
    keep_candidates = max(0, min(int(candidates.numel()), keep_candidates))
    keep_mask = protected.clone()

    mode = str(assignment_mode).lower()
    if keep_candidates > 0 and candidates.numel() > 0:
        if mode == "uniform":
            if keep_candidates == candidates.numel():
                chosen = candidates
            else:
                pos = torch.linspace(0, int(candidates.numel()) - 1, keep_candidates, device=device).round().long()
                pos = torch.unique(pos, sorted=True)
                if int(pos.numel()) < keep_candidates:
                    used = torch.zeros(candidates.numel(), dtype=torch.bool, device=device)
                    used[pos] = True
                    fill = (~used).nonzero(as_tuple=False).flatten()[: keep_candidates - int(pos.numel())]
                    pos = torch.sort(torch.cat([pos, fill]))[0]
                chosen = candidates.index_select(0, pos[:keep_candidates])
        elif mode == "random":
            generator = torch.Generator(device=device)
            generator.manual_seed(int(random_seed))
            chosen = candidates.index_select(0, torch.randperm(candidates.numel(), generator=generator, device=device)[:keep_candidates])
        elif mode == "importance":
            if q_importance is None:
                raise ValueError("assignment_mode='importance' requires q_importance")
            score = q_importance.to(device=device, dtype=torch.float32).flatten()
            if int(score.numel()) != seq_len:
                raise ValueError(f"q_importance length {score.numel()} != seq_len {seq_len}")
            order = torch.argsort(score.index_select(0, candidates), descending=True, stable=True)
            chosen = candidates.index_select(0, order[:keep_candidates])
        else:
            raise ValueError("assignment_mode must be one of: uniform, random, importance")
        keep_mask[chosen] = True
    elif mode not in {"uniform", "random", "importance"}:
        raise ValueError("assignment_mode must be one of: uniform, random, importance")

    bits = torch.where(keep_mask, torch.full((seq_len,), 16, dtype=torch.long, device=device), torch.zeros(seq_len, dtype=torch.long, device=device))
    stats = assignment_stats(bits, modality)
    stats.update({
        "assignment_mode": mode,
        "requested_keep_ratio": keep_ratio,
        "protect_text": bool(protect_text),
    })
    return bits, stats


def _decode_text(processor: Any, generated_ids: torch.Tensor) -> str:
    if processor is None:
        return " ".join(str(int(x)) for x in generated_ids.flatten().detach().cpu().tolist())
    if hasattr(processor, "batch_decode"):
        return processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "decode"):
        return tokenizer.decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return " ".join(str(int(x)) for x in generated_ids.flatten().detach().cpu().tolist())


def _position_kwargs(
    mode: str,
    *,
    original_seq_len: int,
    effective_seq_len: int,
    step: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if mode == "hf_default":
        return {"cache_position": torch.tensor([effective_seq_len + step], dtype=torch.long, device=device)}
    if mode == "compacted":
        return {"cache_position": torch.tensor([effective_seq_len + step], dtype=torch.long, device=device)}
    if mode == "original":
        return {"cache_position": torch.tensor([original_seq_len + step], dtype=torch.long, device=device)}
    raise ValueError("cache_position_mode must be one of: hf_default, compacted, original")


def qwen_mrope_decode_position_ids(
    thinker: Any,
    *,
    original_seq_len: int,
    step: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Keep Qwen semantic RoPE positions independent of compact KV slots.

    Qwen stores ``rope_deltas = next_semantic_position - prompt_length``
    after prefill.  A physically compacted cache must use ``L' + step`` as its
    cache position, but deriving RoPE from that value would shift every decode
    token backwards by ``L - L'``.  Reconstruct the continuation from the
    original prefill length and pass it explicitly.

    The zero-delta fallback covers ordinary text-only positions and lightweight
    test doubles.  Real Qwen2.5-Omni multimodal prefill exposes ``rope_deltas``.
    """
    if original_seq_len < 0:
        raise ValueError("original_seq_len must be non-negative")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    rope_deltas = getattr(thinker, "rope_deltas", None)
    if torch.is_tensor(rope_deltas):
        values = rope_deltas.detach().to(device=device, dtype=torch.long).reshape(-1)
        if int(values.numel()) == 1 and batch_size > 1:
            values = values.expand(batch_size)
        if int(values.numel()) != batch_size:
            raise ValueError(
                "Qwen rope delta batch does not match decode batch: "
                f"deltas={values.numel()}, batch={batch_size}"
            )
    else:
        values = torch.zeros(batch_size, dtype=torch.long, device=device)
    values = values + int(original_seq_len) + int(step)
    return values.view(1, batch_size, 1).expand(3, -1, -1)


def _prefill_kwargs_for_model(thinker: Any, model_inputs: dict[str, Any], use_audio_in_video: bool | None) -> dict[str, Any]:
    kwargs = dict(model_inputs)
    kwargs.update({"use_cache": True})
    if use_audio_in_video is not None:
        kwargs["use_audio_in_video"] = bool(use_audio_in_video)
    if hasattr(thinker, "prepare_inputs_for_generation"):
        input_ids = kwargs.pop("input_ids")
        cache_position = torch.arange(input_ids.shape[1], device=input_ids.device, dtype=torch.long)
        prepared = thinker.prepare_inputs_for_generation(
            input_ids,
            past_key_values=None,
            cache_position=cache_position,
            is_first_iteration=True,
            **kwargs,
        )
        prepared.pop("is_first_iteration", None)
        prepared["use_cache"] = True
        prepared["return_dict"] = True
        return prepared
    kwargs["return_dict"] = True
    return kwargs


def _decode_kwargs_for_model(
    thinker: Any,
    *,
    running_input_ids: torch.Tensor,
    next_token: torch.Tensor,
    attention_mask: torch.Tensor | None,
    past_key_values: Any,
    do_compact: bool,
    cache_position_mode: str,
    original_kv_tokens: int,
    effective_attention_length: int,
    step: int,
    device: torch.device,
    position_ids: torch.Tensor | None = None,
) -> dict[str, Any]:
    del thinker, running_input_ids
    # Decode must pass only the last token. Passing the full running sequence
    # together with past_key_values duplicates the prefix in DynamicCache.
    decode_kwargs: dict[str, Any] = {
        "input_ids": next_token,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "use_cache": True,
        "return_dict": True,
    }
    decode_kwargs.update(_position_kwargs(
        cache_position_mode,
        original_seq_len=original_kv_tokens,
        effective_seq_len=effective_attention_length,
        step=step,
        device=device,
    ))
    if position_ids is not None:
        decode_kwargs["position_ids"] = position_ids
    return decode_kwargs
