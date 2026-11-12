from __future__ import annotations

from typing import Any

import torch

from ..policy.modality_metadata import AUDIO, OTHER, TEXT, VIDEO


def _first_tensor_input_ids(inputs: Any) -> torch.Tensor:
    if isinstance(inputs, dict) and torch.is_tensor(inputs.get("input_ids")):
        return inputs["input_ids"]
    if hasattr(inputs, "input_ids") and torch.is_tensor(inputs.input_ids):
        return inputs.input_ids
    raise ValueError("inputs must contain tensor input_ids")


def _candidate_attr_values(obj: Any, names: tuple[str, ...]) -> list[int]:
    values: list[int] = []
    if obj is None:
        return values
    for name in names:
        value = getattr(obj, name, None)
        if value is None and isinstance(obj, dict):
            value = obj.get(name)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            raw_values = value
        else:
            raw_values = (value,)
        for raw in raw_values:
            try:
                values.append(int(raw))
            except (TypeError, ValueError):
                continue
    return values


def _tokenizer_token_ids(tokenizer: Any, token_strings: tuple[str, ...]) -> list[int]:
    ids: list[int] = []
    if tokenizer is None:
        return ids
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if convert is None:
        return ids
    for token in token_strings:
        try:
            value = convert(token)
        except Exception:
            continue
        if value is None:
            continue
        if isinstance(value, list):
            raw_values = value
        else:
            raw_values = (value,)
        for raw in raw_values:
            try:
                token_id = int(raw)
            except (TypeError, ValueError):
                continue
            if token_id >= 0:
                ids.append(token_id)
    return ids


def resolve_qwen_omni_special_token_ids(
    tokenizer: Any = None,
    processor: Any = None,
    model_config: Any = None,
) -> dict[str, list[int]]:
    """Resolve best-effort Qwen-Omni placeholder token ids."""
    tokenizer = tokenizer or getattr(processor, "tokenizer", None)
    cfgs = [
        model_config,
        getattr(model_config, "thinker_config", None),
        getattr(getattr(model_config, "thinker_config", None), "text_config", None),
        getattr(model_config, "text_config", None),
    ]
    video_names = ("video_token_id", "video_token_index", "vision_token_id", "vision_token_index")
    audio_names = ("audio_token_id", "audio_token_index")
    image_names = ("image_token_id", "image_token_index")
    result = {"video": [], "audio": [], "image": []}
    for cfg in cfgs:
        result["video"].extend(_candidate_attr_values(cfg, video_names))
        result["audio"].extend(_candidate_attr_values(cfg, audio_names))
        result["image"].extend(_candidate_attr_values(cfg, image_names))
    result["video"].extend(_tokenizer_token_ids(tokenizer, ("<|video_pad|>", "<|video|>", "<video>")))
    result["audio"].extend(_tokenizer_token_ids(tokenizer, ("<|audio_pad|>", "<|audio|>", "<audio>")))
    result["image"].extend(_tokenizer_token_ids(tokenizer, ("<|image_pad|>", "<|image|>", "<image>")))
    return {key: sorted(set(value)) for key, value in result.items()}


def modality_stats(vector: torch.Tensor, *, method: str, confidence: str, notes: list[str] | None = None) -> dict[str, Any]:
    vector = vector.flatten()
    return {
        "num_text_tokens": int((vector == TEXT).sum().item()),
        "num_video_tokens": int((vector == VIDEO).sum().item()),
        "num_audio_tokens": int((vector == AUDIO).sum().item()),
        "num_other_tokens": int((vector == OTHER).sum().item()),
        "modality_confidence": confidence,
        "modality_inference_method": method,
        "notes": notes or [],
    }


def infer_qwen_omni_modality_vector(
    inputs: Any,
    tokenizer: Any = None,
    processor: Any = None,
    model_config: Any = None,
    fallback_text_protect: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return a [seq_len] TEXT/VIDEO/AUDIO/OTHER vector for Qwen-Omni inputs.

    This is best-effort. Qwen-Omni processors usually materialize one
    placeholder token per AV feature token, so token-id matching can identify
    video/audio positions. If ids are unavailable, text is protected and
    non-text deletion is disabled by marking everything TEXT.
    """
    input_ids = _first_tensor_input_ids(inputs)
    ids = input_ids[0] if input_ids.ndim == 2 else input_ids
    ids = ids.detach()
    seq_len = int(ids.numel())
    device = ids.device
    modality = torch.full((seq_len,), TEXT if fallback_text_protect else OTHER, dtype=torch.long, device=device)
    token_ids = resolve_qwen_omni_special_token_ids(tokenizer=tokenizer, processor=processor, model_config=model_config)
    notes: list[str] = []

    matched = 0
    if token_ids["video"]:
        video_ids = torch.tensor(token_ids["video"], dtype=ids.dtype, device=device)
        video_mask = (ids[:, None] == video_ids[None, :]).any(dim=1)
        modality[video_mask] = VIDEO
        matched += int(video_mask.sum().item())
    else:
        notes.append("video placeholder token id unavailable")

    if token_ids["audio"]:
        audio_ids = torch.tensor(token_ids["audio"], dtype=ids.dtype, device=device)
        audio_mask = (ids[:, None] == audio_ids[None, :]).any(dim=1)
        modality[audio_mask] = AUDIO
        matched += int(audio_mask.sum().item())
    else:
        notes.append("audio placeholder token id unavailable")

    # Image placeholders are non-text but not separately targeted this round.
    if token_ids["image"]:
        image_ids = torch.tensor(token_ids["image"], dtype=ids.dtype, device=device)
        image_mask = (ids[:, None] == image_ids[None, :]).any(dim=1)
        modality[image_mask] = OTHER
        matched += int(image_mask.sum().item())

    if matched > 0:
        confidence = "placeholder_exact"
        method = "qwen_placeholder_token_ids"
        notes.append("text is inferred as all non-audio/video/image placeholder tokens")
    else:
        confidence = "fallback_text_only" if fallback_text_protect else "fallback_unknown"
        method = "fallback_no_placeholder_match"
        if fallback_text_protect:
            notes.append("all tokens marked TEXT to avoid unsafe deletion")
        else:
            notes.append("all tokens marked OTHER because fallback_text_protect=False")
    stats = modality_stats(modality, method=method, confidence=confidence, notes=notes)
    stats["special_token_ids"] = token_ids
    stats["text_span_available"] = True
    stats["video_span_available"] = bool(token_ids["video"] and stats["num_video_tokens"] > 0)
    stats["audio_span_available"] = bool(token_ids["audio"] and stats["num_audio_tokens"] > 0)
    return modality, stats
