from __future__ import annotations

from typing import Any, Callable

import torch

from ...stc.qwen_stc import (
    compact_stc_attention_mask,
    stc_decode_position_ids,
)
from .qwen_omni_greedy_decode import (
    _decode_kwargs_for_model,
    _decode_text,
    _prefill_kwargs_for_model,
    cache_seq_len,
    clone_inputs_for_model,
    get_output_value,
    get_thinker_model,
    move_inputs_to_device,
    prefill_last_token_logits,
)


def _preallocate_decode_attention_mask(
    attention_mask: torch.Tensor | None,
    decode_tokens: int,
) -> tuple[torch.Tensor | None, int]:
    """Allocate the complete decode mask once and return its prompt length."""
    if attention_mask is None:
        return None, 0
    base_length = int(attention_mask.shape[-1])
    shape = (*attention_mask.shape[:-1], base_length + max(0, int(decode_tokens)))
    buffer = torch.ones(
        shape,
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    buffer[..., :base_length].copy_(attention_mask)
    return buffer, base_length


def _external_kept_indices(thinker: Any) -> tuple[torch.Tensor | None, str | None]:
    """Read the original-token index map exported by a token compressor."""
    candidates = (
        (thinker, "thinker"),
        (getattr(thinker, "model", None), "thinker.model"),
    )
    for owner, owner_name in candidates:
        if owner is None:
            continue
        indices = getattr(owner, "_r2omni_external_kept_indices", None)
        if torch.is_tensor(indices):
            source = getattr(
                owner,
                "_r2omni_external_compression_source",
                owner_name,
            )
            return indices, str(source)
    return None, None


def _compact_external_attention_mask(
    attention_mask: torch.Tensor | None,
    kept_indices: torch.Tensor,
    *,
    semantic_prompt_tokens: int,
    cache_prompt_tokens: int,
) -> torch.Tensor | None:
    """Make the decode mask follow an externally compacted prefill cache."""
    kept = kept_indices.detach().flatten().to(dtype=torch.long)
    if int(kept.numel()) != cache_prompt_tokens:
        raise ValueError(
            "external keep-index count does not match the prefill cache: "
            f"indices={kept.numel()}, cache={cache_prompt_tokens}"
        )
    if kept.numel() == 0:
        raise ValueError("external token compressor exported no retained tokens")
    if bool((kept < 0).any()) or bool((kept >= semantic_prompt_tokens).any()):
        raise ValueError("external token compressor keep indices are out of range")
    if kept.numel() > 1 and bool((kept[1:] <= kept[:-1]).any()):
        raise ValueError("external token compressor keep indices must be strictly increasing")
    if attention_mask is None:
        return None
    if int(attention_mask.shape[-1]) != semantic_prompt_tokens:
        raise ValueError(
            "external decode mask does not match the uncompressed prompt: "
            f"mask={attention_mask.shape[-1]}, prompt={semantic_prompt_tokens}"
        )
    compact = attention_mask.index_select(
        -1,
        kept.to(device=attention_mask.device),
    )
    if int(compact.shape[-1]) != cache_prompt_tokens:
        raise ValueError("external compact decode mask does not match the prefill cache")
    return compact


def r2omni_greedy_decode(
    model: Any,
    processor: Any,
    inputs: dict[str, Any],
    *,
    max_new_tokens: int = 2,
    backend: str = "baseline",
    cache_position_mode: str = "hf_default",
    use_audio_in_video: bool | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    stc_state: Any | None = None,
    prefill_logits_only: bool = True,
    stop_on_eos: bool = False,
    min_new_tokens: int = 0,
    decode_batch_size: int | None = None,
    activate_decode_batch: Callable[[int], None] | None = None,
    clone_baseline_decode_cache: bool = False,
) -> dict[str, Any]:
    """Run controlled greedy decode with an opt-in post-prefill batch path."""
    if backend not in {"baseline", "r2omni"}:
        raise ValueError("backend must be baseline or r2omni")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")
    if not 0 <= min_new_tokens <= max_new_tokens:
        raise ValueError("min_new_tokens must be in [0, max_new_tokens]")
    thinker = get_thinker_model(model)
    thinker.eval()
    if device is None:
        device = next(thinker.parameters()).device
    device = torch.device(device)
    model_inputs = move_inputs_to_device(
        clone_inputs_for_model(inputs),
        device,
        dtype,
        fused_dtype_transfer=True,
        non_blocking=True,
    )
    input_ids = model_inputs.get("input_ids")
    if not torch.is_tensor(input_ids) or input_ids.ndim != 2:
        raise ValueError("R2Omni greedy decode requires rank-2 tensor input_ids")
    semantic_prompt_tokens = int(input_ids.shape[-1])
    prefill_batch_size = int(input_ids.shape[0])
    target_batch_size = (
        prefill_batch_size
        if decode_batch_size is None
        else int(decode_batch_size)
    )
    if target_batch_size < 1:
        raise ValueError("decode_batch_size must be positive")
    if prefill_batch_size not in {1, target_batch_size}:
        raise ValueError(
            "prefill batch must equal decode batch or be one for cache cloning"
        )
    if target_batch_size > 1 and stop_on_eos:
        raise ValueError("throughput batching currently requires stop_on_eos=False")
    attention_mask = model_inputs.get("attention_mask")
    with torch.inference_mode():
        with prefill_last_token_logits(thinker, enabled=prefill_logits_only):
            outputs = thinker(
                **_prefill_kwargs_for_model(
                    thinker, model_inputs, use_audio_in_video
                )
            )
    past = get_output_value(outputs, "past_key_values")
    logits = get_output_value(outputs, "logits")
    cache_prompt_tokens = cache_seq_len(past)
    if stc_state is not None and bool(getattr(stc_state, "applied", False)):
        attention_mask = compact_stc_attention_mask(
            attention_mask,
            getattr(stc_state, "last_keep_mask", None),
        )
        if attention_mask is not None and int(attention_mask.shape[-1]) != cache_prompt_tokens:
            raise ValueError(
                "STC compact decode mask length does not match prefill cache"
            )
        keep_mask = getattr(stc_state, "last_keep_mask", None)
    external_indices, external_source = _external_kept_indices(thinker)
    if external_indices is not None:
        if stc_state is not None and bool(getattr(stc_state, "applied", False)):
            raise ValueError("STC and external token compaction cannot be combined")
        attention_mask = _compact_external_attention_mask(
            attention_mask,
            external_indices,
            semantic_prompt_tokens=semantic_prompt_tokens,
            cache_prompt_tokens=cache_prompt_tokens,
        )
    if target_batch_size > 1 and prefill_batch_size == 1:
        if activate_decode_batch is not None:
            activate_decode_batch(target_batch_size)
        elif clone_baseline_decode_cache:
            repeat = getattr(past, "batch_repeat_interleave", None)
            if not callable(repeat):
                raise TypeError(
                    "baseline cache does not support batch_repeat_interleave"
                )
            repeat(target_batch_size)
        else:
            raise ValueError(
                "decode cache cloning requires an activate_decode_batch callback"
            )
        input_ids = input_ids.expand(target_batch_size, -1).contiguous()
        if attention_mask is not None:
            attention_mask = attention_mask.expand(
                target_batch_size, *attention_mask.shape[1:]
            ).contiguous()
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    if int(next_token.shape[0]) == 1 and target_batch_size > 1:
        next_token = next_token.expand(target_batch_size, -1).contiguous()
    if int(next_token.shape[0]) != target_batch_size:
        raise ValueError("prefill logits do not match requested decode batch")
    generated_ids = torch.empty(
        (int(input_ids.shape[0]), max_new_tokens),
        dtype=next_token.dtype,
        device=device,
    )
    generated_ids[:, :1].copy_(next_token)
    generated_count = 1
    eos_token_id = getattr(getattr(processor, "tokenizer", None), "eos_token_id", None)
    eos_token_ids = (
        {int(value) for value in eos_token_id}
        if isinstance(eos_token_id, (list, tuple, set))
        else set() if eos_token_id is None else {int(eos_token_id)}
    )

    def reached_eos(token: torch.Tensor, count: int) -> bool:
        return bool(
            stop_on_eos
            and count >= min_new_tokens
            and eos_token_ids
            and int(token.reshape(-1)[0].item()) in eos_token_ids
        )
    # Prefill logits are [B, prompt, vocab] and can be much larger than all
    # decode logits combined. Release them before cache packing/decode.
    del outputs, logits

    compact_metadata: dict[str, Any] = {
        "original_seq_len": cache_prompt_tokens,
        "compressed_seq_len": cache_prompt_tokens,
        "kept_tokens": cache_prompt_tokens,
        "evicted_tokens": 0,
        "keep_ratio": 1.0,
    }
    effective_length = cache_prompt_tokens
    decode_tokens = 0 if reached_eos(next_token, generated_count) else max_new_tokens - 1
    mask_buffer, mask_base_length = _preallocate_decode_attention_mask(
        attention_mask,
        decode_tokens,
    )
    with torch.inference_mode():
        for step in range(decode_tokens):
            step_attention_mask = (
                mask_buffer[..., : mask_base_length + step + 1]
                if mask_buffer is not None
                else None
            )
            semantic_position_ids = stc_decode_position_ids(
                stc_state,
                step=step,
                batch_size=int(input_ids.shape[0]),
                device=device,
            )
            if semantic_position_ids is None and external_indices is not None:
                semantic_position_ids = qwen_mrope_decode_position_ids(
                    thinker,
                    original_seq_len=semantic_prompt_tokens,
                    step=step,
                    batch_size=int(input_ids.shape[0]),
                    device=device,
                )
            kwargs = _decode_kwargs_for_model(
                thinker,
                # The helper intentionally sends only next_token when a cache
                # is present. Avoid growing an otherwise unused ID tensor.
                running_input_ids=input_ids,
                next_token=next_token,
                attention_mask=step_attention_mask,
                past_key_values=past,
                do_compact=False,
                cache_position_mode=cache_position_mode,
                original_kv_tokens=cache_prompt_tokens,
                effective_attention_length=effective_length,
                step=step,
                device=device,
                position_ids=semantic_position_ids,
            )
            outputs = thinker(**kwargs)
            past = get_output_value(outputs, "past_key_values")
            logits = get_output_value(outputs, "logits")
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids[:, step + 1 : step + 2].copy_(next_token)
            generated_count = step + 2
            del outputs, logits
            if reached_eos(next_token, generated_count):
                break

    generated_ids = generated_ids[:, :generated_count].contiguous()
    return {
        "backend": backend,
        "prefill_logits_only": bool(prefill_logits_only),
        "generated_ids": generated_ids,
        "output_text": _decode_text(processor, generated_ids),
        "original_kv_tokens": cache_prompt_tokens,
        "initial_effective_attention_length": effective_length,
        "effective_attention_length": effective_length + generated_count - 1,
        "compact_metadata": {
            **compact_metadata,
            **(
                {
                    "external_compression_source": external_source,
                    "semantic_prompt_tokens": semantic_prompt_tokens,
                    "cache_prompt_tokens": cache_prompt_tokens,
                }
                if external_indices is not None
                else {}
            ),
        },
        "final_past_key_values": past,
    }


def official_model_generate_decode(
    model: Any,
    processor: Any,
    inputs: dict[str, Any],
    *,
    max_new_tokens: int,
    use_audio_in_video: bool | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    stop_on_eos: bool = False,
    min_new_tokens: int = 0,
) -> dict[str, Any]:
    """Run the official Qwen-Omni generation path without instrumentation."""
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")
    if not 0 <= min_new_tokens <= max_new_tokens:
        raise ValueError("min_new_tokens must be in [0, max_new_tokens]")
    thinker = get_thinker_model(model)
    thinker.eval()
    if device is None:
        device = next(thinker.parameters()).device
    device = torch.device(device)
    model_inputs = move_inputs_to_device(clone_inputs_for_model(inputs), device, dtype)
    input_ids = model_inputs.get("input_ids")
    if (
        not torch.is_tensor(input_ids)
        or input_ids.ndim != 2
        or int(input_ids.shape[0]) < 1
    ):
        raise ValueError("official model.generate benchmark requires batched tensor input_ids")

    cache_factory = getattr(model, "_r2omni_cache_factory", None)
    native_cache_kwargs = (
        {"thinker_past_key_values": cache_factory()}
        if callable(cache_factory)
        else {}
    )
    with torch.inference_mode():
        generated = model.generate(
            **model_inputs,
            use_audio_in_video=bool(use_audio_in_video),
            return_audio=False,
            thinker_max_new_tokens=max_new_tokens,
            thinker_min_new_tokens=(min_new_tokens if stop_on_eos else max_new_tokens),
            thinker_do_sample=False,
            thinker_use_cache=True,
            thinker_return_dict_in_generate=True,
            **native_cache_kwargs,
        )

    sequences = generated.sequences if hasattr(generated, "sequences") else generated
    generated_ids = sequences[:, int(input_ids.shape[-1]) :]
    generated_token_count = int(generated_ids.shape[-1])
    if (
        (not stop_on_eos and generated_token_count != max_new_tokens)
        or (stop_on_eos and not 1 <= generated_token_count <= max_new_tokens)
    ):
        raise RuntimeError(
            f"official model.generate produced {generated_ids.shape[-1]} tokens; "
            f"expected {'at most' if stop_on_eos else 'exactly'} {max_new_tokens}"
        )
    past = getattr(generated, "past_key_values", None)
    if past is None:
        raise RuntimeError("official model.generate did not return past_key_values")

    final_cache_length = cache_seq_len(past)
    original_cache_length = final_cache_length - max(0, generated_token_count - 1)
    return {
        "backend": "official_generate",
        "generated_ids": generated_ids,
        "output_text": _decode_text(processor, generated_ids),
        "original_kv_tokens": original_cache_length,
        "initial_effective_attention_length": original_cache_length,
        "effective_attention_length": final_cache_length,
        "compact_metadata": {
            "original_seq_len": original_cache_length,
            "compressed_seq_len": original_cache_length,
            "kept_tokens": original_cache_length,
            "evicted_tokens": 0,
            "keep_ratio": 1.0,
            "decode_driver": "transformers_model_generate",
        },
        "final_past_key_values": past,
    }
