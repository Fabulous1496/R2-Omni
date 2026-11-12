"""Integrate STC and packed KV caches at MiniCPM-o's LLM boundary.

MiniCPM keeps its official vision/audio preprocessing and constructs the final
multimodal embedding before calling ``model.llm.generate``.  This adapter only
intercepts that last boundary, so encoder behavior and chat prompts remain
upstream MiniCPM behavior.
"""

from __future__ import annotations

from types import MethodType
from typing import Any, Callable

import torch

from r2omni.sspa.policy.modality_metadata import AUDIO, TEXT, VIDEO
from r2omni.stc.pre_llm_units import apply_stc


def _bounds(value: Any) -> list[tuple[int, int]]:
    """Normalize MiniCPM's batch-one bound representation."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    result: list[tuple[int, int]] = []
    for item in value or []:
        if torch.is_tensor(item):
            item = item.detach().cpu().tolist()
        result.append((int(item[0]), int(item[1])))
    return result


class MiniCPMGenerationAdapter:
    """Patch MiniCPM's private decode boundary for one inference method."""

    VIDEO_SENTINEL = -101
    AUDIO_SENTINEL = -102

    def __init__(
        self,
        model: Any,
        *,
        cache_factory: Callable[[], Any] | None = None,
        sspa_manager: Any | None = None,
        stc_config: dict[str, Any] | None = None,
        token_pruner: Callable[..., Any] | None = None,
    ) -> None:
        self.model = model
        self.cache_factory = cache_factory
        self.sspa_manager = sspa_manager
        self.stc_config = stc_config
        self.token_pruner = token_pruner
        self._original_generate = model.generate
        self._original_decode = model._decode
        self._original_update_model_kwargs = (
            model.llm._update_model_kwargs_for_generation
        )
        self._original_prepare_inputs = model.llm.prepare_inputs_for_generation
        self._request: dict[str, Any] = {}
        self.last_cache: Any | None = None
        self.last_stc_metadata: dict[str, Any] | None = None
        self.last_token_pruning_metadata: dict[str, Any] | None = None
        self.last_generated_tokens: int | None = None
        self.last_modality: torch.Tensor | None = None
        self.installed = False

    def install(self) -> "MiniCPMGenerationAdapter":
        if self.installed:
            return self

        adapter = self

        def wrapped_generate(model_self: Any, *args: Any, **kwargs: Any) -> Any:
            adapter._request = {
                "image_bound": kwargs.get("image_bound"),
                "audio_bounds": kwargs.get("audio_bounds"),
            }
            return adapter._original_generate(*args, **kwargs)

        def wrapped_decode(
            model_self: Any,
            inputs_embeds: torch.Tensor,
            tokenizer: Any,
            attention_mask: torch.Tensor | None,
            **kwargs: Any,
        ) -> Any:
            if int(inputs_embeds.shape[0]) != 1:
                raise ValueError("MiniCPM low-bit adapters support batch size one")
            seq_len = int(inputs_embeds.shape[1])
            modality = torch.full(
                (seq_len,), TEXT, dtype=torch.long, device=inputs_embeds.device
            )
            synthetic_ids = torch.zeros(
                (1, seq_len), dtype=torch.long, device=inputs_embeds.device
            )
            image_bounds = _bounds(adapter._request.get("image_bound"))
            audio_bounds = _bounds(adapter._request.get("audio_bounds"))
            for start, end in image_bounds:
                modality[start:end] = VIDEO
                synthetic_ids[0, start:end] = adapter.VIDEO_SENTINEL
            for start, end in audio_bounds:
                modality[start:end] = AUDIO
                synthetic_ids[0, start:end] = adapter.AUDIO_SENTINEL

            if adapter.token_pruner is not None:
                inputs_embeds, keep_mask, adapter.last_token_pruning_metadata = (
                    adapter.token_pruner(
                        inputs_embeds, modality, image_bounds, audio_bounds
                    )
                )
                inputs_embeds = inputs_embeds[:, keep_mask]
                synthetic_ids = synthetic_ids[:, keep_mask]
                if attention_mask is not None:
                    attention_mask = attention_mask[:, keep_mask]
                modality = modality[keep_mask]

            if adapter.stc_config is not None:
                visual_tokens = sum(end - start for start, end in image_bounds)
                if visual_tokens % 64:
                    raise ValueError(
                        f"MiniCPM visual token count {visual_tokens} is not 8x8 aligned"
                    )
                grid = (
                    torch.tensor(
                        [[visual_tokens // 64, 8, 8]],
                        dtype=torch.long,
                        device=inputs_embeds.device,
                    )
                    if visual_tokens
                    else None
                )
                (
                    inputs_embeds,
                    synthetic_ids,
                    attention_mask,
                    _,
                    keep_mask,
                    adapter.last_stc_metadata,
                ) = apply_stc(
                    inputs_embeds,
                    synthetic_ids,
                    attention_mask,
                    None,
                    adapter.VIDEO_SENTINEL,
                    adapter.AUDIO_SENTINEL,
                    adapter.stc_config,
                    video_grid_thw=grid,
                    video_merge_size=1,
                )
                modality = modality[keep_mask]

            adapter.last_modality = modality.detach()
            if adapter.sspa_manager is not None:
                adapter.sspa_manager.prepare(modality)
            if adapter.cache_factory is not None:
                if "past_key_values" in kwargs:
                    raise ValueError("MiniCPM adapter owns past_key_values")
                adapter.last_cache = adapter.cache_factory()
                kwargs["past_key_values"] = adapter.last_cache
            # MiniCPM leaves this unset. Transformers 4.51 then grows
            # cache_position as [t, t+1, ...] instead of advancing one token,
            # which is incompatible with every native single-token decoder.
            model_self.llm.config.use_cache = True
            model_self.llm.generation_config.use_cache = True
            kwargs["use_cache"] = True
            outputs = adapter._original_decode(
                inputs_embeds, tokenizer, attention_mask, **kwargs
            )
            sequences = getattr(outputs, "sequences", None)
            adapter.last_generated_tokens = (
                int(sequences.shape[-1]) if torch.is_tensor(sequences) else None
            )
            if (
                adapter.last_token_pruning_metadata is not None
                and adapter.last_generated_tokens is not None
            ):
                stats = adapter.last_token_pruning_metadata
                generated = adapter.last_generated_tokens
                original = int(stats["original_prefill_tokens"])
                retained = int(stats["retained_prefill_tokens"])
                stats["generated_tokens"] = generated
                stats["final_active_compression"] = (
                    (original + generated) / max(retained + generated, 1)
                )
            if adapter.last_cache is None:
                adapter.last_cache = getattr(outputs, "past_key_values", None)
            return outputs

        self.model.generate = MethodType(wrapped_generate, self.model)
        self.model._decode = MethodType(wrapped_decode, self.model)
        if self.cache_factory is not None:
            original_update = self._original_update_model_kwargs
            original_prepare = self._original_prepare_inputs

            def wrapped_update(llm_self: Any, *args: Any, **kwargs: Any) -> Any:
                updated = original_update(*args, **kwargs)
                position = updated.get("cache_position")
                if torch.is_tensor(position) and int(position.numel()) > 1:
                    updated["cache_position"] = position[-1:]
                return updated

            self.model.llm._update_model_kwargs_for_generation = MethodType(
                wrapped_update, self.model.llm
            )

            def wrapped_prepare(
                llm_self: Any,
                input_ids: torch.Tensor,
                past_key_values: Any | None = None,
                attention_mask: torch.Tensor | None = None,
                inputs_embeds: torch.Tensor | None = None,
                cache_position: torch.Tensor | None = None,
                **kwargs: Any,
            ) -> Any:
                prepared = original_prepare(
                    input_ids,
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    cache_position=cache_position,
                    **kwargs,
                )
                cache = prepared.get("past_key_values")
                cached_length = (
                    int(cache.get_seq_length())
                    if cache is not None and hasattr(cache, "get_seq_length")
                    else 0
                )
                if cached_length:
                    input_ids = prepared.get("input_ids")
                    if torch.is_tensor(input_ids) and int(input_ids.shape[1]) > 1:
                        prepared["input_ids"] = input_ids[:, -1:]
                    prepared["inputs_embeds"] = None
                    device = (
                        prepared["input_ids"].device
                        if torch.is_tensor(prepared.get("input_ids"))
                        else next(llm_self.parameters()).device
                    )
                    prepared["cache_position"] = torch.tensor(
                        [cached_length], dtype=torch.long, device=device
                    )
                    position_ids = prepared.get("position_ids")
                    if torch.is_tensor(position_ids):
                        prepared["position_ids"] = position_ids[:, -1:].fill_(
                            cached_length
                        )
                return prepared

            self.model.llm.prepare_inputs_for_generation = MethodType(
                wrapped_prepare, self.model.llm
            )
        self.installed = True
        return self

    def uninstall(self) -> None:
        if not self.installed:
            return
        self.model.generate = self._original_generate
        self.model._decode = self._original_decode
        self.model.llm._update_model_kwargs_for_generation = (
            self._original_update_model_kwargs
        )
        self.model.llm.prepare_inputs_for_generation = self._original_prepare_inputs
        self.installed = False
