from __future__ import annotations

import math
import re
from pathlib import Path
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F

MIN_PIXELS = 128 * 28 * 28
MAX_PIXELS = 768 * 28 * 28
SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)
TEST_PROMPT = (
    "You will be provided with video and audio, and a question about the video.\n"
    "Please answer the question based on your understanding of the video and audio. "
    "Respond with the letter corresponding to the correct option directly.\n"
)


def load_worldsense_rows(parquet_path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    return pq.read_table(parquet_path).to_pylist()


def resolve_video_path(dataset_root: Path, row: dict[str, Any]) -> str:
    path = Path(row["video_path"])
    if not path.is_absolute():
        path = dataset_root / str(row["video_path"]).removeprefix("./")
    return str(path.resolve())


def resolve_audio_path(
    dataset_root: Path,
    row: dict[str, Any],
    audio_root: Path | None = None,
) -> Path | None:
    root = audio_root if audio_root is not None else dataset_root / "audios"
    video_name = str(row.get("video") or Path(row["video_path"]).stem)
    candidate = root / f"{video_name}.wav"
    return candidate.resolve() if candidate.exists() else None


def build_question(row: dict[str, Any]) -> str:
    if row.get("open_ended"):
        return str(row["question"])
    candidates = "\n".join(row.get("candidates") or [])
    return f"{TEST_PROMPT}\nQuestion: {row['question']}\n{candidates}"


def build_message(
    row: dict[str, Any],
    dataset_root: Path,
    nframes: int,
    audio_root: Path | None = None,
    *,
    include_audio: bool = True,
) -> list[dict[str, Any]]:
    media: list[dict[str, Any]] = [{
        "type": "video",
        "video": resolve_video_path(dataset_root, row),
        "min_pixels": MIN_PIXELS,
        "max_pixels": MAX_PIXELS,
        "total_pixels": nframes * MAX_PIXELS,
        "max_frames": nframes,
    }]
    audio_path = (
        resolve_audio_path(dataset_root, row, audio_root) if include_audio else None
    )
    if audio_path is not None:
        media.append({"type": "audio", "audio": str(audio_path)})
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": media + [{"type": "text", "text": build_question(row)}]},
    ]


def extract_choice(text: str) -> str:
    normalized = text.strip().upper()
    # MCQ generation is constrained conceptually to the leading option.  A
    # tokenizer may continue the first letter into a word (for example,
    # ``C`` + ``urtain``), so later standalone letters must not override it.
    if normalized[:1] in {"A", "B", "C", "D"}:
        return normalized[0]
    match = re.search(r"\b([A-D])\b", normalized)
    if match:
        return match.group(1)
    return ""


def build_processor_inputs(
    processor: Any,
    text: str,
    audios: Any,
    images: Any,
    videos: Any,
    video_kwargs: dict[str, Any] | None,
    use_audio_in_video: bool,
) -> Any:
    kwargs: dict[str, Any] = {
        "text": text,
        "return_tensors": "pt",
        "padding": False,
        "use_audio_in_video": use_audio_in_video,
    }
    if video_kwargs:
        normalized_video_kwargs = dict(video_kwargs)
        fps = normalized_video_kwargs.get("fps")
        if isinstance(fps, (list, tuple)):
            if len(fps) != 1:
                raise ValueError(f"MCQ evaluation expects one video per sample, got fps={fps!r}")
            normalized_video_kwargs["fps"] = fps[0]
        kwargs.update(normalized_video_kwargs)
    if audios is not None:
        kwargs["audio"] = audios
    if images is not None:
        kwargs["images"] = images
    if videos is not None:
        kwargs["videos"] = videos
    return processor(**kwargs)


def get_video_grid_thw(inputs: Any) -> Any:
    return inputs.get("video_grid_thw", None)


def sanitize_fp16_attention_logits(weights: torch.Tensor) -> torch.Tensor:
    """Stabilize FP16 overflow without erasing causal-mask ``-inf`` values."""
    if weights.dtype != torch.float16:
        return weights
    weights = torch.where(
        torch.isposinf(weights),
        torch.full_like(weights, torch.finfo(weights.dtype).max),
        weights,
    )
    return torch.where(torch.isnan(weights), torch.zeros_like(weights), weights)


def force_text_decoder_eager(model: torch.nn.Module, query_chunk_size: int = 512) -> None:
    """Install bounded-memory eager attention on the text decoder only."""
    from transformers.models.qwen2_5_omni import modeling_qwen2_5_omni as qwen

    text_model = getattr(model, "thinker", model).model
    for layer in text_model.layers:
        def chunked_eager(
            self,
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=None,
            position_embeddings=None,
            **kwargs,
        ):
            batch, query_length, _ = hidden_states.shape
            query = self.q_proj(hidden_states).view(batch, query_length, -1, self.head_dim).transpose(1, 2)
            key = self.k_proj(hidden_states).view(batch, query_length, -1, self.head_dim).transpose(1, 2)
            value = self.v_proj(hidden_states).view(batch, query_length, -1, self.head_dim).transpose(1, 2)
            cos, sin = position_embeddings
            query, key = qwen.apply_multimodal_rotary_pos_emb(
                query, key, cos, sin, self.rope_scaling["mrope_section"]
            )
            if past_key_value is not None:
                key, value = past_key_value.update(
                    key, value, self.layer_idx,
                    {"sin": sin, "cos": cos, "cache_position": cache_position},
                )
            key = qwen.repeat_kv(key, self.num_key_value_groups)
            value = qwen.repeat_kv(value, self.num_key_value_groups)
            output = torch.empty_like(query)
            saved_weights = [] if output_attentions else None
            chunk = max(1, query_chunk_size)
            for start in range(0, query_length, chunk):
                end = min(query_length, start + chunk)
                weights = torch.matmul(query[:, :, start:end], key.transpose(2, 3)) / math.sqrt(self.head_dim)
                if attention_mask is not None:
                    weights = weights + attention_mask[..., start:end, : key.shape[-2]]
                weights = sanitize_fp16_attention_logits(weights)
                weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
                output[:, :, start:end] = torch.matmul(weights, value)
                if saved_weights is not None:
                    saved_weights.append(weights)
            output = self.o_proj(output.transpose(1, 2).contiguous().reshape(batch, query_length, -1))
            return output, torch.cat(saved_weights, dim=2) if saved_weights is not None else None, past_key_value

        layer.self_attn.forward = MethodType(chunked_eager, layer.self_attn)
        layer.self_attn.config._attn_implementation_internal = "eager"
    text_model.config._attn_implementation_internal = "eager"
    text_model._attn_implementation = "eager"


def force_vision_segmented_sdpa(model: torch.nn.Module) -> None:
    """Avoid constructing a global square mask for the vision encoder."""
    from transformers.models.qwen2_5_omni import modeling_qwen2_5_omni as qwen

    visual = getattr(getattr(model, "thinker", model), "visual", None)
    if visual is None:
        return
    for block in visual.blocks:
        def segmented_forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None):
            length = hidden_states.shape[0]
            query = self.q(hidden_states).reshape(length, self.num_heads, -1)
            key = self.k(hidden_states).reshape(length, self.num_heads, -1)
            value = self.v(hidden_states).reshape(length, self.num_heads, -1)
            query = qwen.apply_rotary_pos_emb_vision(query.unsqueeze(0), rotary_pos_emb).squeeze(0).transpose(0, 1)
            key = qwen.apply_rotary_pos_emb_vision(key.unsqueeze(0), rotary_pos_emb).squeeze(0).transpose(0, 1)
            value = value.transpose(0, 1)
            output = torch.empty_like(query)
            boundaries = cu_seqlens.detach().cpu().tolist()
            for start, end in zip(boundaries, boundaries[1:]):
                if end > start:
                    output[:, start:end] = F.scaled_dot_product_attention(
                        query[:, start:end], key[:, start:end], value[:, start:end], dropout_p=0.0
                    )
            return self.proj(output.transpose(0, 1).reshape(length, -1))

        block.attn.forward = MethodType(segmented_forward, block.attn)
