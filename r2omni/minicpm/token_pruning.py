"""MiniCPM-o adapters for the released multimodal token-pruning cores."""

from __future__ import annotations

import importlib.util
import math
import sys
import types
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from r2omni.common.paths import resolve_third_party

import numpy as np
import torch

from r2omni.sspa.policy.modality_metadata import AUDIO, TEXT, VIDEO


def _load_source(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _positions(bounds: list[tuple[int, int]], device: torch.device) -> torch.Tensor:
    parts = [torch.arange(start, end, device=device) for start, end in bounds if end > start]
    return torch.cat(parts) if parts else torch.empty(0, dtype=torch.long, device=device)


def _target_modal_ratio(modality: torch.Tensor, compression: float) -> float:
    total = int(modality.numel())
    modal = int((modality != TEXT).sum().item())
    text = total - modal
    target_total = max(text, int(math.ceil(total / compression)))
    return min(1.0, max(0.0, (target_total - text) / max(modal, 1)))


class MiniCPMTokenPruner:
    """Select MiniCPM multimodal embeddings with an upstream method's core.

    MiniCPM uses embedding bounds instead of Qwen placeholder IDs.  The adapter
    changes only that layout interface: selection remains before LLM prefill,
    text is always retained, and each method calls its released selector.
    """

    def __init__(
        self,
        method: str,
        workspace: Path,
        device: torch.device,
        *,
        target_compression: float = 5.0,
        omniselect_threshold: float = 2.0,
        omniselect_checkpoint: Path | None = None,
        omnizip_contextual_ratio: float = 0.05,
        omnizip_g: int = 3,
    ) -> None:
        if method not in {"omniselect", "omnizip", "seats"}:
            raise ValueError(f"unsupported MiniCPM token-pruning method: {method}")
        self.method = method
        self.workspace = workspace
        self.device = device
        self.target_compression = float(target_compression)
        self.omniselect_threshold = float(omniselect_threshold)
        self.omnizip_contextual_ratio = float(omnizip_contextual_ratio)
        self.omnizip_g = int(omnizip_g)
        self.media: dict[str, Any] = {}
        self.selector: Any | None = None

        if method == "omniselect":
            # Install AudioCLIP's Python dependencies in the active environment.
            audio_clip = resolve_third_party("OmniSelect", workspace) / "AudioCLIP"
            sys.path.insert(0, str(audio_clip))
            from model import AudioCLIP

            checkpoint = omniselect_checkpoint or (
                audio_clip / "assets/AudioCLIP-Full-Training.pt"
            )
            self.selector = AudioCLIP(pretrained=str(checkpoint)).to(device).eval()

    def set_media(
        self,
        question: str,
        frames: list[Any],
        audio_segments: list[Any],
    ) -> None:
        self.media = {
            "question": str(question),
            "frames": frames,
            "audio_segments": audio_segments,
        }

    def __call__(
        self,
        inputs_embeds: torch.Tensor,
        modality: torch.Tensor,
        image_bounds: list[tuple[int, int]],
        audio_bounds: list[tuple[int, int]],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        ratio = _target_modal_ratio(modality, self.target_compression)
        if self.method == "seats":
            updated, mask = self._seats(inputs_embeds, image_bounds, audio_bounds, ratio)
        elif self.method == "omnizip":
            updated, mask = self._omnizip(inputs_embeds, image_bounds, audio_bounds, ratio)
        else:
            updated, mask = self._omniselect(
                inputs_embeds, modality, image_bounds, audio_bounds, ratio
            )
        original = int(mask.numel())
        retained = int(mask.sum().item())
        original_modal = int((modality != TEXT).sum().item())
        retained_modal = int(((modality != TEXT) & mask).sum().item())
        metadata = {
            "method": self.method,
            "adapter": "minicpmo_embedding_bounds_v1",
            "target_compression": self.target_compression,
            "original_prefill_tokens": original,
            "retained_prefill_tokens": retained,
            "prefill_token_keep_ratio": retained / max(original, 1),
            "prefill_token_compression": original / max(retained, 1),
            "original_multimodal_tokens": original_modal,
            "retained_multimodal_tokens": retained_modal,
            "multimodal_keep_ratio": retained_modal / max(original_modal, 1),
        }
        return updated, mask, metadata

    def _seats(
        self,
        embeds: torch.Tensor,
        image_bounds: list[tuple[int, int]],
        audio_bounds: list[tuple[int, int]],
        ratio: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        utils = _load_source(
            "baselines.utils",
            resolve_third_party("SEATS", self.workspace) / "baselines/utils.py",
        )
        sys.modules["baselines.utils"] = utils
        source = resolve_third_party("SEATS", self.workspace) / "seats/pre_llm_units.py"
        core = _load_source("r2omni_seats_pre_llm", source)
        mask = torch.ones(embeds.shape[1], dtype=torch.bool, device=embeds.device)
        for start, end in image_bounds + audio_bounds:
            length = end - start
            if length <= 0:
                continue
            keep = max(1, min(length, int(round(length * ratio))))
            selected = core.weighted_MMD_batch(
                embeds[:, start:end].float(), None, keep
            )[0]
            mask[start:end] = False
            mask[start + selected] = True
        return embeds, mask

    def _omnizip(
        self,
        embeds: torch.Tensor,
        image_bounds: list[tuple[int, int]],
        audio_bounds: list[tuple[int, int]],
        ratio: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from omnizip.omnizip_core import omnizip_audio_attn, omnizip_istm

        flat = embeds[0]
        mask = torch.ones(flat.shape[0], dtype=torch.bool, device=flat.device)
        video_pos = _positions(image_bounds, flat.device)
        audio_pos = _positions(audio_bounds, flat.device)
        video = flat[video_pos]
        audio = flat[audio_pos]
        discard = 1.0 - ratio

        if audio.numel():
            if video.numel():
                an = torch.nn.functional.normalize(audio.float(), dim=-1)
                vn = torch.nn.functional.normalize(video.float(), dim=-1)
                importance = (an @ vn.T).max(dim=1).values
            else:
                importance = audio.float().norm(dim=-1)
            audio_mask, merge_plan = omnizip_audio_attn(
                audio, video if video.numel() else None, importance,
                merging_ratio=discard,
                contextual_ratio=self.omnizip_contextual_ratio,
                g=self.omnizip_g,
            )
            mask[audio_pos] = audio_mask
            if merge_plan:
                normalized_audio = torch.nn.functional.normalize(audio.float(), dim=-1)
                normalized_video = (
                    torch.nn.functional.normalize(video.float(), dim=-1)
                    if video.numel() else None
                )
                for anchor, merged_indices in merge_plan.items():
                    if not merged_indices:
                        continue
                    merged = torch.tensor(merged_indices, device=flat.device)
                    scores = (
                        (normalized_audio[merged] @ normalized_video.T).max(dim=1).values
                        if normalized_video is not None
                        else normalized_audio[merged] @ normalized_audio[anchor]
                    )
                    weights = torch.softmax(scores, dim=0).to(audio.dtype)
                    flat[audio_pos[anchor]] = (
                        audio[anchor] + (audio[merged] * weights[:, None]).sum(dim=0)
                    ) / (1.0 + weights.sum())

        if video.numel():
            lengths = [end - start for start, end in image_bounds]
            if lengths and len(set(lengths)) == 1:
                video_mask = omnizip_istm(
                    video, num_tokens_per_frame=lengths[0], merging_ratio=discard
                )
                mask[video_pos] = video_mask
            else:
                for start, end in image_bounds:
                    local = omnizip_istm(
                        flat[start:end], end - start, merging_ratio=discard
                    )
                    mask[start:end] = local
        return embeds, mask

    @torch.no_grad()
    def _omniselect_media_scores(self) -> tuple[torch.Tensor, torch.Tensor, bool | str]:
        import torchvision as tv
        import nltk
        from nltk import pos_tag, word_tokenize

        frames = self.media["frames"]
        audio_segments = self.media["audio_segments"]
        words = word_tokenize(self.media["question"])
        keywords = [word for word, tag in pos_tag(words) if tag.startswith(("NN", "JJ"))]
        text = keywords or words or ["video"]
        transform = tv.transforms.Compose([
            tv.transforms.Resize(224, interpolation=tv.transforms.InterpolationMode.BICUBIC),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ])
        images = torch.stack([transform(frame.convert("RGB")) for frame in frames]).to(self.device)
        valid_audio = [np.asarray(item, dtype=np.float32).reshape(-1) for item in audio_segments if item is not None]
        if not valid_audio:
            raise ValueError("OmniSelect requires audio")
        width = min(len(item) for item in valid_audio)
        audio = torch.from_numpy(np.stack([item[:width] for item in valid_audio]))[:, None].to(self.device)
        ((audio_features, _, _), _), _ = self.selector(audio=audio)
        ((_, image_features, _), _), _ = self.selector(image=images)
        ((_, _, text_features), _), _ = self.selector(text=text)
        audio_features = torch.nn.functional.normalize(audio_features, dim=-1)
        image_features = torch.nn.functional.normalize(image_features, dim=-1)
        text_features = torch.nn.functional.normalize(text_features, dim=-1)
        logits_a = (self.selector.logit_scale_at.exp().clamp(1, 100) * audio_features @ text_features.T).mean(-1)
        logits_v = (self.selector.logit_scale.exp().clamp(1, 100) * image_features @ text_features.T).mean(-1)
        audio_score, video_score = logits_a.mean(), logits_v.mean()
        if abs(int(video_score) - int(audio_score)) <= self.omniselect_threshold or not keywords:
            video_first: bool | str = "uniform"
        else:
            video_first = bool(audio_score < video_score)
        return logits_a, logits_v, video_first

    def _omniselect(
        self,
        embeds: torch.Tensor,
        modality: torch.Tensor,
        image_bounds: list[tuple[int, int]],
        audio_bounds: list[tuple[int, int]],
        ratio: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The official file imports matplotlib only for optional visualizations.
        # Supply an inert module so the selector core can be loaded without
        # adding plotting packages to the inference environment.
        if "matplotlib" not in sys.modules:
            matplotlib = types.ModuleType("matplotlib")
            pyplot = types.ModuleType("matplotlib.pyplot")
            matplotlib.pyplot = pyplot
            sys.modules["matplotlib"] = matplotlib
            sys.modules["matplotlib.pyplot"] = pyplot
        core = _load_source(
            "r2omni_omniselect_core",
            resolve_third_party("OmniSelect", self.workspace) / "OmniSelect/OmniSelect.py",
        )
        video_pos = _positions(image_bounds, embeds.device)
        audio_pos = _positions(audio_bounds, embeds.device)
        if not video_pos.numel() or not audio_pos.numel():
            return embeds, torch.ones(embeds.shape[1], dtype=torch.bool, device=embeds.device)
        logits_a, logits_v, video_first = self._omniselect_media_scores()
        synthetic = torch.zeros(embeds.shape[1], dtype=torch.long, device=embeds.device)
        synthetic[video_pos] = -101
        synthetic[audio_pos] = -102
        token_to_group: defaultdict[int, int] = defaultdict(int)
        for bounds in (image_bounds, audio_bounds):
            for group, (start, end) in enumerate(bounds, start=1):
                for position in range(start, end):
                    token_to_group[position] = group
        video_features = torch.nn.functional.normalize(embeds[0, video_pos].float(), dim=-1)
        audio_features = embeds[0, audio_pos].float()
        if video_features.numel():
            audio_importance = (
                torch.nn.functional.normalize(audio_features, dim=-1) @ video_features.T
            ).max(dim=1).values
        else:
            audio_importance = audio_features.norm(dim=-1)
        prune_need = {
            "args": SimpleNamespace(prune=True, prune_ratio_a=1.0 - ratio, prune_ratio_v=1.0 - ratio),
            "token_id2group": token_to_group,
            "logits_a": logits_a,
            "logits_v": logits_v,
            "video_first": video_first,
            "v_lst": video_pos.tolist(),
            "a_lst": audio_pos.tolist(),
        }
        mask = core.omniselect(
            None, embeds, audio_features, video_features,
            audio_importance, video_features, synthetic, prune_need
        )
        return embeds, mask
