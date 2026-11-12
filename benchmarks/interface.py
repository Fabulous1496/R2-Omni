"""Stable dataset-adapter interface for R2Omni benchmark runners.

Benchmark data lives in the workspace-level ``Benchmarks`` directory.  This
module keeps dataset-specific loading and media conventions out of the model
runner and provides one registration point for downstream benchmarks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class BenchmarkPaths:
    annotation: Path
    video_root: Path
    audio_root: Path | None = None


class BenchmarkAdapter(ABC):
    """Interface implemented by every dataset consumed by R2Omni."""

    name: str
    aliases: tuple[str, ...] = ()
    annotation_relative: str
    video_root_relative: str
    audio_root_relative: str | None = None
    prediction_mode: str = "first_choice"
    embedded_audio: bool = False

    def paths(self, workspace_root: Path) -> BenchmarkPaths:
        benchmark_root = workspace_root / "Benchmarks"
        return BenchmarkPaths(
            annotation=benchmark_root / self.annotation_relative,
            video_root=benchmark_root / self.video_root_relative,
            audio_root=(
                None
                if self.audio_root_relative is None
                else benchmark_root / self.audio_root_relative
            ),
        )

    @abstractmethod
    def load_rows(self, annotation: Path) -> list[dict[str, Any]]:
        """Load and normalize annotations into runner records."""

    @abstractmethod
    def resolve_video(self, video_root: Path, row: dict[str, Any]) -> str:
        """Return the media path used for availability checks and cache keys."""

    @abstractmethod
    def build_message(
        self,
        row: dict[str, Any],
        video_root: Path,
        nframes: int,
        audio_root: Path | None,
        *,
        include_audio: bool,
        video_resolution_policy: str = "benchmark_cap",
        video_fps: float | None = None,
        max_video_frames: int | None = None,
    ) -> list[dict[str, Any]]:
        """Build the Qwen-Omni conversation for one normalized record."""

    def resolve_audio(
        self,
        video_root: Path,
        row: dict[str, Any],
        audio_root: Path | None,
    ) -> Path | None:
        return None

    def record_metadata(self, row: dict[str, Any]) -> dict[str, Any]:
        return {}

    def aggregate(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        return {}


class WorldSenseAdapter(BenchmarkAdapter):
    name = "worldsense"
    annotation_relative = "WorldSense/data/test-00000-of-00001.parquet"
    video_root_relative = "WorldSense"

    def load_rows(self, annotation: Path) -> list[dict[str, Any]]:
        from r2omni.common.worldsense import load_worldsense_rows

        return load_worldsense_rows(annotation)

    def resolve_video(self, video_root: Path, row: dict[str, Any]) -> str:
        from r2omni.common.worldsense import resolve_video_path

        return resolve_video_path(video_root, row)

    def resolve_audio(
        self,
        video_root: Path,
        row: dict[str, Any],
        audio_root: Path | None,
    ) -> Path | None:
        from r2omni.common.worldsense import resolve_audio_path

        return resolve_audio_path(video_root, row, audio_root)

    def build_message(
        self,
        row: dict[str, Any],
        video_root: Path,
        nframes: int,
        audio_root: Path | None,
        *,
        include_audio: bool,
        video_resolution_policy: str = "benchmark_cap",
        video_fps: float | None = None,
        max_video_frames: int | None = None,
    ) -> list[dict[str, Any]]:
        from r2omni.common.worldsense import build_message

        return build_message(
            row,
            video_root,
            nframes,
            audio_root,
            include_audio=include_audio,
        )


class VideoMMEAdapter(WorldSenseAdapter):
    name = "video_mme"
    aliases = ("videomme",)
    annotation_relative = "Video-MME/videomme/test-00000-of-00001.parquet"
    video_root_relative = "Video-MME"

    def load_rows(self, annotation: Path) -> list[dict[str, Any]]:
        from r2omni.common.videomme import load_videomme_rows

        return load_videomme_rows(annotation)

    def record_metadata(self, row: dict[str, Any]) -> dict[str, Any]:
        from r2omni.common.videomme import videomme_record_metadata

        return videomme_record_metadata(row)

    def aggregate(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        from r2omni.common.videomme import aggregate_videomme_groups

        return aggregate_videomme_groups(records)


class DailyOmniAdapter(BenchmarkAdapter):
    name = "daily_omni"
    aliases = ("daily-omni",)
    annotation_relative = "Daily-Omni/qa.json"
    video_root_relative = "Daily-Omni"
    prediction_mode = "full_choice"

    def load_rows(self, annotation: Path) -> list[dict[str, Any]]:
        from r2omni.common.daily_omni import load_daily_omni_rows

        return load_daily_omni_rows(annotation)

    def resolve_video(self, video_root: Path, row: dict[str, Any]) -> str:
        from r2omni.common.daily_omni import resolve_video_path

        return resolve_video_path(video_root, row)

    def resolve_audio(
        self,
        video_root: Path,
        row: dict[str, Any],
        audio_root: Path | None,
    ) -> Path | None:
        from r2omni.common.daily_omni import resolve_audio_path

        return resolve_audio_path(video_root, row)

    def build_message(
        self,
        row: dict[str, Any],
        video_root: Path,
        nframes: int,
        audio_root: Path | None,
        *,
        include_audio: bool,
        video_resolution_policy: str = "benchmark_cap",
        video_fps: float | None = None,
        max_video_frames: int | None = None,
    ) -> list[dict[str, Any]]:
        from r2omni.common.daily_omni import build_message

        return build_message(
            row, video_root, nframes, include_audio=include_audio
        )

    def record_metadata(self, row: dict[str, Any]) -> dict[str, Any]:
        from r2omni.common.daily_omni import daily_omni_record_metadata

        return daily_omni_record_metadata(row)

    def aggregate(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        from r2omni.common.daily_omni import aggregate_daily_omni_groups

        return aggregate_daily_omni_groups(records)


class VideoSALMONN2Adapter(BenchmarkAdapter):
    name = "video_salmonn2"
    aliases = ("videosalmonn2", "video-salmonn2")
    annotation_relative = "videoSALMONN2/video_salmonn2_test.json"
    video_root_relative = "videoSALMONN2"
    prediction_mode = "text"
    embedded_audio = True

    def load_rows(self, annotation: Path) -> list[dict[str, Any]]:
        from r2omni.common.videosalmonn2 import load_videosalmonn2_rows

        return load_videosalmonn2_rows(annotation)

    def resolve_video(self, video_root: Path, row: dict[str, Any]) -> str:
        from r2omni.common.videosalmonn2 import resolve_video_path

        return resolve_video_path(video_root, row)

    def build_message(
        self,
        row: dict[str, Any],
        video_root: Path,
        nframes: int,
        audio_root: Path | None,
        *,
        include_audio: bool,
        video_resolution_policy: str = "benchmark_cap",
        video_fps: float | None = None,
        max_video_frames: int | None = None,
    ) -> list[dict[str, Any]]:
        from r2omni.common.videosalmonn2 import build_message

        return build_message(
            row,
            video_root,
            nframes,
            resolution_policy=video_resolution_policy,
            video_fps=video_fps,
            max_video_frames=max_video_frames,
        )

    def record_metadata(self, row: dict[str, Any]) -> dict[str, Any]:
        from r2omni.common.videosalmonn2 import videosalmonn2_record_metadata

        return videosalmonn2_record_metadata(row)


class LongVALEAdapter(BenchmarkAdapter):
    name = "longvale"
    annotation_relative = "LongVALE/longvale-annotations-eval.json"
    video_root_relative = "LongVALE"
    prediction_mode = "text"
    embedded_audio = True

    def load_rows(self, annotation: Path) -> list[dict[str, Any]]:
        from r2omni.common.longvale import load_longvale_rows

        return load_longvale_rows(annotation)

    def resolve_video(self, video_root: Path, row: dict[str, Any]) -> str:
        from r2omni.common.longvale import resolve_video_path

        return resolve_video_path(video_root, row)

    def build_message(
        self,
        row: dict[str, Any],
        video_root: Path,
        nframes: int,
        audio_root: Path | None,
        *,
        include_audio: bool,
        video_resolution_policy: str = "benchmark_cap",
        video_fps: float | None = None,
        max_video_frames: int | None = None,
    ) -> list[dict[str, Any]]:
        from r2omni.common.longvale import build_message

        return build_message(row, video_root, nframes)

    def record_metadata(self, row: dict[str, Any]) -> dict[str, Any]:
        from r2omni.common.longvale import longvale_record_metadata

        return longvale_record_metadata(row)


class OmniDCBenchAdapter(BenchmarkAdapter):
    name = "omnidcbench"
    aliases = ("omnidc", "omni_dc_bench")
    annotation_relative = "OmniDCBench/ours_gt_file.json"
    video_root_relative = "OmniDCBench"
    prediction_mode = "text"
    embedded_audio = True

    def load_rows(self, annotation: Path) -> list[dict[str, Any]]:
        from r2omni.common.omnidcbench import load_omnidcbench_rows

        return load_omnidcbench_rows(annotation)

    def resolve_video(self, video_root: Path, row: dict[str, Any]) -> str:
        from r2omni.common.omnidcbench import resolve_video_path

        return resolve_video_path(video_root, row)

    def build_message(
        self,
        row: dict[str, Any],
        video_root: Path,
        nframes: int,
        audio_root: Path | None,
        *,
        include_audio: bool,
        video_resolution_policy: str = "benchmark_cap",
        video_fps: float | None = None,
        max_video_frames: int | None = None,
    ) -> list[dict[str, Any]]:
        from r2omni.common.omnidcbench import build_message

        return build_message(row, video_root, nframes)

    def record_metadata(self, row: dict[str, Any]) -> dict[str, Any]:
        from r2omni.common.omnidcbench import omnidcbench_record_metadata

        return omnidcbench_record_metadata(row)


class AVSDAdapter(BenchmarkAdapter):
    name = "avsd"
    # The official test media URL is no longer publicly readable.  The
    # released validation split has identical 10-turn task structure and local
    # references, so small-sample reproducibility uses its final dialog turn.
    annotation_relative = "AVSD/AVSD/valid_set4DSTC7-AVSD.json"
    video_root_relative = "AVSD/Charades_v1_480"
    prediction_mode = "text"
    embedded_audio = True

    def load_rows(self, annotation: Path) -> list[dict[str, Any]]:
        from r2omni.common.avsd import load_avsd_rows

        return load_avsd_rows(annotation)

    def resolve_video(self, video_root: Path, row: dict[str, Any]) -> str:
        from r2omni.common.avsd import resolve_video_path

        return resolve_video_path(video_root, row)

    def build_message(
        self,
        row: dict[str, Any],
        video_root: Path,
        nframes: int,
        audio_root: Path | None,
        *,
        include_audio: bool,
        video_resolution_policy: str = "benchmark_cap",
        video_fps: float | None = None,
        max_video_frames: int | None = None,
    ) -> list[dict[str, Any]]:
        from r2omni.common.avsd import build_message

        return build_message(row, video_root, nframes)

    def record_metadata(self, row: dict[str, Any]) -> dict[str, Any]:
        from r2omni.common.avsd import avsd_record_metadata

        return avsd_record_metadata(row)


_REGISTRY: dict[str, BenchmarkAdapter] = {}


def register_benchmark(adapter: BenchmarkAdapter, *, replace: bool = False) -> None:
    """Register a built-in or downstream adapter under its name and aliases."""

    keys: Iterable[str] = (adapter.name, *adapter.aliases)
    normalized = tuple(str(key).strip().lower() for key in keys)
    if any(not key for key in normalized):
        raise ValueError("benchmark names and aliases must be non-empty")
    collisions = [key for key in normalized if key in _REGISTRY]
    if collisions and not replace:
        raise ValueError(f"benchmark name already registered: {collisions[0]}")
    for key in normalized:
        _REGISTRY[key] = adapter


def get_benchmark(name: str) -> BenchmarkAdapter:
    key = str(name).strip().lower()
    try:
        return _REGISTRY[key]
    except KeyError as exc:
        raise ValueError(
            f"unsupported benchmark {name!r}; available: {', '.join(benchmark_names())}"
        ) from exc


def benchmark_names() -> tuple[str, ...]:
    return tuple(sorted({adapter.name for adapter in _REGISTRY.values()}))


for _adapter in (
    WorldSenseAdapter(),
    VideoMMEAdapter(),
    DailyOmniAdapter(),
    VideoSALMONN2Adapter(),
    LongVALEAdapter(),
    OmniDCBenchAdapter(),
    AVSDAdapter(),
):
    register_benchmark(_adapter)
