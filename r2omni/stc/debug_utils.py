from __future__ import annotations

from typing import Any


def log_stc_summary(metadata: dict[str, Any], _config: dict[str, Any]) -> None:
    summary = metadata["summary"]
    original = int(summary["original_seq_len"])
    kept = int(summary["new_seq_len"])
    video_original = int(summary["video_original_tokens"])
    video_kept = int(summary["video_kept_tokens"])
    audio_original = int(summary["audio_original_tokens"])
    audio_kept = int(summary["audio_kept_tokens"])
    print(
        "[STC] "
        f"tokens {original}->{kept} ({kept / original:.3f}), "
        f"video {video_original}->{video_kept}, "
        f"audio {audio_original}->{audio_kept}"
    )
