"""Aggregation utilities converting track states into segments and totals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from screentime.types import TrackState


@dataclass
class Segment:
    track_id: int
    label: str
    start_ms: float
    end_ms: float
    duration_ms: float

    def to_dict(self) -> Dict:
        return {
            "track_id": self.track_id,
            "label": self.label,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duration_ms": self.duration_ms,
        }


def _frames_to_segments(
    track: TrackState,
    fps: float,
    max_gap_ms: float,
    min_run_ms: float,
) -> List[Segment]:
    if not track.frames or track.label is None:
        return []

    gap_frames = int(round((max_gap_ms / 1000.0) * fps))
    min_run_frames = int(round((min_run_ms / 1000.0) * fps))

    segments: List[Segment] = []
    start_idx = 0
    frames = track.frames
    timestamps = track.timestamps_ms

    for i in range(1, len(frames)):
        if frames[i] - frames[i - 1] <= gap_frames:
            continue
        seg = _create_segment(track, start_idx, i - 1, fps)
        if seg.duration_ms >= min_run_ms and (frames[i - 1] - frames[start_idx] + 1) >= min_run_frames:
            segments.append(seg)
        start_idx = i

    # final segment
    seg = _create_segment(track, start_idx, len(frames) - 1, fps)
    if seg.duration_ms >= min_run_ms and (frames[-1] - frames[start_idx] + 1) >= min_run_frames:
        segments.append(seg)

    return segments


def _create_segment(track: TrackState, start_idx: int, end_idx: int, fps: float) -> Segment:
    start_frame = track.frames[start_idx]
    end_frame = track.frames[end_idx]
    start_ts = track.timestamps_ms[start_idx]
    end_ts = track.timestamps_ms[end_idx]
    duration_ms = (end_frame - start_frame + 1) / fps * 1000.0
    return Segment(track_id=track.track_id, label=track.label or "", start_ms=start_ts, end_ms=end_ts, duration_ms=duration_ms)


def tracks_to_segments(
    tracks: Iterable[TrackState],
    fps: float,
    max_gap_ms: float,
    min_run_ms: float,
) -> List[Segment]:
    segments: List[Segment] = []
    for track in tracks:
        segments.extend(_frames_to_segments(track, fps, max_gap_ms, min_run_ms))
    return segments


def segments_to_totals(segments: Iterable[Segment]) -> pd.DataFrame:
    data = [seg.to_dict() for seg in segments]
    if not data:
        return pd.DataFrame(columns=["label", "duration_ms"])
    df = pd.DataFrame(data)
    totals = df.groupby("label")["duration_ms"].sum().reset_index().sort_values("duration_ms", ascending=False)
    return totals


def segments_to_timeline(segments: Iterable[Segment], fps: float, video_duration_ms: float) -> pd.DataFrame:
    timeline: Dict[str, List[int]] = {}
    total_seconds = int(np.ceil(video_duration_ms / 1000.0))
    timeline_seconds = np.zeros((total_seconds,), dtype=np.int32)

    # build second resolution presence map per label
    label_presence: Dict[str, np.ndarray] = {}
    for seg in segments:
        start_sec = int(np.floor(seg.start_ms / 1000.0))
        end_sec = int(np.ceil(seg.end_ms / 1000.0))
        arr = label_presence.setdefault(seg.label, np.zeros((total_seconds,), dtype=np.int32))
        arr[start_sec:end_sec] = 1

    if not label_presence:
        return pd.DataFrame({"second": np.arange(total_seconds)})

    data = {"second": np.arange(total_seconds)}
    for label, arr in label_presence.items():
        data[label] = arr
    return pd.DataFrame(data)
