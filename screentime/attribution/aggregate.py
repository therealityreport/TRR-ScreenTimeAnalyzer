"""Aggregation utilities converting track states into segments and totals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from screentime.types import Subtrack, TrackState


@dataclass
class Segment:
    byte_track_id: int
    subtrack_id: int
    label: str
    start_ms: float
    end_ms: float
    duration_ms: float
    frames: int
    avg_similarity: float

    def to_dict(self) -> Dict:
        return {
            "byte_track_id": self.byte_track_id,
            "subtrack_id": self.subtrack_id,
            "label": self.label,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duration_ms": self.duration_ms,
            "frames": self.frames,
            "avg_similarity": self.avg_similarity,
        }


def _subtrack_to_segments(
    track_id: int,
    subtrack_idx: int,
    subtrack: Subtrack,
    fps: float,
    max_gap_ms: float,
    min_run_ms: float,
) -> List[Segment]:
    """Convert a finalized subtrack window into a single segment if it passes thresholds."""
    import logging
    LOGGER = logging.getLogger(__name__)

    if not subtrack.label:
        LOGGER.debug(
            "      Skipping subtrack %d.%d: no label",
            track_id,
            subtrack_idx,
        )
        return []

    start_frame = subtrack.start_frame
    end_frame = subtrack.end_frame
    if end_frame < start_frame:
        LOGGER.debug(
            "      Skipping subtrack %d.%d: invalid window (start=%d end=%d)",
            track_id,
            subtrack_idx,
            start_frame,
            end_frame,
        )
        return []

    frame_count = max(1, end_frame - start_frame + 1)
    if fps <= 0:
        duration_ms = 0.0
        start_ms = 0.0
    else:
        duration_ms = frame_count / fps * 1000.0
        start_ms = start_frame / fps * 1000.0
    end_ms = start_ms + duration_ms
    if duration_ms < min_run_ms:
        LOGGER.info(
            "      Filtered (too_short) subtrack %d.%d: %s window %d→%d duration_ms=%.1f (need %.1f)",
            track_id,
            subtrack_idx,
            subtrack.label,
            start_frame,
            end_frame,
            duration_ms,
            min_run_ms,
        )
        return []

    avg_similarity = subtrack.avg_similarity
    if np.isnan(avg_similarity):
        avg_similarity = 0.0

    segment = Segment(
        byte_track_id=track_id,
        subtrack_id=subtrack_idx,
        label=subtrack.label or "UNKNOWN",
        start_ms=start_ms,
        end_ms=end_ms,
        duration_ms=duration_ms,
        frames=frame_count,
        avg_similarity=avg_similarity,
    )
    LOGGER.info(
        "      Appended segment %d.%d: %s window %d→%d duration_ms=%.1f labelled_frames=%d",
        track_id,
        subtrack_idx,
        segment.label,
        start_frame,
        end_frame,
        duration_ms,
        len(subtrack.frame_scores),
    )
    return [segment]


def _labels_to_segments(
    track: TrackState,
    fps: float,
    max_gap_ms: float,
    min_run_ms: float,
) -> List[Segment]:
    """Derive segments directly from per-frame label assignments."""
    import logging

    LOGGER = logging.getLogger(__name__)
    if not track.label_scores:
        return []

    entries = sorted(track.label_scores.items())
    max_gap_frames = int(round((max_gap_ms / 1000.0) * fps)) if fps > 0 else 0
    min_run_frames = int(round((min_run_ms / 1000.0) * fps)) if fps > 0 else 0

    segments: List[Segment] = []
    current_label: Optional[str] = None
    run_frames: List[int] = []
    run_scores: List[float] = []

    def finalize_run() -> None:
        if not run_frames or current_label is None:
            return
        start_frame = run_frames[0]
        end_frame = run_frames[-1]
        frame_span = max(1, end_frame - start_frame + 1)
        duration_ms = frame_span / fps * 1000.0 if fps > 0 else 0.0
        if duration_ms < min_run_ms or frame_span < max(1, min_run_frames):
            LOGGER.info(
                "      Filtered (labels) %d: %s window %d→%d duration_ms=%.1f (need %.1f)",
                track.track_id,
                current_label,
                start_frame,
                end_frame,
                duration_ms,
                min_run_ms,
            )
            return
        avg_similarity = float(np.mean(run_scores)) if run_scores else float("nan")
        start_ms = start_frame / fps * 1000.0 if fps > 0 else 0.0
        duration_ms = frame_span / fps * 1000.0 if fps > 0 else 0.0
        end_ms = start_ms + duration_ms
        segment = Segment(
            byte_track_id=track.byte_track_id,
            subtrack_id=len(segments),
            label=current_label or "UNKNOWN",
            start_ms=start_ms,
            end_ms=end_ms,
            duration_ms=duration_ms,
            frames=frame_span,
            avg_similarity=avg_similarity if not np.isnan(avg_similarity) else 0.0,
        )
        LOGGER.info(
            "      Appended fallback segment %d.%d: %s frames=%d duration_ms=%.1f",
            segment.byte_track_id,
            segment.subtrack_id,
            segment.label,
            frame_span,
            duration_ms,
        )
        segments.append(segment)

    prev_frame: Optional[int] = None
    for frame_idx, (label, score) in entries:
        if current_label is None:
            current_label = label
            run_frames = [frame_idx]
            run_scores = [score]
            prev_frame = frame_idx
            continue

        frame_gap = (frame_idx - prev_frame) if prev_frame is not None else 0
        if label == current_label and (max_gap_frames <= 0 or frame_gap <= max_gap_frames):
            run_frames.append(frame_idx)
            run_scores.append(score)
            prev_frame = frame_idx
            continue

        finalize_run()
        current_label = label
        run_frames = [frame_idx]
        run_scores = [score]
        prev_frame = frame_idx

    finalize_run()
    return segments


def _frames_to_segments(
    track: TrackState,
    fps: float,
    max_gap_ms: float,
    min_run_ms: float,
) -> List[Segment]:
    """Legacy method: convert track with single label to segments."""
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
    """Legacy method: create segment from track."""
    start_frame = track.frames[start_idx]
    end_frame = track.frames[end_idx]
    start_ts = track.timestamps_ms[start_idx]
    if fps <= 0:
        duration_ms = 0.0
    else:
        duration_ms = (end_frame - start_frame + 1) / fps * 1000.0
    end_ts = start_ts + duration_ms
    frame_count = end_frame - start_frame + 1
    return Segment(
        byte_track_id=track.track_id,
        subtrack_id=0,
        label=track.label or "UNKNOWN",
        start_ms=start_ts,
        end_ms=end_ts,
        duration_ms=duration_ms,
        frames=frame_count,
        avg_similarity=0.0,
    )


def tracks_to_segments(
    tracks: Iterable[TrackState],
    fps: float,
    max_gap_ms: float,
    min_run_ms: float,
    use_subtracks: bool = True,
) -> List[Segment]:
    """Convert tracks to segments, using subtracks if available."""
    import logging
    LOGGER = logging.getLogger(__name__)

    segments: List[Segment] = []
    for track in tracks:
        track_segments: List[Segment] = []
        track_id = getattr(track, "byte_track_id", track.track_id)

        if use_subtracks and track.subtracks:
            LOGGER.info(
                "Track %d: Processing %d subtracks (final_label=%s)",
                track_id,
                len(track.subtracks),
                track.label,
            )
            for idx, subtrack in enumerate(track.subtracks):
                LOGGER.info(
                    "  Subtrack %d.%d: %s (frames %d-%d, %d labeled frames)",
                    track_id,
                    idx,
                    subtrack.label,
                    subtrack.start_frame,
                    subtrack.end_frame,
                    len(subtrack.frame_scores),
                )
                segs = _subtrack_to_segments(track_id, idx, subtrack, fps, max_gap_ms, min_run_ms)
                LOGGER.info("    → Generated %d segments", len(segs))
                if not segs:
                    duration_ms = (
                        (subtrack.end_frame - subtrack.start_frame + 1) / fps * 1000.0
                        if fps > 0
                        else 0.0
                    )
                    LOGGER.warning(
                        "    → FILTERED OUT: duration_ms=%.1f, min_run_ms=%.1f",
                        duration_ms,
                        min_run_ms,
                    )
                track_segments.extend(segs)
            if not track_segments and track.label_scores:
                LOGGER.info(
                    "Track %d: Subtracks produced no segments; falling back to label_scores (%d labelled frames)",
                    track_id,
                    len(track.label_scores),
                )
                track_segments.extend(_labels_to_segments(track, fps, max_gap_ms, min_run_ms))
        elif track.label_scores:
            LOGGER.info(
                "Track %d: Deriving segments from label_scores (%d labelled frames)",
                track_id,
                len(track.label_scores),
            )
            track_segments.extend(_labels_to_segments(track, fps, max_gap_ms, min_run_ms))
        elif track.label and track.frames:
            LOGGER.info(
                "Track %d: Using track-level label %s for fallback segment conversion (%d frames)",
                track_id,
                track.label,
                len(track.frames),
            )
            track_segments.extend(_frames_to_segments(track, fps, max_gap_ms, min_run_ms))
        else:
            LOGGER.debug(
                "Track %d: No subtracks or label_scores available; skipping segment conversion",
                track_id,
            )

        segments.extend(track_segments)

    deduped: Dict[tuple, Segment] = {}
    for seg in segments:
        key = (
            seg.byte_track_id,
            seg.subtrack_id,
            seg.label,
            round(seg.start_ms),
            round(seg.end_ms),
        )
        if key not in deduped:
            deduped[key] = seg
    return list(deduped.values())


def segments_to_totals(segments: Iterable[Segment]) -> pd.DataFrame:
    data = [seg.to_dict() for seg in segments]
    if not data:
        return pd.DataFrame(columns=["label", "duration_ms"])
    df = pd.DataFrame(data)
    totals = df.groupby("label")["duration_ms"].sum().reset_index().sort_values("duration_ms", ascending=False)
    return totals


def segments_to_timeline(segments: Iterable[Segment], fps: float, video_duration_ms: float) -> pd.DataFrame:
    """Create per-second timeline showing which labels are present."""
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


def segments_to_track_timeline(segments: Iterable[Segment], video_duration_ms: float) -> pd.DataFrame:
    """Create per-second timeline showing track_id, subtrack_id, and label."""
    total_seconds = int(np.ceil(video_duration_ms / 1000.0))
    
    # Build list of (second, byte_track_id, subtrack_id, label) tuples
    rows = []
    for seg in segments:
        start_sec = int(np.floor(seg.start_ms / 1000.0))
        end_sec = int(np.ceil(seg.end_ms / 1000.0))
        for sec in range(start_sec, end_sec):
            if sec < total_seconds:
                rows.append({
                    "second": sec,
                    "byte_track_id": seg.byte_track_id,
                    "subtrack_id": seg.subtrack_id,
                    "label": seg.label,
                })
    
    if not rows:
        return pd.DataFrame(columns=["second", "byte_track_id", "subtrack_id", "label"])
    
    return pd.DataFrame(rows)
