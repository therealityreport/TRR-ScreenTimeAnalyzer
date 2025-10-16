"""Helpers for associating embeddings with track states."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from screentime.recognition.matcher import TrackVotingMatcher
from screentime.types import Subtrack, TrackState


def apply_embedding(
    track: TrackState,
    embedding: np.ndarray,
    matcher: TrackVotingMatcher,
    frame_idx: Optional[int] = None,
) -> Optional[str]:
    """Apply a face embedding to a track and return current label (if any)."""

    matcher.update_track(track, embedding, frame_idx)
    return track.label


def finalize_label(track: TrackState) -> Optional[str]:
    """Return the strongest label for a track when processing is complete."""

    if track.label:
        return track.label
    if not track.label_votes:
        return None
    label, _ = max(track.label_votes.items(), key=lambda item: item[1])
    return label


def finalize_track_subtracks(track: TrackState, matcher: TrackVotingMatcher) -> None:
    """Finalize any remaining subtrack for a completed track."""

    if matcher.identity_split_enabled and track.current_subtrack_label:
        matcher._finalize_subtrack(track, track.current_subtrack_label)
        track.current_subtrack_label = None
        track.current_subtrack_start_frame = None

    if track.subtracks or not track.label_scores:
        return

    # Build subtracks directly from recorded label assignments.
    sorted_frames = sorted(track.label_scores.items())
    if not sorted_frames:
        return

    current_label: Optional[str] = None
    current_start: Optional[int] = None
    current_scores: Dict[int, float] = {}
    prev_frame_idx: Optional[int] = None
    subtracks: list[Subtrack] = []

    for frame_idx, (label, score) in sorted_frames:
        if current_label is None:
            current_label = label
            current_start = frame_idx
            current_scores = {frame_idx: score}
            prev_frame_idx = frame_idx
            continue

        if label == current_label:
            current_scores[frame_idx] = score
            prev_frame_idx = frame_idx
            continue

        end_frame = prev_frame_idx if prev_frame_idx is not None else frame_idx
        subtracks.append(
            Subtrack(
                start_frame=current_start if current_start is not None else frame_idx,
                end_frame=end_frame,
                label=current_label,
                frame_scores=dict(current_scores),
            )
        )
        current_label = label
        current_start = frame_idx
        current_scores = {frame_idx: score}
        prev_frame_idx = frame_idx

    if current_label is not None and current_start is not None:
        end_frame = prev_frame_idx if prev_frame_idx is not None else current_start
        subtracks.append(
            Subtrack(
                start_frame=current_start,
                end_frame=end_frame,
                label=current_label,
                frame_scores=dict(current_scores),
            )
        )

    track.subtracks.extend(subtracks)
