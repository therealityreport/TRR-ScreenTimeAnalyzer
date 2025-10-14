"""Helpers for associating embeddings with track states."""

from __future__ import annotations

from typing import Optional

import numpy as np

from screentime.recognition.matcher import TrackVotingMatcher
from screentime.types import TrackState


def apply_embedding(
    track: TrackState,
    embedding: np.ndarray,
    matcher: TrackVotingMatcher,
) -> Optional[str]:
    """Apply a face embedding to a track and return current label (if any)."""

    matcher.update_track(track, embedding)
    return track.label


def finalize_label(track: TrackState) -> Optional[str]:
    """Return the strongest label for a track when processing is complete."""

    if track.label:
        return track.label
    if not track.label_scores:
        return None
    label, _ = max(track.label_scores.items(), key=lambda item: item[1])
    return label
