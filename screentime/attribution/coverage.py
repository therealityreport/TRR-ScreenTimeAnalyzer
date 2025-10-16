"""Helpers for bridging coverage gaps when exporting segments."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from screentime.attribution.aggregate import Segment
from screentime.types import TrackState


def missing_labels_with_coverage(
    tracks: Iterable[TrackState],
    segments: Sequence[Segment],
    known_labels: Optional[Set[str]] = None,
    min_label_frames: int = 1,
    min_vote_weight: float = 1e-3,
) -> Set[str]:
    """Return labels that have evidence but no exported segments."""

    covered_labels = {
        seg.label
        for seg in segments
        if seg.label and seg.label != "UNKNOWN"
    }

    frame_evidence: Dict[str, int] = {}
    vote_evidence: Dict[str, float] = {}

    for track in tracks:
        if track.label and track.label != "UNKNOWN":
            frame_evidence[track.label] = max(frame_evidence.get(track.label, 0), len(track.frames))

        for _, (label, _score) in track.label_scores.items():
            if not label or label == "UNKNOWN":
                continue
            frame_evidence[label] = frame_evidence.get(label, 0) + 1

        for label, weight in track.label_votes.items():
            if not label or label == "UNKNOWN":
                continue
            vote_evidence[label] = vote_evidence.get(label, 0.0) + float(weight)

    candidate_labels: Set[str] = set()
    for label, frame_count in frame_evidence.items():
        if frame_count >= min_label_frames:
            candidate_labels.add(label)

    for label, weight in vote_evidence.items():
        if weight >= min_vote_weight:
            candidate_labels.add(label)

    if known_labels is not None:
        candidate_labels &= known_labels

    return {label for label in candidate_labels if label not in covered_labels}


def merge_segments_with_fallback(
    base_segments: Sequence[Segment],
    fallback_segments: Sequence[Segment],
    eligible_labels: Set[str],
) -> Tuple[List[Segment], List[Segment]]:
    """Merge fallback segments for eligible labels, avoiding duplicates."""

    merged: List[Segment] = list(base_segments)
    added: List[Segment] = []

    def _key(seg: Segment) -> Tuple[int, str, int, int]:
        return (
            seg.byte_track_id,
            seg.label or "UNKNOWN",
            int(round(seg.start_ms)),
            int(round(seg.end_ms)),
        )

    seen = {_key(seg) for seg in merged}

    for seg in fallback_segments:
        if not seg.label or seg.label == "UNKNOWN":
            continue
        if seg.label not in eligible_labels:
            continue
        key = _key(seg)
        if key in seen:
            continue
        merged.append(seg)
        added.append(seg)
        seen.add(key)

    merged.sort(key=lambda s: (s.start_ms, s.byte_track_id, s.subtrack_id))
    return merged, added
