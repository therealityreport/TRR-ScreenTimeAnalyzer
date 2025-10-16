"""Utilities for analysing and repairing label coverage in tracker output."""

from __future__ import annotations

from dataclasses import replace
from typing import List, Optional, Sequence, Set, Tuple

from screentime.attribution.aggregate import Segment
from screentime.types import TrackState


def _track_labels(track: TrackState) -> Set[str]:
    labels: Set[str] = set()
    if track.label:
        labels.add(track.label)
    for _, (label, _score) in track.label_scores.items():
        if label and label != "UNKNOWN":
            labels.add(label)
    for subtrack in track.subtracks:
        if subtrack.label and subtrack.label != "UNKNOWN":
            labels.add(subtrack.label)
    return labels


def missing_labels_with_coverage(
    tracks: Sequence[TrackState],
    segments: Sequence[Segment],
    known_labels: Optional[Set[str]] = None,
) -> Set[str]:
    """Return labels that have labelled frames but no exported segments."""

    covered_labels: Set[str] = {
        seg.label
        for seg in segments
        if seg.label and seg.label != "UNKNOWN"
    }

    observed: Set[str] = set()
    for track in tracks:
        observed.update(_track_labels(track))

    if known_labels is not None:
        observed.intersection_update(known_labels)

    return observed - covered_labels


def merge_segments_with_fallback(
    *,
    base_segments: Sequence[Segment],
    fallback_segments: Sequence[Segment],
    eligible_labels: Set[str],
) -> Tuple[List[Segment], List[Segment]]:
    """Merge fallback segments into the base set, avoiding duplicates."""

    merged: List[Segment] = list(base_segments)
    added: List[Segment] = []

    seen_keys = {
        (seg.byte_track_id, seg.label, round(seg.start_ms, 3), round(seg.end_ms, 3))
        for seg in merged
        if seg.label
    }

    segments_by_track: dict[int, List[Segment]] = {}
    for seg in merged:
        segments_by_track.setdefault(seg.byte_track_id, []).append(seg)

    for seg in fallback_segments:
        if not seg.label or seg.label == "UNKNOWN" or seg.label not in eligible_labels:
            continue
        key = (seg.byte_track_id, seg.label, round(seg.start_ms, 3), round(seg.end_ms, 3))
        if key in seen_keys:
            continue

        track_segments = segments_by_track.setdefault(seg.byte_track_id, [])
        next_subtrack_id = (
            max((existing.subtrack_id for existing in track_segments), default=-1) + 1
        )
        replacement = replace(seg, subtrack_id=next_subtrack_id)

        merged.append(replacement)
        added.append(replacement)
        track_segments.append(replacement)
        seen_keys.add(key)

    return merged, added


__all__ = [
    "merge_segments_with_fallback",
    "missing_labels_with_coverage",
]

