from screentime.attribution.aggregate import Segment
from screentime.attribution.coverage import (
    merge_segments_with_fallback,
    missing_labels_with_coverage,
)
from screentime.types import TrackState


def make_segment(track_id: int, subtrack_id: int, label: str, start: float, end: float) -> Segment:
    return Segment(
        byte_track_id=track_id,
        subtrack_id=subtrack_id,
        label=label,
        start_ms=start,
        end_ms=end,
        duration_ms=end - start,
        frames=10,
        avg_similarity=0.9,
    )


def test_missing_labels_with_coverage_identifies_labels_without_segments():
    track_a = TrackState(track_id=1)
    track_a.add_label(frame_idx=5, label="LVP", score=0.88)

    track_b = TrackState(track_id=2)
    track_b.add_label(frame_idx=10, label="KYLE", score=0.93)

    segments = [make_segment(2, 0, "KYLE", 0.0, 800.0)]

    missing = missing_labels_with_coverage(
        tracks=[track_a, track_b],
        segments=segments,
        known_labels={"LVP", "KYLE", "EILEEN"},
    )

    assert missing == {"LVP"}


def test_merge_segments_with_fallback_appends_unique_segments_per_track():
    base = [make_segment(1, 0, "KYLE", 0.0, 1000.0)]
    fallback = [
        make_segment(1, 0, "LVP", 1200.0, 1600.0),
        make_segment(1, 0, "LVP", 1200.0, 1600.0),  # duplicate should be ignored
    ]

    merged, added = merge_segments_with_fallback(
        base_segments=base,
        fallback_segments=fallback,
        eligible_labels={"LVP"},
    )

    assert len(merged) == 2
    assert len(added) == 1
    assert added[0].label == "LVP"
    assert added[0].subtrack_id == 1  # sequential assignment for track 1
