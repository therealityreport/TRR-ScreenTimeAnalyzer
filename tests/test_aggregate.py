import pytest

from screentime.attribution.aggregate import segments_to_totals, tracks_to_segments
from screentime.types import Subtrack, TrackState


def build_track(track_id: int, label: str, frames: range, fps: float) -> TrackState:
    track = TrackState(track_id=track_id, label=label)
    for frame in frames:
        track.add_observation(frame_idx=frame, bbox=(0, 0, 10, 10), score=0.9, timestamp_ms=frame / fps * 1000.0)
    return track


def test_segments_bridge_short_gaps():
    fps = 30.0
    track = TrackState(track_id=1, label="Alice")
    frames = list(range(0, 30)) + list(range(35, 60))
    for frame in frames:
        track.add_observation(frame, (0, 0, 10, 10), 0.8, frame / fps * 1000.0)
        track.add_label(frame, "Alice", 0.9)

    segments = tracks_to_segments([track], fps=fps, max_gap_ms=500, min_run_ms=200)
    assert len(segments) == 1
    totals = segments_to_totals(segments)
    assert totals.iloc[0]["label"] == "Alice"


def test_segments_keep_subtracks():
    fps = 30.0
    track = TrackState(track_id=42)
    for frame in range(0, 60):
        track.add_observation(frame, (0, 0, 10, 10), 0.9, frame / fps * 1000.0)

    subtrack_a = Subtrack(
        start_frame=0,
        end_frame=29,
        label="RINNA",
        frame_scores={0: 0.92, 10: 0.88, 20: 0.90},
    )
    subtrack_b = Subtrack(
        start_frame=30,
        end_frame=59,
        label="KYLE",
        frame_scores={30: 0.94, 45: 0.90},
    )
    track.subtracks = [subtrack_a, subtrack_b]

    segments = tracks_to_segments(
        [track],
        fps=fps,
        max_gap_ms=300,
        min_run_ms=600,
        use_subtracks=True,
    )
    assert [seg.label for seg in segments] == ["RINNA", "KYLE"]
    assert [seg.subtrack_id for seg in segments] == [0, 1]
    assert segments[0].frames == 30
    assert segments[1].frames == 30
    assert segments[0].duration_ms == pytest.approx(1000.0, rel=1e-6)
    assert segments[1].duration_ms == pytest.approx(1000.0, rel=1e-6)


def test_segments_totals_consistency_with_multiple_identities():
    fps = 25.0
    track = TrackState(track_id=7)
    for frame in range(0, 90):
        track.add_observation(frame, (0, 0, 12, 12), 0.85, frame / fps * 1000.0)

    labels = [
        ("LVP", 0, 29, {0: 0.91, 10: 0.90}),
        ("BRANDI", 30, 59, {35: 0.88, 55: 0.89}),
        ("KYLE", 60, 89, {62: 0.93, 80: 0.92}),
    ]
    for idx, (label, start, end, scores) in enumerate(labels):
        track.subtracks.append(
            Subtrack(
                start_frame=start,
                end_frame=end,
                label=label,
                frame_scores=scores,
            )
        )

    segments = tracks_to_segments(
        [track],
        fps=fps,
        max_gap_ms=300,
        min_run_ms=600,
        use_subtracks=True,
    )
    assert len(segments) >= 3
    totals = segments_to_totals(segments)
    totals_map = {row["label"]: row["duration_ms"] for _, row in totals.iterrows()}
    for label, *_ in labels:
        seg_sum = sum(seg.duration_ms for seg in segments if seg.label == label)
        assert totals_map[label] == pytest.approx(seg_sum, rel=1e-6)
