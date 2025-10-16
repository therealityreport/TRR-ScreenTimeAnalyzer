from screentime.attribution import aggregate
from screentime.attribution.coverage import merge_segments_with_fallback, missing_labels_with_coverage
from screentime.types import TrackState


def build_track_with_labels(track_id: int, label: str, frame_count: int, fps: float) -> TrackState:
    track = TrackState(track_id=track_id)
    for frame in range(frame_count):
        timestamp_ms = frame / fps * 1000.0
        track.add_observation(frame, (0, 0, 10, 10), 0.9, timestamp_ms)
        track.add_label(frame, label, 0.95)
    return track


def test_missing_labels_detects_vote_only_tracks():
    track = TrackState(track_id=1)
    track.record_vote("ALICE", 1.25)

    missing = missing_labels_with_coverage(
        tracks=[track],
        segments=[],
        known_labels={"ALICE", "BOB"},
    )
    assert missing == {"ALICE"}


def test_merge_segments_recovers_short_runs():
    fps = 30.0
    track = build_track_with_labels(track_id=7, label="ALICE", frame_count=5, fps=fps)

    base_segments = aggregate.tracks_to_segments(
        [track],
        fps=fps,
        max_gap_ms=0,
        min_run_ms=500.0,
        use_subtracks=True,
    )
    assert not base_segments

    fallback_segments = aggregate.tracks_to_segments(
        [track],
        fps=fps,
        max_gap_ms=0,
        min_run_ms=0.0,
        use_subtracks=False,
    )
    assert len(fallback_segments) == 1

    missing = missing_labels_with_coverage(
        tracks=[track],
        segments=base_segments,
        known_labels={"ALICE"},
    )
    assert missing == {"ALICE"}

    merged, added = merge_segments_with_fallback(
        base_segments=base_segments,
        fallback_segments=fallback_segments,
        eligible_labels=missing,
    )

    assert len(merged) == 1
    assert len(added) == 1
    assert merged[0].label == "ALICE"

    merged_again, added_again = merge_segments_with_fallback(
        base_segments=merged,
        fallback_segments=fallback_segments,
        eligible_labels=missing,
    )
    assert len(added_again) == 0
    assert len(merged_again) == 1
