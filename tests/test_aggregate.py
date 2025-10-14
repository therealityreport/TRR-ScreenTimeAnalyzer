from screentime.attribution.aggregate import tracks_to_segments, segments_to_totals
from screentime.types import TrackState


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

    segments = tracks_to_segments([track], fps=fps, max_gap_ms=500, min_run_ms=200)
    assert len(segments) == 1
    totals = segments_to_totals(segments)
    assert totals.iloc[0]["label"] == "Alice"
