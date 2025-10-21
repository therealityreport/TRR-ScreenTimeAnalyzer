from pathlib import Path

from screentime.harvest.harvest import HarvestConfig, TrackSamplingState
from screentime.types import FaceSample


def _make_sample(track_id: int, frame_idx: int, frontalness: float, sharpness: float, quality: float) -> FaceSample:
    return FaceSample(
        track_id=track_id,
        frame_idx=frame_idx,
        timestamp_ms=frame_idx * 33.0,
        path=Path(f"sample_{track_id}_{frame_idx}.jpg"),
        score=0.9,
        bbox=(0.0, 0.0, 10.0, 10.0),
        quality=quality,
        sharpness=sharpness,
        orientation="frontal" if frontalness >= 0.5 else "profile",
        frontalness=frontalness,
        area_frac=0.1,
    )


def test_profile_bucket_admits_subthreshold_faces():
    config = HarvestConfig(
        samples_per_track=2,
        samples_min=1,
        min_gap_frames=0,
        min_frontalness=0.5,
        profile_quota=1,
        profile_min_frontalness=0.1,
    )

    state = TrackSamplingState()
    frontal_sample = _make_sample(track_id=1, frame_idx=0, frontalness=0.8, sharpness=40.0, quality=0.9)
    profile_sample = _make_sample(track_id=1, frame_idx=5, frontalness=0.2, sharpness=42.0, quality=0.8)

    state.add_candidate(frontal_sample, frontal_sample.quality, frontal_sample.sharpness, frontal_sample.frontalness, frontal_sample.area_frac, frontal_sample.orientation, config)
    state.add_candidate(profile_sample, profile_sample.quality, profile_sample.sharpness, profile_sample.frontalness, profile_sample.area_frac, profile_sample.orientation, config)

    selected = state.export_samples(config)

    assert len(selected) == 2, "Profile quota should allow filling the second slot."
    assert any(abs(sample.frontalness - 0.2) < 1e-6 for sample in selected)

    reasons = {candidate.sample.frame_idx: candidate.reason for candidate in state.candidates}
    assert reasons[5] == "picked_profile"
