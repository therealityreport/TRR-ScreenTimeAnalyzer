from pathlib import Path

import pytest

pytest.importorskip("cv2")

from screentime.harvest.harvest import HarvestConfig, HarvestRunner, TrackSamplingState
from screentime.types import FaceSample


class DummyDetector:
    def detect(self, frame, frame_idx):  # pragma: no cover - stub
        return []


class DummyTracker:
    def set_frame_rate(self, fps):  # pragma: no cover
        pass


def build_runner() -> HarvestRunner:
    config = HarvestConfig()
    return HarvestRunner(DummyDetector(), DummyDetector(), DummyTracker(), config)


def test_best_k_with_spacing_prioritizes_quality():
    config = HarvestConfig(samples_per_track=3, min_gap_frames=5, samples_max=10)
    state = TrackSamplingState()

    def add_candidate(frame_idx: int, quality: float) -> None:
        sample = FaceSample(
            track_id=1,
            frame_idx=frame_idx,
            timestamp_ms=float(frame_idx * 40),
            path=Path(f"frame_{frame_idx}.jpg"),
            score=0.9,
            bbox=(0.0, 0.0, 10.0, 10.0),
            quality=quality,
            sharpness=200.0,
            orientation="frontal",
            frontalness=1.0,
            area_frac=config.target_area_frac,
        )
        state.add_candidate(sample, quality, 200.0, 1.0, config.target_area_frac, "frontal", config)

    add_candidate(0, 0.95)
    add_candidate(2, 0.94)   # should be skipped due to min gap
    add_candidate(10, 0.92)
    add_candidate(20, 0.80)
    add_candidate(30, 0.60)

    selected = state.export_samples(config)
    selected_frames = {sample.frame_idx for sample in selected}
    assert selected_frames == {0, 10, 20}

    debug_rows = list(state.iter_debug_rows(track_id=1))
    picked_frames = {row["frame"] for row in debug_rows if row["picked"]}
    assert picked_frames == selected_frames


def test_threshold_rejects_small_area_and_blur():
    runner = build_runner()
    runner.config.min_area_frac = 0.003
    runner.config.min_sharpness_laplacian = 120.0

    frame_area = 10000.0  # e.g., 100x100 frame
    small_bbox = (0.0, 0.0, 5.0, 5.0)  # area = 25 -> area_frac = 0.0025 < 0.003
    assert not runner._passes_area_threshold(small_bbox, frame_area)

    sharpness_good = 150.0
    sharpness_bad = 80.0
    assert runner._passes_sharpness_threshold(sharpness_good)
    assert not runner._passes_sharpness_threshold(sharpness_bad)


def test_frontalness_gate_marks_reason():
    config = HarvestConfig(min_frontalness=0.6)
    state = TrackSamplingState()
    sample = FaceSample(
        track_id=1,
        frame_idx=0,
        timestamp_ms=0.0,
        path=Path("/tmp/example.jpg"),
        score=0.9,
        bbox=(0, 0, 10, 10),
        quality=0.5,
        sharpness=200.0,
        orientation="profile",
        frontalness=0.2,
        area_frac=config.target_area_frac,
    )
    state.add_candidate(sample, 0.5, 200.0, 0.2, config.target_area_frac, "profile", config)
    exported = state.export_samples(config)
    assert not exported
    assert state.candidates[0].reason == "rejected_frontalness"


def test_sharpness_percentile_gate():
    config = HarvestConfig(min_sharpness_pct=50.0, min_frontalness=0.0)
    state = TrackSamplingState()

    def make_sample(idx: int, sharp: float) -> FaceSample:
        return FaceSample(
            track_id=1,
            frame_idx=idx,
            timestamp_ms=float(idx * 40),
            path=Path(f"/tmp/sample_{idx}.jpg"),
            score=0.9,
            bbox=(0, 0, 10, 10),
            quality=0.5 + idx * 0.1,
            sharpness=sharp,
            orientation="frontal",
            frontalness=1.0,
            area_frac=config.target_area_frac,
        )

    state.add_candidate(make_sample(0, 50.0), 0.6, 50.0, 1.0, config.target_area_frac, "frontal", config)
    state.add_candidate(make_sample(10, 200.0), 0.9, 200.0, 1.0, config.target_area_frac, "frontal", config)

    exported = state.export_samples(config)
    assert len(exported) == 1
    assert exported[0].frame_idx == 10
    reasons = {cand.sample.frame_idx: cand.reason for cand in state.candidates}
    assert reasons[0] == "rejected_sharpness"


def test_export_samples_ensures_frontal_quota():
    config = HarvestConfig(
        samples_per_track=4,
        samples_min=4,
        min_frontalness=0.1,
        frontal_pctile=50.0,
        min_frontal_picks=2,
    )
    state = TrackSamplingState()

    def add_candidate(frame_idx: int, frontal: float, quality: float) -> None:
        sample = FaceSample(
            track_id=1,
            frame_idx=frame_idx,
            timestamp_ms=frame_idx * 40.0,
            path=Path(f"/tmp/sample_{frame_idx}.jpg"),
            score=0.9,
            bbox=(0, 0, 10, 10),
            sharpness=200.0,
            frontalness=frontal,
            quality=quality,
        )
        state.add_candidate(sample, quality, 200.0, frontal, config.target_area_frac, "frontal", config)

    add_candidate(0, 0.9, 0.95)
    add_candidate(10, 0.8, 0.9)
    add_candidate(20, 0.3, 0.85)
    add_candidate(30, 0.2, 0.80)
    add_candidate(40, 0.15, 0.70)

    selected = state.export_samples(config)
    assert len(selected) == 4
    assert sum(1 for sample in selected if sample.frontalness >= 0.3) >= config.min_frontal_picks
