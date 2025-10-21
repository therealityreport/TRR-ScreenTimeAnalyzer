from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pytest

cv2 = pytest.importorskip(
    "cv2", reason="OpenCV is required for harvest runner integration tests", exc_type=ImportError
)
import numpy as np

from screentime.harvest.harvest import HarvestConfig, HarvestRunner
from screentime.io_utils import infer_video_stem
from screentime.tracking.bytetrack_wrap import TrackObservation
from screentime.types import Detection


class _NullPersonDetector:
    def detect(self, frame, frame_idx):  # noqa: D401 - simple stub
        return []


class _NullFaceDetector:
    def detect(self, frame, frame_idx):
        return []


class _NullTracker:
    def set_frame_rate(self, fps: float) -> None:
        return None

    def update(self, detections, shape):
        return []


class _CutScenePersonDetector:
    def __init__(self, schedule: Dict[int, float]):
        self.schedule = schedule

    def detect(self, frame, frame_idx):
        score = self.schedule.get(frame_idx)
        if score is None:
            return []
        bbox = (10.0, 10.0, 30.0, 40.0)
        return [Detection(frame_idx=frame_idx, bbox=bbox, score=score)]


class _CutAwareTracker:
    def __init__(self) -> None:
        self.track_buffer = 10
        self.new_track_thresh = 0.6
        self.configure_calls = []
        self.active_id: Optional[int] = None
        self.last_frame_seen: Optional[int] = None
        self.next_id = 1
        self.reentries_survived = 0

    def set_frame_rate(self, fps: float) -> None:
        return None

    def describe_settings(self):
        return {"track_buffer": self.track_buffer, "new_track_thresh": self.new_track_thresh}

    def configure(self, *, track_buffer: Optional[int] = None, new_track_thresh: Optional[float] = None) -> None:
        if track_buffer is not None:
            self.track_buffer = int(track_buffer)
        if new_track_thresh is not None:
            self.new_track_thresh = float(new_track_thresh)
        self.configure_calls.append({"track_buffer": track_buffer, "new_track_thresh": new_track_thresh})

    def update(self, detections, shape):
        outputs = []
        if not detections:
            return outputs
        detection = detections[0]
        frame_idx = detection.frame_idx
        if self.active_id is None:
            if detection.score < self.new_track_thresh:
                return []
            self.active_id = self.next_id
            self.next_id += 1
            self.last_frame_seen = frame_idx
        else:
            gap = frame_idx - (self.last_frame_seen if self.last_frame_seen is not None else frame_idx)
            if gap > self.track_buffer:
                if detection.score < self.new_track_thresh:
                    self.last_frame_seen = frame_idx
                    return []
                self.active_id = self.next_id
                self.next_id += 1
            else:
                if gap > 0:
                    self.reentries_survived += 1
            self.last_frame_seen = frame_idx
        outputs.append(
            TrackObservation(
                track_id=self.active_id,
                bbox=detection.bbox,
                score=float(detection.score),
            )
        )
        return outputs


def _create_blank_video(path: Path, frame_count: int = 3, fps: int = 30) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (64, 64))
    for _ in range(frame_count):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _create_high_cut_video(path: Path, frame_count: int = 90, fps: int = 60) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (64, 64))
    for idx in range(frame_count):
        if idx % 2 == 0:
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
        else:
            frame = np.full((64, 64, 3), 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_harvest_runner_run_no_crash(tmp_path):
    video_path = tmp_path / "blank.mp4"
    _create_blank_video(video_path)

    person_detector = _NullPersonDetector()
    face_detector = _NullFaceDetector()
    tracker = _NullTracker()

    config = HarvestConfig(
        identity_guard=False,
        identity_split=False,
        samples_per_track=1,
        write_candidates=False,
        reindex_harvest_tracks=True,
        stitch_identities=False,
    )

    runner = HarvestRunner(person_detector, face_detector, tracker, config)
    output_root = tmp_path / "output"
    manifest_path = runner.run(video_path, output_root)

    assert manifest_path.exists(), "Harvest runner should produce manifest without errors."


def test_harvest_runner_respects_explicit_root(tmp_path):
    video_path = tmp_path / "blank.mp4"
    _create_blank_video(video_path)

    config = HarvestConfig(
        identity_guard=False,
        identity_split=False,
        samples_per_track=1,
        write_candidates=False,
        reindex_harvest_tracks=True,
        stitch_identities=False,
    )
    runner = HarvestRunner(_NullPersonDetector(), _NullFaceDetector(), _NullTracker(), config)

    explicit_dir = tmp_path / "flat"
    manifest_path = runner.run(video_path, explicit_dir, legacy_layout=False)

    assert manifest_path.parent == explicit_dir
    nested_path = explicit_dir / infer_video_stem(video_path)
    assert not nested_path.exists(), "Legacy-style nested directory should not be created for explicit roots."


def test_harvest_runner_skips_embedder_when_guard_disabled(tmp_path, monkeypatch: pytest.MonkeyPatch):
    video_path = tmp_path / "blank.mp4"
    _create_blank_video(video_path)

    calls = {"init": 0, "embed": 0}

    class _SentinelEmbedder:
        def __init__(self, *args, **kwargs):
            calls["init"] += 1

        def embed(self, image):
            calls["embed"] += 1
            return np.zeros(512, dtype=np.float32)

    monkeypatch.setattr("screentime.harvest.harvest.ArcFaceEmbedder", _SentinelEmbedder)

    config = HarvestConfig(
        identity_guard=False,
        identity_split=False,
        samples_per_track=1,
        write_candidates=False,
        reindex_harvest_tracks=True,
        stitch_identities=False,
    )

    runner = HarvestRunner(_NullPersonDetector(), _NullFaceDetector(), _NullTracker(), config)
    output_root = tmp_path / "output"
    runner.run(video_path, output_root)

    assert calls["init"] == 0, "ArcFace embedder should not be constructed when identity_guard is disabled."
    assert calls["embed"] == 0, "Embeddings should not be computed when identity_guard is disabled."
    assert runner.embedder is None, "HarvestRunner should leave embedder unset when identity_guard is disabled."


def test_harvest_runner_autotunes_tracker_for_high_cut(tmp_path):
    video_path = tmp_path / "high_cut.mp4"
    _create_high_cut_video(video_path, frame_count=90, fps=60)

    schedule = {0: 0.75, 20: 0.48, 40: 0.48, 60: 0.48}
    person_detector = _CutScenePersonDetector(schedule)
    face_detector = _NullFaceDetector()
    tracker = _CutAwareTracker()

    config = HarvestConfig(
        identity_guard=False,
        identity_split=False,
        samples_per_track=1,
        write_candidates=False,
        reindex_harvest_tracks=True,
        stitch_identities=False,
        max_new_tracks_per_sec=None,
    )

    runner = HarvestRunner(person_detector, face_detector, tracker, config)
    output_root = tmp_path / "output"
    runner.run(video_path, output_root)

    assert tracker.track_buffer > 10
    assert tracker.new_track_thresh <= 0.5
    assert tracker.reentries_survived >= 2

    stats = runner.last_run_stats
    assert stats["auto_tracker_tuning"]["requested"]
    assert stats["auto_tracker_tuning"]["applied"]
    assert stats["max_new_tracks_auto"] is True
    assert stats["max_new_tracks_per_sec"] > 2.0
