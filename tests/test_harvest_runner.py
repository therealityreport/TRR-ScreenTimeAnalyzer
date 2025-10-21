from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from screentime.harvest.harvest import HarvestConfig, HarvestRunner
from screentime.io_utils import infer_video_stem


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


def _create_blank_video(path: Path, frame_count: int = 3, fps: int = 30) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (64, 64))
    for _ in range(frame_count):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
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
