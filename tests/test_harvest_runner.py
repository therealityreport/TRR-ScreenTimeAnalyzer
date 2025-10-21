from __future__ import annotations

import json
from pathlib import Path

import sys
import types
from typing import Any, Dict, List

import numpy as np
import pytest

_FAKE_VIDEOS: Dict[str, Dict[str, Any]] = {}

try:  # pragma: no cover - exercised implicitly
    import cv2  # type: ignore
except Exception:  # pragma: no cover - dependency guard
    CAP_PROP_FPS = 5
    CAP_PROP_FRAME_COUNT = 7
    CAP_PROP_FRAME_HEIGHT = 3
    CAP_PROP_FRAME_WIDTH = 4

    class _FakeVideoWriter:
        def __init__(self, path, fourcc, fps, size):
            self.path = str(path)
            self.fps = float(fps)
            self.size = size
            self.frames: List[np.ndarray] = []

        def write(self, frame: np.ndarray) -> None:
            self.frames.append(frame)

        def release(self) -> None:
            _FAKE_VIDEOS[self.path] = {
                "frames": list(self.frames),
                "fps": self.fps,
                "size": self.size,
            }

    class _FakeVideoCapture:
        def __init__(self, path):
            self.path = str(path)
            self.data = _FAKE_VIDEOS.get(self.path, {
                "frames": [],
                "fps": 30.0,
                "size": (64, 64),
            })
            self._idx = 0
            self._opened = self.path in _FAKE_VIDEOS

        def isOpened(self):
            return self._opened

        def read(self):
            if not self._opened or self._idx >= len(self.data["frames"]):
                return False, None
            frame = self.data["frames"][self._idx]
            self._idx += 1
            return True, frame

        def get(self, prop):
            if prop == CAP_PROP_FPS:
                return self.data["fps"]
            if prop == CAP_PROP_FRAME_COUNT:
                return len(self.data["frames"])
            if prop == CAP_PROP_FRAME_HEIGHT:
                return self.data["size"][1]
            if prop == CAP_PROP_FRAME_WIDTH:
                return self.data["size"][0]
            return 0.0

        def release(self):
            return None

    def _fake_fourcc(*_args):
        return 0

    def _fake_imwrite(_path, _image):
        return True

    cv2 = types.SimpleNamespace(
        VideoWriter=_FakeVideoWriter,
        VideoWriter_fourcc=_fake_fourcc,
        VideoCapture=_FakeVideoCapture,
        CAP_PROP_FPS=CAP_PROP_FPS,
        CAP_PROP_FRAME_COUNT=CAP_PROP_FRAME_COUNT,
        CAP_PROP_FRAME_HEIGHT=CAP_PROP_FRAME_HEIGHT,
        CAP_PROP_FRAME_WIDTH=CAP_PROP_FRAME_WIDTH,
        imwrite=_fake_imwrite,
    )
    sys.modules["cv2"] = cv2

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
    assert not runner.identity_guard_degraded


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
    assert not runner.identity_guard_degraded, "Degraded flag should remain false when guard is disabled explicitly."


def test_harvest_runner_continues_when_embedder_init_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_path = tmp_path / "blank.mp4"
    _create_blank_video(video_path)

    class _ExplodingEmbedder:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr("screentime.harvest.harvest.ArcFaceEmbedder", _ExplodingEmbedder)

    config = HarvestConfig(
        identity_guard=True,
        identity_split=True,
        samples_per_track=1,
        write_candidates=False,
        reindex_harvest_tracks=True,
        stitch_identities=False,
        debug_rejections=True,
    )

    runner = HarvestRunner(_NullPersonDetector(), _NullFaceDetector(), _NullTracker(), config)
    output_root = tmp_path / "output"
    manifest_path = runner.run(video_path, output_root)

    assert manifest_path.exists(), "Harvest runner should finish even if the embedder fails to initialize."
    assert runner.identity_guard_degraded, "Degraded flag should be set when embedder initialization fails."
    assert not config.identity_guard, "Identity guard should be disabled when embedder initialization fails."
    assert not config.identity_split, "Identity split should be disabled when embedder initialization fails."

    debug_path = manifest_path.parent / "harvest_debug.json"
    assert debug_path.exists(), "Debug payload should be written when debug_rejections is enabled."

    with debug_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    assert payload.get("identity_guard_degraded") is True, "Debug payload should record degraded identity guard mode."
