from __future__ import annotations

import json
import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from screentime.harvest.harvest import HarvestConfig, HarvestRunner
from screentime.tracking.bytetrack_wrap import TrackObservation
from screentime.types import Detection
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


class _StaticPersonDetector:
    def __init__(self, bbox):
        self._bbox = bbox

    def detect(self, frame, frame_idx):
        return [Detection(frame_idx=frame_idx, bbox=self._bbox, score=0.95)]


class _BoxFaceDetector:
    def __init__(self, bbox):
        self._bbox = bbox

    def detect(self, frame, frame_idx):
        return [Detection(frame_idx=frame_idx, bbox=self._bbox, score=0.95)]

    @staticmethod
    def align_to_112(image, landmarks, bbox):  # noqa: D401 - simple stub
        return np.zeros((112, 112, 3), dtype=np.uint8)


class _StaticTracker:
    def __init__(self, bbox, track_id: int = 1):
        self._bbox = bbox
        self._track_id = track_id

    def set_frame_rate(self, fps: float) -> None:
        return None

    def update(self, detections, shape):
        return [TrackObservation(track_id=self._track_id, bbox=self._bbox, score=0.9)]


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


def test_harvest_runner_disables_guard_when_embedder_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_path = tmp_path / "blank.mp4"
    _create_blank_video(video_path)

    class _FailingEmbedder:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("no provider")

    monkeypatch.setattr("screentime.harvest.harvest.ArcFaceEmbedder", _FailingEmbedder)

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

    assert manifest_path.exists(), "Harvest should still produce a manifest when embeddings fail."
    assert runner.embedder is None, "Embedder should remain unset after a fallback."
    assert runner.config.identity_guard is False
    assert runner.config.identity_split is False

    debug_path = manifest_path.parent / "harvest_debug.json"
    assert debug_path.exists(), "Debug log should be written when debug_rejections is enabled."

    with debug_path.open("r", encoding="utf-8") as fh:
        debug_payload = json.load(fh)

    guard_status = debug_payload.get("identity_guard_status")
    assert guard_status is not None, "Debug payload should record identity guard status."
    assert guard_status["requested"] is True
    assert guard_status["active_guard"] is False
    assert guard_status["fallback_reason"]


def test_hybrid_writes_manifest_and_tracks(tmp_path: Path) -> None:
    video_path = tmp_path / "hybrid.mp4"
    _create_blank_video(video_path, frame_count=2)

    bbox = (5.0, 5.0, 40.0, 40.0)
    person_detector = _StaticPersonDetector(bbox)
    face_detector = _BoxFaceDetector(bbox)
    tracker = _StaticTracker(bbox)

    config = HarvestConfig(
        identity_guard=False,
        identity_split=False,
        samples_per_track=1,
        write_candidates=False,
        reindex_harvest_tracks=True,
        stitch_identities=False,
        min_frontalness=0.0,
        min_area_frac=0.0,
        min_face_px=0,
        face_in_track_iou=0.0,
        allow_face_center=True,
        dilate_track_px=0.0,
        temporal_iou_tolerance=0,
        frame_selector=lambda frame_idx: frame_idx == 0,
        face_only_tracking=False,
    )

    runner = HarvestRunner(person_detector, face_detector, tracker, config)
    output_root = tmp_path / "hybrid_out"
    manifest_path = runner.run(video_path, output_root)

    assert manifest_path.exists(), "Manifest should be written for hybrid run."
    selected_csv = manifest_path.parent / "selected_samples.csv"
    assert selected_csv.exists(), "selected_samples.csv should always be emitted."
    identity_path = manifest_path.parent / "identity_tracks.json"
    assert identity_path.exists(), "identity_tracks.json should be produced for stitching summary."

    track_dirs = list(manifest_path.parent.glob("track_*"))
    assert track_dirs, "Hybrid harvest should create track directories."


def test_closeup_face_only_tracking_continues_track(tmp_path: Path) -> None:
    video_path = tmp_path / "face_only.mp4"
    _create_blank_video(video_path, frame_count=3)

    bbox = (10.0, 10.0, 48.0, 48.0)

    class _FaceDetectorStream(_BoxFaceDetector):
        def detect(self, frame, frame_idx):
            return [Detection(frame_idx=frame_idx, bbox=self._bbox, score=0.92)]

    face_detector = _FaceDetectorStream(bbox)

    config = HarvestConfig(
        identity_guard=False,
        identity_split=False,
        samples_per_track=1,
        write_candidates=False,
        reindex_harvest_tracks=True,
        stitch_identities=False,
        min_frontalness=0.0,
        min_area_frac=0.0,
        min_face_px=0,
        face_in_track_iou=0.0,
        allow_face_center=True,
        dilate_track_px=0.0,
        temporal_iou_tolerance=0,
        face_only_tracking=True,
        frame_selector=lambda _: True,
    )

    runner = HarvestRunner(_NullPersonDetector(), face_detector, _NullTracker(), config)
    output_root = tmp_path / "face_only_out"
    manifest_path = runner.run(video_path, output_root)

    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    assert manifest, "Face-only fallback should still produce a track manifest."
    entry = manifest[0]
    assert entry["byte_track_id"] >= 1_000_000, "Fallback track should originate from face-only tracker namespace."


def test_no_head_fallback_crops_when_no_face(tmp_path: Path) -> None:
    video_path = tmp_path / "no_face.mp4"
    _create_blank_video(video_path, frame_count=2)

    bbox = (12.0, 12.0, 50.0, 50.0)
    person_detector = _StaticPersonDetector(bbox)
    tracker = _StaticTracker(bbox)

    config = HarvestConfig(
        identity_guard=False,
        identity_split=False,
        samples_per_track=1,
        write_candidates=False,
        reindex_harvest_tracks=True,
        stitch_identities=False,
        min_frontalness=0.0,
        min_area_frac=0.0,
        min_face_px=0,
        face_in_track_iou=0.0,
        allow_face_center=False,
        dilate_track_px=0.0,
        temporal_iou_tolerance=0,
        fallback_head_pct=0.0,
        frame_selector=lambda _: True,
    )

    runner = HarvestRunner(person_detector, _NullFaceDetector(), tracker, config)
    output_root = tmp_path / "no_face_out"
    manifest_path = runner.run(video_path, output_root)

    selected_csv = manifest_path.parent / "selected_samples.csv"
    with selected_csv.open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    assert not rows, "No samples should be produced when no faces are detected and head fallback is disabled."
