from __future__ import annotations

import cv2
import numpy as np

from pathlib import Path

from screentime.harvest.harvest import HarvestConfig, HarvestRunner


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
