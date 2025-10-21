import json
import sys
import types
from pathlib import Path

import numpy as np

if "cv2" not in sys.modules:
    fake_cv2 = types.ModuleType("cv2")
    _VIDEO_REGISTRY = {}

    fake_cv2.CAP_PROP_FPS = 5
    fake_cv2.CAP_PROP_FRAME_COUNT = 7
    fake_cv2.CAP_PROP_FRAME_HEIGHT = 3
    fake_cv2.CAP_PROP_FRAME_WIDTH = 4
    fake_cv2.COLOR_RGB2GRAY = 11
    fake_cv2.CV_64F = np.float64

    class _FakeVideoCapture:
        def __init__(self, path: str):
            payload = _VIDEO_REGISTRY.get(path)
            self._frames = [frame.copy() for frame in payload["frames"]] if payload else []
            self._fps = payload.get("fps", 30.0) if payload else 30.0
            self._index = 0

        def isOpened(self) -> bool:
            return bool(self._frames)

        def read(self):
            if self._index >= len(self._frames):
                return False, None
            frame = self._frames[self._index]
            self._index += 1
            return True, frame.copy()

        def get(self, prop):
            if not self._frames:
                return 0.0
            if prop == fake_cv2.CAP_PROP_FPS:
                return self._fps
            if prop == fake_cv2.CAP_PROP_FRAME_COUNT:
                return len(self._frames)
            if prop == fake_cv2.CAP_PROP_FRAME_HEIGHT:
                return self._frames[0].shape[0]
            if prop == fake_cv2.CAP_PROP_FRAME_WIDTH:
                return self._frames[0].shape[1]
            return 0.0

        def release(self) -> None:
            return None

    def _register_video(path: str, frames, fps: float = 30.0):
        _VIDEO_REGISTRY[path] = {"frames": list(frames), "fps": float(fps)}

    def imwrite(path: str, image) -> bool:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"stub")
        return True

    def cvtColor(image, code):
        if code != fake_cv2.COLOR_RGB2GRAY:
            raise ValueError("Unsupported color conversion in stub")
        if image.ndim == 2:
            return image
        return image.mean(axis=2).astype(np.float32)

    def Laplacian(image, ddepth):
        img = image.astype(np.float64)
        padded = np.pad(img, 1, mode="edge")
        out = np.zeros_like(img)
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                patch = padded[y : y + 3, x : x + 3]
                out[y, x] = float((patch * kernel).sum())
        return out.astype(np.float64 if ddepth == fake_cv2.CV_64F else img.dtype)

    fake_cv2.VideoCapture = _FakeVideoCapture
    fake_cv2.imwrite = imwrite
    fake_cv2.cvtColor = cvtColor
    fake_cv2.Laplacian = Laplacian
    fake_cv2._register_video = _register_video
    sys.modules["cv2"] = fake_cv2

from screentime.harvest import harvest
from screentime.tracking.bytetrack_wrap import TrackAccumulator
from screentime.types import Detection, TrackState

HarvestConfig = harvest.HarvestConfig
HarvestRunner = harvest.HarvestRunner
cv2 = harvest.cv2


class DummyDetector:
    def __init__(self):
        self.outputs = []

    def detect(self, frame, frame_idx):  # pragma: no cover - stub
        return self.outputs.pop(0) if self.outputs else []


class DummyTracker:
    def __init__(self):
        self.frame_rate = None

    def set_frame_rate(self, fps):  # pragma: no cover
        self.frame_rate = fps


def build_runner():
    config = HarvestConfig()
    runner = HarvestRunner(DummyDetector(), DummyDetector(), DummyTracker(), config)
    return runner


def test_best_iou_track_picks_highest():
    runner = build_runner()
    track_lookup = {
        1: type("Obs", (), {"bbox": (0, 0, 80, 80)}),
        2: type("Obs", (), {"bbox": (40, 40, 160, 160)}),
    }
    face_bbox = (90, 90, 120, 120)
    accumulator = TrackAccumulator()
    state1 = TrackState(track_id=1, frames=[0], bboxes=[(0, 0, 80, 80)])
    state2 = TrackState(track_id=2, frames=[0], bboxes=[(40, 40, 160, 160)])
    accumulator.active[1] = state1
    accumulator.active[2] = state2
    track_id, iou_val, best_bbox = runner._best_iou_track(
        face_bbox,
        track_lookup,
        accumulator,
        frame_idx=0,
        frame_width=1920,
        frame_height=1080,
    )
    assert track_id == 2
    assert iou_val > 0.03
    assert best_bbox == (40, 40, 160, 160)


def test_face_center_in_bbox():
    runner = build_runner()
    person_bbox = (0, 0, 100, 200)
    face_inside_head = (10, 20, 30, 60)
    face_outside_head = (10, 120, 30, 160)
    assert runner._face_center_in_bbox(face_inside_head, person_bbox)
    outside = (150, 150, 200, 200)
    assert not runner._face_center_in_bbox(outside, person_bbox)


def test_quality_scores_frontal_higher():
    runner = build_runner()
    frontal = runner._score_quality(300.0, 1.0, runner.config.target_area_frac)
    profile = runner._score_quality(300.0, 0.2, runner.config.target_area_frac)
    assert frontal > profile


def test_default_face_in_track_iou_threshold():
    config = HarvestConfig()
    assert config.face_in_track_iou == 0.25


def test_center_fallback_requires_unique_candidate():
    runner = build_runner()
    faces = [
        (10, 10, 30, 30),
        (40, 40, 60, 60),
    ]
    person_bbox = (0, 0, 80, 80)
    candidate_faces = [idx for idx, bbox in enumerate(faces) if runner._face_center_in_bbox(bbox, person_bbox)]
    assert len(candidate_faces) == 2
    assert len(candidate_faces) != 1


def test_face_without_track_harvested_with_fallback(tmp_path):
    frame_size = (64, 64)
    video_path = tmp_path / "fallback.mp4"
    frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.float32)
    cv2._register_video(str(video_path), [frame], fps=30.0)

    class _NullPersonDetector:
        def detect(self, frame, frame_idx):
            return []

    class _FallbackFaceDetector:
        def __init__(self, bbox):
            self._bbox = bbox

        def detect(self, frame, frame_idx):
            return [
                Detection(
                    frame_idx=frame_idx,
                    bbox=self._bbox,
                    score=0.95,
                    class_id=0,
                    landmarks=None,
                )
            ]

        def align_to_112(self, frame, landmarks, bbox):
            return np.zeros((112, 112, 3), dtype=np.uint8)

    class _NullTracker:
        def set_frame_rate(self, fps):
            return None

        def update(self, detections, shape):
            return []

    face_bbox = (20.0, 20.0, 44.0, 44.0)
    person_detector = _NullPersonDetector()
    face_detector = _FallbackFaceDetector(face_bbox)
    tracker = _NullTracker()

    config = HarvestConfig(
        identity_guard=False,
        identity_split=False,
        samples_per_track=1,
        samples_min=1,
        samples_max=1,
        min_gap_frames=0,
        fallback_head_pct=0.5,
        debug_rejections=True,
    )

    runner = HarvestRunner(person_detector, face_detector, tracker, config)
    manifest_path = runner.run(video_path, tmp_path / "output")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload, "Expected fallback assignment to produce a harvest track"
    samples = payload[0]["samples"]
    assert len(samples) == 1
    sample = samples[0]
    assert sample["match_mode"] == "fallback"
    assert sample["byte_track_id"] < 0

    expected_bbox = [8.0, 8.0, 56.0, 56.0]
    assert sample["person_bbox"] == expected_bbox
    assert Path(sample["path"]).exists()
