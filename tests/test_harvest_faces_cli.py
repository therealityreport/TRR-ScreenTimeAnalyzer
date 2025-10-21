from __future__ import annotations

from pathlib import Path
import sys
import types

import pytest

try:  # pragma: no cover - prefer real cv2 when available
    import cv2  # type: ignore
except Exception:  # pragma: no cover - testing fallback for headless envs
    class _VideoCaptureStub:
        def __init__(self, *args, **kwargs):
            self._opened = True

        def isOpened(self) -> bool:
            return self._opened

        def set(self, *args, **kwargs):
            return True

        def read(self):
            return False, None

        def get(self, *args, **kwargs):
            return 0

        def release(self):
            self._opened = False

    cv2 = types.SimpleNamespace(  # type: ignore
        setNumThreads=lambda *args, **kwargs: None,
        VideoCapture=_VideoCaptureStub,
        imread=lambda *args, **kwargs: None,
        imwrite=lambda *args, **kwargs: True,
        resize=lambda image, size, interpolation=None: image,
        CAP_PROP_FRAME_COUNT=7,
        CAP_PROP_POS_FRAMES=1,
    )
    sys.modules["cv2"] = cv2  # type: ignore

try:  # pragma: no cover - prefer real scenedetect when available
    import scenedetect  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover - testing fallback for headless envs
    scenes_stub = types.ModuleType("screentime.segmentation.scenes")

    def _detect_scenes_stub(*args, **kwargs):
        return []

    scenes_stub.detect_scenes = _detect_scenes_stub  # type: ignore[attr-defined]
    sys.modules["screentime.segmentation.scenes"] = scenes_stub

from scripts import harvest_faces


def test_cpu_preset_applies_defaults_for_cpu_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video_path = tmp_path / "episode.mp4"
    video_path.write_bytes(b"0")
    harvest_dir = tmp_path / "harvest_out"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "harvest_faces.py",
            str(video_path),
            "--person-weights",
            "weights.pt",
            "--harvest-dir",
            str(harvest_dir),
        ],
    )
    args = harvest_faces.parse_args()
    args.threads = 1
    args.session_options = None
    args._retina_size_override = False  # mirrors main() initialization

    def fake_load_yaml(path: Path) -> dict:
        path_str = str(path)
        if "pipeline" in path_str:
            return {"stride": 1, "det_size": [960, 960]}
        if "bytetrack" in path_str:
            return {}
        return {}

    monkeypatch.setattr(harvest_faces, "load_yaml", fake_load_yaml)

    face_ctor: dict[str, object] = {}

    class DummyFaceDetector:
        def __init__(self, det_size, **kwargs):
            face_ctor["det_size"] = det_size
            face_ctor["user_override"] = kwargs.get("user_det_size_override")

    class DummyPersonDetector:
        def __init__(self, **kwargs):
            self.conf_thres = kwargs.get("conf_thres")

    class DummyTracker:
        def __init__(self, **kwargs):
            return None

    results: dict[str, object] = {}

    class DummyRunner:
        def __init__(self, person_detector, face_detector, tracker, config):
            results["config_stride"] = config.stride

        def run(self, video, output_root, legacy_layout=True):
            manifest_root = Path(output_root)
            manifest_root.mkdir(parents=True, exist_ok=True)
            manifest = manifest_root / "manifest.json"
            manifest.write_text("[]")
            return manifest

    monkeypatch.setattr(harvest_faces, "RetinaFaceDetector", DummyFaceDetector)
    monkeypatch.setattr(harvest_faces, "YOLOPersonDetector", DummyPersonDetector)
    monkeypatch.setattr(harvest_faces, "ByteTrackWrapper", DummyTracker)
    monkeypatch.setattr(harvest_faces, "HarvestRunner", DummyRunner)

    harvest_faces.run_standard_harvest(args)

    assert getattr(args, "_cpu_preset", False) is True
    assert int(args.stride) >= 2
    assert face_ctor.get("det_size") == (640, 640)
    assert results.get("config_stride", 0) >= 2


def test_pipeline_min_frontalness_threads_through_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_path = tmp_path / "episode.mp4"
    video_path.write_bytes(b"0")
    harvest_dir = tmp_path / "harvest_out"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "harvest_faces.py",
            str(video_path),
            "--person-weights",
            "weights.pt",
            "--harvest-dir",
            str(harvest_dir),
        ],
    )
    args = harvest_faces.parse_args()
    args.threads = 1
    args.session_options = None
    args._retina_size_override = False

    pipeline_min_frontalness = 0.10

    def fake_load_yaml(path: Path) -> dict:
        path_str = str(path)
        if "pipeline" in path_str:
            return {"stride": 1, "det_size": [960, 960], "min_frontalness": pipeline_min_frontalness}
        if "bytetrack" in path_str:
            return {}
        return {}

    monkeypatch.setattr(harvest_faces, "load_yaml", fake_load_yaml)

    class DummyFaceDetector:
        def __init__(self, det_size, **kwargs):
            self.det_size = det_size

    class DummyPersonDetector:
        def __init__(self, **kwargs):
            self.conf_thres = kwargs.get("conf_thres")

    class DummyTracker:
        def __init__(self, **kwargs):
            return None

    captured: dict[str, object] = {}

    class DummyRunner:
        def __init__(self, person_detector, face_detector, tracker, config):
            captured["min_frontalness"] = config.min_frontalness

        def run(self, video, output_root, legacy_layout=True):
            manifest_root = Path(output_root)
            manifest_root.mkdir(parents=True, exist_ok=True)
            manifest = manifest_root / "manifest.json"
            manifest.write_text("[]")
            return manifest

    monkeypatch.setattr(harvest_faces, "RetinaFaceDetector", DummyFaceDetector)
    monkeypatch.setattr(harvest_faces, "YOLOPersonDetector", DummyPersonDetector)
    monkeypatch.setattr(harvest_faces, "ByteTrackWrapper", DummyTracker)
    monkeypatch.setattr(harvest_faces, "HarvestRunner", DummyRunner)

    harvest_faces.run_standard_harvest(args)

    assert captured.get("min_frontalness") == pytest.approx(pipeline_min_frontalness)
