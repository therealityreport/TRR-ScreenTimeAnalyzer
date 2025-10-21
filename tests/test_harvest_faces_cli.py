from __future__ import annotations

import math
import sys
from pathlib import Path

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - fallback for headless test envs
    class _StubCapture:
        def __init__(self, *_args, **_kwargs) -> None:
            self._opened = True

        def isOpened(self) -> bool:
            return self._opened

        def get(self, _prop: int) -> float:
            return 0.0

        def release(self) -> None:
            self._opened = False

        def read(self):
            return False, None

        def set(self, *_args, **_kwargs) -> None:
            pass

    class _Cv2Stub:  # pragma: no cover - fallback for CI
        CAP_PROP_FPS = 5
        CAP_PROP_FRAME_COUNT = 7
        CAP_PROP_FRAME_HEIGHT = 1
        CAP_PROP_FRAME_WIDTH = 2
        CAP_PROP_POS_FRAMES = 0

        def __init__(self) -> None:
            self.VideoCapture = _StubCapture

        def setNumThreads(self, *_args, **_kwargs) -> None:
            pass

        def cvtColor(self, *_args, **_kwargs):
            return None

        def __getattr__(self, name: str):
            if name.isupper():
                return 0
            raise AttributeError(name)

    cv2 = _Cv2Stub()  # type: ignore
    sys.modules["cv2"] = cv2

import pytest

from scripts import harvest_faces


def test_cpu_preset_applies_defaults_for_cpu_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video_path = tmp_path / "episode.mp4"
    video_path.write_bytes(b"0")
    harvest_dir = tmp_path / "harvest_out"

    args = harvest_faces.parse_args(
        [
            str(video_path),
            "--person-weights",
            "weights.pt",
            "--harvest-dir",
            str(harvest_dir),
        ]
    )
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

        def run(self, video, output_root, legacy_layout=True, cap=None):
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


def test_samples_per_sec_derives_stride(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video_path = tmp_path / "episode.mp4"
    video_path.write_bytes(b"0")

    args = harvest_faces.parse_args(
        [
            str(video_path),
            "--person-weights",
            "weights.pt",
            "--harvest-dir",
            str(tmp_path / "harvest"),
            "--samples-per-sec",
            "5",
        ]
    )

    monkeypatch.setattr(harvest_faces, "parse_args", lambda: args)

    class DummyCapture:
        def __init__(self) -> None:
            self.released = False

        def isOpened(self) -> bool:
            return True

        def get(self, prop: int) -> float:
            if prop == harvest_faces.cv2.CAP_PROP_FPS:
                return 29.97
            return 0.0

        def release(self) -> None:
            self.released = True

    capture = DummyCapture()
    monkeypatch.setattr(harvest_faces.cv2, "VideoCapture", lambda path: capture)

    monkeypatch.setattr(harvest_faces, "configure_threads", lambda threads: None)
    monkeypatch.setattr(harvest_faces, "build_session_options", lambda threads: None)
    monkeypatch.setattr(harvest_faces, "setup_logging", lambda: None)

    def unexpected_scene(*_args, **_kwargs):
        raise AssertionError("scene aware not expected")

    monkeypatch.setattr(harvest_faces, "run_scene_aware_harvest", unexpected_scene)

    observed: dict[str, object] = {}

    def fake_run_standard(args_param, cap):
        observed["stride"] = args_param.stride
        observed["cap"] = cap

    monkeypatch.setattr(harvest_faces, "run_standard_harvest", fake_run_standard)

    harvest_faces.main()

    expected_stride = math.ceil(29.97 / 5.0)
    assert observed["stride"] == expected_stride
    assert observed["cap"] is capture
    assert capture.released is True
