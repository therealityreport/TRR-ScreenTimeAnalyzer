from __future__ import annotations

from pathlib import Path

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
