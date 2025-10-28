from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

__stub__ = False

try:  # pragma: no cover - prefer real OpenCV when available
    _real_cv2 = importlib.import_module("cv2.cv2")
except Exception:  # pragma: no cover - exercised in tests via stub
    __stub__ = True

    _STUB_VIDEO_STORAGE: Dict[str, Dict[str, Any]] = {}

    CAP_PROP_FPS = 5
    CAP_PROP_FRAME_COUNT = 7
    CAP_PROP_FRAME_HEIGHT = 1
    CAP_PROP_FRAME_WIDTH = 2
    CAP_PROP_POS_FRAMES = 0

    class VideoWriter:
        def __init__(self, path: str, _fourcc: int, fps: float, size: Tuple[int, int]) -> None:
            self.path = path
            self.fps = float(fps or 0.0)
            self.size = tuple(int(v) for v in size)
            self.frames: List[np.ndarray] = []
            _STUB_VIDEO_STORAGE[path] = {"fps": self.fps, "size": self.size, "frames": self.frames}

        def isOpened(self) -> bool:
            return True

        def write(self, frame: np.ndarray) -> None:
            self.frames.append(np.array(frame, copy=True))

        def release(self) -> None:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.path).write_bytes(b"")

    class VideoCapture:
        def __init__(self, path: str) -> None:
            info = _STUB_VIDEO_STORAGE.get(path)
            self._info = info or {"fps": 30.0, "size": (64, 64), "frames": []}
            self._idx = 0
            self._opened = True

        def isOpened(self) -> bool:
            return self._opened

        def get(self, prop: int) -> float:
            if prop == CAP_PROP_FPS:
                return float(self._info.get("fps", 0.0))
            if prop == CAP_PROP_FRAME_COUNT:
                return float(len(self._info.get("frames", [])))
            if prop == CAP_PROP_FRAME_HEIGHT:
                return float(self._info.get("size", (0, 0))[1])
            if prop == CAP_PROP_FRAME_WIDTH:
                return float(self._info.get("size", (0, 0))[0])
            if prop == CAP_PROP_POS_FRAMES:
                return float(self._idx)
            return 0.0

        def read(self) -> Tuple[bool, Optional[np.ndarray]]:
            frames: List[np.ndarray] = self._info.get("frames", [])
            if self._idx >= len(frames):
                self._opened = False
                return False, None
            frame = np.array(frames[self._idx], copy=True)
            self._idx += 1
            return True, frame

        def release(self) -> None:
            self._opened = False

        def set(self, *_args: object, **_kwargs: object) -> None:
            return None

    def VideoWriter_fourcc(*_args: object) -> int:
        return 0

    def imwrite(path: str, image: np.ndarray) -> bool:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = np.asarray(image, dtype=np.uint8)
        Path(path).write_bytes(data.tobytes() if data.size else b"")
        return True

    def imread(path: str) -> Optional[np.ndarray]:
        if not Path(path).exists():
            return None
        return np.zeros((1, 1, 3), dtype=np.uint8)

    def cvtColor(image: np.ndarray, _code: int) -> np.ndarray:
        return image

    def Laplacian(image: np.ndarray, _depth: int) -> np.ndarray:
        return np.zeros_like(image, dtype=np.float32)

    def setNumThreads(*_args: object, **_kwargs: object) -> None:
        return None

    def __getattr__(name: str) -> Any:  # pragma: no cover - fallback
        if name.isupper():
            return 0
        raise AttributeError(name)
else:  # pragma: no cover - passthrough for real OpenCV
    import sys

    module = sys.modules[__name__]
    for attr in dir(_real_cv2):
        setattr(module, attr, getattr(_real_cv2, attr))
    __stub__ = False

__all__ = [name for name in globals() if not name.startswith("_")]
