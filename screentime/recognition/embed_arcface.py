"""ArcFace embedding utilities."""

from __future__ import annotations

import logging
import os
import platform
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from screentime.types import l2_normalize

LOGGER = logging.getLogger("screentime.recognition.embed")


def _default_providers() -> Tuple[str, ...]:
    """Choose default ONNX providers based on platform."""
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin" and machine in {"arm64", "aarch64"}:
        return ("CoreMLExecutionProvider", "CPUExecutionProvider")
    return ("CPUExecutionProvider",)


class ArcFaceEmbedder:
    """Loads an ArcFace ONNX model via InsightFace for embedding extraction."""

    def __init__(self, model_path: Optional[str] = None, providers: Optional[Sequence[str]] = None) -> None:
        os.environ.setdefault("OMP_NUM_THREADS", "2")
        os.environ.setdefault("MKL_NUM_THREADS", "2")
        os.environ.setdefault("ORT_INTRA_OP_NUM_THREADS", "2")
        try:
            from insightface.model_zoo import get_model
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "insightface is required for ArcFaceEmbedder. "
                "Install it via `pip install insightface`."
            ) from exc

        if model_path:
            resolved = str(Path(model_path).expanduser())
        else:
            resolved = "arcface_r100_v1"
        provider_list: Tuple[str, ...]
        if providers is None:
            provider_list = _default_providers()
        else:
            provider_list = tuple(providers)
        LOGGER.info("Loading ArcFace model %s providers=%s", resolved, provider_list)
        model = get_model(resolved, download=True, providers=list(provider_list))
        if model is None:
            LOGGER.info("Falling back to FaceAnalysis recognition model")
            from insightface.app import FaceAnalysis

            analysis = FaceAnalysis(name="buffalo_l", providers=list(provider_list))
            analysis.prepare(ctx_id=0)
            model = analysis.models.get("recognition")
            if model is None:
                raise RuntimeError("Unable to load ArcFace recognition model via insightface FaceAnalysis")
        if hasattr(model, "prepare"):
            model.prepare(ctx_id=0)
        self.model = model
        self.providers = provider_list
        self.backend = None
        try:
            session = getattr(model, "session", None)
            if session is not None:
                self.backend = session.get_providers()[0]
        except Exception:  # pragma: no cover - provider introspection best-effort
            self.backend = None

    def embed(self, aligned_face: np.ndarray) -> np.ndarray:
        """Compute L2-normalized embedding for an aligned 112x112 face image."""
        feat = self.model.get_feat(aligned_face)
        embedding = l2_normalize(np.asarray(feat, dtype=np.float32).reshape(-1))
        return embedding
