"""ArcFace embedding utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from screentime.types import l2_normalize

LOGGER = logging.getLogger("screentime.recognition.embed")


class ArcFaceEmbedder:
    """Loads an ArcFace ONNX model via InsightFace for embedding extraction."""

    def __init__(self, model_path: Optional[str] = None, providers: Optional[list[str]] = None) -> None:
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
        LOGGER.info("Loading ArcFace model %s", resolved)
        model = get_model(resolved, download=True, providers=providers)
        if model is None:
            LOGGER.info("Falling back to FaceAnalysis recognition model")
            from insightface.app import FaceAnalysis

            analysis = FaceAnalysis(name="buffalo_l", providers=providers)
            analysis.prepare(ctx_id=0)
            model = analysis.models.get("recognition")
            if model is None:
                raise RuntimeError("Unable to load ArcFace recognition model via insightface FaceAnalysis")
        if hasattr(model, "prepare"):
            model.prepare(ctx_id=0)
        self.model = model

    def embed(self, aligned_face: np.ndarray) -> np.ndarray:
        """Compute L2-normalized embedding for an aligned 112x112 face image."""
        feat = self.model.get_feat(aligned_face)
        embedding = l2_normalize(np.asarray(feat, dtype=np.float32).reshape(-1))
        return embedding
