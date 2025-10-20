"""Nearest-neighbour suggestions for facebank labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from screentime.recognition.embed_arcface import ArcFaceEmbedder
from screentime.recognition.facebank import load_facebank


@dataclass
class Suggestion:
    label: str
    score: float


def load_facebank_embeddings(parquet_path: Path) -> Dict[str, np.ndarray]:
    """Load facebank parquet to a label->embedding mapping."""
    if not parquet_path.exists():
        return {}
    return load_facebank(parquet_path)


def create_embedder(model_path: Optional[str] = None, providers: Optional[Sequence[str]] = None) -> ArcFaceEmbedder:
    """Instantiate ArcFace embedder, raising RuntimeError if insightface is missing."""
    return ArcFaceEmbedder(model_path=model_path, providers=providers)


def embed_sample(embedder: ArcFaceEmbedder, image_path: Path) -> np.ndarray:
    """Embed a face crop into a normalized vector."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Unable to load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    aligned = cv2.resize(image, (112, 112))
    return embedder.embed(aligned)


def top_k(facebank: Dict[str, np.ndarray], embedding: np.ndarray, k: int = 3) -> List[Suggestion]:
    """Return top-k cosine similarity suggestions given a normalized embedding."""
    if not facebank:
        return []
    scores: List[Tuple[str, float]] = []
    for label, vec in facebank.items():
        try:
            score = float(np.dot(vec, embedding))
        except Exception:
            continue
        scores.append((label, score))
    scores.sort(key=lambda item: item[1], reverse=True)
    return [Suggestion(label=label, score=score) for label, score in scores[:k]]
