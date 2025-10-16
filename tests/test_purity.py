from pathlib import Path

import numpy as np

from scripts.check_purity import compute_track_stats


class StubEmbedder:
    def embed(self, image):  # pragma: no cover - not used in tests
        raise AssertionError("Embedder should not be called when similarities exist")


def test_compute_track_stats_no_samples():
    entry = {"track_id": 1, "samples": []}
    stats = compute_track_stats(entry, StubEmbedder())
    assert stats["sample_count"] == 0
    assert stats["min_similarity"] is None


def test_compute_track_stats_uses_manifest_values():
    entry = {
        "track_id": 2,
        "samples": [
            {"path": str(Path("frame1.jpg")), "similarity_to_centroid": 0.9},
            {"path": str(Path("frame2.jpg")), "similarity_to_centroid": 0.8},
            {"path": str(Path("frame3.jpg")), "similarity_to_centroid": 0.85},
        ],
    }
    stats = compute_track_stats(entry, StubEmbedder())
    assert stats["sample_count"] == 3
    assert np.isclose(stats["min_similarity"], 0.8)
    assert np.isclose(stats["median_similarity"], 0.85)
