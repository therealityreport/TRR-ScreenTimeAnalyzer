import numpy as np

from screentime.harvest.harvest import HarvestConfig
from screentime.types import FaceSample, ManifestEntry


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    num = float((a * b).sum())
    den = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    return num / den


def test_identity_guard_split_decision_logic():
    cfg = HarvestConfig(identity_guard=True, identity_split=True, identity_sim_threshold=0.62, identity_min_picks=2)

    # Embeddings for two different identities (orthogonal vectors)
    e_a = np.zeros((512,), dtype=np.float32)
    e_a[0] = 1.0
    e_b = np.zeros((512,), dtype=np.float32)
    e_b[1] = 1.0

    # Simulate centroid accumulation for harvest id 1 with two samples of A
    centroid = e_a.copy()
    count = 2

    # New sample B should trigger a split under the threshold
    sim = _cosine(e_b, centroid)
    assert sim < cfg.identity_sim_threshold
    should_split = cfg.identity_guard and cfg.identity_split and (count >= cfg.identity_min_picks) and (
        sim < cfg.identity_sim_threshold
    )
    assert should_split


def test_manifest_entry_includes_byte_track_id():
    samples = [
        FaceSample(
            track_id=1,
            byte_track_id=7,
            frame_idx=10,
            timestamp_ms=400.0,
            path=__file__,
            score=0.9,
            bbox=(0.0, 0.0, 10.0, 10.0),
        )
    ]

    entry = ManifestEntry(
        track_id=1,
        byte_track_id=7,
        total_frames=100,
        avg_conf=0.8,
        avg_area=123.0,
        first_ts_ms=0.0,
        last_ts_ms=4000.0,
        samples=samples,
    )
    payload = entry.to_dict()
    assert payload["track_id"] == 1
    assert payload["byte_track_id"] == 7
    assert payload["samples"][0]["byte_track_id"] == 7

