import numpy as np

from scripts import cluster_tracks


def test_cluster_tracks_groups_similar_embeddings():
    embeddings = {
        1: np.array([1.0, 0.0, 0.0], dtype=np.float32),
        2: np.array([0.98, 0.05, 0.0], dtype=np.float32),
        3: np.array([0.0, 1.0, 0.0], dtype=np.float32),
    }
    grouped = cluster_tracks.cluster_tracks(embeddings, distance_thresh=0.3)
    assert len(grouped) == 2
    assert grouped[0] == [1, 2]
    assert grouped[1] == [3]


def test_cluster_tracks_single_track_returns_single_cluster():
    embeddings = {42: np.array([0.0, 0.0, 1.0], dtype=np.float32)}
    grouped = cluster_tracks.cluster_tracks(embeddings, distance_thresh=0.3)
    assert grouped == {0: [42]}
