import numpy as np

from screentime.recognition.matcher import TrackVotingMatcher
from screentime.types import TrackState


def test_track_vote_persistence_sets_label():
    facebank = {
        "Alice": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "Bob": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    }
    matcher = TrackVotingMatcher(facebank, similarity_th=0.1, vote_decay=1.0)
    track = TrackState(track_id=1)
    matcher.update_track(track, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert track.label == "Alice"
    matcher.update_track(track, np.array([0.7, 0.0, 0.0], dtype=np.float32))
    assert track.label == "Alice"
