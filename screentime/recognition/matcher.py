"""Cosine similarity matcher with vote persistence for tracks."""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np

from screentime.types import TrackState, l2_normalize

LOGGER = logging.getLogger("screentime.recognition.matcher")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError("Embedding shapes do not match")
    return float(np.dot(a, b))


class TrackVotingMatcher:
    """Maintains per-track vote state given incoming embeddings."""

    def __init__(
        self,
        facebank: Dict[str, np.ndarray],
        similarity_th: float = 0.82,
        vote_decay: float = 0.99,
        flip_tolerance: float = 0.30,
        dominance_ratio: float = 2.0,
    ) -> None:
        self.facebank = {label: l2_normalize(vec) for label, vec in facebank.items()}
        self.similarity_th = similarity_th
        self.vote_decay = vote_decay
        self.flip_tolerance = flip_tolerance
        self.dominance_ratio = dominance_ratio

    def best_match(self, embedding: np.ndarray) -> Optional[Tuple[str, float]]:
        if not self.facebank:
            return None
        embedding = l2_normalize(embedding)
        scores = {
            label: cosine_similarity(vec, embedding)
            for label, vec in self.facebank.items()
        }
        label, score = max(scores.items(), key=lambda item: item[1])
        if score < self.similarity_th:
            return None
        return label, score

    def update_track(self, track: TrackState, embedding: np.ndarray) -> None:
        match = self.best_match(embedding)
        # Decay existing votes
        for label in list(track.label_scores.keys()):
            track.label_scores[label] *= self.vote_decay

        if match is None:
            return

        label, score = match
        track.record_vote(label, score)

        sorted_votes = sorted(track.label_scores.items(), key=lambda item: item[1], reverse=True)
        if not sorted_votes:
            return

        top_label, top_weight = sorted_votes[0]
        runner_up_weight = sorted_votes[1][1] if len(sorted_votes) > 1 else 0.0

        # Check if dominance condition holds to set/keep identity
        if top_weight >= runner_up_weight * self.dominance_ratio:
            previous_label = track.label
            if previous_label is None:
                track.label = top_label
                return
            if previous_label == top_label:
                # reinforce the label
                return

            # Evaluate flip tolerance
            current_weight = track.label_scores.get(previous_label, 0.0)
            if top_weight >= current_weight * (1.0 + self.flip_tolerance):
                LOGGER.debug(
                    "Track %s flipping label %s -> %s (%.3f vs %.3f)",
                    track.track_id,
                    previous_label,
                    top_label,
                    top_weight,
                    current_weight,
                )
                track.label = top_label
        else:
            LOGGER.debug(
                "Track %s votes not dominant: %s",
                track.track_id,
                sorted_votes,
            )
