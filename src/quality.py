from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import TARGET_DECK_SIZE, MIN_QUALITY_SCORE, MIN_DECKS


class DeckQualityScorer:
    """Score and filter candidate decks by semantic coherence and size."""

    @staticmethod
    def score(embeddings: np.ndarray, indices: list[int]) -> float:
        """Return Q ∈ [0, 1] for a candidate deck."""
        cluster_embs = embeddings[indices]
        centroid = cluster_embs.mean(axis=0)

        # Coherence: mean cosine similarity of every word to the centroid
        sims = cosine_similarity(cluster_embs, [centroid]).flatten()
        coherence = float(sims.mean())

        # Size factor: Gaussian penalty for deviation from target size
        size = len(indices)
        size_factor = float(np.exp(-0.5 * ((size - TARGET_DECK_SIZE) / 5.0) ** 2))

        return coherence * size_factor

    def filter_decks(
        self,
        embeddings: np.ndarray,
        clusters: list[list[int]],
    ) -> tuple[list[tuple[list[int], float]], list[int]]:
        """Return (accepted, rejected_indices).

        *accepted* is a list of (indices, quality_score) sorted best-first.
        If strict filtering would drop below MIN_DECKS, the threshold is
        relaxed progressively so coverage stays above the floor.
        """
        scored = [(c, self.score(embeddings, c)) for c in clusters]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Adaptive threshold: try the configured minimum first, then relax
        threshold = MIN_QUALITY_SCORE
        while True:
            accepted = [(c, q) for c, q in scored if q >= threshold]
            if len(accepted) >= MIN_DECKS or threshold <= 0.05:
                break
            threshold -= 0.05

        rejected_indices: list[int] = []
        accepted_set = {id(c) for c, _ in accepted}
        for c, _ in scored:
            if id(c) not in accepted_set:
                rejected_indices.extend(c)

        return accepted, rejected_indices
