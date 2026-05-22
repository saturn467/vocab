from __future__ import annotations

import math

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    TARGET_DECK_SIZE,
    MIN_DECK_SIZE,
    MAX_DECK_SIZE,
    OUTLIER_SIMILARITY_THRESHOLD,
)


class QualityClusterer:
    """KMeans clustering with outlier ejection and size balancing."""

    def cluster(
        self,
        embeddings: np.ndarray,
    ) -> tuple[list[list[int]], list[int]]:
        """Return (accepted_clusters, rejected_indices).

        Each cluster is a list of integer indices into *embeddings*.
        """
        n = len(embeddings)
        n_clusters = max(1, math.ceil(n / TARGET_DECK_SIZE))

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        centroids = kmeans.cluster_centers_

        # ── Build raw clusters ────────────────────────────────────────────
        raw: dict[int, list[int]] = {}
        for idx, label in enumerate(labels):
            raw.setdefault(int(label), []).append(idx)

        rejected: list[int] = []
        kept_clusters: list[list[int]] = []

        # ── Per-cluster outlier ejection ──────────────────────────────────
        for label, indices in raw.items():
            centroid = centroids[label]
            cluster_embs = embeddings[indices]
            sims = cosine_similarity(cluster_embs, [centroid]).flatten()

            kept: list[int] = []
            for i, sim in zip(indices, sims):
                if sim >= OUTLIER_SIMILARITY_THRESHOLD:
                    kept.append(i)
                else:
                    rejected.append(i)

            if len(kept) >= MIN_DECK_SIZE:
                kept_clusters.append(kept)
            else:
                rejected.extend(kept)

        # ── Split oversized clusters ──────────────────────────────────────
        final_clusters: list[list[int]] = []
        for cluster in kept_clusters:
            if len(cluster) <= MAX_DECK_SIZE:
                final_clusters.append(cluster)
                continue

            sub_n = max(2, math.ceil(len(cluster) / TARGET_DECK_SIZE))
            sub_embs = embeddings[cluster]
            sub_km = KMeans(n_clusters=sub_n, random_state=42, n_init=5)
            sub_labels = sub_km.fit_predict(sub_embs)

            sub_groups: dict[int, list[int]] = {}
            for i, sl in enumerate(sub_labels):
                sub_groups.setdefault(int(sl), []).append(cluster[i])

            for sg in sub_groups.values():
                if len(sg) >= MIN_DECK_SIZE:
                    final_clusters.append(sg)
                else:
                    rejected.extend(sg)

        return final_clusters, rejected
