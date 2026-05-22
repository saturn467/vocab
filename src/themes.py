from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import THEME_NAMES

if TYPE_CHECKING:
    from embeddings import EmbeddingGenerator


class ThemeClassifier:
    """Assign one or more themes to a deck based on its centroid embedding."""

    def __init__(self, embedder: EmbeddingGenerator):
        self.theme_names = THEME_NAMES
        self.theme_embeddings: np.ndarray = embedder.encode_texts(self.theme_names)

    def classify_deck(
        self,
        deck_embedding: np.ndarray,
        max_themes: int = 3,
        threshold: float = 0.30,
    ) -> list[str]:
        """Return 1–*max_themes* theme names sorted by relevance."""
        sims = cosine_similarity([deck_embedding], self.theme_embeddings)[0]
        pairs = sorted(
            zip(self.theme_names, sims), key=lambda x: x[1], reverse=True
        )
        selected = [name for name, score in pairs if score >= threshold][:max_themes]
        if not selected:
            selected = [pairs[0][0]]
        return selected
