from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import SUBCATEGORIES

if TYPE_CHECKING:
    from embeddings import EmbeddingGenerator
    from loader import Word


class DeckNamer:
    """Derive a descriptive deck name from subcategory matching with diversity.

    When multiple decks match the same subcategory, later decks are steered
    toward the next-best alternative so the catalog reads more distinctly.
    """

    _MAX_REUSE = 2  # allow a subcategory at most twice before penalising

    def __init__(self, embedder: EmbeddingGenerator):
        self.subcategory_labels = [name for name, _ in SUBCATEGORIES]
        self.subcategory_embeddings: np.ndarray = embedder.encode_texts(
            self.subcategory_labels
        )
        self._usage: dict[str, int] = {}

    def name_deck(
        self,
        deck_centroid: np.ndarray,
        deck_words: list[Word] | None = None,
    ) -> str:
        sims = cosine_similarity([deck_centroid], self.subcategory_embeddings)[0]

        # Penalise overused subcategories so naming spreads out
        adjusted = sims.copy()
        for i, label in enumerate(self.subcategory_labels):
            used = self._usage.get(label, 0)
            if used >= self._MAX_REUSE:
                adjusted[i] *= max(0.1, 1.0 - 0.3 * (used - self._MAX_REUSE + 1))

        best_idx = int(np.argmax(adjusted))
        base_name = self.subcategory_labels[best_idx]

        self._usage[base_name] = self._usage.get(base_name, 0) + 1
        count = self._usage[base_name]

        if count <= 1:
            return base_name
        return f"{base_name} {count}"
