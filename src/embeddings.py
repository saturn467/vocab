from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL

if TYPE_CHECKING:
    from loader import Word


class EmbeddingGenerator:
    """Generate semantic embeddings for words (optionally enriched with context)."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)

    def encode_words(self, words: list[Word], batch_size: int = 256) -> np.ndarray:
        texts: list[str] = []
        for w in words:
            if w.context_sentence:
                texts.append(f"{w.lemma}: {w.context_sentence}")
            else:
                texts.append(w.lemma)
        return self.model.encode(texts, show_progress_bar=True, batch_size=batch_size)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=False)
