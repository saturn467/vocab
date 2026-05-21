from __future__ import annotations

from config import CEFR_BANDS


class CEFRClassifier:
    """Estimate CEFR level from frequency rank (primary signal)."""

    def classify_word(self, frequency_rank: int) -> str:
        for level, low, high in CEFR_BANDS:
            if low <= frequency_rank <= high:
                return level
        return "C2"

    def classify_deck(self, frequency_ranks: list[int]) -> str:
        """Assign a single CEFR level to a deck based on the median rank."""
        if not frequency_ranks:
            return "B1"
        sorted_ranks = sorted(frequency_ranks)
        median_rank = sorted_ranks[len(sorted_ranks) // 2]
        return self.classify_word(median_rank)
