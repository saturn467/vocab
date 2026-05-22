from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config import (
    EXCEL_FILE,
    LEMMA_COLUMN,
    FREQUENCY_COLUMN,
    CONTEXT_COLUMN,
    CONTEXT_PLACEHOLDER,
)


@dataclass
class Word:
    lemma: str
    frequency_rank: int
    context_sentence: str | None = None


class WordLoader:
    """Load vocabulary words from an Excel spreadsheet."""

    def load(self, path: str = EXCEL_FILE) -> list[Word]:
        df = pd.read_excel(path, engine="openpyxl")

        words: list[Word] = []
        for _, row in df.iterrows():
            lemma = str(row[LEMMA_COLUMN]).strip()
            if not lemma or lemma == "nan":
                continue

            freq = int(row[FREQUENCY_COLUMN])

            ctx_raw = str(row.get(CONTEXT_COLUMN, "")).strip()
            context = ctx_raw if ctx_raw and ctx_raw != CONTEXT_PLACEHOLDER else None

            words.append(Word(lemma=lemma, frequency_rank=freq, context_sentence=context))

        return words
