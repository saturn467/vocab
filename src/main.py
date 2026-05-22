"""
Vocabulary Taxonomer – Main Pipeline

Reads an Excel vocabulary list, clusters words into semantically coherent
study decks, assigns CEFR levels and themes, and exports a JSON catalog
compatible with the Drizzle schema (decks, decks_themes, cards).
"""

from __future__ import annotations

import numpy as np

from config import SOURCE_LANGUAGE, TARGET_LANGUAGE
from loader import WordLoader
from embeddings import EmbeddingGenerator
from clustering import QualityClusterer
from quality import DeckQualityScorer
from cefr import CEFRClassifier
from themes import ThemeClassifier
from namer import DeckNamer
from exporter import Exporter


def main() -> None:
    # ── 1. Load ───────────────────────────────────────────────────────────
    print("Loading vocabulary …")
    words = WordLoader().load()
    print(f"  {len(words)} words loaded")

    # ── 2. Embed ──────────────────────────────────────────────────────────
    print("Generating embeddings …")
    embedder = EmbeddingGenerator()
    embeddings = embedder.encode_words(words)
    print(f"  embedding shape: {embeddings.shape}")

    # ── 3. Cluster ────────────────────────────────────────────────────────
    print("Clustering …")
    clusterer = QualityClusterer()
    clusters, rejected = clusterer.cluster(embeddings)
    print(f"  {len(clusters)} raw clusters, {len(rejected)} words ejected as outliers")

    # ── 4. Quality gate ───────────────────────────────────────────────────
    print("Scoring deck quality …")
    scorer = DeckQualityScorer()
    accepted, quality_rejected = scorer.filter_decks(embeddings, clusters)
    total_rejected = len(rejected) + len(quality_rejected)
    print(f"  {len(accepted)} decks accepted, {total_rejected} words unused")

    # ── 5. Classify, name, and assemble ───────────────────────────────────
    print("Assigning CEFR levels, themes, and names …")
    cefr = CEFRClassifier()
    theme_clf = ThemeClassifier(embedder)
    namer = DeckNamer(embedder)

    decks_data: list[dict] = []
    level_dist: dict[str, int] = {}

    for indices, q_score in accepted:
        deck_words = [words[i] for i in indices]
        deck_lemmas = [w.lemma for w in deck_words]
        freq_ranks = [w.frequency_rank for w in deck_words]

        centroid = embeddings[indices].mean(axis=0)
        level = cefr.classify_deck(freq_ranks)
        themes = theme_clf.classify_deck(centroid)
        name = namer.name_deck(centroid)

        level_dist[level] = level_dist.get(level, 0) + 1

        decks_data.append({
            "name": name,
            "description": (
                f"{level}-level vocabulary deck. "
                f"Themes: {', '.join(themes)}. "
                f"Contains {len(deck_lemmas)} words."
            ),
            "level": level,
            "source_language_code": SOURCE_LANGUAGE,
            "target_language_code": TARGET_LANGUAGE,
            "themes": themes,
            "words": deck_lemmas,
            "quality_score": round(q_score, 4),
        })

    # ── 6. Export ─────────────────────────────────────────────────────────
    print("Exporting catalog …")
    exporter = Exporter()
    catalog = exporter.export(decks_data)

    # ── 7. Summary ────────────────────────────────────────────────────────
    total_cards = len(catalog["cards"])
    avg_q = (
        sum(d["quality_score"] for d in decks_data) / len(decks_data)
        if decks_data
        else 0
    )

    print("\n── Summary ──────────────────────────────────────────────")
    print(f"  Total words loaded:    {len(words)}")
    print(f"  Words in decks:        {total_cards}")
    print(f"  Words unused:          {len(words) - total_cards}")
    print(f"  Decks created:         {len(decks_data)}")
    print(f"  Themes used:           {len(catalog['themes'])}")
    print(f"  Average quality (Q):   {avg_q:.3f}")
    print(f"  CEFR distribution:     {level_dist}")
    print(f"  Output file:           catalog_seed.json")
    print("─────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
