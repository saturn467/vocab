from __future__ import annotations

import json
import re
import unicodedata
import uuid
from typing import Any

from config import _PROJECT_ROOT


def _slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text


def _new_id() -> str:
    return str(uuid.uuid4())


class Exporter:
    """Export the taxonomy to a JSON file matching the Drizzle schema."""

    def export(
        self,
        decks_data: list[dict[str, Any]],
        filename: str = str(_PROJECT_ROOT / "catalog_seed.json"),
    ) -> dict[str, Any]:
        """Build and write the full catalog.

        *decks_data* is a list of dicts, each with keys:
            name, description, level, source_language_code,
            target_language_code, themes (list[str]), words (list[str])
        """
        # ── Themes ────────────────────────────────────────────────────────
        theme_id_map: dict[str, str] = {}
        themes_rows: list[dict[str, Any]] = []
        for deck in decks_data:
            for theme_name in deck["themes"]:
                if theme_name not in theme_id_map:
                    tid = _new_id()
                    theme_id_map[theme_name] = tid
                    themes_rows.append({"id": tid, "name": theme_name})

        # ── Decks, DecksThemes, Cards ─────────────────────────────────────
        decks_rows: list[dict[str, Any]] = []
        decks_themes_rows: list[dict[str, Any]] = []
        cards_rows: list[dict[str, Any]] = []

        used_slugs: set[str] = set()

        for deck in decks_data:
            deck_id = _new_id()

            slug = _slugify(f"{deck['name']}-{deck['level']}")
            if slug in used_slugs:
                slug = f"{slug}-{deck_id[:8]}"
            used_slugs.add(slug)

            decks_rows.append({
                "id": deck_id,
                "slug": slug,
                "name": deck["name"],
                "description": deck.get("description"),
                "source_language_code": deck["source_language_code"],
                "target_language_code": deck["target_language_code"],
                "level": deck["level"],
            })

            for theme_name in deck["themes"]:
                decks_themes_rows.append({
                    "deck_id": deck_id,
                    "theme_id": theme_id_map[theme_name],
                })

            for position, word in enumerate(deck["words"], start=1):
                cards_rows.append({
                    "id": _new_id(),
                    "deck_id": deck_id,
                    "position": position,
                    "source_text": "",
                    "target_text": word,
                    "source_audio_url": None,
                    "target_audio_url": None,
                    "image_url": None,
                    "hint": None,
                    "notes": None,
                })

        catalog = {
            "themes": themes_rows,
            "decks": decks_rows,
            "decks_themes": decks_themes_rows,
            "cards": cards_rows,
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(catalog, f, indent=2, ensure_ascii=False)

        return catalog
