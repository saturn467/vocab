# Card Enrichment Pipeline

Production-ready script that enriches language-learning flashcard objects with
English translations, French pronunciation audio, and optional images.

## Folder Structure

```
enrichment/
├── enrich.py              # Main enrichment script
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variable template
├── README.md              # This file
├── examples/
│   ├── input.json         # Example input (4 cards)
│   └── output.json        # Example output (enriched)
│
├── audio/                 # Generated at runtime — French TTS files
│   └── {card_id}.mp3
├── images/                # Generated at runtime — vocabulary images
│   └── {card_id}.jpg
└── .enrichment_checkpoint.json  # Auto-created for resumability
```

## Installation

```bash
cd enrichment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

## Configuration

```bash
cp .env.example .env
# Edit .env and add your API keys
```

### Provider Options

| Feature     | Provider  | API Key Required       | Cost                |
|-------------|-----------|------------------------|---------------------|
| Translation | `openai`  | `OPENAI_API_KEY`       | ~$0.01 / 50 words   |
| Translation | `deepl`   | `DEEPL_API_KEY`        | Free tier: 500K chars/month |
| Audio       | `gtts`    | None (free)            | Free                |
| Audio       | `openai`  | `OPENAI_API_KEY`       | $0.015 / 1K chars   |
| Images      | `unsplash`| `UNSPLASH_ACCESS_KEY`  | Free: 50 req/hour   |
| Images      | `pexels`  | `PEXELS_API_KEY`       | Free: 200 req/hour  |
| Images      | `none`    | —                      | —                   |

**Recommended starting configuration:**
- Translation: `openai` (includes concreteness classification for image selection)
- Audio: `gtts` (free, decent quality)
- Images: `none` (enable later if needed)

## How to Run

```bash
# Basic usage (reads catalog.json, writes catalog_enriched.json)
python enrich.py

# Custom input/output
python enrich.py --input cards.json --output cards_enriched.json

# Verbose logging
python enrich.py -v --input cards.json --output enriched.json
```

### Input Format

The input JSON file can be either:

1. The full catalog object from the taxonomer (recommended):
```json
{
  "themes": [...],
  "decks": [{"id": "...", "name": "...", "level": "A2", ...}],
  "decks_themes": [{"deck_id": "...", "theme_id": "..."}],
  "cards": [{"id": "...", "target_text": "maison", "source_text": "", ...}]
}
```

2. A flat array of card objects (decks/themes will be empty in the output):
```json
[{"id": "...", "target_text": "maison", "source_text": "", ...}]
```

The output preserves the complete catalog structure — `themes`, `decks`,
`decks_themes`, and enriched `cards` — so the result is ready for database
seeding without needing to reassemble from separate files.

### Resumability

The script saves a checkpoint every 50 cards (configurable via `CHECKPOINT_INTERVAL`).
If interrupted, re-run the same command — it will resume from the checkpoint automatically.

To force a fresh start, delete `.enrichment_checkpoint.json`.

## Performance & Scaling

### 3,000 Cards (Typical Run)

| Phase       | Provider  | Time       | Cost        |
|-------------|-----------|------------|-------------|
| Translation | OpenAI    | ~2 min     | ~$0.60      |
| Audio       | gTTS      | ~15 min    | Free        |
| Audio       | OpenAI    | ~5 min     | ~$0.45      |
| Images      | Unsplash  | ~30 min    | Free        |
| **Total**   | OpenAI+gTTS | **~17 min** | **~$0.60** |

### 10,000 Cards

| Phase       | Provider  | Time       | Cost        |
|-------------|-----------|------------|-------------|
| Translation | OpenAI    | ~7 min     | ~$2.00      |
| Audio       | gTTS      | ~50 min    | Free        |
| Audio       | OpenAI    | ~15 min    | ~$1.50      |
| Images      | Unsplash  | ~2 hr      | Free        |

### Tuning Knobs

| Variable                | Default | Effect |
|-------------------------|---------|--------|
| `TRANSLATION_BATCH_SIZE`| 50      | Words per translation API call. Larger = fewer calls but higher per-call risk. Max ~80 for OpenAI. |
| `AUDIO_CONCURRENCY`     | 10      | Parallel audio generation tasks. gTTS is rate-sensitive; lower to 5 if you get 429s. |
| `IMAGE_CONCURRENCY`     | 5       | Parallel image downloads. Unsplash free tier allows 50 req/hour. |
| `CHECKPOINT_INTERVAL`   | 50      | Cards between checkpoint saves. Lower = safer but more disk I/O. |

## API Cost Notes

### OpenAI
- **Translation** (`gpt-4o-mini`): ~$0.15 per 1M input tokens, ~$0.60 per 1M output tokens.
  Batch of 50 words ≈ 500 tokens → ~$0.01 per batch.
- **TTS** (`tts-1`): $0.015 per 1,000 characters. Single word ≈ 8 chars → ~$0.00012 per card.
  3,000 cards ≈ $0.36.

### DeepL
- Free tier: 500,000 characters/month. 10,000 words ≈ 80,000 characters → well within limits.
- Pro: €5.49/month + €25 per 1M characters.

### gTTS
- Free (uses Google Translate's TTS endpoint). No API key required.
- Rate-limited; the script handles 429 responses with retries.

### Unsplash / Pexels
- Both offer free tiers suitable for development.
- Unsplash: 50 requests/hour (demo), 5,000/hour (production key).
- Pexels: 200 requests/hour, 20,000/month.

## Notes

- **DeepL + Images**: DeepL does not classify word concreteness, so no images will
  be fetched when using DeepL for translation. Switch to OpenAI if you need images,
  or set `IMAGE_PROVIDER=none`.
- **Encoding**: All file I/O uses UTF-8. French diacritics (é, è, ê, ç, etc.) are
  preserved throughout.
- **Idempotent**: Re-running the script on already-enriched cards is safe — it skips
  cards that already have translations, audio files, and images.
- **Extending**: To add `hint` or `notes` generation, add a new phase in
  `enrich.py:run()` following the same pattern as `translate_cards`.
