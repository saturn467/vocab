#!/usr/bin/env python3
"""
Card Enrichment Pipeline
========================

Enriches language-learning flashcard objects with:
  • English translations           → source_text
  • French pronunciation audio     → target_audio_url
  • Optional images (concrete nouns) → image_url

Usage:
    python enrich.py --input cards.json --output cards_enriched.json

Environment variables (see .env.example):
    TRANSLATION_PROVIDER   openai | deepl           (default: openai)
    AUDIO_PROVIDER         gtts   | openai          (default: gtts)
    IMAGE_PROVIDER         unsplash | pexels | none (default: none)
    OPENAI_API_KEY         Required for OpenAI translation / TTS
    DEEPL_API_KEY          Required for DeepL translation
    UNSPLASH_ACCESS_KEY    Required for Unsplash images
    PEXELS_API_KEY         Required for Pexels images
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import aiohttp
from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

TRANSLATION_PROVIDER: str = os.getenv("TRANSLATION_PROVIDER", "openai").lower()
AUDIO_PROVIDER: str = os.getenv("AUDIO_PROVIDER", "gtts").lower()
IMAGE_PROVIDER: str = os.getenv("IMAGE_PROVIDER", "none").lower()

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TTS_MODEL: str = os.getenv("OPENAI_TTS_MODEL", "tts-1")
OPENAI_TTS_VOICE: str = os.getenv("OPENAI_TTS_VOICE", "alloy")

DEEPL_API_KEY: str = os.getenv("DEEPL_API_KEY", "")

UNSPLASH_ACCESS_KEY: str = os.getenv("UNSPLASH_ACCESS_KEY", "")
PEXELS_API_KEY: str = os.getenv("PEXELS_API_KEY", "")

BATCH_SIZE: int = int(os.getenv("TRANSLATION_BATCH_SIZE", "50"))
AUDIO_CONCURRENCY: int = int(os.getenv("AUDIO_CONCURRENCY", "10"))
IMAGE_CONCURRENCY: int = int(os.getenv("IMAGE_CONCURRENCY", "5"))
CHECKPOINT_EVERY: int = int(os.getenv("CHECKPOINT_INTERVAL", "50"))

AUDIO_DIR = Path(os.getenv("AUDIO_DIR", "audio"))
IMAGE_DIR = Path(os.getenv("IMAGE_DIR", "images"))
CHECKPOINT_FILE = Path(os.getenv("CHECKPOINT_FILE", ".enrichment_checkpoint.json"))

log = logging.getLogger("enrich")


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def load_catalog(path: Path) -> dict[str, Any]:
    """Load the full catalog JSON.

    Accepts either:
      • A full catalog object ``{"themes": [...], "decks": [...], ...}``
      • A flat array of card objects (wrapped into a catalog automatically)
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return {"themes": [], "decks": [], "decks_themes": [], "cards": data}
    if isinstance(data, dict) and "cards" in data:
        return data
    raise ValueError(f"Unrecognised input format in {path}")


def save_catalog(catalog: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def save_checkpoint(catalog: dict[str, Any]) -> None:
    save_catalog(catalog, CHECKPOINT_FILE)
    log.debug("Checkpoint saved (%d cards)", len(catalog["cards"]))


def load_checkpoint() -> dict[str, Any] | None:
    if CHECKPOINT_FILE.exists():
        log.info("Resuming from checkpoint %s", CHECKPOINT_FILE)
        return load_catalog(CHECKPOINT_FILE)
    return None


def needs_translation(card: dict[str, Any]) -> bool:
    return not card.get("source_text")


def needs_audio(card: dict[str, Any]) -> bool:
    url = card.get("target_audio_url")
    if url and Path(url).exists():
        return False
    return True


def needs_image(card: dict[str, Any]) -> bool:
    url = card.get("image_url")
    if url and Path(url).exists():
        return False
    return True


def _validate_config() -> None:
    """Exit early if required API keys are missing."""
    errors: list[str] = []
    if TRANSLATION_PROVIDER == "openai" and not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY is required when TRANSLATION_PROVIDER=openai")
    if TRANSLATION_PROVIDER == "deepl" and not DEEPL_API_KEY:
        errors.append("DEEPL_API_KEY is required when TRANSLATION_PROVIDER=deepl")
    if AUDIO_PROVIDER == "openai" and not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY is required when AUDIO_PROVIDER=openai")
    if IMAGE_PROVIDER == "unsplash" and not UNSPLASH_ACCESS_KEY:
        errors.append("UNSPLASH_ACCESS_KEY is required when IMAGE_PROVIDER=unsplash")
    if IMAGE_PROVIDER == "pexels" and not PEXELS_API_KEY:
        errors.append("PEXELS_API_KEY is required when IMAGE_PROVIDER=pexels")
    if errors:
        for e in errors:
            log.error(e)
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════
#  TRANSLATION
# ═══════════════════════════════════════════════════════════════════════════

_RETRYABLE = (aiohttp.ClientError, asyncio.TimeoutError, ValueError)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_exception_type(_RETRYABLE),
)
async def _translate_openai(
    session: aiohttp.ClientSession,
    words: list[str],
    semaphore: asyncio.Semaphore,
) -> list[dict[str, Any]]:
    """Batch-translate French → English via OpenAI, with concreteness flag."""
    prompt = (
        "Translate each French word/phrase below into English.\n"
        "For each word, indicate whether it is a concrete, visualisable concept "
        "(a noun that could be depicted in a photograph).\n\n"
        'Return ONLY a JSON object: {"translations": [{"word": "…", '
        '"translation": "…", "concrete": true/false}, …]}\n'
        "Maintain the EXACT input order.\n\n"
        f"French words:\n{json.dumps(words, ensure_ascii=False)}"
    )
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }

    async with semaphore:
        async with session.post(
            "https://api.openai.com/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            if resp.status == 429:
                retry_after = int(resp.headers.get("Retry-After", "10"))
                log.warning("OpenAI rate-limited — waiting %ds", retry_after)
                await asyncio.sleep(retry_after)
                raise aiohttp.ClientError("rate limited")
            resp.raise_for_status()
            body = await resp.json()

    content: str = body["choices"][0]["message"]["content"]
    parsed = json.loads(content)

    if isinstance(parsed, dict) and "translations" in parsed:
        result = parsed["translations"]
    elif isinstance(parsed, list):
        result = parsed
    else:
        raise ValueError(f"Unexpected OpenAI response: {content[:200]}")

    # Pad if the model returned fewer items than requested
    while len(result) < len(words):
        idx = len(result)
        result.append({"word": words[idx], "translation": "", "concrete": False})

    return result


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_exception_type(_RETRYABLE),
)
async def _translate_deepl(
    session: aiohttp.ClientSession,
    words: list[str],
    semaphore: asyncio.Semaphore,
) -> list[dict[str, Any]]:
    """Batch-translate French → English via DeepL."""
    base = (
        "https://api-free.deepl.com"
        if DEEPL_API_KEY.endswith(":fx")
        else "https://api.deepl.com"
    )
    headers = {"Authorization": f"DeepL-Auth-Key {DEEPL_API_KEY}"}
    payload = {"text": words, "source_lang": "FR", "target_lang": "EN"}

    async with semaphore:
        async with session.post(
            f"{base}/v2/translate",
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            if resp.status == 429:
                await asyncio.sleep(10)
                raise aiohttp.ClientError("rate limited")
            resp.raise_for_status()
            body = await resp.json()

    return [
        {"word": w, "translation": t["text"], "concrete": False}
        for w, t in zip(words, body["translations"])
    ]


async def translate_cards(
    catalog: dict[str, Any],
    session: aiohttp.ClientSession,
) -> None:
    """Phase 1: fill ``source_text`` for every card that needs it."""
    cards = catalog["cards"]
    pending = [(i, c) for i, c in enumerate(cards) if needs_translation(c)]
    if not pending:
        log.info("Translation: all cards already translated — skipping")
        return

    log.info("Translating %d / %d cards (provider=%s) …",
             len(pending), len(cards), TRANSLATION_PROVIDER)

    semaphore = asyncio.Semaphore(3)
    translate_fn = (
        _translate_openai if TRANSLATION_PROVIDER == "openai" else _translate_deepl
    )

    batches = [pending[i : i + BATCH_SIZE] for i in range(0, len(pending), BATCH_SIZE)]
    processed = 0

    for batch in tqdm(batches, desc="Translating", unit="batch"):
        words = [c["target_text"] for _, c in batch]
        try:
            results = await translate_fn(session, words, semaphore)
            for (_, card), result in zip(batch, results):
                card["source_text"] = result.get("translation", "")
                card["_concrete"] = result.get("concrete", False)
        except Exception:
            log.exception("Batch translation failed — leaving source_text empty")

        processed += len(batch)
        if processed % CHECKPOINT_EVERY < BATCH_SIZE:
            save_checkpoint(catalog)

    save_checkpoint(catalog)
    log.info("Translation complete")


# ═══════════════════════════════════════════════════════════════════════════
#  AUDIO GENERATION
# ═══════════════════════════════════════════════════════════════════════════

async def _generate_gtts(word: str, output_path: Path) -> None:
    """Generate French pronunciation audio with gTTS (free, no key)."""
    from gtts import gTTS  # imported here so gtts is optional at module level

    def _blocking() -> None:
        tts = gTTS(text=word, lang="fr")
        tts.save(str(output_path))

    await asyncio.to_thread(_blocking)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(_RETRYABLE),
)
async def _generate_openai_tts(
    session: aiohttp.ClientSession,
    word: str,
    output_path: Path,
    semaphore: asyncio.Semaphore,
) -> None:
    """Generate French pronunciation audio with OpenAI TTS."""
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENAI_TTS_MODEL,
        "voice": OPENAI_TTS_VOICE,
        "input": word,
    }
    async with semaphore:
        async with session.post(
            "https://api.openai.com/v1/audio/speech",
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            if resp.status == 429:
                await asyncio.sleep(5)
                raise aiohttp.ClientError("rate limited")
            resp.raise_for_status()
            output_path.write_bytes(await resp.read())


async def generate_audio(
    catalog: dict[str, Any],
    session: aiohttp.ClientSession,
) -> None:
    """Phase 2: generate pronunciation audio for every card that needs it."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    cards = catalog["cards"]

    pending = [(i, c) for i, c in enumerate(cards) if needs_audio(c)]
    if not pending:
        log.info("Audio: all cards already have audio — skipping")
        return

    log.info("Generating audio for %d / %d cards (provider=%s) …",
             len(pending), len(cards), AUDIO_PROVIDER)

    semaphore = asyncio.Semaphore(AUDIO_CONCURRENCY)

    async def _one(card: dict[str, Any]) -> None:
        audio_path = AUDIO_DIR / f"{card['id']}.mp3"
        if audio_path.exists():
            card["target_audio_url"] = str(audio_path)
            return
        try:
            if AUDIO_PROVIDER == "openai":
                await _generate_openai_tts(session, card["target_text"],
                                           audio_path, semaphore)
            else:
                async with semaphore:
                    await _generate_gtts(card["target_text"], audio_path)
            card["target_audio_url"] = str(audio_path)
        except Exception:
            log.exception("Audio failed for '%s'", card["target_text"])

    tasks = [_one(c) for _, c in pending]
    pbar = tqdm(total=len(tasks), desc="Audio", unit="card")
    completed = 0
    for coro in asyncio.as_completed(tasks):
        await coro
        completed += 1
        pbar.update(1)
        if completed % CHECKPOINT_EVERY == 0:
            save_checkpoint(catalog)
    pbar.close()

    save_checkpoint(catalog)
    log.info("Audio generation complete")


# ═══════════════════════════════════════════════════════════════════════════
#  IMAGE FETCHING (optional)
# ═══════════════════════════════════════════════════════════════════════════

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(_RETRYABLE),
)
async def _fetch_unsplash(
    session: aiohttp.ClientSession,
    query: str,
    output_path: Path,
    semaphore: asyncio.Semaphore,
) -> bool:
    params = {
        "query": query,
        "per_page": "1",
        "orientation": "squarish",
        "client_id": UNSPLASH_ACCESS_KEY,
    }
    async with semaphore:
        async with session.get(
            "https://api.unsplash.com/search/photos",
            params=params,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            if resp.status == 429:
                await asyncio.sleep(10)
                raise aiohttp.ClientError("rate limited")
            resp.raise_for_status()
            data = await resp.json()

    results = data.get("results", [])
    if not results:
        return False

    url = results[0]["urls"].get("small", results[0]["urls"]["regular"])
    async with semaphore:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            resp.raise_for_status()
            output_path.write_bytes(await resp.read())
    return True


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(_RETRYABLE),
)
async def _fetch_pexels(
    session: aiohttp.ClientSession,
    query: str,
    output_path: Path,
    semaphore: asyncio.Semaphore,
) -> bool:
    headers = {"Authorization": PEXELS_API_KEY}
    params = {"query": query, "per_page": "1", "size": "small"}
    async with semaphore:
        async with session.get(
            "https://api.pexels.com/v1/search",
            headers=headers,
            params=params,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            if resp.status == 429:
                await asyncio.sleep(10)
                raise aiohttp.ClientError("rate limited")
            resp.raise_for_status()
            data = await resp.json()

    photos = data.get("photos", [])
    if not photos:
        return False

    url = photos[0]["src"]["small"]
    async with semaphore:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            resp.raise_for_status()
            output_path.write_bytes(await resp.read())
    return True


async def fetch_images(
    catalog: dict[str, Any],
    session: aiohttp.ClientSession,
) -> None:
    """Phase 3: fetch images for concrete nouns (optional)."""
    if IMAGE_PROVIDER == "none":
        log.info("Images: provider set to 'none' — skipping")
        return

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    cards = catalog["cards"]

    pending = [
        (i, c) for i, c in enumerate(cards)
        if needs_image(c) and c.get("_concrete", False)
    ]
    if not pending:
        log.info("Images: no concrete nouns remaining — skipping")
        return

    log.info("Fetching images for %d concrete nouns (provider=%s) …",
             len(pending), IMAGE_PROVIDER)

    semaphore = asyncio.Semaphore(IMAGE_CONCURRENCY)
    fetch_fn = _fetch_unsplash if IMAGE_PROVIDER == "unsplash" else _fetch_pexels

    async def _one(card: dict[str, Any]) -> None:
        image_path = IMAGE_DIR / f"{card['id']}.jpg"
        if image_path.exists():
            card["image_url"] = str(image_path)
            return
        query = card.get("source_text") or card["target_text"]
        try:
            found = await fetch_fn(session, query, image_path, semaphore)
            if found:
                card["image_url"] = str(image_path)
        except Exception:
            log.exception("Image fetch failed for '%s'", query)

    tasks = [_one(c) for _, c in pending]
    pbar = tqdm(total=len(tasks), desc="Images", unit="card")
    completed = 0
    for coro in asyncio.as_completed(tasks):
        await coro
        completed += 1
        pbar.update(1)
        if completed % CHECKPOINT_EVERY == 0:
            save_checkpoint(catalog)
    pbar.close()

    save_checkpoint(catalog)
    log.info("Image fetching complete")


# ═══════════════════════════════════════════════════════════════════════════
#  PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

async def run(input_path: Path, output_path: Path) -> None:
    catalog = load_checkpoint() or load_catalog(input_path)
    cards = catalog["cards"]
    log.info("Loaded %d cards (%d decks, %d themes)",
             len(cards), len(catalog.get("decks", [])),
             len(catalog.get("themes", [])))

    connector = aiohttp.TCPConnector(limit=20)
    async with aiohttp.ClientSession(connector=connector) as session:
        await translate_cards(catalog, session)
        await generate_audio(catalog, session)
        await fetch_images(catalog, session)

    # Strip internal bookkeeping fields before writing output
    for card in cards:
        card.pop("_concrete", None)

    catalog["cards"] = cards
    save_catalog(catalog, output_path)
    log.info("Saved enriched catalog → %s", output_path)

    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        log.info("Checkpoint removed — run complete")

    # Print summary
    translated = sum(1 for c in cards if c.get("source_text"))
    with_audio = sum(1 for c in cards if c.get("target_audio_url"))
    with_image = sum(1 for c in cards if c.get("image_url"))
    print(f"\n── Enrichment Summary ──────────────────────────────────")
    print(f"  Total cards:    {len(cards)}")
    print(f"  Translated:     {translated}")
    print(f"  With audio:     {with_audio}")
    print(f"  With image:     {with_image}")
    print(f"  Decks:          {len(catalog.get('decks', []))}")
    print(f"  Themes:         {len(catalog.get('themes', []))}")
    print(f"  Deck-themes:    {len(catalog.get('decks_themes', []))}")
    print(f"  Output file:    {output_path}")
    print(f"────────────────────────────────────────────────────────\n")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich flashcard objects with translations, audio, and images",
    )
    parser.add_argument(
        "-i", "--input",
        type=Path,
        default=Path("catalog.json"),
        help="Input JSON file (default: catalog.json)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("catalog_enriched.json"),
        help="Output JSON file (default: catalog_enriched.json)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    _validate_config()

    log.info("Providers — translation: %s  audio: %s  images: %s",
             TRANSLATION_PROVIDER, AUDIO_PROVIDER, IMAGE_PROVIDER)

    asyncio.run(run(args.input, args.output))


if __name__ == "__main__":
    main()
