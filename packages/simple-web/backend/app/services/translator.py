"""Language detection + translation to English via Google Translate."""
from __future__ import annotations

import re


def is_likely_english(text: str) -> bool:
    """Quick heuristic: if >60% of chars are ASCII letters, it's English."""
    ascii_letters = sum(1 for c in text if c.isascii() and c.isalpha())
    all_letters = sum(1 for c in text if c.isalpha())
    if all_letters == 0:
        return True
    return ascii_letters / all_letters > 0.6


def translate_to_english(texts: list[str]) -> tuple[list[str], bool]:
    """Translate texts to English if non-English detected.

    Returns (translated_texts, was_translated).
    """
    if not texts:
        return texts, False

    # Check first few texts
    sample = " ".join(texts[:3])
    if is_likely_english(sample):
        return texts, False

    try:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source="auto", target="en")
        translated = []
        for t in texts:
            if is_likely_english(t):
                translated.append(t)
            else:
                result = translator.translate(t[:5000])
                translated.append(result or t)
        return translated, True
    except Exception:
        # Fallback: return original
        return texts, False
