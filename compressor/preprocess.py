from __future__ import annotations

import re


WHITESPACE_RE = re.compile(r"\s+")


def preprocess_text(text: str) -> str:
    """Normalize whitespace while preserving the user's wording."""
    return WHITESPACE_RE.sub(" ", text).strip()
