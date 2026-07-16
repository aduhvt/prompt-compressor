from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import tiktoken


@dataclass(slots=True)
class TokenStats:
    original_tokens: int
    compressed_tokens: int
    reduction_percentage: float


@lru_cache(maxsize=4)
def _get_encoding(encoding_name: str) -> tiktoken.Encoding | None:
    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception:
        return None


def _fallback_count(text: str) -> int:
    # A stable offline approximation for environments where tiktoken assets
    # were not pre-cached yet.
    words = [word for word in text.split() if word]
    punctuation_units = sum(1 for char in text if not char.isalnum() and not char.isspace())
    return max(1, len(words) + punctuation_units)


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    encoding = _get_encoding(encoding_name)
    if encoding is None:
        return _fallback_count(text)
    return len(encoding.encode(text))


def compare_token_counts(
    original_text: str,
    compressed_text: str,
    encoding_name: str = "cl100k_base",
) -> TokenStats:
    original_tokens = count_tokens(original_text, encoding_name=encoding_name)
    compressed_tokens = count_tokens(compressed_text, encoding_name=encoding_name)

    if original_tokens == 0:
        reduction_percentage = 0.0
    else:
        reduction_percentage = ((original_tokens - compressed_tokens) / original_tokens) * 100

    return TokenStats(
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        reduction_percentage=reduction_percentage,
    )
