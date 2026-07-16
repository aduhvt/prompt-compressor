from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from compressor.preprocess import preprocess_text

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(MODULE_DIR)

DEFAULT_SIMILARITY_MODEL = os.getenv(
    "SIMILARITY_MODEL",
    os.path.join(PROJECT_ROOT, "models", "sentence-transformers", "all-MiniLM-L6-v2"),
)


@dataclass(slots=True)
class SimilarityResult:
    score: float
    method: str


def _tokenize(text: str) -> set[str]:
    return {token for token in preprocess_text(text).lower().split(" ") if token}


def _fallback_similarity(text_a: str, text_b: str) -> SimilarityResult:
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)
    if not tokens_a and not tokens_b:
        return SimilarityResult(score=1.0, method="token-overlap-fallback")
    if not tokens_a or not tokens_b:
        return SimilarityResult(score=0.0, method="token-overlap-fallback")

    intersection = len(tokens_a & tokens_b)
    magnitude = math.sqrt(len(tokens_a) * len(tokens_b))
    score = intersection / magnitude if magnitude else 0.0
    return SimilarityResult(score=max(0.0, min(score, 1.0)), method="token-overlap-fallback")


@lru_cache(maxsize=1)
def _load_model(model_name_or_path: str = DEFAULT_SIMILARITY_MODEL) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name_or_path, local_files_only=True)


def calculate_similarity(
    original_text: str,
    compressed_text: str,
    model_name_or_path: str = DEFAULT_SIMILARITY_MODEL,
) -> SimilarityResult:
    original = preprocess_text(original_text)
    compressed = preprocess_text(compressed_text)
    if not original or not compressed:
        return SimilarityResult(score=0.0, method="empty-input")

    if not os.path.isdir(model_name_or_path):
        return _fallback_similarity(original, compressed)

    try:
        model = _load_model(model_name_or_path)
        embeddings = model.encode([original, compressed], normalize_embeddings=True)
        score = float(embeddings[0] @ embeddings[1])
        return SimilarityResult(score=max(0.0, min(score, 1.0)), method="sentence-transformers")
    except Exception:
        return _fallback_similarity(original, compressed)
