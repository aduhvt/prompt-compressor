from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import Literal

import ollama

from compressor.preprocess import preprocess_text

CompressionMode = Literal["lossless", "balanced", "aggressive"]


MODE_INSTRUCTIONS: dict[CompressionMode, str] = {
    "lossless": (
        "Preserve every explicit requirement, constraint, and piece of intent. "
        "Only remove redundancy and rewrite for concision."
    ),
    "balanced": (
        "Preserve the core meaning and important constraints while compressing "
        "aggressively enough to reduce token count noticeably."
    ),
    "aggressive": (
        "Maximize brevity while keeping the primary task, key constraints, and "
        "critical context intact."
    ),
}

SYSTEM_PROMPT = (
    "You compress prompts for downstream LLM use. Return only the compressed prompt. "
    "Do not add commentary, bullets, quotes, headings, or explanations."
)


@dataclass(slots=True)
class CompressionResult:
    original_text: str
    compressed_text: str
    mode: CompressionMode
    model: str


class CompressionError(RuntimeError):
    """Raised when the local compression model cannot be used."""


@dataclass(slots=True)
class OllamaStatus:
    installed: bool
    running: bool
    available_models: list[str]


def build_compression_prompt(text: str, mode: CompressionMode) -> str:
    instruction = MODE_INSTRUCTIONS[mode]
    return (
        f"Compression mode: {mode}\n"
        f"Instruction: {instruction}\n\n"
        "Rewrite the following prompt into a shorter version:\n"
        f"{text}"
    )


def get_ollama_status() -> OllamaStatus:
    installed = shutil.which("ollama") is not None

    try:
        response = ollama.list()
        models = sorted(
            {
                model.get("model") or model.get("name")
                for model in response.get("models", [])
                if model.get("model") or model.get("name")
            }
        )
        return OllamaStatus(installed=installed, running=True, available_models=models)
    except Exception:
        return OllamaStatus(installed=installed, running=False, available_models=[])


def compress_prompt(
    text: str,
    mode: CompressionMode = "balanced",
    model: str = "qwen3:4b",
) -> CompressionResult:
    cleaned_text = preprocess_text(text)
    if not cleaned_text:
        raise ValueError("Input text is empty after preprocessing.")

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_compression_prompt(cleaned_text, mode)},
            ],
            options={"temperature": 0.2},
        )
    except Exception as exc:  # pragma: no cover - depends on local Ollama runtime
        status = get_ollama_status()
        if not status.installed:
            raise CompressionError(
                "Ollama does not appear to be installed or available on PATH. "
                "Install Ollama locally, start it, then pull the model you want to use."
            ) from exc
        if not status.running:
            raise CompressionError(
                "Ollama appears to be installed but the local runtime is not responding on its API. "
                "Start Ollama, then retry."
            ) from exc
        if status.available_models and model not in status.available_models:
            available = ", ".join(status.available_models)
            raise CompressionError(
                f"Ollama is running, but the requested model `{model}` is not available. "
                f"Installed models: {available}"
            ) from exc
        raise CompressionError(
            "Ollama is reachable, but the requested compression call failed. "
            f"Verify that `{model}` is pulled and usable."
        ) from exc

    compressed_text = preprocess_text(response["message"]["content"])
    if not compressed_text:
        raise CompressionError("The model returned an empty compression result.")

    return CompressionResult(
        original_text=cleaned_text,
        compressed_text=compressed_text,
        mode=mode,
        model=model,
    )
