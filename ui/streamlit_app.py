from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from compressor.compress import CompressionError, compress_prompt, get_ollama_status
from compressor.similarity import DEFAULT_SIMILARITY_MODEL, calculate_similarity
from compressor.token_counter import compare_token_counts


st.set_page_config(
    page_title="Prompt Compressor",
    page_icon="PC",
    layout="wide",
)


def render_header() -> None:
    st.title("Prompt Compressor")
    st.caption("Local prompt compression with Ollama, token analysis, and similarity scoring.")


def render_sidebar() -> tuple[str, str]:
    with st.sidebar:
        st.header("Settings")
        mode = st.selectbox(
            "Compression mode",
            options=["lossless", "balanced", "aggressive"],
            index=1,
            help="Choose how aggressively the prompt should be shortened.",
        )
        model = st.text_input(
            "Ollama model",
            value=os.getenv("OLLAMA_MODEL", "qwen3:4b"),
            help="This model must exist in your local Ollama installation.",
        )
        st.markdown("**Similarity model path**")
        st.code(DEFAULT_SIMILARITY_MODEL)
        st.caption("Place the sentence-transformer model there for fully offline similarity scoring.")
        return mode, model


def render_runtime_status(selected_model: str) -> None:
    status = get_ollama_status()
    if not status.installed:
        st.error("Ollama is not installed or not available on PATH.")
        return
    if not status.running:
        st.warning("Ollama is installed, but its local runtime is not responding.")
        return

    st.success("Ollama runtime detected.")
    if status.available_models:
        models = ", ".join(status.available_models)
        st.caption(f"Installed Ollama models: {models}")
        if selected_model not in status.available_models:
            st.warning(f"Selected model `{selected_model}` is not currently installed.")


def render_metrics(original_text: str, compressed_text: str) -> None:
    token_stats = compare_token_counts(original_text, compressed_text)
    similarity = calculate_similarity(original_text, compressed_text)

    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("Original tokens", token_stats.original_tokens)
    metric_b.metric("Compressed tokens", token_stats.compressed_tokens)
    metric_c.metric("Reduction", f"{token_stats.reduction_percentage:.1f}%")

    st.progress(similarity.score)
    st.caption(f"Semantic similarity: {similarity.score:.3f} via `{similarity.method}`")

    if similarity.method != "sentence-transformers":
        st.warning(
            "Sentence-transformer weights were not found locally, so similarity is using a token-overlap fallback."
        )


def main() -> None:
    render_header()
    mode, model = render_sidebar()
    render_runtime_status(model)

    default_prompt = (
        "Write a concise launch announcement for our offline prompt compression app. "
        "Mention that it runs locally with Ollama, measures token savings, and keeps meaning intact."
    )
    prompt_text = st.text_area("Input prompt", value=default_prompt, height=220, placeholder="Paste a prompt here...")

    if st.button("Compress", type="primary", use_container_width=True):
        if not prompt_text.strip():
            st.error("Enter a prompt before compressing.")
            return

        with st.spinner("Compressing prompt locally..."):
            try:
                result = compress_prompt(prompt_text, mode=mode, model=model)
            except ValueError as exc:
                st.error(str(exc))
                return
            except CompressionError as exc:
                st.error(str(exc))
                return

        left_col, right_col = st.columns(2)
        with left_col:
            st.subheader("Original")
            st.text_area("Original prompt", value=result.original_text, height=260, disabled=True)
        with right_col:
            st.subheader("Compressed")
            st.text_area("Compressed prompt", value=result.compressed_text, height=260, disabled=True)

        render_metrics(result.original_text, result.compressed_text)


if __name__ == "__main__":
    main()
