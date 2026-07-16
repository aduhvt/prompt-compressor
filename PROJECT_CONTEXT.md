# Prompt Compressor Project

## Goal
Build a local AI-powered prompt compression application.

Input:
- User enters text in natural language.

Output:
- Compressed prompt that preserves meaning.
- Token reduction percentage.
- Semantic similarity score.

## Tech Stack

### AI Model
- Qwen 3 4B (Ollama)

### Backend
- Python 3.12

### Libraries
- ollama
- sentence-transformers
- tiktoken
- streamlit

### UI
- Streamlit

## Architecture

User Input
    ↓
Prompt Compressor (Qwen)
    ↓
Compressed Prompt
    ↓
Token Counter
    ↓
Similarity Checker
    ↓
Streamlit UI

## Development Roadmap

### V1
- Text input
- Compress button
- Output compressed text

### V2
- Token counting

### V3
- Semantic similarity

### V4
- Compression modes:
  - Lossless
  - Balanced
  - Aggressive

## Constraints

- Must work fully offline after model download.
- Must run on GTX 1650.
- Must run locally.
- No cloud APIs.

## Current Status

- Project initialized.
- Tech stack finalized.
- Requirements defined.