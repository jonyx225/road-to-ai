"""Central settings. Override any of them with environment variables."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = Path(os.getenv("RAG_DOCS_DIR", BASE_DIR / "data" / "docs"))
INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR", BASE_DIR / "index"))
EVAL_PATH = BASE_DIR / "data" / "eval_questions.jsonl"

# ------------------------------------------------------------------ chunking
# Sizes are in WORDS (English averages roughly 1.3 tokens per word).
# 160 words is about 210 tokens: small enough to fit inside all-MiniLM-L6-v2's
# input limit (it truncates at 256 word pieces), big enough to hold a full idea.
CHUNK_WORDS = int(os.getenv("RAG_CHUNK_WORDS", "160"))
CHUNK_OVERLAP_WORDS = int(os.getenv("RAG_CHUNK_OVERLAP_WORDS", "30"))

# ------------------------------------------------------------------ embeddings
# "sbert" = sentence-transformers model (real semantic search, downloads ~90 MB)
# "tfidf" = offline character n-gram TF-IDF (no downloads; used by the tests)
EMBEDDER = os.getenv("RAG_EMBEDDER", "sbert")
SBERT_MODEL = os.getenv("RAG_SBERT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANK_MODEL = os.getenv("RAG_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# ------------------------------------------------------------------ retrieval
DEFAULT_MODE = "hybrid"      # "dense", "bm25" or "hybrid"
TOP_K = 4                    # chunks handed to the LLM
CANDIDATE_K = 20             # how many candidates each retriever contributes before fusing
RRF_K = 60                   # reciprocal-rank-fusion constant (60 is the usual default)

# ------------------------------------------------------------------ generation
LLM_BACKEND = os.getenv("RAG_LLM", "anthropic")   # "anthropic" or "fake" (offline demo)
LLM_MODEL = os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001")
# Set LLM_TEMPERATURE=none if your chosen model rejects the temperature parameter.
_temperature = os.getenv("LLM_TEMPERATURE", "0")
LLM_TEMPERATURE = None if _temperature.lower() == "none" else float(_temperature)
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "600"))

# USD per million tokens. Defaults are Claude Haiku 4.5's list prices at the time of
# writing. PRICES CHANGE: check https://docs.claude.com and update if you switch models.
LLM_PRICE_INPUT_PER_MTOK = float(os.getenv("LLM_PRICE_INPUT_PER_MTOK", "1.00"))
LLM_PRICE_OUTPUT_PER_MTOK = float(os.getenv("LLM_PRICE_OUTPUT_PER_MTOK", "5.00"))
