"""Shared configuration for Lab 18."""

import os
from dotenv import load_dotenv

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT_DIR, ".env"))


def _env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


# Cached models work without a download. Enable downloads explicitly when needed.
OFFLINE_MODE = _env_bool("RAG_OFFLINE")
ALLOW_MODEL_DOWNLOAD = _env_bool("RAG_ALLOW_MODEL_DOWNLOAD")
ENRICHMENT_CACHE_ENABLED = _env_bool("RAG_ENRICHMENT_CACHE", True)
ENRICHMENT_CACHE_DIR = os.path.join(ROOT_DIR, ".cache", "enrichment")

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if OPENAI_API_KEY.lower() in {"sk-...", "your-api-key", "your_api_key_here", "changeme"}:
    OPENAI_API_KEY = ""
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "20"))

# --- Qdrant ---
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "lab18_production"
NAIVE_COLLECTION = "lab18_naive"

# --- Embedding ---
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# --- Chunking ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Paths ---
DATA_DIR = os.path.join(ROOT_DIR, "data")
TEST_SET_PATH = os.path.join(ROOT_DIR, "test_set.json")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
