"""Unit tests use local fallbacks; neural and API paths are tested with mocks."""

import os

os.environ["RAG_OFFLINE"] = "1"
os.environ["RAG_ALLOW_MODEL_DOWNLOAD"] = "0"
os.environ["RAG_ENRICHMENT_CACHE"] = "0"
