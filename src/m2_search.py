from __future__ import annotations

"""Vietnamese BM25, dense retrieval, and reciprocal rank fusion.

Models use the local cache by default. Without a cached embedding model, dense
retrieval uses explicit token-hash vectors, a lexical fallback rather than a
semantic embedding model.
"""

import hashlib
import json
import math
import os
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from functools import cache, lru_cache

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from config import (
    BM25_TOP_K,
    COLLECTION_NAME,
    DENSE_TOP_K,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    HYBRID_TOP_K,
    QDRANT_HOST,
    QDRANT_PORT,
)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str


@lru_cache(maxsize=1)
def _word_tokenizer():
    try:
        from underthesea import word_tokenize

        return word_tokenize
    except Exception:  # noqa: BLE001 - Optional NLP backends may fail during import.
        # Vietnamese whitespace tokens remain useful without the optional model.
        return None


def segment_vietnamese(text: str) -> str:
    """Segment words while keeping compounds compatible with query whitespace."""
    text = unicodedata.normalize("NFC", text)
    if not text.strip():
        return ""
    tokenizer = _word_tokenizer()
    if tokenizer is not None:
        try:
            text = tokenizer(text, format="text")
        except Exception:  # noqa: BLE001 - Preserve tokenization if the optional model fails.
            return text.replace("_", " ")
    return text.replace("_", " ")


def _tokens(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", segment_vietnamese(text).casefold())


class _LocalBM25:
    """Standard positive-IDF BM25 for installations without rank-bm25."""

    def __init__(self, corpus: list[list[str]]):
        self.frequencies = [Counter(tokens) for tokens in corpus]
        self.lengths = [len(tokens) for tokens in corpus]
        self.average_length = sum(self.lengths) / len(corpus)
        document_frequency = Counter(
            term for frequencies in self.frequencies for term in frequencies
        )
        self.idf = {
            term: math.log(1 + (len(corpus) - count + 0.5) / (count + 0.5))
            for term, count in document_frequency.items()
        }

    def get_scores(self, query: list[str]) -> list[float]:
        scores = []
        for frequencies, length in zip(self.frequencies, self.lengths):
            score = 0.0
            for term in query:
                frequency = frequencies.get(term, 0)
                denominator = frequency + 1.5 * (
                    0.25 + 0.75 * length / self.average_length
                )
                score += self.idf.get(term, 0.0) * frequency * 2.5 / denominator
            scores.append(score)
        return scores


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        self.documents = list(chunks)
        self.corpus_tokens = [_tokens(chunk["text"]) for chunk in self.documents]
        self.bm25 = None
        if not any(self.corpus_tokens):
            return
        try:
            from rank_bm25 import BM25Okapi

            self.bm25 = BM25Okapi(self.corpus_tokens)
            # The original Okapi IDF can be zero/negative for tiny corpora. Use
            # the positive variant so a relevant singleton remains retrievable.
            self.bm25.idf = _LocalBM25(self.corpus_tokens).idf
        except ImportError:
            self.bm25 = _LocalBM25(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        if self.bm25 is None or top_k <= 0:
            return []
        query_tokens = _tokens(query)
        if not query_tokens:
            return []
        scores = self.bm25.get_scores(query_tokens)
        indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [
            SearchResult(
                text=self.documents[i]["text"],
                score=float(scores[i]),
                metadata=dict(self.documents[i].get("metadata", {})),
                method="bm25",
            )
            for i in indices[:top_k]
            if scores[i] > 0
        ]


@cache
def _notice(message: str) -> None:
    print(message, flush=True)


@lru_cache(maxsize=4)
def _qdrant_client(host: str, port: int, offline: bool):
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        _notice("Dense search: Qdrant unavailable; using local cosine search.")
        return None
    if not offline:
        try:
            client = QdrantClient(
                host=host, port=port, timeout=2, check_compatibility=False
            )
            client.get_collections()
            return client
        except Exception:  # noqa: BLE001 - A missing service must allow local retrieval.
            _notice("Dense search: Qdrant server unavailable; using in-memory Qdrant.")
    return QdrantClient(":memory:")


@lru_cache(maxsize=4)
def _load_encoder(model: str, allow_download: bool, offline: bool):
    if offline:
        return None
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model, local_files_only=not allow_download)
    except Exception:  # noqa: BLE001 - Optional model backends raise varied loader errors.
        # Cache failures too: each query must not retry imports or downloads.
        return None


def _hash_vectors(texts: list[str]) -> np.ndarray:
    vectors = np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)
    for row, text in enumerate(texts):
        for token, count in Counter(_tokens(text)).items():
            bucket = int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:8], "big")
            vectors[row, bucket % EMBEDDING_DIM] += math.log1p(count)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


@dataclass
class _DenseIndex:
    documents: list[dict]
    vectors: np.ndarray
    mode: str
    stored_in_qdrant: bool = False


class DenseSearch:
    def __init__(self):
        self.client = _qdrant_client(QDRANT_HOST, QDRANT_PORT, config.OFFLINE_MODE)
        self._encoder = None
        self._encoder_checked = False
        self._indexes: dict[str, _DenseIndex] = {}

    def _get_encoder(self):
        if self._encoder is None and not self._encoder_checked:
            self._encoder = _load_encoder(
                EMBEDDING_MODEL, config.ALLOW_MODEL_DOWNLOAD, config.OFFLINE_MODE
            )
            self._encoder_checked = True
        return self._encoder

    def _encode(self, texts: list[str], force_hash: bool = False) -> tuple[np.ndarray, str]:
        encoder = None if force_hash else self._get_encoder()
        if encoder is not None:
            try:
                vectors = np.asarray(
                    encoder.encode(texts, show_progress_bar=False, normalize_embeddings=True),
                    dtype=np.float32,
                )
                if vectors.ndim == 1 and len(texts) == 1:
                    vectors = vectors.reshape(1, -1)
                if (
                    vectors.ndim != 2
                    or vectors.shape[0] != len(texts)
                    or vectors.shape[1] == 0
                    or not np.isfinite(vectors).all()
                ):
                    raise ValueError("Encoder returned invalid vectors")
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                return vectors / np.maximum(norms, 1e-12), "encoder"
            except Exception:  # noqa: BLE001 - Recover from optional encoder backend failures.
                self._encoder = None
                self._encoder_checked = True
        _notice(
            "Dense search: using local token-hash vectors (lexical fallback; "
            "semantic model unavailable or offline)."
        )
        return _hash_vectors(texts), "hash"

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Replace only the selected collection, with its actual vector dimension."""
        documents = [
            {"text": chunk["text"], "metadata": dict(chunk.get("metadata", {}))}
            for chunk in chunks
            if chunk["text"].strip()
        ]
        if documents:
            vectors, mode = self._encode([chunk["text"] for chunk in documents])
        else:
            vectors, mode = np.empty((0, EMBEDDING_DIM), dtype=np.float32), "hash"
        index = _DenseIndex(documents, vectors, mode)
        self._indexes[collection] = index
        if self.client is None:
            return
        try:
            from qdrant_client.models import Distance, PointStruct, VectorParams

            if self.client.collection_exists(collection):
                self.client.delete_collection(collection)
            if not documents:
                return
            self.client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=int(vectors.shape[1]), distance=Distance.COSINE),
            )
            for start in range(0, len(documents), 128):
                points = [
                    PointStruct(id=i, vector=vectors[i].tolist(), payload=documents[i])
                    for i in range(start, min(start + 128, len(documents)))
                ]
                self.client.upsert(collection_name=collection, points=points, wait=True)
            index.stored_in_qdrant = True
        except Exception:  # noqa: BLE001 - Keep the local index available after service failures.
            _notice("Dense search: Qdrant operation failed; using local cosine search.")

    def search(
        self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME
    ) -> list[SearchResult]:
        index = self._indexes.get(collection)
        if top_k <= 0 or not query.strip() or index is None or not index.documents:
            return []
        vectors, mode = self._encode([query], force_hash=index.mode == "hash")
        if mode != index.mode or vectors.shape[1] != index.vectors.shape[1]:
            # If the model becomes unavailable, re-embed the corpus too. Comparing
            # token-hash queries with semantic document vectors is meaningless.
            self.index(index.documents, collection)
            index = self._indexes[collection]
            vectors, _ = self._encode([query], force_hash=index.mode == "hash")
        query_vector = vectors[0]
        if not np.any(query_vector):
            return []
        if index.stored_in_qdrant:
            try:
                response = self.client.query_points(
                    collection_name=collection,
                    query=query_vector.tolist(),
                    limit=top_k,
                    with_payload=True,
                )
                return [
                    SearchResult(
                        text=point.payload["text"],
                        score=float(point.score),
                        metadata=dict(point.payload.get("metadata", {})),
                        method="dense",
                    )
                    for point in response.points
                    if point.payload and point.score > 0
                ]
            except Exception:  # noqa: BLE001 - Fall back to the already computed local vectors.
                index.stored_in_qdrant = False
                _notice("Dense search: Qdrant operation failed; using local cosine search.")
        scores = index.vectors @ query_vector
        indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [
            SearchResult(
                text=index.documents[i]["text"],
                score=float(scores[i]),
                metadata=dict(index.documents[i]["metadata"]),
                method="dense",
            )
            for i in indices[:top_k]
            if scores[i] > 0
        ]


def reciprocal_rank_fusion(
    results_list: list[list[SearchResult]], k: int = 60, top_k: int = HYBRID_TOP_K
) -> list[SearchResult]:
    """Fuse one-based ranks without merging distinct sources or repeated votes."""
    if top_k <= 0:
        return []
    if k < 0:
        raise ValueError("RRF k must be nonnegative")
    scores: dict[tuple[str, str], float] = {}
    representatives: dict[tuple[str, str], SearchResult] = {}
    for results in results_list:
        seen = set()
        for rank, result in enumerate(results, start=1):
            identity = (
                result.text,
                json.dumps(result.metadata, sort_keys=True, ensure_ascii=False, default=str),
            )
            if identity in seen:
                continue
            seen.add(identity)
            representatives.setdefault(identity, result)
            scores[identity] = scores.get(identity, 0.0) + 1.0 / (k + rank)
    ranked = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [
        SearchResult(
            text=representatives[identity].text,
            score=scores[identity],
            metadata=dict(representatives[identity].metadata),
            method="hybrid",
        )
        for identity in ranked
    ]


class HybridSearch:
    """Combine BM25 and dense rankings through reciprocal rank fusion."""

    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        if top_k <= 0:
            return []
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
