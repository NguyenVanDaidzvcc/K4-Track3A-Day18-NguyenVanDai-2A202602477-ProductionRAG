from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import math
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from config import RERANK_TOP_K

_cross_encoder_cache = {}
_unavailable_cross_encoders = set()
_fallback_warning_shown = False


def _warn_fallback() -> None:
    global _fallback_warning_shown
    if not _fallback_warning_shown:
        print("  ℹ️ Reranker model chưa có hoặc đang chạy offline; xếp hạng bằng độ khớp từ vựng.", flush=True)
        _fallback_warning_shown = True


def _lexical_scores(query: str, documents: list[dict]) -> list[float]:
    stopwords = {"là", "của", "và", "có", "được", "cho", "trong", "một", "các",
                 "với", "những", "bao", "nhiêu", "nào", "the", "a", "an", "is", "of"}
    stopwords.update({"như", "thế", "khi", "từ", "về", "gì", "how", "what", "which", "does"})

    def tokens(text: str) -> list[str]:
        normalized = unicodedata.normalize("NFC", text).casefold()
        return [token for token in re.findall(r"[^\W_]+", normalized) if token not in stopwords]

    query_tokens = set(tokens(query))
    frequencies = [Counter(tokens(document["text"])) for document in documents]
    if not query_tokens or not frequencies:
        return [0.0] * len(documents)
    document_frequency = Counter(term for counts in frequencies for term in counts)
    idf = {
        term: math.log(1 + (len(documents) - document_frequency[term] + 0.5)
                       / (document_frequency[term] + 0.5))
        for term in query_tokens
    }
    # Squared IDF makes topic-specific terms outweigh words repeated across the
    # candidate corpus, without a domain-specific list of favored entities.
    coverage_weights = {term: weight * weight for term, weight in idf.items()}
    total_weight = sum(coverage_weights.values())
    lengths = [sum(counts.values()) for counts in frequencies]
    average_length = max(sum(lengths) / len(lengths), 1)
    scores = []
    for counts, length in zip(frequencies, lengths):
        coverage = sum(coverage_weights[term] for term in query_tokens if counts[term]) / total_weight
        bm25 = sum(
            coverage_weights[term] * counts[term]
            / (counts[term] + 1.2 * (0.4 + 0.6 * length / average_length))
            for term in query_tokens
        ) / total_weight
        scores.append(0.8 * coverage + 0.2 * bm25)
    return scores


def _rank_results(documents: list[dict], scores, top_k: int) -> list[RerankResult]:
    if hasattr(scores, "tolist"):
        scores = scores.tolist()
    if isinstance(scores, (int, float)):
        scores = [scores]
    values = []
    for score in scores:
        while isinstance(score, (list, tuple)) and len(score) == 1:
            score = score[0]
        value = float(score)
        if not math.isfinite(value):
            raise ValueError("Reranker scores must be finite")
        values.append(value)
    if len(values) != len(documents):
        raise ValueError("Reranker must return one score for every document")
    scored = sorted(zip(values, documents), key=lambda item: item[0], reverse=True)
    return [RerankResult(
        text=document["text"], original_score=float(document.get("score", 0.0) or 0.0),
        rerank_score=score, metadata=dict(document.get("metadata") or {}), rank=rank,
    ) for rank, (score, document) in enumerate(scored[:top_k], start=1)]


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if config.OFFLINE_MODE or self.model_name in _unavailable_cross_encoders:
            return None
        if self._model is None:
            self._model = _cross_encoder_cache.get(self.model_name)
        if self._model is None:
            try:
                if not config.ALLOW_MODEL_DOWNLOAD and not os.path.isdir(self.model_name):
                    from huggingface_hub import try_to_load_from_cache
                    cached_config = try_to_load_from_cache(self.model_name, "config.json")
                    if not isinstance(cached_config, str) or not os.path.isfile(cached_config):
                        _unavailable_cross_encoders.add(self.model_name)
                        return None
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(
                    self.model_name, local_files_only=not config.ALLOW_MODEL_DOWNLOAD,
                )
                _cross_encoder_cache[self.model_name] = self._model
            except Exception:  # noqa: BLE001 - Optional model dependencies may raise backend-specific errors.
                _unavailable_cross_encoders.add(self.model_name)
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not documents or top_k == 0:
            return []
        model = self._load_model()
        if model is not None:
            try:
                scores = model.predict([(query, document["text"]) for document in documents])
                return _rank_results(documents, scores, top_k)
            except Exception:  # noqa: BLE001 - Keep retrieval usable if an optional model backend fails.
                _unavailable_cross_encoders.add(self.model_name)
                _cross_encoder_cache.pop(self.model_name, None)
                self._model = None
        _warn_fallback()
        return _rank_results(documents, _lexical_scores(query, documents), top_k)


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self, model_name: str = "ms-marco-TinyBERT-L-2-v2", cache_dir: str | None = None):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else Path(__file__).resolve().parent.parent / ".cache" / "flashrank"
        self._model = None
        self._unavailable = False

    def _load_model(self):
        if config.OFFLINE_MODE or self._unavailable:
            return None
        if self._model is None:
            # Ranker downloads automatically when its model directory is absent.
            if not config.ALLOW_MODEL_DOWNLOAD and not (self.cache_dir / self.model_name).is_dir():
                self._unavailable = True
                return None
            try:
                from flashrank import Ranker
                self._model = Ranker(model_name=self.model_name, cache_dir=str(self.cache_dir))
            except Exception:  # noqa: BLE001 - Optional ONNX model dependencies can fail independently.
                self._unavailable = True
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not documents or top_k == 0:
            return []
        model = self._load_model()
        if model is not None:
            try:
                from flashrank import RerankRequest
                passages = [{"id": i, "text": document["text"]} for i, document in enumerate(documents)]
                results = model.rerank(RerankRequest(query=query, passages=passages))
                ranked_docs = [documents[int(result["id"])] for result in results]
                return _rank_results(ranked_docs, [result["score"] for result in results], top_k)
            except Exception:  # noqa: BLE001 - Recover from optional reranker backend failures.
                self._unavailable = True
                self._model = None
        _warn_fallback()
        return _rank_results(documents, _lexical_scores(query, documents), top_k)


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs. (Đã implement sẵn)"""
    if n_runs <= 0:
        raise ValueError("n_runs must be greater than zero")
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
