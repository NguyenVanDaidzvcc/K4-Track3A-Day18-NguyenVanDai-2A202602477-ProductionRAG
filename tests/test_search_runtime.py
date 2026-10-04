"""Regressions for real indexing, offline operation, and metadata preservation."""

import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import config
from src import m2_search as search_module
from src.m2_search import BM25Search, DenseSearch, SearchResult, reciprocal_rank_fusion


@pytest.fixture(autouse=True)
def simple_tokenizer(monkeypatch):
    # These cases exercise retrieval behavior without loading segmentation models.
    monkeypatch.setattr(search_module, "_word_tokenizer", lambda: None)


def test_bm25_singleton_normalizes_case_punctuation_and_compounds():
    search = BM25Search()
    metadata = {"source": "policy", "parent_id": "parent-1"}
    search.index([{"text": "NGHỈ_PHÉP: 12 ngày!", "metadata": metadata}])
    results = search.search("nghỉ phép")
    assert len(results) == 1
    assert results[0].score > 0
    assert results[0].metadata == metadata
    assert search.search("không có") == []


def test_bm25_missing_dependency_still_retrieves(monkeypatch):
    monkeypatch.setitem(sys.modules, "rank_bm25", None)
    search = BM25Search()
    search.index([
        {"text": "nghỉ phép năm", "metadata": {"source": "policy"}},
        {"text": "mật khẩu bảo mật", "metadata": {"source": "it"}},
    ])
    assert search.search("nghỉ phép")[0].metadata["source"] == "policy"


@pytest.mark.parametrize("chunks", [[], [{"text": "   "}], [{"text": "...!?"}]])
def test_bm25_handles_empty_corpora(chunks):
    search = BM25Search()
    search.index(chunks)
    assert search.search("anything") == []


def test_rrf_keeps_sources_and_counts_each_list_once():
    first = SearchResult("same text", 0.9, {"source": "first"}, "bm25")
    second = SearchResult("same text", 0.8, {"source": "second"}, "bm25")
    dense_first = SearchResult("same text", 0.5, {"source": "first"}, "dense")
    results = reciprocal_rank_fusion([[first, first, second], [dense_first]], k=60)
    assert len(results) == 2
    assert results[0].metadata["source"] == "first"
    assert results[0].score == pytest.approx(2 / 61)
    assert results[1].score == pytest.approx(1 / 63)
    assert all(result.method == "hybrid" for result in results)
    assert reciprocal_rank_fusion([[first]], top_k=0) == []


class ThreeDimensionalEncoder:
    def encode(self, texts, **kwargs):
        return np.array([
            [float("alpha" in text), float("beta" in text), 0.1]
            for text in texts
        ])


@pytest.fixture
def memory_dense(monkeypatch):
    qdrant = pytest.importorskip("qdrant_client")
    client = qdrant.QdrantClient(":memory:")
    monkeypatch.setattr(search_module, "_qdrant_client", lambda *args: client)
    search = DenseSearch()
    search._encoder = ThreeDimensionalEncoder()
    yield search
    client.close()


def test_dense_real_qdrant_uses_model_dimension_and_keeps_metadata(memory_dense):
    metadata = {"source": "a", "parent_id": "parent-a", "nested": {"value": 3}}
    memory_dense.index([
        {"text": "alpha policy", "metadata": metadata},
        {"text": "beta guide", "metadata": {"source": "b"}},
    ], collection="test-alpha")
    collection = memory_dense.client.get_collection("test-alpha")
    assert collection.config.params.vectors.size == 3
    assert memory_dense._indexes["test-alpha"].stored_in_qdrant
    results = memory_dense.search("alpha", top_k=1, collection="test-alpha")
    assert len(results) == 1
    assert results[0].text == "alpha policy"
    assert results[0].metadata == metadata
    assert results[0].score > 0.9


def test_dense_replacing_one_collection_preserves_the_other(memory_dense):
    memory_dense.index([{"text": "alpha old"}], collection="first")
    memory_dense.index([{"text": "beta keep"}], collection="second")
    memory_dense.index([{"text": "alpha new"}], collection="first")
    assert memory_dense.search("alpha", collection="first")[0].text == "alpha new"
    assert memory_dense.search("beta", collection="second")[0].text == "beta keep"
    memory_dense.index([], collection="first")
    assert memory_dense.search("alpha", collection="first") == []
    assert memory_dense.client.collection_exists("second")
    assert memory_dense.search("beta", top_k=0, collection="second") == []
    assert memory_dense.search("  ", collection="second") == []
    assert memory_dense.search("alpha", collection="missing") == []


def test_dense_without_qdrant_uses_cosine_and_preserves_parent(monkeypatch):
    monkeypatch.setattr(search_module, "_qdrant_client", lambda *args: None)
    search = DenseSearch()
    search._encoder = ThreeDimensionalEncoder()
    search.index([
        {"text": "alpha", "metadata": {"parent_id": "a"}},
        {"text": "beta", "metadata": {"parent_id": "b"}},
    ])
    results = search.search("alpha", top_k=1)
    assert results[0].text == "alpha"
    assert results[0].metadata == {"parent_id": "a"}


def test_encoder_failure_is_cached_and_downloads_are_disabled(monkeypatch):
    calls = []

    def missing_model(model, **kwargs):
        calls.append((model, kwargs))
        raise OSError("not cached")

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=missing_model)
    )
    monkeypatch.setattr(config, "OFFLINE_MODE", False)
    monkeypatch.setattr(config, "ALLOW_MODEL_DOWNLOAD", False)
    monkeypatch.setattr(search_module, "_qdrant_client", lambda *args: None)
    search_module._load_encoder.cache_clear()
    for _ in range(2):
        search = DenseSearch()
        search.index([{"text": "alpha policy"}, {"text": "beta guide"}])
        assert search.search("alpha")[0].text == "alpha policy"
    assert len(calls) == 1
    assert calls[0][1]["local_files_only"] is True
    search_module._load_encoder.cache_clear()


def test_offline_skips_model_loading_and_hash_vectors_are_normalized(monkeypatch):
    def forbidden_model(*args, **kwargs):
        pytest.fail("Offline mode attempted to load a model")

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=forbidden_model)
    )
    monkeypatch.setattr(config, "OFFLINE_MODE", True)
    monkeypatch.setattr(search_module, "_qdrant_client", lambda *args: None)
    search = DenseSearch()
    search.index([{"text": "alpha alpha"}, {"text": "beta"}])
    first = search_module._hash_vectors(["alpha alpha", "beta"])
    second = search_module._hash_vectors(["alpha alpha", "beta"])
    assert np.array_equal(first, second)
    assert np.allclose(np.linalg.norm(first, axis=1), 1)
    assert search.search("alpha")[0].score == pytest.approx(1)
    assert search.search("...") == []


def test_qdrant_connection_is_probed_only_once_and_skipped_offline(monkeypatch):
    created = []

    class FakeClient:
        def __init__(self, *args, **kwargs):
            created.append((args, kwargs))

        def get_collections(self):
            raise ConnectionError("server unavailable")

    monkeypatch.setitem(sys.modules, "qdrant_client", SimpleNamespace(QdrantClient=FakeClient))
    search_module._qdrant_client.cache_clear()
    client = search_module._qdrant_client("localhost", 6333, False)
    assert search_module._qdrant_client("localhost", 6333, False) is client
    assert len(created) == 2
    assert created[0][1]["check_compatibility"] is False
    created.clear()
    search_module._qdrant_client("localhost", 6333, True)
    assert created == [((":memory:",), {})]
    search_module._qdrant_client.cache_clear()


def test_query_encoder_failure_reembeds_documents_consistently(memory_dense):
    class FailingQueryEncoder(ThreeDimensionalEncoder):
        def encode(self, texts, **kwargs):
            if texts == ["alpha"]:
                raise RuntimeError("model became unavailable")
            return super().encode(texts, **kwargs)

    memory_dense._encoder = FailingQueryEncoder()
    memory_dense.index([{"text": "alpha policy"}, {"text": "beta guide"}])
    results = memory_dense.search("alpha", top_k=1)
    assert results[0].text == "alpha policy"
    assert memory_dense._indexes[config.COLLECTION_NAME].mode == "hash"
    assert math.isfinite(results[0].score)
