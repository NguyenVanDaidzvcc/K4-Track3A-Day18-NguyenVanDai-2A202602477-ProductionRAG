"""Runtime regressions without downloads or external services."""

import re
import sys
from types import SimpleNamespace

import pytest

import config
from src import m1_chunking as chunking
from src import m3_rerank as reranking


def test_hierarchical_bounds_preserves_words_and_unique_document_links():
    text = " ".join(f"word{i}" for i in range(120))
    metadata = {"source": "policy-a.md", "category": "hr"}
    parents, children = chunking.chunk_hierarchical(text, 130, 35, metadata)
    other_parents, _ = chunking.chunk_hierarchical(text, 130, 35, {"source": "policy-b.md"})
    ids = {parent.metadata["parent_id"] for parent in parents}

    assert all(0 < len(parent.text) <= 130 for parent in parents)
    assert all(0 < len(child.text) <= 35 for child in children)
    assert " ".join(parent.text for parent in parents).split() == text.split()
    assert " ".join(child.text for child in children).split() == text.split()
    assert all(child.parent_id in ids for child in children)
    assert ids.isdisjoint(parent.metadata["parent_id"] for parent in other_parents)
    assert all(child.metadata["category"] == "hr" for child in children)
    assert metadata == {"source": "policy-a.md", "category": "hr"}
    repeated_parents, _ = chunking.chunk_hierarchical(text, 130, 35, metadata)
    assert ids == {parent.metadata["parent_id"] for parent in repeated_parents}


def test_hierarchical_bounds_long_unbroken_text():
    text = "x" * 251
    parents, children = chunking.chunk_hierarchical(text, 100, 25)
    assert all(len(parent.text) <= 100 for parent in parents)
    assert all(len(child.text) <= 25 for child in children)
    assert re.sub(r"\s+", "", "".join(child.text for child in children)) == text


@pytest.mark.parametrize("parent_size,child_size", [(0, 10), (10, 0), (-1, 10)])
def test_hierarchical_rejects_invalid_sizes(parent_size, child_size):
    with pytest.raises(ValueError):
        chunking.chunk_hierarchical("document", parent_size, child_size)


def test_structure_keeps_code_headings_tables_and_lists_together():
    text = """# Policy
````markdown
## Heading inside code
```
# Still code
````

| Name | Value |
| --- | --- |
| Leave | 15 |

- First item
- Second item

## Security
MFA is required.
"""
    chunks = chunking.chunk_structure_aware(text, {"source": "policy.md"})
    assert len(chunks) == 2
    assert "# Still code" in chunks[0].text
    assert "| Leave | 15 |" in chunks[0].text
    assert "- Second item" in chunks[0].text
    assert chunks[1].metadata["section"] == "## Security"
    assert all(chunk.metadata["source"] == "policy.md" for chunk in chunks)


def test_semantic_groups_using_embedding_cosine(monkeypatch):
    encoder = SimpleNamespace(encode=lambda sentences: [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]])
    monkeypatch.setattr(chunking, "_get_semantic_encoder", lambda: encoder)
    monkeypatch.setattr(chunking, "_semantic_encoder_unavailable", False)
    chunks = chunking.chunk_semantic("Annual leave. Leave policy. Password security.", 0.8)
    assert [chunk.text for chunk in chunks] == ["Annual leave. Leave policy.", "Password security."]
    assert all(chunk.metadata["similarity_method"] == "embedding" for chunk in chunks)


def test_semantic_failed_local_model_load_is_only_attempted_once(monkeypatch, capsys):
    calls = []

    def unavailable(*args, **kwargs):
        calls.append(kwargs)
        raise OSError("No model in local cache")

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=unavailable))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(try_to_load_from_cache=lambda *args: __file__))
    monkeypatch.setattr(config, "OFFLINE_MODE", False)
    monkeypatch.setattr(config, "ALLOW_MODEL_DOWNLOAD", False)
    monkeypatch.setattr(chunking, "_semantic_encoder", None)
    monkeypatch.setattr(chunking, "_semantic_encoder_unavailable", False)
    monkeypatch.setattr(chunking, "_semantic_warning_shown", False)
    first = chunking.chunk_semantic("Leave policy. Password policy.")
    second = chunking.chunk_semantic("Leave policy. Password policy.")
    assert first and second
    assert calls == [{"local_files_only": True}]
    assert all(chunk.metadata["similarity_method"] == "lexical" for chunk in second)
    assert capsys.readouterr().out.count("độ giống từ vựng") == 1


def test_cross_encoder_order_scores_metadata_and_single_scalar(monkeypatch):
    monkeypatch.setattr(config, "OFFLINE_MODE", False)
    monkeypatch.setattr(reranking, "_unavailable_cross_encoders", set())
    reranker = reranking.CrossEncoderReranker("test-model")
    reranker._model = SimpleNamespace(predict=lambda pairs: [[0.2], [0.9], [0.5]])
    documents = [{"text": f"doc{i}", "score": i / 10, "metadata": {"source": str(i)}} for i in range(3)]
    results = reranker.rerank("question", documents, top_k=2)
    assert [result.text for result in results] == ["doc1", "doc2"]
    assert [result.rank for result in results] == [1, 2]
    assert [result.original_score for result in results] == [0.1, 0.2]
    assert results[0].metadata == {"source": "1"}
    reranker._model = SimpleNamespace(predict=lambda pairs: 0.75)
    assert reranker.rerank("question", documents[:1])[0].rerank_score == 0.75


def test_cross_encoder_caches_failed_load_and_uses_lexical_scores(monkeypatch, capsys):
    calls = []

    def unavailable(*args, **kwargs):
        calls.append(kwargs)
        raise OSError("No model in local cache")

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=unavailable))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(try_to_load_from_cache=lambda *args: __file__))
    monkeypatch.setattr(config, "OFFLINE_MODE", False)
    monkeypatch.setattr(config, "ALLOW_MODEL_DOWNLOAD", False)
    monkeypatch.setattr(reranking, "_cross_encoder_cache", {})
    monkeypatch.setattr(reranking, "_unavailable_cross_encoders", set())
    monkeypatch.setattr(reranking, "_fallback_warning_shown", False)
    documents = [{"text": "password security"}, {"text": "annual leave policy"}]
    results = reranking.CrossEncoderReranker().rerank("annual leave", documents)
    second = reranking.CrossEncoderReranker().rerank("annual leave", documents)
    assert results[0].text == "annual leave policy"
    assert second[0].rerank_score > second[1].rerank_score
    assert calls == [{"local_files_only": True}]
    assert capsys.readouterr().out.count("độ khớp từ vựng") == 1


def test_offline_skips_all_model_imports(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Offline code must not load a model")

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(
        SentenceTransformer=forbidden, CrossEncoder=forbidden,
    ))
    monkeypatch.setitem(sys.modules, "flashrank", SimpleNamespace(Ranker=forbidden))
    monkeypatch.setattr(config, "OFFLINE_MODE", True)
    documents = [{"text": "annual leave policy"}]
    assert chunking.chunk_semantic("Annual leave policy.")
    assert reranking.CrossEncoderReranker().rerank("annual leave", documents)
    assert reranking.FlashrankReranker().rerank("annual leave", documents)


def test_missing_model_cache_skips_heavyweight_constructors(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Missing model caches must not trigger model loading or downloads")

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(try_to_load_from_cache=lambda *args: None))
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(
        SentenceTransformer=forbidden, CrossEncoder=forbidden,
    ))
    monkeypatch.setattr(config, "OFFLINE_MODE", False)
    monkeypatch.setattr(config, "ALLOW_MODEL_DOWNLOAD", False)
    monkeypatch.setattr(chunking, "_semantic_encoder", None)
    monkeypatch.setattr(chunking, "_semantic_encoder_unavailable", False)
    monkeypatch.setattr(reranking, "_cross_encoder_cache", {})
    monkeypatch.setattr(reranking, "_unavailable_cross_encoders", set())
    assert chunking.chunk_semantic("Annual leave policy.")
    assert reranking.CrossEncoderReranker().rerank("annual leave", [{"text": "annual leave policy"}])


def test_flashrank_retains_original_document_mapping(monkeypatch):
    monkeypatch.setattr(config, "OFFLINE_MODE", False)
    monkeypatch.setitem(sys.modules, "flashrank", SimpleNamespace(RerankRequest=SimpleNamespace))
    reranker = reranking.FlashrankReranker()
    reranker._model = SimpleNamespace(rerank=lambda request: [
        {"id": 1, "score": 0.9}, {"id": 0, "score": 0.2},
    ])
    documents = [{"text": "first", "metadata": {"source": "a"}},
                 {"text": "second", "metadata": {"source": "b"}}]
    results = reranker.rerank("question", documents, top_k=1)
    assert results[0].text == "second"
    assert results[0].metadata == {"source": "b"}


def test_rerank_zero_limit_empty_documents_and_invalid_benchmark():
    reranker = reranking.CrossEncoderReranker()
    assert reranker.rerank("question", [], top_k=3) == []
    assert reranker.rerank("question", [{"text": "document"}], top_k=0) == []
    with pytest.raises(ValueError):
        reranking.benchmark_reranker(reranker, "question", [], n_runs=0)
