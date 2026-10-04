"""Regression coverage for context recovery and bounded API failure handling."""

from types import SimpleNamespace

import config
import check_lab
from src import llm, pipeline
from src.m2_search import SearchResult


def test_query_recovers_parent_and_deduplicates_context(monkeypatch):
    retrieved = [SearchResult("child one", 1.0, {"parent_id": "p1"}, "hybrid"),
                 SearchResult("child two", 0.9, {"parent_id": "p1"}, "hybrid")]
    search = SimpleNamespace(parent_contexts={"p1": "Complete original policy context"},
                             search=lambda query: retrieved)
    reranker = SimpleNamespace(rerank=lambda query, docs, top_k: retrieved)
    monkeypatch.setattr(pipeline, "complete", lambda *args, **kwargs: None)
    answer, contexts = pipeline.run_query("policy?", search, reranker)
    assert contexts == ["Complete original policy context"]
    assert answer == contexts[0]


def test_empty_query_results_do_not_call_llm(monkeypatch):
    def forbidden_call(*args, **kwargs):
        raise AssertionError("LLM must not be called without contexts")
    monkeypatch.setattr(pipeline, "complete", forbidden_call)
    search = SimpleNamespace(search=lambda query: [])
    reranker = SimpleNamespace(rerank=lambda query, docs, top_k: [])
    answer, contexts = pipeline.run_query("missing", search, reranker)
    assert contexts == []
    assert "Không tìm thấy" in answer


def test_query_fills_distinct_parent_contexts_after_duplicate_children(monkeypatch):
    candidates = [
        SearchResult(f"leave child {i}", 1.0 - i / 10, {"parent_id": "leave"}, "hybrid")
        for i in range(3)
    ] + [
        SearchResult("salary child", 0.6, {"parent_id": "salary"}, "hybrid"),
        SearchResult("probation child", 0.5, {"parent_id": "probation"}, "hybrid"),
        SearchResult("other child", 0.4, {"parent_id": "other"}, "hybrid"),
    ]
    search = SimpleNamespace(
        parent_contexts={"leave": "Original leave policy", "salary": "Original salary table",
                         "probation": "Original probation policy", "other": "Other policy"},
        search=lambda query: candidates,
    )
    def rank_parents(query, docs, top_k):
        assert len(docs) == 4
        assert docs[0]["text"] == "Original leave policy"
        return [SimpleNamespace(text=doc["text"], metadata=doc["metadata"])
                for doc in docs[:top_k]]

    reranker = SimpleNamespace(rerank=rank_parents)
    messages_seen = []

    def generate(messages, **kwargs):
        messages_seen.extend(messages)
        return "Answer based on leave and salary"

    monkeypatch.setattr(pipeline, "complete", generate)
    answer, contexts = pipeline.run_query("Leave entitlement and salary?", search, reranker)
    assert contexts == ["Original leave policy", "Original salary table", "Original probation policy"]
    assert answer == "Answer based on leave and salary"
    assert all(context in messages_seen[1]["content"] for context in contexts)
    assert "Other policy" not in messages_seen[1]["content"]


def test_query_fills_contexts_when_reranker_returns_no_results(monkeypatch):
    candidates = [
        SearchResult("child one", 1, {"parent_id": "p1"}, "hybrid"),
        SearchResult("child two", 0.9, {"parent_id": "p1"}, "hybrid"),
        SearchResult("child three", 0.8, {"parent_id": "p1"}, "hybrid"),
        SearchResult("distinct child", 0.7, {"parent_id": "p2"}, "hybrid"),
    ]
    search = SimpleNamespace(parent_contexts={"p1": "first original", "p2": "second original"},
                             search=lambda query: candidates)
    reranker = SimpleNamespace(rerank=lambda query, docs, top_k: [])
    monkeypatch.setattr(pipeline, "complete", lambda *args, **kwargs: None)
    _, contexts = pipeline.run_query("question", search, reranker)
    assert contexts == ["first original", "second original"]


def test_query_routes_multiple_sources_when_child_search_misses_them(monkeypatch):
    parents = {
        "generic": {"text": "Unrelated document", "metadata": {"source": "generic.md", "parent_id": "generic"}},
        "travel": {"text": "Original travel rules", "metadata": {"source": "travel.md", "parent_id": "travel"}},
        "approval": {"text": "Original approval limits", "metadata": {"source": "approval.md", "parent_id": "approval"}},
    }
    search = SimpleNamespace(
        parent_documents=parents,
        parent_contexts={key: doc["text"] for key, doc in parents.items()},
        documents=[],
        search=lambda query: [SearchResult("Unrelated child", 1.0, parents["generic"]["metadata"], "hybrid")],
    )
    monkeypatch.setattr(pipeline, "plan_retrieval", lambda *args: {
        "queries": ["Travel cost and approval?"], "sources": ["travel.md", "approval.md"]})
    reranker = SimpleNamespace(rerank=lambda query, docs, top_k: [
        SimpleNamespace(text=doc["text"], metadata=doc["metadata"]) for doc in docs])
    monkeypatch.setattr(pipeline, "complete", lambda *args, **kwargs: "Grounded answer")
    answer, contexts = pipeline.run_query("Travel cost and approval?", search, reranker)
    assert answer == "Grounded answer"
    assert contexts == ["Original travel rules", "Original approval limits"]


def test_query_routes_sources_in_rerank_order_with_parent_diversity(monkeypatch):
    parents = {
        "generic": {"text": "Unrelated policy", "metadata": {"source": "generic.md", "parent_id": "generic"}},
        "travel-best": {"text": "Best travel evidence", "metadata": {"source": "travel.md", "parent_id": "travel-best"}},
        "travel-other": {"text": "Other travel passage", "metadata": {"source": "travel.md", "parent_id": "travel-other"}},
        "approval": {"text": "Approval evidence", "metadata": {"source": "approval.md", "parent_id": "approval"}},
    }
    candidates = [SearchResult(doc["text"], 1.0, doc["metadata"], "hybrid")
                  for doc in parents.values()]
    search = SimpleNamespace(
        parent_documents=parents,
        parent_contexts={key: doc["text"] for key, doc in parents.items()},
        search=lambda query: candidates,
    )
    monkeypatch.setattr(pipeline, "plan_retrieval", lambda *args: {
        "queries": ["Travel and approval?"], "sources": ["approval.md", "travel.md"]})
    monkeypatch.setattr(pipeline, "RERANK_TOP_K", 2)
    # The unrelated passage ranks first; two travel parents precede approval.
    # Routing must filter the unrelated source and reserve room for approval.
    reranker = SimpleNamespace(rerank=lambda query, docs, top_k: [
        SimpleNamespace(text=doc["text"], metadata=doc["metadata"])
        for doc in parents.values()])
    messages_seen = []

    def generate(messages, **kwargs):
        messages_seen.extend(messages)
        return "Grounded answer"

    monkeypatch.setattr(pipeline, "complete", generate)
    answer, contexts = pipeline.run_query("Travel and approval?", search, reranker)
    assert answer == "Grounded answer"
    assert contexts == ["Best travel evidence", "Approval evidence"]
    assert messages_seen[1]["content"].startswith(
        "Context:\nBest travel evidence\n\nApproval evidence\n\n")
    assert "Unrelated policy" not in messages_seen[1]["content"]
    assert "Other travel passage" not in messages_seen[1]["content"]


def test_query_keeps_reranked_contexts_when_planned_sources_are_unavailable(monkeypatch):
    candidates = [
        SearchResult("Less relevant evidence", 1.0, {"source": "first.md"}, "hybrid"),
        SearchResult("Best available evidence", 0.8, {"source": "second.md"}, "hybrid"),
    ]
    search = SimpleNamespace(search=lambda query: candidates)
    monkeypatch.setattr(pipeline, "plan_retrieval", lambda *args: {
        "queries": ["Policy?"], "sources": ["unavailable.md"]})
    reranker = SimpleNamespace(rerank=lambda query, docs, top_k: list(reversed(candidates)))
    monkeypatch.setattr(pipeline, "complete", lambda *args, **kwargs: None)
    answer, contexts = pipeline.run_query("Policy?", search, reranker)
    assert contexts == ["Best available evidence", "Less relevant evidence"]
    assert answer == "Best available evidence"


def test_llm_failure_disables_further_requests_without_exposing_exception(monkeypatch, capsys):
    calls = []

    def fail(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("sensitive request data")

    monkeypatch.setattr(config, "OFFLINE_MODE", False)
    monkeypatch.setattr(config, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(llm, "_unavailable", False)
    monkeypatch.setattr(llm, "_client", SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fail))))
    assert llm.complete([]) is None
    assert llm.complete([]) is None
    assert len(calls) == 1
    assert not llm.is_available()
    assert "sensitive request data" not in capsys.readouterr().out


def test_check_lab_counts_collection_errors(monkeypatch):
    monkeypatch.setattr(check_lab.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(stdout="2 passed, 1 error in 0.1s", stderr="", returncode=1))
    assert check_lab.run_tests() == (2, 3)


def test_check_lab_does_not_report_success_when_pytest_cannot_run(monkeypatch):
    monkeypatch.setattr(check_lab.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(stdout="", stderr="No module named pytest", returncode=1))
    assert check_lab.run_tests() == (0, 0)
