"""Quality and persistent enrichment-cache checks without external requests."""

import json
from unittest.mock import Mock

import pytest

import config
from src import llm, m3_rerank, m5_enrichment


@pytest.fixture(autouse=True)
def isolated_local_runtime(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "OFFLINE_MODE", True)
    monkeypatch.setattr(config, "ENRICHMENT_CACHE_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "ENRICHMENT_CACHE_DIR", str(tmp_path / "cache"), raising=False)
    monkeypatch.setattr(llm, "is_available", lambda: False)
    monkeypatch.setattr(llm, "complete", lambda *args, **kwargs: None)


def test_lexical_rerank_weights_rare_topic_terms_over_common_query_words():
    documents = [
        {"text": f"Employee dates and days for annual review number {index}."}
        for index in range(8)
    ]
    target = {
        "text": "Senior laptop allocation rules. " + "Detailed supporting explanation. " * 40,
        "score": 0.3, "metadata": {"source": "equipment", "parent_id": "equipment-1"},
    }
    documents.insert(4, target)
    results = m3_rerank.CrossEncoderReranker().rerank(
        "Employee dates days: SENIOR laptop?", documents, top_k=3
    )
    assert results[0].text == target["text"]
    assert results[0].metadata == target["metadata"]
    assert results[0].original_score == 0.3
    assert [result.rank for result in results] == [1, 2, 3]
    assert results[0].rerank_score > results[1].rerank_score


def test_lexical_rerank_normalizes_compounds_and_accents():
    scores = m3_rerank._lexical_scores("NGHỈ PHÉP", [
        {"text": "Nghỉ_phép hằng năm."},
        {"text": "Mật khẩu bảo mật."},
    ])
    assert scores[0] > scores[1]
    assert m3_rerank._lexical_scores("...!?", [{"text": "content"}]) == [0]
    assert m3_rerank._lexical_scores("content", []) == []


@pytest.fixture
def live_enrichment(monkeypatch):
    monkeypatch.setattr(config, "OFFLINE_MODE", False)
    monkeypatch.setattr(config, "OPENAI_MODEL", "model-for-cache-tests")
    monkeypatch.setattr(config, "OPENAI_API_KEY", "secret-must-never-enter-cache")
    monkeypatch.setattr(llm, "is_available", lambda: True)
    payload = {
        "summary": "Equipment is supplied.",
        "questions": ["What equipment is supplied?"],
        "context": "Equipment policy.",
        "metadata": {"topic": "Equipment", "entities": [], "category": "it", "language": "en"},
    }
    complete = Mock(return_value=json.dumps(payload))
    monkeypatch.setattr(llm, "complete", complete)
    chunk = {"text": "A laptop is supplied.", "metadata": {
        "source": "equipment.md", "document_title": "Equipment policy", "parent_id": "p1",
    }}
    return complete, chunk, payload


def test_enrichment_cache_reuses_api_fields_but_preserves_new_source_metadata(live_enrichment, capsys):
    complete, chunk, _ = live_enrichment
    first = m5_enrichment.enrich_chunks([chunk])[0]
    updated = {"text": chunk["text"], "metadata": {**chunk["metadata"], "parent_id": "p2", "page": 9}}
    second = m5_enrichment.enrich_chunks([updated])[0]
    assert complete.call_count == 1
    assert not first.cache_hit and second.cache_hit
    assert first.enrichment_mode == second.enrichment_mode == "api"
    assert second.auto_metadata["parent_id"] == "p2"
    assert second.auto_metadata["page"] == 9
    assert second.auto_metadata["source"] == "equipment.md"
    assert "1 cache hits" in capsys.readouterr().out
    cache_files = list(m5_enrichment.Path(config.ENRICHMENT_CACHE_DIR).glob("*.json"))
    assert len(cache_files) == 1
    assert config.OPENAI_API_KEY not in cache_files[0].read_text(encoding="utf-8")
    record = json.loads(cache_files[0].read_text(encoding="utf-8"))
    assert record["mode"] == "api"


@pytest.mark.parametrize("changed_field", ["text", "source", "title", "model", "prompt"])
def test_enrichment_cache_invalidates_changed_inputs(live_enrichment, monkeypatch, changed_field):
    complete, chunk, _ = live_enrichment
    m5_enrichment.enrich_chunks([chunk])
    changed = {"text": chunk["text"], "metadata": dict(chunk["metadata"])}
    if changed_field == "text":
        changed["text"] += " A monitor is also supplied."
    elif changed_field == "source":
        changed["metadata"]["source"] = "another.md"
    elif changed_field == "title":
        changed["metadata"]["document_title"] = "Updated equipment policy"
    elif changed_field == "model":
        monkeypatch.setattr(config, "OPENAI_MODEL", "another-model")
    elif changed_field == "prompt":
        monkeypatch.setattr(m5_enrichment, "ENRICHMENT_PROMPT_VERSION", "another-prompt")
    result = m5_enrichment.enrich_chunks([changed])[0]
    assert complete.call_count == 2
    assert not result.cache_hit


def test_offline_ignores_live_cache_and_reuses_only_local_outputs(live_enrichment, monkeypatch):
    complete, chunk, payload = live_enrichment
    m5_enrichment.enrich_chunks([chunk])
    monkeypatch.setattr(config, "OFFLINE_MODE", True)
    complete.return_value = None
    local = m5_enrichment.enrich_chunks([chunk])[0]
    cached_local = m5_enrichment.enrich_chunks([chunk])[0]
    assert local.summary == chunk["text"]
    assert local.summary != payload["summary"]
    assert local.enrichment_mode == cached_local.enrichment_mode == "local"
    assert not local.cache_hit and cached_local.cache_hit
    assert complete.call_count == 2


def test_api_failure_is_not_cached_as_success_and_later_success_is_used(live_enrichment):
    complete, chunk, payload = live_enrichment
    complete.side_effect = [None, json.dumps(payload)]
    fallback = m5_enrichment.enrich_chunks([chunk])[0]
    api_path = m5_enrichment._cache_path(
        chunk["text"], chunk["metadata"]["source"], chunk["metadata"]["document_title"], "api"
    )
    assert not api_path.exists()
    live = m5_enrichment.enrich_chunks([chunk])[0]
    cached = m5_enrichment.enrich_chunks([chunk])[0]
    assert fallback.enrichment_mode == "local"
    assert live.enrichment_mode == cached.enrichment_mode == "api"
    assert cached.cache_hit and complete.call_count == 2


@pytest.mark.parametrize("corruption", ["not json", '{"schema_version": 1}', "[]"])
def test_invalid_cache_is_ignored_and_replaced_atomically(live_enrichment, corruption):
    complete, chunk, _ = live_enrichment
    m5_enrichment.enrich_chunks([chunk])
    path = m5_enrichment._cache_path(
        chunk["text"], chunk["metadata"]["source"], chunk["metadata"]["document_title"], "api"
    )
    path.write_text(corruption, encoding="utf-8")
    result = m5_enrichment.enrich_chunks([chunk])[0]
    assert complete.call_count == 2
    assert not result.cache_hit
    assert json.loads(path.read_text(encoding="utf-8"))["mode"] == "api"
    assert not list(path.parent.glob("*.tmp"))


def test_cache_write_failure_keeps_enrichment_usable_and_cleans_temp_file(live_enrichment, monkeypatch):
    complete, chunk, payload = live_enrichment

    def cannot_replace(*args):
        raise OSError("filesystem unavailable")

    monkeypatch.setattr(m5_enrichment.os, "replace", cannot_replace)
    result = m5_enrichment.enrich_chunks([chunk])[0]
    assert result.summary == payload["summary"]
    assert not result.cache_hit and complete.call_count == 1
    assert list(m5_enrichment.Path(config.ENRICHMENT_CACHE_DIR).iterdir()) == []


def test_partial_api_output_is_not_persisted_as_a_complete_result(live_enrichment):
    complete, chunk, _ = live_enrichment
    complete.return_value = json.dumps({"summary": "Equipment is supplied."})
    first = m5_enrichment.enrich_chunks([chunk])[0]
    second = m5_enrichment.enrich_chunks([chunk])[0]
    assert first.enrichment_mode == second.enrichment_mode == "mixed"
    assert not second.cache_hit and complete.call_count == 2
    assert not m5_enrichment.Path(config.ENRICHMENT_CACHE_DIR).exists()
