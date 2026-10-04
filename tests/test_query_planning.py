"""Query plans must be grounded in the supplied document catalog."""

import json
from unittest.mock import Mock

from src import llm
from src.query_planning import plan_retrieval


def test_planner_keeps_original_query_and_validates_selected_sources(monkeypatch):
    monkeypatch.setattr(llm, "is_available", lambda: True)
    call = Mock(return_value=json.dumps({
        "sources": ["salary.md", "salary.md", "invented.md", "leave.md"],
        "queries": ["Senior salary", "annual leave", "annual leave"],
    }))
    monkeypatch.setattr(llm, "complete", call)
    documents = [{"metadata": {"source": source, "document_title": source}}
                 for source in ("salary.md", "leave.md")]
    plan = plan_retrieval("Leave and salary?", documents)
    assert plan == {"queries": ["Leave and salary?", "Senior salary", "annual leave"],
                    "sources": ["salary.md", "leave.md"]}
    assert call.call_args.kwargs["json_mode"] is True


def test_planner_handles_bad_json_without_losing_query(monkeypatch):
    monkeypatch.setattr(llm, "is_available", lambda: True)
    monkeypatch.setattr(llm, "complete", lambda *args, **kwargs: "invalid")
    assert plan_retrieval("question", [{"metadata": {"source": "doc.md"}}]) == {
        "queries": ["question"], "sources": []}


def test_planner_offline_does_not_make_requests(monkeypatch):
    monkeypatch.setattr(llm, "is_available", lambda: False)
    call = Mock(side_effect=AssertionError("Offline planner attempted a request"))
    monkeypatch.setattr(llm, "complete", call)
    assert plan_retrieval("question", [{"metadata": {"source": "doc.md"}}])["sources"] == []
    call.assert_not_called()
