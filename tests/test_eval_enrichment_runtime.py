"""Regression checks for API failures, evaluation status and indexed enrichment."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import config
from src import llm, m4_eval, m5_enrichment


@pytest.fixture
def fake_ragas(monkeypatch):
    monkeypatch.setattr(config, "OFFLINE_MODE", False)
    monkeypatch.setattr(config, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(config, "OPENAI_MODEL", "test-model")
    monkeypatch.setattr(config, "OPENAI_TIMEOUT_SECONDS", 13)
    monkeypatch.setattr(llm, "is_available", lambda: True)
    rows = [
        dict(faithfulness=0.8, answer_relevancy=0.6, context_precision=0.4, context_recall=0.2),
        dict(faithfulness=0.6, answer_relevancy=0.8, context_precision=0.2, context_recall=0.4),
    ]
    frame = SimpleNamespace(to_dict=lambda **kwargs: rows)
    evaluate = Mock(return_value=SimpleNamespace(to_pandas=lambda: frame))
    chat = Mock()
    embeddings = Mock()

    class FakeAnswerRelevancy:
        name = "answer_relevancy"

        def __init__(self, question_generation=None, strictness=3):
            self.question_generation = question_generation or SimpleNamespace(output_format_instruction="JSON")
            self.strictness = strictness
            self.similarity_calls = 0

        def calculate_similarity(self, question, generated_questions):
            self.similarity_calls += 1
            return [0.8, 0.6, 0.4]

        def _calculate_score(self, answers, row):
            questions = [answer.question for answer in answers]
            if all(question == "" for question in questions):
                return float("nan")
            similarities = self.calculate_similarity(row["question"], questions)
            return sum(similarities) / len(similarities) * int(not any(answer.noncommittal for answer in answers))

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(
        Dataset=SimpleNamespace(from_dict=lambda values: values)))
    monkeypatch.setitem(sys.modules, "langchain_openai", SimpleNamespace(
        ChatOpenAI=chat, OpenAIEmbeddings=embeddings))
    monkeypatch.setitem(sys.modules, "ragas", SimpleNamespace(evaluate=evaluate))
    monkeypatch.setitem(sys.modules, "ragas.metrics", SimpleNamespace(
        AnswerRelevancy=FakeAnswerRelevancy,
        Faithfulness=lambda: SimpleNamespace(name="faithfulness"),
        ContextPrecision=lambda: SimpleNamespace(name="context_precision"),
        ContextRecall=lambda: SimpleNamespace(name="context_recall")))
    monkeypatch.setitem(sys.modules, "ragas.llms.prompt", SimpleNamespace(
        Prompt=lambda **kwargs: SimpleNamespace(**kwargs)))
    monkeypatch.setitem(sys.modules, "ragas.run_config", SimpleNamespace(
        RunConfig=lambda **kwargs: SimpleNamespace(**kwargs)))
    return rows, evaluate, chat, embeddings


def test_ragas_aggregates_actual_rows_and_bounds_api_runtime(fake_ragas):
    _, evaluate, chat, embeddings = fake_ragas
    result = m4_eval.evaluate_ragas(["q1", "q2"], ["a1", "a2"], [["c1"], ["c2"]], ["g1", "g2"])
    assert result["evaluation_status"] == "completed"
    assert result["faithfulness"] == pytest.approx(0.7)
    assert result["context_recall"] == pytest.approx(0.3)
    assert result["per_question"][1].question == "q2"
    run_config = evaluate.call_args.kwargs["run_config"]
    assert (run_config.timeout, run_config.max_retries, run_config.max_workers, run_config.max_wait) == (13, 1, 2, 1)
    assert evaluate.call_args.kwargs["raise_exceptions"] is True
    assert chat.call_args.kwargs["max_retries"] == 0
    assert chat.call_args.kwargs["model"] == "test-model"
    assert chat.call_args.kwargs["timeout"] == 13
    assert embeddings.call_args.kwargs["max_retries"] == 0
    assert embeddings.call_args.kwargs["request_timeout"] == 13
    assert embeddings.call_args.kwargs["model"] == "text-embedding-3-small"
    metrics = evaluate.call_args.kwargs["metrics"]
    assert [metric.name for metric in metrics] == list(m4_eval.METRICS)
    assert metrics[1].strictness == 3
    assert metrics[1].question_generation.language == "vietnamese"
    assert metrics[1].question_generation.input_keys == ["answer", "context"]
    assert result["evaluation_config"]["llm_model"] == "test-model"
    assert result["evaluation_config"]["answer_relevancy"]["prompt_language"] == "vi"


def test_fresh_localized_metric_instances_preserve_uncertainty_example(fake_ragas):
    first, _ = m4_eval._build_evaluation_metrics()
    second, _ = m4_eval._build_evaluation_metrics()
    assert all(left is not right for left, right in zip(first, second))
    first_prompt = first[1].question_generation
    assert first_prompt is not second[1].question_generation
    assert "tiếng Việt" in first_prompt.instruction
    assert any(example["output"]["noncommittal"] == 1 for example in first_prompt.examples)
    assert first_prompt.examples[0]["output"]["question"] == "Albert Einstein sinh ra ở đâu?"
    assert m4_eval.RELEVANCY_STRICTNESS == 3


def test_relevancy_diagnostics_capture_formula_once_and_map_duplicate_questions(fake_ragas, tmp_path):
    rows, evaluate, *_ = fake_ragas

    def traced_evaluate(dataset, *, metrics, **kwargs):
        relevance = metrics[1]
        for index, answer in enumerate(dataset["answer"]):
            generations = [
                SimpleNamespace(question=f"{answer}: câu hỏi {number}?", noncommittal=int(index == 1 and number == 2))
                for number in range(3)
            ]
            input_row = {key: values[index] for key, values in dataset.items()}
            rows[index]["answer_relevancy"] = relevance._calculate_score(generations, input_row)
        assert relevance.similarity_calls == len(dataset["question"])
        return SimpleNamespace(to_pandas=lambda: SimpleNamespace(to_dict=lambda **kwargs: rows))

    evaluate.side_effect = traced_evaluate
    result = m4_eval.evaluate_ragas(["same question", "same question"], ["first answer", "second answer"],
                                  [["first context"], ["second context"]], ["g1", "g2"])
    first, second = result["per_question"]
    assert first.answer_relevancy == pytest.approx(0.6)
    assert second.answer_relevancy == 0.0
    first_trace = first.diagnostics["answer_relevancy"]
    second_trace = second.diagnostics["answer_relevancy"]
    assert first_trace["cosine_similarities"] == [0.8, 0.6, 0.4]
    assert first_trace["noncommittal_labels"] == [0, 0, 0]
    assert first_trace["noncommittal_gate"] is False
    assert second_trace["noncommittal_labels"] == [0, 0, 1]
    assert second_trace["noncommittal_gate"] is True
    assert first_trace["generated_questions"][0].startswith("first answer")
    assert second_trace["generated_questions"][0].startswith("second answer")
    assert second_trace["score"] == 0.0
    path = tmp_path / "diagnostics.json"
    m4_eval.save_report(result, [], path)
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["per_question"][1]["diagnostics"]["answer_relevancy"] == second_trace
    assert report["evaluation_config"]["embedding_model"] == "text-embedding-3-small"
    assert report["evaluation_config"]["answer_relevancy"]["strictness"] == 3
    assert set(report["aggregate"]) == set(m4_eval.METRICS)


def test_non_finite_scores_are_marked_and_never_written_as_nan(fake_ragas, tmp_path):
    rows, *_ = fake_ragas
    rows[0]["faithfulness"] = float("nan")
    rows[1]["context_recall"] = float("inf")
    result = m4_eval.evaluate_ragas(["q1", "q2"], ["a1", "a2"], [["c1"], ["c2"]], ["g1", "g2"])
    assert result["evaluation_status"] == "partial"
    assert result["faithfulness"] == pytest.approx(0.6)
    assert all(row.evaluation_status == "partial" for row in result["per_question"])
    assert m4_eval.failure_analysis(result["per_question"]) == []
    path = tmp_path / "report.json"
    m4_eval.save_report(result, [], path)
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["evaluation_status"] == "partial"
    assert report["per_question"][0]["faithfulness"] == 0.0
    assert "NaN" not in path.read_text(encoding="utf-8")


def test_api_exception_preserves_count_and_does_not_expose_key(fake_ragas, tmp_path):
    _, evaluate, *_ = fake_ragas
    evaluate.side_effect = ValueError("private-key-value")
    result = m4_eval.evaluate_ragas(["q"], ["a"], [["c"]], ["g"])
    assert result["evaluation_status"] == "failed"
    assert result["num_questions"] == 1
    assert "private-key-value" not in result["error"]
    assert "ValueError" in result["error"]
    assert m4_eval.failure_analysis(result["per_question"]) == []
    path = tmp_path / "report.json"
    m4_eval.save_report(result, [], path)
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["num_questions"] == 1
    assert report["evaluation_status"] == "failed"
    assert report["per_question"][0]["question"] == "q"
    assert set(report["aggregate"]) == set(m4_eval.METRICS)


def test_offline_eval_preserves_input_rows(monkeypatch):
    monkeypatch.setattr(config, "OFFLINE_MODE", True)
    result = m4_eval.evaluate_ragas(["q"], ["a"], [["c"]], ["g"])
    assert result["evaluation_status"] == "skipped"
    assert result["skipped"] is True
    assert result["per_question"][0].contexts == ["c"]
    assert result["num_questions"] == 1
    assert m4_eval.failure_analysis(result["per_question"]) == []


def test_mismatched_eval_input_lengths_fail_before_calling_api(monkeypatch):
    available = Mock(side_effect=AssertionError("API availability should not be checked"))
    monkeypatch.setattr(llm, "is_available", available)
    with pytest.raises(ValueError, match="equal lengths"):
        m4_eval.evaluate_ragas(["q"], [], [["c"]], ["g"])
    available.assert_not_called()


def test_failure_analysis_orders_by_average_and_finds_weak_metric():
    results = [
        m4_eval.EvalResult("strong", "a", ["c"], "g", 0.9, 0.9, 0.9, 0.8),
        m4_eval.EvalResult("weak", "a", ["c"], "g", 0.5, 0.6, 0.4, 0.1),
    ]
    failure = m4_eval.failure_analysis(results, bottom_n=1)[0]
    assert failure["question"] == "weak"
    assert failure["worst_metric"] == "context_recall"
    assert failure["score"] == 0.1
    assert "chunks" in failure["diagnosis"]
    assert m4_eval.failure_analysis(results, bottom_n=0) == []


def test_combined_enrichment_uses_one_call_and_indexes_every_field(monkeypatch):
    payload = {
        "summary": "Nghỉ phép 12 ngày.",
        "questions": ["Nhân viên được nghỉ phép bao nhiêu ngày?"],
        "context": "Quy định nghỉ phép.",
        "metadata": {"topic": "Nghỉ phép", "entities": [], "category": "hr",
                     "language": "vi", "source": "incorrect-source"},
    }
    complete = Mock(return_value=json.dumps(payload))
    monkeypatch.setattr(llm, "complete", complete)
    text = "Nhân viên được nghỉ phép 12 ngày mỗi năm."
    chunk = m5_enrichment.enrich_chunks([
        {"text": text, "metadata": {"source": "policy.md", "parent_id": "parent-1", "page": 2}},
    ])[0]
    assert complete.call_count == 1
    assert complete.call_args.kwargs["json_mode"] is True
    assert text in chunk.enriched_text
    assert chunk.summary in chunk.enriched_text
    assert chunk.hypothesis_questions[0] in chunk.enriched_text
    assert "policy.md" in chunk.enriched_text
    assert chunk.auto_metadata["source"] == "policy.md"
    assert chunk.auto_metadata["parent_id"] == "parent-1"
    assert chunk.auto_metadata["page"] == 2
    assert chunk.auto_metadata["category"] == "hr"


@pytest.mark.parametrize("response", [
    None, "invalid JSON", "[]",
    '{"summary": [], "questions": {}, "context": 1, "metadata": []}',
])
def test_bad_enrichment_responses_use_complete_local_fallbacks(monkeypatch, response):
    complete = Mock(return_value=response)
    monkeypatch.setattr(llm, "complete", complete)
    text = "Mật khẩu thay đổi mỗi 90 ngày. Sử dụng VPN. Không chia sẻ tài khoản."
    chunk = m5_enrichment.enrich_chunks([{"text": text, "metadata": {"source": "it.md"}}])[0]
    assert complete.call_count == 1
    assert chunk.original_text == text
    assert text in chunk.enriched_text
    assert chunk.summary and len(chunk.summary) < len(text)
    assert chunk.hypothesis_questions
    assert chunk.auto_metadata["category"] == "it"
    assert chunk.auto_metadata["source"] == "it.md"


def test_individual_fallbacks_handle_short_text_and_zero_questions(monkeypatch):
    monkeypatch.setattr(llm, "complete", lambda *args, **kwargs: None)
    text = "Nhân viên được nghỉ phép 12 ngày."
    assert m5_enrichment.summarize_chunk(text) == text
    assert m5_enrichment.generate_hypothesis_questions(text, 1)[0].endswith("?")
    assert m5_enrichment.generate_hypothesis_questions(text, 0) == []
    assert text in m5_enrichment.contextual_prepend(text, "policy.md")
    assert m5_enrichment.extract_metadata(text)["category"] == "hr"
    assert m5_enrichment.summarize_chunk("") == ""
    with pytest.raises(ValueError, match="Unknown"):
        m5_enrichment.enrich_chunks([], methods=["misspelled"])
