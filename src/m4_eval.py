"""Module 4: RAGAS evaluation and failure analysis."""

from __future__ import annotations

import json
import math
import os
import sys
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src import llm

METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
EVALUATION_EMBEDDING_MODEL = "text-embedding-3-small"
RELEVANCY_PROMPT_VERSION = "vi-question-generation-v1"
RELEVANCY_STRICTNESS = 3


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    evaluation_status: str = "completed"
    evaluation_error: str | None = None
    diagnostics: dict = field(default_factory=dict)


def _evaluation_config() -> dict:
    try:
        ragas_version = version("ragas")
    except PackageNotFoundError:
        ragas_version = None
    return {
        "ragas_version": ragas_version,
        "llm_model": config.OPENAI_MODEL,
        "embedding_model": EVALUATION_EMBEDDING_MODEL,
        "metrics": list(METRICS),
        "answer_relevancy": {
            "prompt_language": "vi",
            "prompt_version": RELEVANCY_PROMPT_VERSION,
            "strictness": RELEVANCY_STRICTNESS,
            "formula": "mean(cosine_similarities) * int(not any(noncommittal_labels))",
        },
    }


def _evaluation_row_key(row: dict) -> tuple:
    return row["question"], row["answer"], tuple(row["contexts"])


def _build_evaluation_metrics():
    """Use fresh metric instances and trace the unchanged RAGAS score calculation."""
    from ragas.llms.prompt import Prompt
    from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

    stock_prompt = AnswerRelevancy().question_generation
    question_generation = Prompt(
        name="question_generation",
        instruction=(
            "Tạo một câu hỏi bằng tiếng Việt cho câu trả lời đã cho và xác định câu trả lời có "
            "thiếu cam kết (noncommittal) hay không. Đặt noncommittal là 1 nếu câu trả lời né tránh, "
            "mơ hồ hoặc không xác định, và là 0 nếu câu trả lời dứt khoát. Ví dụ, "
            "'Tôi không biết' hoặc 'Tôi không chắc' là những câu trả lời thiếu cam kết."
        ),
        output_format_instruction=stock_prompt.output_format_instruction,
        examples=[
            {
                "answer": "Albert Einstein sinh ra ở Đức.",
                "context": "Albert Einstein là nhà vật lý lý thuyết sinh ra ở Đức, được coi là một trong những nhà khoa học có ảnh hưởng nhất.",
                "output": {"question": "Albert Einstein sinh ra ở đâu?", "noncommittal": 0},
            },
            {
                "answer": "Nó có thể đổi màu da theo nhiệt độ của môi trường.",
                "context": "Một nghiên cứu phát hiện loài ếch mới trong rừng Amazon có khả năng đổi màu da theo nhiệt độ của môi trường.",
                "output": {"question": "Loài ếch mới được phát hiện có khả năng đặc biệt gì?", "noncommittal": 0},
            },
            {
                "answer": "Everest.",
                "context": "Ngọn núi cao nhất thế giới tính từ mực nước biển nằm trong dãy Himalaya.",
                "output": {"question": "Ngọn núi cao nhất thế giới là gì?", "noncommittal": 0},
            },
            {
                "answer": "Tôi không biết tính năng đột phá của chiếc điện thoại được phát minh năm 2023 vì không có thông tin sau năm 2022.",
                "context": "Năm 2023, một chiếc điện thoại có thời lượng pin một tháng được phát minh.",
                "output": {"question": "Chiếc điện thoại được phát minh năm 2023 có tính năng đột phá gì?", "noncommittal": 1},
            },
        ],
        input_keys=["answer", "context"],
        output_key="output",
        output_type="json",
        language="vietnamese",
    )

    class TracedAnswerRelevancy(AnswerRelevancy):
        def __init__(self):
            super().__init__(question_generation=question_generation, strictness=RELEVANCY_STRICTNESS)
            self.traces = {}
            self._active_trace = ContextVar("answer_relevancy_trace", default=None)

        def calculate_similarity(self, question, generated_questions):
            similarities = super().calculate_similarity(question, generated_questions)
            trace = self._active_trace.get()
            if trace is not None:
                trace["cosine_similarities"] = [_finite_score(value) for value in similarities]
            return similarities

        def _calculate_score(self, answers, row):
            trace = {
                "generated_questions": [answer.question for answer in answers],
                "noncommittal_labels": [int(answer.noncommittal) for answer in answers],
                "noncommittal_gate": any(answer.noncommittal for answer in answers),
                "cosine_similarities": [],
                "score": None,
            }
            token = self._active_trace.set(trace)
            try:
                # Keep RAGAS's mean-cosine formula and any-noncommittal zero gate.
                score = super()._calculate_score(answers, row)
                trace["score"] = _finite_score(score)
                return score
            finally:
                self._active_trace.reset(token)
                self.traces.setdefault(_evaluation_row_key(row), []).append(trace)

    relevance = TracedAnswerRelevancy()
    return [Faithfulness(), relevance, ContextPrecision(), ContextRecall()], relevance


def load_test_set(path: str = config.TEST_SET_PATH) -> list[dict]:
    """Load evaluation questions and reference answers from JSON."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _unavailable_result(questions, answers, contexts, ground_truths, status, error):
    return {
        **dict.fromkeys(METRICS, 0.0),
        "evaluation_status": status,
        "skipped": status == "skipped",
        "error": error,
        "num_questions": len(questions),
        "evaluation_config": _evaluation_config(),
        "per_question": [
            EvalResult(question, answer, list(context), truth, 0.0, 0.0, 0.0, 0.0,
                       evaluation_status=status, evaluation_error=error)
            for question, answer, context, truth in zip(questions, answers, contexts, ground_truths)
        ],
    }


def _finite_score(value) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return score if math.isfinite(score) else None


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Evaluate with RAGAS 0.1; unavailable evaluation is explicitly marked."""
    if len({len(questions), len(answers), len(contexts), len(ground_truths)}) != 1:
        raise ValueError("questions, answers, contexts and ground_truths must have equal lengths")
    if not all(isinstance(value, str) for values in (questions, answers, ground_truths) for value in values):
        raise ValueError("questions, answers and ground_truths must contain strings")
    if not all(isinstance(context, list) and all(isinstance(text, str) for text in context)
               for context in contexts):
        raise ValueError("contexts must contain lists of strings")
    if not questions or config.OFFLINE_MODE or not llm.is_available():
        reason = "No evaluation questions." if not questions else "Offline mode, missing API key, or unavailable API."
        return _unavailable_result(questions, answers, contexts, ground_truths, "skipped", reason)

    try:
        # Evaluation only needs API access; disable optional RAGAS telemetry.
        os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
        from datasets import Dataset
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import evaluate
        from ragas.run_config import RunConfig

        metrics, relevance = _build_evaluation_metrics()
        dataset = Dataset.from_dict({
            "question": questions, "answer": answers,
            "contexts": contexts, "ground_truth": ground_truths,
        })
        timeout = config.OPENAI_TIMEOUT_SECONDS
        run_config = RunConfig(timeout=timeout, max_retries=1, max_workers=2, max_wait=1)
        evaluator = ChatOpenAI(model=config.OPENAI_MODEL, api_key=config.OPENAI_API_KEY,
                               temperature=0, timeout=timeout, max_retries=0)
        embeddings = OpenAIEmbeddings(model=EVALUATION_EMBEDDING_MODEL, api_key=config.OPENAI_API_KEY,
                                      request_timeout=timeout, max_retries=0)
        result = evaluate(dataset, metrics=metrics,
                          llm=evaluator, embeddings=embeddings,
                          run_config=run_config, raise_exceptions=True)
        rows = result.to_pandas().to_dict(orient="records")
        if len(rows) != len(questions):
            raise ValueError("RAGAS returned a different number of rows than the evaluation inputs")

        scores_by_metric = {metric: [] for metric in METRICS}
        per_question = []
        invalid_rows = 0
        for question, answer, context, truth, row in zip(questions, answers, contexts, ground_truths, rows):
            scores = {metric: _finite_score(row.get(metric)) for metric in METRICS}
            invalid = [metric for metric, score in scores.items() if score is None]
            for metric, score in scores.items():
                if score is not None:
                    scores_by_metric[metric].append(score)
            invalid_rows += bool(invalid)
            input_row = {"question": question, "answer": answer, "contexts": context}
            traces = relevance.traces.get(_evaluation_row_key(input_row), [])
            diagnostics = {"answer_relevancy": traces.pop(0)} if traces else {}
            per_question.append(EvalResult(
                question, answer, list(context), truth,
                **{metric: score if score is not None else 0.0 for metric, score in scores.items()},
                evaluation_status="partial" if invalid else "completed",
                evaluation_error=f"Missing or non-finite metrics: {', '.join(invalid)}" if invalid else None,
                diagnostics=diagnostics,
            ))
        return {
            **{metric: sum(scores) / len(scores) if scores else 0.0
               for metric, scores in scores_by_metric.items()},
            "evaluation_status": "partial" if invalid_rows else "completed",
            "skipped": False,
            "error": f"{invalid_rows} question(s) have missing or non-finite scores." if invalid_rows else None,
            "num_questions": len(questions),
            "evaluation_config": _evaluation_config(),
            "per_question": per_question,
        }
    except Exception as exc:
        # SDK validation messages can contain configuration values; retain only the type.
        error = f"RAGAS evaluation failed ({type(exc).__name__})."
        print(f"  {error}")
        return _unavailable_result(questions, answers, contexts, ground_truths, "failed", error)


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Diagnose the lowest average scores, excluding unavailable evaluations."""
    if bottom_n <= 0:
        return []
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    }
    ranked = []
    for result in eval_results:
        row = asdict(result) if isinstance(result, EvalResult) else result
        if row.get("evaluation_status", "completed") != "completed":
            continue
        scores = {metric: _finite_score(row.get(metric)) for metric in METRICS}
        if any(score is None for score in scores.values()):
            continue
        worst_metric = min(scores, key=scores.get)
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        ranked.append({
            "question": row["question"], "worst_metric": worst_metric,
            "score": scores[worst_metric], "average_score": sum(scores.values()) / len(METRICS),
            "diagnosis": diagnosis, "suggested_fix": suggested_fix,
        })
    return sorted(ranked, key=lambda row: row["average_score"])[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str | None = None):
    """Save numeric aggregates, evaluation status and individual rows as JSON."""
    path = path or os.path.join(config.REPORTS_DIR, "ragas_report.json")
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    per_question = [asdict(row) if isinstance(row, EvalResult) else row
                    for row in results.get("per_question", [])]
    report = {
        "aggregate": {metric: _finite_score(results.get(metric)) or 0.0 for metric in METRICS},
        "num_questions": results.get("num_questions", len(per_question)),
        "evaluation_status": results.get("evaluation_status", "completed"),
        "skipped": results.get("skipped", False),
        "error": results.get("error"),
        "evaluation_config": results.get("evaluation_config", _evaluation_config()),
        "per_question": per_question,
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run python main.py to generate answers and evaluate the pipeline.")
