"""Module 5: chunk enrichment with API and deterministic offline fallbacks."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src import llm

ENRICHMENT_PROMPT_VERSION = "combined-v2"
_COMBINED_PROMPT = """Analyze the supplied passage using only its stated facts.
Use the passage's language. Return JSON with these fields: summary (at most two
short sentences), questions (at most three questions answerable from the passage),
context (one short sentence locating the passage in its document), and metadata
with topic (string), entities (string array), category (policy|hr|it|finance),
language (vi|en), date_range (string array). Preserve names, qualifiers, numbers,
units, and dates. Do not infer missing policy rules or add unsupported details."""
_cache_warning_shown = False


@dataclass
class EnrichedChunk:
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str
    enrichment_mode: str = "local"
    cache_hit: bool = False


def _sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+|\n+", text) if sentence.strip()]


def _summary_fallback(text: str) -> str:
    return " ".join(_sentences(text)[:2])


def _questions_fallback(text: str, n_questions: int = 3) -> list[str]:
    questions = []
    for sentence in _sentences(text):
        statement = sentence.rstrip(".!?")
        if statement:
            questions.append(f"Thông tin về {statement[:180]} là gì?")
        if len(questions) >= n_questions:
            break
    return questions


def _context_fallback(source: str) -> str:
    return f"Trích từ tài liệu {source}." if source else "Nội dung trích từ tài liệu."


def _metadata_fallback(text: str) -> dict:
    lowered = text.casefold()
    categories = {
        "it": ("mật khẩu", "password", "vpn", "phần mềm", "software", "bảo mật", "security"),
        "finance": ("thuế", "tax", "chi phí", "expense", "ngân sách", "budget", "thanh toán"),
        "hr": ("nhân viên", "employee", "nghỉ phép", "leave", "tuyển dụng", "lương", "salary"),
    }
    category = next((category for category, words in categories.items()
                     if any(word in lowered for word in words)), "policy")
    entities = list(dict.fromkeys(re.findall(r"\b[A-ZĐ][A-Za-zÀ-ỹĐđ]+(?:\s+[A-ZĐ][A-Za-zÀ-ỹĐđ]+)+\b", text)))
    dates = list(dict.fromkeys(re.findall(r"\b(?:\d{1,2}[/.-]\d{1,2}[/.-]\d{4}|(?:19|20)\d{2})\b", text)))
    return {
        "topic": " ".join(text.split()[:12]), "entities": entities,
        "category": category, "language": "vi" if re.search(r"[À-ỹĐđ]", text) else "en",
        "date_range": dates,
    }


def _request_json(system_prompt: str, text: str, max_tokens: int = 400) -> dict | None:
    response = llm.complete([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ], max_tokens=max_tokens, json_mode=True)
    if not response:
        return None
    try:
        parsed = json.loads(response)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _valid_questions(value, n_questions: int = 3) -> list[str] | None:
    if not isinstance(value, list) or not all(isinstance(question, str) for question in value):
        return None
    questions = [question.strip() for question in value if question.strip()]
    return questions[:n_questions] or None


def _valid_metadata(value) -> dict | None:
    if not isinstance(value, dict):
        return None
    if not isinstance(value.get("topic"), str) or not isinstance(value.get("entities"), list):
        return None
    if not all(isinstance(entity, str) for entity in value["entities"]):
        return None
    if value.get("category") not in {"policy", "hr", "it", "finance"}:
        return None
    if value.get("language") not in {"vi", "en"}:
        return None
    # Only validated fields are merged into chunk metadata.
    metadata = {key: value[key] for key in ("topic", "entities", "category", "language")}
    dates = value.get("date_range")
    if isinstance(dates, str) or (isinstance(dates, list) and all(isinstance(date, str) for date in dates)):
        metadata["date_range"] = dates
    return metadata


def _cache_path(text: str, source: str, title: str, mode: str) -> Path:
    identity = {
        "text": text, "source": source, "document_title": title,
        "model": config.OPENAI_MODEL, "prompt_version": ENRICHMENT_PROMPT_VERSION,
        "prompt": _COMBINED_PROMPT, "mode": mode,
    }
    digest = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    directory = getattr(
        config, "ENRICHMENT_CACHE_DIR", Path(config.ROOT_DIR) / ".cache" / "enrichment"
    )
    return Path(directory) / f"{digest}.json"


def _read_cached(path: Path, mode: str) -> dict | None:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return None
    if (
        not isinstance(record, dict)
        or record.get("schema_version") != 1
        or record.get("cache_key") != path.stem
        or record.get("prompt_version") != ENRICHMENT_PROMPT_VERSION
        or record.get("model") != config.OPENAI_MODEL
        or record.get("mode") != mode
    ):
        return None
    data = record.get("data")
    if not isinstance(data, dict):
        return None
    summary, context = data.get("summary"), data.get("context")
    questions, metadata = data.get("questions"), _valid_metadata(data.get("metadata"))
    if (
        not isinstance(summary, str) or not isinstance(context, str)
        or not isinstance(questions, list) or len(questions) > 3
        or not all(isinstance(question, str) and question.strip() for question in questions)
        or metadata is None
        or (mode == "api" and (not summary.strip() or not context.strip() or not questions))
    ):
        return None
    return {"summary": summary, "context": context, "questions": questions,
            "metadata": metadata, "mode": mode}


def _write_cached(path: Path, result: dict) -> None:
    global _cache_warning_shown
    temporary_path = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "schema_version": 1, "cache_key": path.stem,
            "prompt_version": ENRICHMENT_PROMPT_VERSION,
            "model": config.OPENAI_MODEL, "mode": result["mode"],
            "data": {field: result[field] for field in ("summary", "context", "questions", "metadata")},
        }
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.stem}-", suffix=".tmp", delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(record, handle, ensure_ascii=False, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except (OSError, TypeError, ValueError):
        if not _cache_warning_shown:
            print("  Enrichment cache unavailable; continuing without cache.", flush=True)
            _cache_warning_shown = True
    finally:
        if temporary_path is not None:
            with suppress(OSError):
                temporary_path.unlink()


def _cached_enrichment(text: str, source: str, title: str) -> tuple[dict, bool]:
    enabled = getattr(config, "ENRICHMENT_CACHE_ENABLED", True)
    mode = "api" if not config.OFFLINE_MODE and llm.is_available() else "local"
    if enabled:
        cached = _read_cached(_cache_path(text, source, title, mode), mode)
        if cached is not None:
            return cached, True
    result = _enrich_single_call(text, source, title)
    result_mode = result["mode"]
    # An API failure is stored only as a local fallback. It cannot suppress
    # a later successful API attempt, and offline never consumes live data.
    if enabled and result_mode in {"api", "local"} and (result_mode == "local" or mode == "api"):
        _write_cached(_cache_path(text, source, title, result_mode), result)
    return result, False


def summarize_chunk(text: str) -> str:
    """Summarize a chunk, falling back to its first two sentences."""
    if not text.strip():
        return ""
    response = llm.complete([
        {"role": "system", "content": "Tóm tắt đoạn văn trong tối đa 2 câu ngắn bằng ngôn ngữ của đoạn văn. Chỉ trả về bản tóm tắt."},
        {"role": "user", "content": text},
    ], max_tokens=150)
    return response.strip() if isinstance(response, str) and response.strip() else _summary_fallback(text)


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """Generate questions that can be answered using this chunk."""
    if n_questions <= 0 or not text.strip():
        return []
    result = _request_json(
        f'Dựa trên đoạn văn, tạo tối đa {n_questions} câu hỏi mà đoạn văn trả lời được. Trả về JSON: {{"questions": ["..."]}}.',
        text, max_tokens=200,
    )
    questions = _valid_questions(result.get("questions"), n_questions) if result else None
    return questions if questions is not None else _questions_fallback(text, n_questions)


def contextual_prepend(text: str, document_title: str = "") -> str:
    """Add document context without changing the original chunk text."""
    response = llm.complete([
        {"role": "system", "content": "Viết 1 câu ngắn mô tả vị trí và chủ đề của đoạn văn trong tài liệu. Chỉ trả về 1 câu."},
        {"role": "user", "content": f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}"},
    ], max_tokens=80) if text.strip() else None
    context = response.strip() if isinstance(response, str) and response.strip() else _context_fallback(document_title)
    if document_title and document_title not in context:
        context = f"{_context_fallback(document_title)} {context}"
    return f"{context}\n\n{text}"


def extract_metadata(text: str) -> dict:
    """Extract validated metadata, or infer basic fields from chunk content."""
    result = _request_json(
        'Trích xuất metadata. Trả về JSON với topic (chuỗi), entities (mảng chuỗi), category (policy|hr|it|finance), language (vi|en), date_range (mảng chuỗi).',
        text, max_tokens=180,
    ) if text.strip() else None
    return _valid_metadata(result) or _metadata_fallback(text)


def _enrich_single_call(text: str, source: str, title: str = "") -> dict:
    """Get all enrichment fields with at most one API call per chunk."""
    result = _request_json(
        _COMBINED_PROMPT,
        f"Tài liệu: {title or source}\nNguồn: {source}\n\nĐoạn văn:\n{text}",
    ) if text.strip() else None
    result = result or {}
    summary = result.get("summary")
    context = result.get("context")
    summary = summary.strip() if isinstance(summary, str) else ""
    context = context.strip() if isinstance(context, str) else ""
    questions = _valid_questions(result.get("questions"))
    metadata = _valid_metadata(result.get("metadata"))
    valid_fields = (bool(summary), bool(context), questions is not None, metadata is not None)
    mode = "api" if all(valid_fields) else "mixed" if any(valid_fields) else "local"
    return {
        "summary": summary or _summary_fallback(text),
        "questions": questions or _questions_fallback(text),
        "context": context or _context_fallback(source),
        "metadata": metadata or _metadata_fallback(text),
        "mode": mode,
    }


def enrich_chunks(chunks: list[dict], methods: list[str] | None = None) -> list[EnrichedChunk]:
    """Enrich indexed text while preserving raw content and original metadata."""
    methods = ["combined"] if methods is None else methods
    allowed_methods = {"summary", "hyqa", "contextual", "metadata", "combined"}
    unknown_methods = set(methods) - allowed_methods
    if unknown_methods:
        raise ValueError(f"Unknown enrichment method(s): {', '.join(sorted(unknown_methods))}")
    enriched = []
    cache_hits = 0
    for index, chunk in enumerate(chunks):
        text = chunk["text"]
        metadata = dict(chunk.get("metadata") or {})
        source = str(metadata.get("source", ""))
        cache_hit = False
        enrichment_mode = "local"
        if "combined" in methods:
            title = str(metadata.get("document_title", source))
            result, cache_hit = _cached_enrichment(text, source, title)
            cache_hits += int(cache_hit)
            enrichment_mode = result["mode"]
            summary, questions = result["summary"], result["questions"]
            context = result["context"]
            if source and source not in context:
                context = f"{_context_fallback(source)} {context}"
            enriched_text = f"{context}\n\n{text}"
            auto_metadata = result["metadata"]
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_metadata = extract_metadata(text) if "metadata" in methods else {}
        additions = []
        if summary:
            additions.append(f"Tóm tắt: {summary}")
        if questions:
            additions.append("Câu hỏi liên quan:\n" + "\n".join(questions))
        if additions:
            enriched_text = "\n\n".join([enriched_text, *additions])
        enriched.append(EnrichedChunk(
            original_text=text, enriched_text=enriched_text, summary=summary,
            hypothesis_questions=questions,
            # Supplied source, page numbers, parent IDs and metadata are authoritative.
            auto_metadata={**auto_metadata, **metadata}, method="+".join(methods),
            enrichment_mode=enrichment_mode, cache_hit=cache_hit,
        ))
        if (index + 1) % 10 == 0 or index + 1 == len(chunks):
            print(f"  Enriched {index + 1}/{len(chunks)} chunks ({cache_hits} cache hits)...", flush=True)
    return enriched


if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."
    for chunk in enrich_chunks([{"text": sample, "metadata": {"source": "Sổ tay nhân viên"}}]):
        print(chunk.enriched_text)
        print(chunk.auto_metadata)
