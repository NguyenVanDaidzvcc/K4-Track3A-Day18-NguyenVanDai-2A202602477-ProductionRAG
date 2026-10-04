from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import glob
import hashlib
import math
import os
import re
import sys
from dataclasses import dataclass, field
from itertools import pairwise

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from config import (
    DATA_DIR,
    HIERARCHICAL_CHILD_SIZE,
    HIERARCHICAL_PARENT_SIZE,
    SEMANTIC_THRESHOLD,
)

_semantic_encoder = None
_semantic_encoder_unavailable = False
_semantic_warning_shown = False


def _warn_semantic_fallback() -> None:
    global _semantic_warning_shown
    if not _semantic_warning_shown:
        print("  ℹ️ Semantic model chưa có hoặc đang chạy offline; dùng độ giống từ vựng để chunk.", flush=True)
        _semantic_warning_shown = True


def _get_semantic_encoder():
    global _semantic_encoder, _semantic_encoder_unavailable
    if config.OFFLINE_MODE or _semantic_encoder_unavailable:
        return None
    if _semantic_encoder is None:
        try:
            if not config.ALLOW_MODEL_DOWNLOAD:
                from huggingface_hub import try_to_load_from_cache
                cached_config = try_to_load_from_cache("sentence-transformers/all-MiniLM-L6-v2", "config.json")
                if not isinstance(cached_config, str) or not os.path.isfile(cached_config):
                    _semantic_encoder_unavailable = True
                    return None
            from sentence_transformers import SentenceTransformer
            _semantic_encoder = SentenceTransformer(
                "all-MiniLM-L6-v2", local_files_only=not config.ALLOW_MODEL_DOWNLOAD,
            )
        except Exception:  # noqa: BLE001 - Optional model dependencies may raise backend-specific errors.
            _semantic_encoder_unavailable = True
    return _semantic_encoder


def _lexical_similarity(left: str, right: str) -> float:
    a = set(re.findall(r"\w+", left.casefold()))
    b = set(re.findall(r"\w+", right.casefold()))
    return len(a & b) / math.sqrt(len(a) * len(b)) if a and b else 0.0


def _bounded_chunks(text: str, size: int) -> list[str]:
    """Pack paragraphs and split long ones at whitespace, bounded in characters."""
    if size <= 0:
        raise ValueError("Chunk size must be greater than zero")
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    pieces = []
    for paragraph in paragraphs:
        while len(paragraph) > size:
            breaks = list(re.finditer(r"\s+", paragraph[:size + 1]))
            boundary = breaks[-1] if breaks and breaks[-1].start() > 0 else None
            if boundary is None:
                pieces.append(paragraph[:size])
                paragraph = paragraph[size:].lstrip()
            else:
                pieces.append(paragraph[:boundary.start()].rstrip())
                paragraph = paragraph[boundary.end():].lstrip()
        if paragraph:
            pieces.append(paragraph)
    chunks = []
    current = ""
    for piece in pieces:
        candidate = f"{current}\n\n{piece}" if current else piece
        if len(candidate) > size:
            chunks.append(current)
            current = piece
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    if not -1 <= threshold <= 1:
        raise ValueError("Semantic threshold must be between -1 and 1")
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n\s*\n", text) if s.strip()]
    if not sentences:
        return []
    metadata = metadata or {}
    encoder = _get_semantic_encoder()
    similarities = None
    if encoder is not None:
        try:
            embeddings = encoder.encode(sentences)
            similarities = []
            for left, right in pairwise(embeddings):
                dot = sum(float(a) * float(b) for a, b in zip(left, right))
                norm_left = math.sqrt(sum(float(a) ** 2 for a in left))
                norm_right = math.sqrt(sum(float(b) ** 2 for b in right))
                similarities.append(dot / (norm_left * norm_right + 1e-9))
        except Exception:  # noqa: BLE001 - Keep chunking usable if an optional model backend fails.
            global _semantic_encoder_unavailable
            _semantic_encoder_unavailable = True
            similarities = None
    if similarities is None:
        _warn_semantic_fallback()
        similarities = [_lexical_similarity(a, b) for a, b in pairwise(sentences)]
    groups = [[sentences[0]]]
    for sentence, similarity in zip(sentences[1:], similarities):
        if similarity < threshold:
            groups.append([])
        groups[-1].append(sentence)
    return [Chunk(
        text=" ".join(group),
        metadata={**metadata, "strategy": "semantic", "chunk_index": i,
                  "similarity_method": "embedding" if encoder is not None and not _semantic_encoder_unavailable else "lexical"},
    ) for i, group in enumerate(groups)]


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    if parent_size <= 0 or child_size <= 0:
        raise ValueError("Parent and child chunk sizes must be greater than zero")
    metadata = metadata or {}
    document_key = f"{metadata.get('source', '')}\0{text}"
    document_id = hashlib.sha256(document_key.encode("utf-8")).hexdigest()[:16]
    parents, children = [], []
    for index, parent_text in enumerate(_bounded_chunks(text, parent_size)):
        parent_id = f"parent_{document_id}_{index}"
        parents.append(Chunk(
            text=parent_text,
            metadata={**metadata, "chunk_type": "parent", "parent_id": parent_id,
                      "chunk_index": index, "strategy": "hierarchical"},
        ))
        for child_text in _bounded_chunks(parent_text, child_size):
            children.append(Chunk(
                text=child_text, parent_id=parent_id,
                metadata={**metadata, "chunk_type": "child", "parent_id": parent_id,
                          "chunk_index": len(children), "strategy": "hierarchical"},
            ))
    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = metadata or {}
    chunks = []
    section, lines = "", []
    fence_character, fence_length = None, 0

    def flush() -> None:
        content = "".join(lines).strip()
        if content:
            chunks.append(Chunk(
                text=content,
                metadata={**metadata, "section": section, "strategy": "structure",
                          "chunk_index": len(chunks)},
            ))

    for line in text.splitlines(keepends=True):
        fence = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line.rstrip("\r\n"))
        if fence_character is not None:
            lines.append(line)
            if (fence and fence.group(1)[0] == fence_character
                    and len(fence.group(1)) >= fence_length and not fence.group(2).strip()):
                fence_character = None
            continue
        if fence:
            fence_character, fence_length = fence.group(1)[0], len(fence.group(1))
            lines.append(line)
            continue
        if re.match(r"^ {0,3}#{1,6}[ \t]+\S", line):
            flush()
            section, lines = line.strip(), [line]
        else:
            lines.append(line)
    flush()
    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
