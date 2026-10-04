from __future__ import annotations

"""Production RAG Pipeline — Ghép toàn bộ M1+M2+M3+M4+M5."""

import os, sys, time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import BM25Search, HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from src.llm import ANSWER_SYSTEM_PROMPT, complete, is_available
from src.query_planning import plan_retrieval
from src.retrieval_policy import annotate_documents, prefer_current_candidates
from config import RERANK_TOP_K


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    # Step 1: Load & Chunk (M1)
    t0 = time.time()
    print("\n[1/4] Chunking documents...", flush=True)
    docs = annotate_documents(load_documents())
    all_chunks = []
    parent_contexts = {}
    parent_documents = {}
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        parent_contexts.update({p.metadata["parent_id"]: p.text for p in parents})
        parent_documents.update({p.metadata["parent_id"]: {"text": p.text, "metadata": p.metadata}
                                 for p in parents})
        for child in children:
            all_chunks.append({"text": child.text, "metadata": {**child.metadata, "parent_id": child.parent_id}})
    print(f"  ✓ {len(all_chunks)} chunks from {len(docs)} documents ({time.time()-t0:.1f}s)", flush=True)
    if not all_chunks:
        raise ValueError("Không có nội dung để lập chỉ mục. Kiểm tra thư mục data/ và text layer của PDF.")

    # Step 2: Enrichment (M5)
    t0 = time.time()
    mode = "LLM, 1 API call/chunk" if is_available() else "trích xuất cục bộ"
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, {mode})...", flush=True)
    enriched = enrich_chunks(all_chunks)
    if enriched:
        all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        print(f"  ✓ Enriched {len(enriched)} chunks ({time.time()-t0:.1f}s)", flush=True)
    else:
        print("  Không có nội dung enrichment; dùng các đoạn gốc.", flush=True)

    # Step 3: Index (M2)
    t0 = time.time()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.parent_contexts = parent_contexts
    search.parent_documents = parent_documents
    search.documents = docs
    search.parent_bm25 = BM25Search()
    search.parent_bm25.index(list(parent_documents.values()))
    search.index(all_chunks)
    print(f"  ✓ Indexed ({time.time()-t0:.1f}s)", flush=True)

    # Step 4: Reranker (M3)
    t0 = time.time()
    print("\n[4/4] Preparing reranker...", flush=True)
    reranker = CrossEncoderReranker()
    print(f"  ✓ Reranker ready ({time.time()-t0:.1f}s)", flush=True)

    return search, reranker


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query through pipeline."""
    plan = plan_retrieval(query, getattr(search, "documents", []))
    results = []
    parent_bm25 = getattr(search, "parent_bm25", None)
    for retrieval_query in plan["queries"]:
        results.extend(search.search(retrieval_query))
        if parent_bm25 is not None:
            results.extend(parent_bm25.search(retrieval_query, top_k=10))
    parent_documents = getattr(search, "parent_documents", {})
    parents = getattr(search, "parent_contexts", {})
    docs = []
    seen = set()
    for result in results:
        parent_id = result.metadata.get("parent_id")
        parent = parent_documents.get(parent_id)
        text = parent["text"] if parent else parents.get(parent_id, result.text)
        metadata = dict(parent["metadata"] if parent else result.metadata)
        identity = (text, metadata.get("source"))
        if identity not in seen:
            seen.add(identity)
            docs.append({"text": text, "score": result.score, "metadata": metadata})
    for parent in parent_documents.values():
        identity = (parent["text"], parent["metadata"].get("source"))
        if parent["metadata"].get("source") in plan["sources"] and identity not in seen:
            seen.add(identity)
            docs.append({**parent, "score": 0.0})
    docs = prefer_current_candidates(query, docs)
    # Rerank original parent passages. Generated enrichment is used for recall,
    # while the answer receives only the actual source evidence.
    reranked = reranker.rerank(query, docs, top_k=len(docs))
    if reranked:
        selected = [{"text": result.text, "metadata": result.metadata} for result in reranked]
    else:
        selected = docs
    routed = []
    planned_sources = set(plan["sources"])
    seen_sources = set()
    # Keep reranker relevance order while choosing the best passage per source.
    for doc in selected:
        source = doc["metadata"].get("source")
        if source in planned_sources and source not in seen_sources:
            routed.append(doc)
            seen_sources.add(source)
    if routed:
        selected = routed
    contexts = []
    for result in selected:
        text = parents.get(result["metadata"].get("parent_id"), result["text"])
        if text and text not in contexts:
            contexts.append(text)
            if len(contexts) == RERANK_TOP_K:
                break
    answer = None
    if contexts:
        context_str = "\n\n".join(contexts)
        answer = complete([
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"},
        ], max_tokens=600)
    answer = answer or (contexts[0] if contexts else "Không tìm thấy thông tin.")
    return answer, contexts


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set."""
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []

    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    t0 = time.time()
    print(f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    eval_status = results.get("evaluation_status", "completed")
    print(f"  RAGAS: {eval_status} ({time.time()-t0:.1f}s)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        if eval_status == "completed":
            s = results.get(m, 0)
            print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")
        else:
            print(f"  {m}: N/A (chưa có đánh giá RAGAS hợp lệ)")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures)
    return results


if __name__ == "__main__":
    import argparse
    import config
    parser = argparse.ArgumentParser(description="Production RAG pipeline")
    parser.add_argument("--offline", action="store_true", help="Chạy cục bộ, không gọi API hay tải model")
    if parser.parse_args().offline:
        config.OFFLINE_MODE = True
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
