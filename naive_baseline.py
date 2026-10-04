"""
Basic RAG Baseline — Chạy TRƯỚC để có scores so sánh.
=====================================================
Basic = paragraph chunking + dense-only search (không hybrid, không rerank, không enrichment).
Đây là RAG đã học ở buổi trước — hôm nay sẽ cải thiện từng bước.
"""

import sys, os, time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.m1_chunking import load_documents, chunk_basic
from src.m2_search import DenseSearch
from src.m4_eval import load_test_set, evaluate_ragas, save_report
from src.llm import ANSWER_SYSTEM_PROMPT, complete
from config import NAIVE_COLLECTION, REPORTS_DIR


def main():
    print("=" * 60)
    print("BASIC RAG BASELINE")
    print("(paragraph chunking + dense-only, no rerank, no enrichment)")
    print("=" * 60)

    docs = load_documents()
    chunks = []
    for doc in docs:
        for c in chunk_basic(doc["text"], metadata=doc["metadata"]):
            chunks.append({"text": c.text, "metadata": c.metadata})
    print(f"  {len(chunks)} basic paragraph chunks")

    search = DenseSearch()
    search.index(chunks, collection=NAIVE_COLLECTION)

    test_set = load_test_set()
    questions, answers, all_contexts, ground_truths = [], [], [], []

    for i, item in enumerate(test_set):
        results = search.search(item["question"], top_k=3, collection=NAIVE_COLLECTION)
        contexts = [r.text for r in results]

        answer = None
        if contexts:
            context_str = "\n\n".join(contexts)
            answer = complete([
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {item['question']}"},
            ], max_tokens=600)
        answer = answer or (contexts[0] if contexts else "Không tìm thấy.")

        answers.append(answer)
        questions.append(item["question"])
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    print("\nBASIC BASELINE SCORES")
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        if results.get("evaluation_status", "completed") == "completed":
            print(f"  {m}: {results.get(m, 0):.4f}")
        else:
            print(f"  {m}: N/A (chưa có đánh giá RAGAS hợp lệ)")
    save_report(results, [], path=os.path.join(REPORTS_DIR, "naive_baseline_report.json"))
    print("\nBaseline hoàn tất. Chạy python main.py để so sánh với production.")
    return results


if __name__ == "__main__":
    import argparse
    import config
    parser = argparse.ArgumentParser(description="Basic RAG baseline")
    parser.add_argument("--offline", action="store_true", help="Chạy cục bộ, không gọi API hay tải model")
    if parser.parse_args().offline:
        config.OFFLINE_MODE = True
    start = time.time()
    main()
    print(f"Total: {time.time() - start:.1f}s")
