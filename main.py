"""
Lab 18: Production RAG Pipeline — Main Entry Point
===================================================
Chạy toàn bộ pipeline: naive baseline → production → so sánh → report.

Usage:
    python main.py
"""

import json
import os
import sys
import time
import config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def main(*, offline: bool = False):
    if offline:
        config.OFFLINE_MODE = True
    print("=" * 60)
    print("LAB 18: PRODUCTION RAG PIPELINE")
    print("=" * 60)
    start = time.time()

    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    if config.OFFLINE_MODE:
        print("Chế độ cục bộ: tìm kiếm dự phòng, câu trả lời trích xuất; RAGAS sẽ được bỏ qua.")
    elif not config.OPENAI_API_KEY:
        print("Chưa có OPENAI_API_KEY hợp lệ: dùng trích xuất cục bộ và bỏ qua RAGAS.")

    # Step 1: Basic Baseline
    print("\n📌 STEP 1: Running Basic RAG Baseline...")
    print("-" * 40)
    from naive_baseline import main as run_baseline
    run_baseline()

    # Step 2: Production Pipeline
    print("\n📌 STEP 2: Running Production Pipeline...")
    print("-" * 40)
    from src.pipeline import build_pipeline, evaluate_pipeline
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)

    # Step 3: Comparison
    print("\n📌 STEP 3: Comparison")
    print("-" * 40)
    naive_path = os.path.join(config.REPORTS_DIR, "naive_baseline_report.json")
    prod_path = os.path.join(config.REPORTS_DIR, "ragas_report.json")

    if os.path.exists(naive_path) and os.path.exists(prod_path):
        with open(naive_path, encoding="utf-8") as f:
            naive = json.load(f)
        with open(prod_path, encoding="utf-8") as f:
            prod = json.load(f)

        print(f"\n{'Metric':<25} {'Basic':>8} {'Production':>12} {'Δ':>8}")
        print("-" * 55)
        for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            if any(r.get("evaluation_status", "completed") != "completed" for r in (naive, prod)):
                print(f"  {m:<23} {'N/A':>8} {'N/A':>12} {'N/A':>8}")
                continue
            n = naive.get("aggregate", {}).get(m, 0)
            p = prod.get("aggregate", {}).get(m, 0)
            d = p - n
            status = "✓" if p >= 0.75 else " "
            print(f"{status} {m:<23} {n:>8.4f} {p:>12.4f} {d:>+8.4f}")

    elapsed = time.time() - start
    print(f"\n⏱️  Total time: {elapsed:.1f}s")
    print("\n📋 Next steps:")
    print("  1. Điền analysis/failure_analysis.md")
    print("  2. Viết analysis/reflections/reflection_[HọTên].md")
    print("  3. Chạy: python check_lab.py")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run basic and production RAG pipelines")
    parser.add_argument("--offline", action="store_true", help="Chạy cục bộ, không gọi API hay tải model")
    main(offline=parser.parse_args().offline)
