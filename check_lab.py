"""
Kiểm tra định dạng bài nộp trước khi submit.
Chạy: python check_lab.py

⚠️ Lỗi định dạng khiến script chấm tự động không chạy → trừ 5 điểm thủ tục.
"""

import json
import os
import sys
import subprocess

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve(path: str) -> str:
    return os.path.join(ROOT_DIR, path)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def check_file(path: str, required: bool = True) -> bool:
    if os.path.exists(_resolve(path)):
        print(f"  ✅ {path}")
        return True
    elif required:
        print(f"  ❌ THIẾU: {path}")
        return False
    else:
        print(f"  ⚠️  Optional: {path}")
        return True


def check_json(path: str, required_keys: list[str]) -> bool:
    try:
        with open(_resolve(path), encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            print(f"  ❌ {path} phải chứa một JSON object")
            return False
        missing = [k for k in required_keys if k not in data]
        if missing:
            print(f"  ❌ {path} thiếu keys: {missing}")
            return False
        print(f"  ✅ {path} — keys OK")
        if data.get("evaluation_status") in {"skipped", "failed", "partial"}:
            print("  ⚠️  Báo cáo chưa có điểm RAGAS hợp lệ; cần API key để đánh giá thật.")
        return True
    except (json.JSONDecodeError, OSError) as e:
        print(f"  ❌ {path} — {e}")
        return False


def check_todos() -> int:
    """Count remaining TODO markers in src/."""
    count = 0
    for root, _, files in os.walk(_resolve("src")):
        for f in files:
            if f.endswith(".py"):
                with open(os.path.join(root, f), encoding="utf-8") as fh:
                    for line in fh:
                        if "# TODO:" in line:
                            count += 1
    return count


def run_tests() -> tuple[int, int]:
    """Run pytest and return (passed, total)."""
    try:
        import re
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=no", "-q"],
            capture_output=True, text=True, timeout=120, encoding="utf-8", errors="replace",
            cwd=ROOT_DIR,
        )
        lines = result.stdout.strip().split("\n")
        summary = lines[-1] if lines else ""
        m_pass = re.search(r"(\d+)\s+passed", summary)
        m_fail = re.search(r"(\d+)\s+failed", summary)
        m_error = re.search(r"(\d+)\s+errors?", summary)
        passed = int(m_pass.group(1)) if m_pass else 0
        failed = int(m_fail.group(1)) if m_fail else 0
        collection_errors = int(m_error.group(1)) if m_error else 0
        total = passed + failed + collection_errors
        if result.returncode != 0:
            print(result.stdout[-2500:])
            if result.stderr:
                print(result.stderr[-1000:])
            if total == passed:
                return 0, 0
        return passed, total
    except Exception as e:
        print(f"  ⚠️  pytest error: {e}")
        return 0, 0


def validate():
    print("🔍 Kiểm tra bài nộp Lab 18: Production RAG\n")
    errors = 0

    # 1. Source files
    print("📁 Source code:")
    for f in ["src/m1_chunking.py", "src/m2_search.py", "src/m3_rerank.py",
              "src/m4_eval.py", "src/m5_enrichment.py", "src/pipeline.py"]:
        if not check_file(f):
            errors += 1

    # 2. Reports
    print("\n📊 Reports:")
    if check_file("reports/ragas_report.json"):
        if not check_json("reports/ragas_report.json", ["aggregate", "num_questions"]):
            errors += 1
    else:
        errors += 1
    check_file("reports/naive_baseline_report.json", required=False)

    # 3. Analysis
    print("\n📝 Analysis:")
    if not check_file("analysis/failure_analysis.md"):
        errors += 1

    # 4. Individual reflections
    print("\n👤 Individual reflections:")
    reflections = []
    ref_dir = "analysis/reflections"
    if os.path.isdir(_resolve(ref_dir)):
        reflections.extend([f"{ref_dir}/{f}" for f in os.listdir(_resolve(ref_dir))
                            if f.startswith("reflection_") and f.endswith(".md") and f != "reflection_TEMPLATE.md"])
    if os.path.isdir(_resolve("analysis")):
        reflections.extend([f"analysis/{f}" for f in os.listdir(_resolve("analysis"))
                            if f.startswith("reflection_") and f.endswith(".md") and f != "reflection_TEMPLATE.md"])

    if reflections:
        for r in set(reflections):
            print(f"  ✅ {r}")
    else:
        print(f"  ⚠️  Chưa có file reflection cá nhân (đặt tại {ref_dir}/reflection_[HọTên].md hoặc analysis/reflection_[HọTên].md)")

    # 5. TODO count
    print("\n🔧 TODO markers:")
    todo_count = check_todos()
    if todo_count == 0:
        print("  ✅ Không còn TODO nào")
    else:
        print(f"  ⚠️  Còn {todo_count} TODO chưa implement")

    # 6. Tests
    print("\n🧪 Auto-tests:")
    passed, total = run_tests()
    if total > 0:
        pct = passed / total * 100
        print(f"  {'✅' if passed == total else '❌'} {passed}/{total} tests passed ({pct:.0f}%)")
        if passed != total:
            errors += 1
    else:
        print("  ⚠️  Không chạy được tests")
        errors += 1

    # 7. Summary
    print("\n" + "=" * 50)
    if errors == 0:
        print("✅ Kiểm tra mã nguồn và định dạng đạt. Xem lưu ý về RAGAS và reflection ở trên.")
    else:
        print(f"❌ Có {errors} lỗi. Sửa trước khi nộp.")
    print("=" * 50)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(validate())
