# Lab 18: Production RAG Pipeline

**Họ và tên:** Nguyễn Văn Đại

**Mã sinh viên:** 2A202602477

**Khóa:** K4 — Track 3A

**Bài thực hành:** Ngày 18 — Production RAG

Tên repository theo quy chuẩn bài nộp:

```text
K4-Track3A-DAY18-NguyenVanDai-2A202602477-ProductionRAG
```

## Tổng quan

Project xây dựng hệ thống hỏi đáp trên tài liệu nội bộ tiếng Việt, so sánh Naive Baseline với Production RAG bằng bốn chỉ số RAGAS trên 20 câu hỏi.

```text
Tài liệu → M1 Chunking → M5 Enrichment → M2 Hybrid Search
         → Lập kế hoạch truy vấn → M3 Reranking → LLM Answer → M4 RAGAS
```

| Module | Triển khai |
|---|---|
| M1 — Chunking | Basic, semantic, hierarchical và structure-aware; Production tìm trên đoạn con và trả về đoạn cha. |
| M2 — Search | BM25 tiếng Việt, dense search và Reciprocal Rank Fusion (RRF). |
| M3 — Reranking | Cross-encoder khi có model; rerank từ vựng khi model chưa khả dụng. |
| M4 — Evaluation | Faithfulness, answer relevancy, context precision, context recall; lưu điểm và chẩn đoán từng câu. |
| M5 — Enrichment | Tóm tắt, câu hỏi giả định, contextual prepend và metadata; chế độ kết hợp dùng tối đa một API call cho mỗi chunk chưa được cache. |

Production lập kế hoạch truy vấn từ danh mục tài liệu để lấy nguồn cho câu hỏi nhiều ý, loại đoạn cha trùng nhau trước khi rerank và đưa văn bản nguồn gốc vào bước trả lời. Chính sách hiện hành được ưu tiên khi có bằng chứng thay thế phiên bản cũ; câu hỏi lịch sử vẫn được tra cứu phiên bản tương ứng.

## Yêu cầu môi trường

| Thành phần | Yêu cầu |
|---|---|
| Python | Python 3.11+ và các thư viện trong `requirements.txt`. |
| API key | `OPENAI_API_KEY` hợp lệ để tạo câu trả lời bằng LLM, enrichment và chấm RAGAS. |
| Docker / Qdrant | Tùy chọn. Khi server chưa chạy, sử dụng Qdrant trong bộ nhớ. |
| Model cục bộ | Tùy chọn. Nếu chưa có model trong cache, sử dụng tìm kiếm và rerank từ vựng dự phòng. |

## Cài đặt trên Windows (PowerShell)

Mở terminal tại thư mục gốc của project, nơi có `main.py` và `requirements.txt`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (-not (Test-Path -LiteralPath .env)) {
    Copy-Item -LiteralPath .env.example -Destination .env
}
```

Điền `OPENAI_API_KEY` trong `.env`. Giá trị `sk-...` chỉ là mẫu và được bỏ qua. Giữ `RAG_OFFLINE=0` để chạy với API. Các lệnh dùng trực tiếp Python trong `.venv` để tránh chọn nhầm môi trường.

Chạy toàn bộ Baseline, Production và bảng so sánh:

```powershell
.\.venv\Scripts\python.exe main.py
```

Chạy riêng từng pipeline nếu cần:

```powershell
.\.venv\Scripts\python.exe naive_baseline.py
.\.venv\Scripts\python.exe src/pipeline.py
```

Lần chạy đầu có thể mất vài phút do enrichment và đánh giá RAGAS gọi API. Enrichment hợp lệ được lưu tại `.cache/enrichment` để tái sử dụng khi nội dung và cấu hình tương ứng không đổi.

### Linux / macOS

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
[ -f .env ] || cp .env.example .env
# Điền OPENAI_API_KEY trong .env trước khi chạy.
.venv/bin/python main.py
```

## Chạy cục bộ khi chưa có API key

```powershell
.\.venv\Scripts\python.exe main.py --offline
```

Chế độ này dùng vector từ vựng, rerank từ vựng và nội dung trích xuất từ tài liệu. Báo cáo vẫn có đủ 20 câu hỏi, nhưng RAGAS có trạng thái `skipped` và bảng điểm hiển thị `N/A` vì chưa thực hiện đánh giá bằng API.

Mỗi lần chạy cập nhật các file báo cáo trong `reports/`. Nếu muốn giữ điểm API của lần chạy trước, sao lưu báo cáo trước khi chạy offline; chạy lại `main.py` với API key hợp lệ để tạo điểm mới.

## Model và cấu hình tùy chọn

Mặc định chỉ nạp model đã có trong cache. Đặt `RAG_ALLOW_MODEL_DOWNLOAD=1` trong `.env` để cho phép tải model. Có thể tải trước bằng Python trong môi trường ảo:

```powershell
.\.venv\Scripts\python.exe -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
.\.venv\Scripts\python.exe -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
.\.venv\Scripts\python.exe -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-v2-m3')"
```

Để sử dụng Qdrant server cục bộ:

```powershell
docker compose up -d
```

| Biến trong `.env` | Mặc định | Tác dụng |
|---|---|---|
| `RAG_OFFLINE` | `0` | Đặt `1` để bỏ qua API và model neural. |
| `RAG_ALLOW_MODEL_DOWNLOAD` | `0` | Đặt `1` để cho phép tải model chưa có trong cache. |
| `RAG_ENRICHMENT_CACHE` | `1` | Đặt `0` để tắt cache enrichment. |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model dùng cho tạo câu trả lời và enrichment. |
| `OPENAI_TIMEOUT_SECONDS` | `20` | Timeout cho mỗi yêu cầu qua helper LLM của pipeline. |
| `QDRANT_HOST` / `QDRANT_PORT` | `localhost` / `6333` | Địa chỉ Qdrant server. |

## Kết quả đánh giá hiện tại

Số liệu từ [Naive Baseline](reports/naive_baseline_report.json) và [Production](reports/ragas_report.json). Cả hai báo cáo có trạng thái `completed`, đánh giá 20 câu và cùng cấu hình bộ chấm.

| Metric | Naive Baseline | Production | Δ Production − Baseline |
|---|---:|---:|---:|
| Faithfulness | 0.7930 | 0.8792 | +0.0862 |
| Answer relevancy | 0.7665 | 0.8544 | +0.0879 |
| Context precision | 0.9417 | 0.9292 | -0.0125 |
| Context recall | 0.8500 | 0.8833 | +0.0333 |

Bộ chấm dùng RAGAS `0.1.22`, LLM `gpt-4o-mini` và embedding đánh giá `text-embedding-3-small`. Prompt tái dựng câu hỏi của answer relevancy dùng tiếng Việt cho cả Baseline và Production, với ba câu hỏi tái dựng và công thức RAGAS gốc. Cấu hình được lưu ở `evaluation_config`; chi tiết câu hỏi tái dựng, nhãn `noncommittal` và cosine được lưu tại `per_question[].diagnostics.answer_relevancy`.

Mã có đường tìm kiếm và rerank từ vựng dự phòng khi model BGE chưa có trong cache, kết hợp LLM cho lập kế hoạch truy vấn, enrichment và trả lời. Hai báo cáo chưa lưu backend retrieval/reranking; cần đối chiếu log để xác nhận backend thực tế của lần chạy. Điểm trên là kết quả của một lần đánh giá trên 20 câu, có thể dao động khi chấm lại bằng LLM. Answer relevancy đo mức liên quan, không phải tỷ lệ trả lời đúng; bộ chấm cũng có thể gán nhãn `noncommittal` cho câu trả lời đủ thông tin. Chỉ so sánh trực tiếp các lần chạy có cùng cấu hình bộ chấm.

Xem [phân tích các câu có điểm thấp](analysis/failure_analysis.md) và [reflection của Nguyễn Văn Đại](analysis/reflections/reflection_NguyenVanDai.md).

Tổng thời gian của lần chạy trong ảnh terminal là **279,4 giây**; chưa có báo cáo
thời gian từng bước để phân bổ con số này. Sau lần đánh giá, mã nguồn được sửa để
giữ thứ tự reranker khi lọc các nguồn đã chọn và giữ một đoạn cha tốt nhất mỗi nguồn.
Prompt trả lời cũng được bổ sung cách tính số ngày quá hạn và yêu cầu nêu giả định
khi tài liệu thiếu quy tắc tính phí cho phần tháng. Bảng điểm trên thuộc lần chạy
trước các sửa đổi này; cần chạy lại đánh giá bằng API để đo tác động thực tế.

## Kiểm tra mã nguồn và bài nộp

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -q
.\.venv\Scripts\python.exe check_lab.py
```

Lần kiểm tra mã nguồn gần nhất: **118 tests passed**, `check_lab.py` hoàn tất với mã thoát `0` và không còn TODO. Bộ kiểm tra gồm năm module, hoạt động pipeline, fallback, cache, truy vấn nhiều nguồn, thứ tự context sau khi lọc nguồn, lựa chọn phiên bản tài liệu và lưu dấu vết đánh giá. `check_lab.py` kiểm tra file, định dạng báo cáo, reflection, TODO và chạy tests; nội dung phân tích cần được đọc lại trước khi nộp.

## Cấu trúc project

```text
K4-Track3A-Day18-Production-RAG/
├── README.md
├── ASSIGNMENT.md                       # Đề bài gốc
├── RUBRIC.md                           # Tiêu chí chấm điểm
├── main.py                             # Baseline → Production → so sánh
├── naive_baseline.py                   # Pipeline cơ bản
├── check_lab.py                        # Kiểm tra bài nộp
├── config.py
├── requirements.txt
├── docker-compose.yml
├── .env.example
├── data/                               # 25 Markdown + 3 PDF
├── test_set.json                        # 20 câu hỏi và đáp án chuẩn
├── src/
│   ├── m1_chunking.py
│   ├── m2_search.py
│   ├── m3_rerank.py
│   ├── m4_eval.py
│   ├── m5_enrichment.py
│   ├── llm.py                          # Prompt và helper gọi LLM
│   ├── query_planning.py               # Lập kế hoạch nguồn và truy vấn
│   ├── retrieval_policy.py             # Metadata và phiên bản chính sách
│   └── pipeline.py
├── tests/                              # Tests module và hồi quy
├── analysis/
│   ├── failure_analysis.md
│   └── reflections/
│       ├── reflection_NguyenVanDai.md
│       └── reflection_TEMPLATE.md
├── reports/
│   ├── naive_baseline_report.json
│   ├── ragas_report.json
│   └── history/                        # Báo cáo trước lần cải thiện
└── templates/                          # Mẫu phân tích gốc
```

Dữ liệu gồm chính sách nhân sự, lương, IT, quy trình và an toàn. Hai PDF scan chưa có lớp văn bản cần OCR trước khi lập chỉ mục; phiên chạy hiện tại đọc được 26 tài liệu và tạo 117 đoạn con. Bộ câu hỏi gồm tra cứu, phiên bản, phủ định, nhiều nguồn, tính toán và câu hỏi mơ hồ.

Quy định nộp bài và thời hạn được ghi trong [ASSIGNMENT.md](ASSIGNMENT.md); tiêu chí chấm điểm nằm ở [RUBRIC.md](RUBRIC.md).
