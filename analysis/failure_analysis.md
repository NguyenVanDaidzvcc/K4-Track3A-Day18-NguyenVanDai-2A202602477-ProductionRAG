# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Nguyễn Văn Đại

**Khóa:** K4 — Track 3A

**Nguồn:** [Basic](../reports/naive_baseline_report.json), [Production](../reports/ragas_report.json), [test set gốc](../test_set.json) và tài liệu trong `data/`.

## RAGAS Scores

Cả hai báo cáo có `evaluation_status: completed`, đủ **20 câu hỏi**, cùng cấu hình bộ chấm: RAGAS `0.1.22`, LLM `gpt-4o-mini`, embedding `text-embedding-3-small`; answer relevancy dùng prompt tiếng Việt `vi-question-generation-v1`, `strictness = 3`.

| Metric | Naive Baseline | Production | Δ (Production − Basic) |
|---|---:|---:|---:|
| Faithfulness | 0.7930 | 0.8792 | +0.0862 |
| Answer Relevancy | 0.7665 | 0.8544 | +0.0879 |
| Context Precision | 0.9417 | 0.9292 | −0.0125 |
| Context Recall | 0.8500 | 0.8833 | +0.0333 |

Các giá trị và chênh lệch được tính từ `aggregate` đầy đủ rồi làm tròn 4 chữ số. Production tăng ba chỉ số, context precision giảm nhẹ. Một lần so sánh trên 20 câu chưa đủ để quy mức tăng cho riêng enrichment, query planning hoặc reranking, hoặc kết luận ý nghĩa thống kê.

Ảnh màn hình người dùng cung cấp ghi **Total time: 279.4s** cho lần chạy so sánh. Đây là tổng thời gian theo ảnh; JSON không lưu thời gian từng bước, nên không suy ra thời gian riêng cho Basic, Production, enrichment hay RAGAS.

## Bottom-5 Failures

Giữ đúng thứ tự năm phần tử đầu của `reports/ragas_report.json → failures`. Theo [hàm `failure_analysis()`](../src/m4_eval.py), các câu được xếp theo **trung bình bốn metric tăng dần**, sau đó ghi metric thấp nhất của từng câu. Nhãn tự động như `LLM hallucinating` hoặc `Missing relevant chunks` chỉ là gợi ý theo metric; cần đối chiếu đầu ra và nguồn trước khi kết luận nguyên nhân.

| Thứ tự | Câu trong test set | Worst metric | Điểm | Trung bình 4 metric |
|---|---:|---|---:|---:|
| #1: Tạm ứng quá hạn | 17 | faithfulness | 0.2500 | 0.6655 |
| #2: Hoàn chi đào tạo | 14 | faithfulness | 0.3333 | 0.7564 |
| #3: MFA | 8 | context_recall | 0.5000 | 0.7824 |
| #4: Thâm niên và ngày phép | 5 | context_recall | 0.5000 | 0.8116 |
| #5: Chu kỳ đổi mật khẩu | 7 | context_recall | 0.5000 | 0.8394 |

### #1 — Tạm ứng quá hạn: lỗi suy luận đã xác nhận

- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Test set ghi hạn 15 ngày, quá hạn 5 ngày; phí `2%/tháng × 15.000.000 = 300.000 VNĐ/tháng`, tính pro-rata khoảng **50.000 VNĐ cho 5 ngày**. Phần pro-rata cần giả định tháng có 30 ngày; nguồn chưa quy định cách tính phần tháng.
- **Got:** Đầu ra nhận đúng “quá hạn 5 ngày”, sau đó viết “số tháng quá hạn là 20 ngày”, dùng `20/30 ≈ 0.67` rồi kết luận **khoảng 200.000 VNĐ**.
- **Worst metric:** `faithfulness = 0.2500`; answer relevancy `0.7454`, context precision `1.0000`, context recall `0.6667`.
- **Error Tree:** Output sai? **Có**, lấy toàn bộ 20 ngày làm thời gian tính phạt, mâu thuẫn với 5 ngày quá hạn đã xác định → Context đúng? **Có**: [chính sách tạm ứng](../data/tam_ung.md) là context đầu tiên, có hạn 15 ngày và phí 2%/tháng → Query OK? Đã lấy được nguồn cần thiết; báo cáo không lưu query plan nên chưa đánh giá được từng rewrite → Lỗi chính nằm ở **suy luận và kiểm tra phép tính**, kèm giả định phần tháng chưa được làm rõ.
- **Root cause:** Bằng chứng trực tiếp là biến thời gian đổi từ 5 sang 20 trong cùng đầu ra. Prompt ở lần đánh giá đã yêu cầu kiểm tra đơn vị, thời gian và phân biệt phí tháng với phí quá hạn, nhưng chưa có bước tính độc lập bảo đảm yêu cầu được tuân thủ. `temperature=0` đã có, nên chỉ “hạ temperature” không giải quyết được vấn đề này. Recall thấp cũng có thể chịu ảnh hưởng bởi pro-rata trong ground truth mà nguồn chưa nêu.
- **Suggested fix:** Hướng dẫn sau lần đánh giá **đã được sửa** trong [prompt trả lời](../src/llm.py): tính `ngày quá hạn = max(0, ngày đã trôi qua − thời hạn)`, dùng nhất quán số ngày đó; khi thiếu quy tắc phần tháng, không khẳng định phí chính thức, và phép minh họa phải nêu giả định. **Chưa có lần đánh giá API mới để xác nhận tác dụng**. Bước tiếp theo là kiểm tra phép tính bằng công cụ xác định và thêm ca mốc 15/16/20 ngày. Minh họa với tháng 30 ngày phải dùng `15.000.000 × 0.02 × 5/30 = 50.000 VNĐ`.

### #2 — Hoàn chi đào tạo: điểm thấp chưa chứng minh câu trả lời sai

- **Question:** Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng hoàn thành khóa học. Phải hoàn trả bao nhiêu?
- **Expected:** Cam kết ít nhất 1 năm sau khi hoàn thành khóa học; nghỉ sau 8 tháng là trước hạn, hoàn trả **100% chi phí, tức 25.000.000 VNĐ**.
- **Got:** “Nhân viên phải hoàn trả **100% chi phí** đào tạo đã được tài trợ, tức là **25.000.000 VNĐ**, vì nhân viên nghỉ việc trước thời hạn cam kết 1 năm.”
- **Worst metric:** `faithfulness = 0.3333`; answer relevancy `0.6924`, context precision `1.0000`, context recall `1.0000`.
- **Error Tree:** Output sai? **Chưa thấy lỗi nội dung** so với câu hỏi và ground truth → Context đúng? **Có**: [chính sách hoàn chi đào tạo](../data/hoan_chi_dao_tao.md) nêu cam kết 1 năm và hoàn trả 100% khi nghỉ sớm → Query OK? Nguồn chính đã có; không có query-plan log để phân tích rewrite → Cần kiểm tra **bộ chấm và cách xử lý suy luận từ dữ kiện trong câu hỏi**, trước khi sửa retrieval.
- **Root cause:** Chưa xác định nguyên nhân faithfulness thấp. Khoản 25 triệu và mốc 8 tháng đến từ câu hỏi; nguồn cung cấp điều kiện 1 năm và tỷ lệ 100%, nên kết luận có căn cứ khi kết hợp chúng. Báo cáo chỉ lưu diagnostics answer relevancy, không có trace từng claim của faithfulness. Nhiễu bộ chấm hoặc khó khăn khi chấm suy luận là **giả thuyết**, chưa phải nguyên nhân đã chứng minh.
- **Suggested fix:** Làm rõ phép suy luận: “Thời hạn cam kết là 12 tháng sau khi hoàn thành khóa học. Vì 8 < 12, hoàn trả 100% × 25.000.000 = 25.000.000 VNĐ.” Rà soát thủ công, lưu trace faithfulness nếu bổ sung được và chạy lặp cùng cấu hình trong lần đánh giá sau. Giữ nguyên điểm đã lưu và test set gốc.

### #3 — MFA: context hiện hành thiếu phần lịch sử trong ground truth

- **Question:** Có cần kích hoạt xác thực đa yếu tố (MFA) không?
- **Expected:** Ground truth ghi v2.0 hiện hành bắt buộc MFA cho email, VPN và hệ thống nội bộ; kèm thông tin **v1.0 không yêu cầu MFA**.
- **Got:** “Có, tất cả nhân viên bắt buộc phải kích hoạt xác thực đa yếu tố (MFA) cho tài khoản email, VPN và các hệ thống nội bộ.”
- **Worst metric:** `context_recall = 0.5000`; faithfulness `1.0000`, answer relevancy `0.6295`, context precision `1.0000`.
- **Error Tree:** Output sai? **Không sai quy định hiện hành**, nhưng không có phần lịch sử trong ground truth → Context đúng? **Có**: [mật khẩu v2.0](../data/mat_khau_v2.md) nêu MFA bắt buộc; context không chứa v1.0 → Query OK? Câu hỏi không yêu cầu lịch sử; [bộ lọc phiên bản](../src/retrieval_policy.py) loại bản đã bị thay thế với ý định hiện hành → Điểm thấp liên quan đến **phạm vi ground truth và context**, chưa chứng minh thiếu bằng chứng cho câu hỏi hiện hành.
- **Root cause:** Ground truth có cả hiện hành và lịch sử, còn câu hỏi chỉ hỏi có cần MFA hay không. Thiếu v1.0 trong context là quan sát trực tiếp và có thể giải thích recall 0.5; chưa có trace để xác nhận chính xác từng mệnh đề. [V1.0](../data/mat_khau_v1.md) không có mục MFA; không nên biến việc không nhắc tới thành kết luận rộng hơn tài liệu. Answer relevancy có ba nhãn `noncommittal = 0`, cosine đều khoảng `0.6295`; điểm thấp không do gate đưa về 0.
- **Suggested fix:** Giữ ưu tiên v2.0 cho câu hỏi hiện hành và ghi rõ phiên bản trong đầu ra. Chú thích sự khác biệt phạm vi đánh giá; nếu tạo bộ bổ sung, thêm câu hỏi so sánh v1.0/v2.0 với bằng chứng từng phiên bản. Không đưa v1.0 vào mọi context chỉ để tăng recall trên bộ gốc.

### #4 — Thâm niên và ngày phép: hiện hành đúng, ground truth có bản cũ

- **Question:** Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?
- **Expected:** V2024 hiện hành: từ **3 năm**, cộng 1 ngày cho mỗi 3 năm; ground truth kèm quy định cũ v2023 là **5 năm**.
- **Got:** “Nhân viên có thâm niên từ 3 năm trở lên sẽ được cộng thêm 1 ngày phép cho mỗi 3 năm làm việc liên tục.”
- **Worst metric:** `context_recall = 0.5000`; faithfulness `1.0000`, answer relevancy `0.7466`, context precision `1.0000`.
- **Error Tree:** Output sai? **Không sai theo v2024** → Context đúng? **Có**: [nghỉ phép năm v2024](../data/nghi_phep_nam_v2024.md) chứa quy định 3 năm và tuyên bố thay thế v1.0; không có [v2023](../data/nghi_phep_nam_v2023.md) trong context → Query OK? Không yêu cầu năm cũ hay so sánh, nên lọc hiện hành phù hợp → Có dấu hiệu **khác biệt phạm vi lịch sử**, chưa có bằng chứng lỗi chunking.
- **Root cause:** Phần lịch sử 5 năm của ground truth không được context hiện hành hỗ trợ. Khác biệt này quan sát được và có thể giải thích recall 0.5; không kết luận tự động rằng BM25 hoặc chunking đã làm mất đoạn hiện hành. Nguồn v2024 đã cung cấp đầy đủ điều kiện và mức cộng thêm.
- **Suggested fix:** Nêu phiên bản áp dụng. Với câu hỏi thực sự so sánh, lấy cả v2023/v2024 và trình bày riêng quy định 5 năm/3 năm. Đánh giá thêm tình huống hiện hành/lịch sử/so sánh ngoài bộ gốc; giữ bộ lọc phiên bản cho câu hỏi hiện hành.

### #5 — Chu kỳ đổi mật khẩu: đúng 120 ngày, không kèm 90 ngày cũ

- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Expected:** V2.0 hiện hành: **120 ngày**; ground truth kèm thông tin bản cũ **90 ngày** đã bị thay thế.
- **Got:** “Mật khẩu phải được thay đổi mỗi 120 ngày.”
- **Worst metric:** `context_recall = 0.5000`; faithfulness `1.0000`, answer relevancy `0.8575`, context precision `1.0000`.
- **Error Tree:** Output sai? **Không sai quy định hiện hành** → Context đúng? **Có**: [mật khẩu v2.0](../data/mat_khau_v2.md) ghi chu kỳ 120 ngày và thay thế v1.0; bản cũ 90 ngày không có trong context → Query OK? Câu hỏi không có ý định lịch sử; ưu tiên hiện hành phù hợp → Cần xem **phạm vi ground truth**, trước khi tăng số context.
- **Root cause:** Context thiếu bằng chứng về chu kỳ 90 ngày so với ground truth có lịch sử. Khác biệt phạm vi là quan sát trực tiếp; việc nó gây đúng recall 0.5 vẫn là diễn giải từ điểm tổng vì không có trace từng mệnh đề. Chưa có bằng chứng câu trả lời 120 ngày sai hoặc retrieval nhầm phiên bản.
- **Suggested fix:** Trả lời “Theo chính sách mật khẩu v2.0 hiện hành, đổi mật khẩu mỗi 120 ngày.” Chỉ lấy [v1.0](../data/mat_khau_v1.md) khi hỏi lịch sử hoặc so sánh; nếu cần đánh giá chuyển tiếp chính sách, thêm câu hỏi yêu cầu rõ ràng và lưu kết quả riêng.

## Case Study (cho presentation)

**Question chọn phân tích:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?

**Error Tree walkthrough:**

1. **Output đúng? → Không.** Đầu ra xác định 5 ngày quá hạn, rồi tính theo 20 ngày và ra khoảng 200.000 VNĐ. Lỗi này xác nhận được từ văn bản, độc lập với nhãn RAGAS.
2. **Context đúng? → Có phần chính sách cần thiết.** Hạn 15 ngày và mức phí 2%/tháng có trong nguồn. Cách tính phần tháng chưa được quy định rõ, nên không thể coi 50.000 VNĐ là mức phí chính thức mà không nêu giả định.
3. **Query rewrite OK? → Chưa đủ log để đánh giá toàn bộ.** Nguồn tạm ứng đã có, nên chưa thấy thiếu nguồn là nguyên nhân chính. Báo cáo không lưu truy vấn rewrite hoặc thứ hạng ứng viên.
4. **Fix ở bước → Tổng hợp và kiểm tra phép tính.** Giữ riêng hạn thanh toán 15 ngày, thời gian đã qua 20 ngày và thời gian quá hạn 5 ngày. Prompt đã được làm rõ sau đánh giá; cần bước tính xác định để kiểm tra đầu ra.

**Câu trả lời đề xuất:** “Khoản tạm ứng đã quá hạn 5 ngày (20 − 15). Chính sách quy định phí 2%/tháng trên 15.000.000 VNĐ, tương đương 300.000 VNĐ/tháng, nhưng chưa nêu cách tính phần tháng. Nếu giả định tính pro-rata với tháng 30 ngày thì phí cho 5 ngày là 50.000 VNĐ; cần xác nhận quy tắc phần tháng với phòng Tài chính.” Đây là phương án đề xuất, chưa phải đầu ra của lần chạy RAGAS mới.

**Sửa đổi đã kiểm tra sau lần đánh giá:** Ngoài hướng dẫn tính quá hạn, [pipeline](../src/pipeline.py) đã giữ thứ tự rerank khi lọc theo nguồn của query plan và chọn một đoạn cha tốt nhất cho mỗi nguồn. `python check_lab.py` đạt **118/118 kiểm thử**, **0 TODO**. Các kiểm tra cục bộ xác nhận hành vi lọc/thứ tự và đường chạy bằng mock; chưa xác nhận câu trả lời LLM mới hoặc mức thay đổi điểm RAGAS. Hai JSON kết quả vẫn là báo cáo của lần chạy trước sửa.

**Nếu có thêm 1 giờ, sẽ optimize:**

| Thời gian dự kiến | Công việc tiếp theo | Kết quả cần kiểm tra |
|---|---|---|
| 0–15 phút | Thiết kế bước kiểm tra phép tính xác định dựa trên dữ kiện và chính sách, bổ sung cho prompt đã sửa. | Ngày quá hạn là `max(0, ngày đã qua − hạn)`, không dùng toàn bộ thời gian đã qua. |
| 15–30 phút | Thêm kiểm tra mốc 15/16/20 ngày, tỷ lệ tháng và giả định pro-rata. | Chưa quá hạn không bị tính phí; không coi 30 ngày/tháng là chính sách khi nguồn chưa nêu. |
| 30–45 phút | Rà soát thủ công Q14, Q18, Q20 có faithfulness thấp dù đáp án có căn cứ; chú thích phạm vi lịch sử của Q8/Q5/Q7. | Đối chiếu từng kết luận với nguồn; chưa quy lỗi hallucination khi thiếu trace bộ chấm. |
| 45–60 phút | Chạy kiểm tra cục bộ sau thay đổi mới; chuẩn bị lưu query plan, backend và trace cho lần đánh giá kế tiếp. Nếu chạy RAGAS lại, lưu báo cáo mới riêng. | Giữ test set và báo cáo gốc; chỉ công bố thay đổi điểm sau khi có phép đo mới. |

Đây là kế hoạch tiếp theo, chưa được thực hiện. Việc biên soạn phân tích và kiểm tra sau sửa không gọi thêm API, không chỉnh điểm hoặc ground truth, và không khẳng định các sửa đổi đã cải thiện bốn chỉ số.
