# Reflection — Lab 18: Production RAG

**Họ và tên:** Nguyễn Văn Đại

**MSSV:** 2A202602477

**Khóa:** K4 — Track 3A

**Project:** Production RAG cho tra cứu tài liệu chính sách nội bộ

Nội dung dưới đây dựa trên mã nguồn hiện tại và hai báo cáo đánh giá đã lưu trong repository. Các kết quả quan sát được trình bày riêng với kế hoạch cải tiến; những hạng mục trong kế hoạch chưa được coi là đã triển khai hoặc đã đo lường.

## Phần 1: Mapping bài giảng vào triển khai

| Khái niệm | Module và hàm | Áp dụng trong project và quan sát |
|---|---|---|
| Semantic chunking | M1: `chunk_semantic()` | Nhóm các câu theo độ tương đồng với ngưỡng mặc định `0.85`. Khi không có model, dùng độ tương đồng từ vựng và ghi nhận phương pháp trong metadata. Đây là chiến lược có thể lựa chọn; pipeline Production hiện dùng hierarchical chunking. Chưa có phép đo trong hai báo cáo để kết luận ngưỡng này tạo số chunk tối ưu. |
| Hierarchical và structure-aware chunking | M1: `chunk_hierarchical()`, `chunk_structure_aware()` | Hierarchical tạo đoạn cha tối đa 2.048 ký tự và đoạn con tối đa 256 ký tự theo cấu hình, liên kết bằng `parent_id`. Truy xuất đoạn con rồi khôi phục đoạn cha giúp lấy đủ điều kiện của chính sách. Structure-aware giữ cấu trúc tiêu đề, bảng, danh sách và code block khi phân chia tài liệu. |
| BM25 và fusion | M2: `segment_vietnamese()`, `BM25Search`, `DenseSearch`, `reciprocal_rank_fusion()` | Chuẩn hóa tiếng Việt, chữ hoa/thường và dấu câu; thay dấu gạch dưới của từ ghép để truy vấn khớp với tài liệu. RRF kết hợp thứ hạng bắt đầu từ 1, không cộng lặp một ứng viên trong cùng danh sách và giữ các nguồn khác nhau dù nội dung giống nhau. Khi BGE chưa có trong cache, nhánh vector dùng token-hash xác định, mang tính từ vựng. |
| Reranking | M3: `CrossEncoderReranker.rerank()`, `_lexical_scores()` | Có đường chạy cross-encoder `BAAI/bge-reranker-v2-m3` khi model khả dụng. Khi thiếu model trong cache, dùng độ bao phủ truy vấn có trọng số IDF kết hợp thành phần BM25. Từ đặc trưng như tên cấp bậc hoặc thiết bị được ưu tiên hơn từ chung xuất hiện ở nhiều ứng viên. Sau lần đánh giá, pipeline đã được sửa để giữ thứ tự rerank khi lọc theo nguồn của query plan và chọn một đoạn cha tốt nhất cho mỗi nguồn. |
| Enrichment | M5: `_enrich_single_call()`, `enrich_chunks()` | Một yêu cầu LLM có thể tạo summary, câu hỏi giả định, context và metadata cho một chunk chưa có cache. Nội dung bổ sung hỗ trợ truy xuất; nguồn, số trang và `parent_id` ban đầu vẫn được giữ. Cache trong `.cache/enrichment` phân biệt kết quả API với fallback cục bộ. |
| RAGAS và phân tích lỗi | M4: `evaluate_ragas()`, `failure_analysis()`, `save_report()` | Đánh giá faithfulness, answer relevancy, context precision và context recall. Báo cáo giữ trạng thái đánh giá, cấu hình và kết quả từng câu. Khi offline hoặc API không khả dụng, kết quả được ghi rõ là chưa đánh giá hợp lệ, không xem điểm mặc định là điểm RAGAS thật. |

Ngoài năm module, [pipeline](../../src/pipeline.py) kết hợp LLM với các thành phần từ vựng. LLM lập kế hoạch truy xuất từ danh mục tài liệu và tạo câu trả lời; BM25, fusion và reranking chọn bằng chứng. Các đoạn con cùng cha được gộp để không chiếm hết số vị trí context. Câu trả lời nhận đoạn cha gốc, thay vì xem summary hoặc câu hỏi do enrichment sinh ra là bằng chứng của chính sách.

[Xử lý phiên bản](../../src/retrieval_policy.py) dựa trên thông tin thay thế được nêu rõ trong tài liệu cùng nhóm chính sách. Ngày hiệu lực mới hơn, đứng riêng, chưa đủ để kết luận bản cũ bị thay thế. Truy vấn hiện hành loại các bản đã bị thay thế; truy vấn yêu cầu năm, phiên bản lịch sử hoặc so sánh vẫn có thể giữ tài liệu cũ. Cách xử lý này đáp ứng ý định câu hỏi và tránh trộn quy định giữa các phiên bản.

### Kết quả đã quan sát

Trong lần cập nhật này, `python check_lab.py` đã được chạy sau sửa mã và đạt **118/118 kiểm thử**, **0 TODO**. Hai kiểm thử mới xác nhận giữ thứ tự rerank khi lọc nguồn, chọn một đoạn cha cho mỗi nguồn và đường fallback. Các kiểm tra trong repository gồm đường chạy bằng mock; không chứng minh model BGE thực tế đã được tải hoặc chất lượng trả lời LLM đã tăng.

Hai báo cáo [Basic](../../reports/naive_baseline_report.json) và [Production](../../reports/ragas_report.json) đều ghi `evaluation_status: completed`, gồm **20 câu hỏi**. Cấu hình bộ chấm giống nhau: RAGAS `0.1.22`, LLM `gpt-4o-mini`, embedding đánh giá `text-embedding-3-small`; prompt answer relevancy tiếng Việt `vi-question-generation-v1`, `strictness = 3`.

| Metric | Basic | Production | Δ (Production − Basic) |
|---|---:|---:|---:|
| Faithfulness | 0.7930 | 0.8792 | +0.0862 |
| Answer Relevancy | 0.7665 | 0.8544 | +0.0879 |
| Context Precision | 0.9417 | 0.9292 | −0.0125 |
| Context Recall | 0.8500 | 0.8833 | +0.0333 |

Bảng là kết quả trong hai JSON hiện tại, làm tròn từ `aggregate` đầy đủ; chênh lệch cũng được tính trước khi làm tròn. Production tăng ba chỉ số và giảm nhẹ context precision. **Tổng thời gian 279.4 giây** lấy từ ảnh màn hình người dùng cung cấp, không phải trường trong JSON. Chưa có số đo thời gian từng bước, số token hoặc chi phí để kết luận về tốc độ và mức tiết kiệm.

Chưa có thí nghiệm tách từng thành phần để quy mức thay đổi cho riêng enrichment, reranking hay query planning; một lần chạy trên 20 câu cũng chưa đủ để xác nhận ý nghĩa thống kê. Embedding dùng để chấm RAGAS khác với embedding của hệ truy xuất. Fallback hash và lexical đã được quan sát ở phiên làm việc trước khi BGE chưa có cache; hai JSON mới không lưu backend retrieval/reranking, nên không tự suy ra backend thực tế của lần chạy mới chỉ từ tên lớp hoặc cấu hình bộ chấm.

**Sửa mã sau đánh giá:** Prompt trả lời đã yêu cầu tính `ngày quá hạn = max(0, ngày đã trôi qua − thời hạn)`, dùng cùng số ngày trong phép tính và nêu rõ giả định phần tháng; thiếu quy tắc trong nguồn thì chưa đủ căn cứ để khẳng định phí chính thức. Pipeline đã giữ thứ tự rerank khi lọc nguồn và một đoạn cha tốt nhất cho mỗi nguồn. Hai sửa đổi đã qua kiểm tra cục bộ nêu trên, nhưng **chưa chạy lại API**; điểm và câu trả lời trong báo cáo vẫn thuộc lần chạy trước sửa, chưa xác nhận hiệu quả chất lượng của mã mới.

## Phần 2: Khó khăn và cách giải quyết

### Lỗi quan sát được và quá trình debug

Hai báo cáo mới ghi `error: null` và ảnh terminal cho thấy chương trình hoàn tất;
không có exception runtime được ghi nhận để trích dẫn. Lỗi đầu ra cụ thể ở câu tạm
ứng là câu **“số tháng quá hạn là 20 ngày”**, dù trước đó xác định quá hạn 5 ngày.
Nhãn **“LLM hallucinating”** trong `failures` là chẩn đoán tự động theo metric,
không phải thông báo exception hay kết luận đã kiểm chứng cho mọi câu điểm thấp.

Quá trình debug bắt đầu từ `failures`, ghép câu hỏi với `per_question` để đọc
`answer`, `contexts` và `ground_truth`, rồi đối chiếu `data/tam_ung.md`. Nguồn đã có
hạn 15 ngày và phí 2%/tháng, nên kiểm tra phép thay số trong đầu ra trước khi đổi
retrieval. Với thứ tự context, đọc bước dựng `selected` rồi `routed` cho thấy vòng
lặp theo danh sách nguồn đã ghi đè thứ tự reranker. Kiểm thử hồi quy đặt thứ tự
nguồn ngược thứ tự rerank, thêm nguồn không liên quan và nhiều đoạn cùng nguồn;
sau sửa, context giữ thứ tự liên quan và đủ các nguồn cần thiết.

Kiến thức cần bổ sung là cách phân biệt thời gian đã qua với thời gian quá hạn,
phạm vi hiện hành/lịch sử và giới hạn của chấm suy luận bằng LLM. Cách bổ sung là
đọc quy tắc trong tài liệu gốc, kiểm tra đơn vị và phép tính độc lập, đồng thời
đối chiếu diagnostics trước khi quy nguyên nhân cho một module.

### Khả năng chạy khi thiếu dịch vụ hoặc model

Một pipeline phụ thuộc Qdrant, model neural và API có nhiều điều kiện môi trường. Triển khai hiện tại kiểm tra Qdrant một lần rồi dùng Qdrant trong bộ nhớ nếu server không khả dụng; nếu không có thư viện Qdrant, có đường cosine cục bộ. Model chỉ được tìm trong cache mặc định, và thất bại tải model được ghi nhớ để không lặp lại trên mỗi truy vấn. Chế độ offline bỏ qua việc tải model và gọi API.

Fallback giúp chương trình tiếp tục xử lý dữ liệu, nhưng chất lượng của vector token-hash không tương đương embedding ngữ nghĩa. Việc diễn giải báo cáo cần nêu backend thực tế, thay vì suy ra rằng tên lớp `DenseSearch` đồng nghĩa với BGE đã chạy. PDF không có text layer cũng cần OCR trước khi có thể đóng góp vào corpus; bỏ qua PDF scan chưa giải quyết được phần dữ liệu thiếu này.

### Độ liên quan, bằng chứng gốc và phiên bản

Những từ chung như “nhân viên” hoặc con số có thể kéo các đoạn không cùng chủ đề lên cao. Reranker từ vựng đã dùng IDF theo tập ứng viên để tăng trọng số từ đặc trưng. Query planning chọn nguồn từ danh mục và tạo truy vấn cho các ý cần kết hợp, trong khi bước khôi phục đoạn cha giúp giữ bảng, điều kiện và ngoại lệ của quy định.

Enrichment bổ sung nhiều từ có ích cho recall nhưng cũng có nguy cơ tạo thông tin không được nguồn hỗ trợ. Pipeline vì vậy dùng văn bản gốc để tạo câu trả lời. Metadata phiên bản và bằng chứng thay thế được áp dụng trước khi chọn context, đồng thời giữ đường truy xuất lịch sử khi câu hỏi yêu cầu.

### Chi phí enrichment và tính hợp lệ của cache

Combined enrichment giới hạn ở một yêu cầu cho mỗi chunk chưa được tái sử dụng. Cache được định danh theo nội dung, nguồn, tiêu đề tài liệu, model, prompt/phiên bản prompt và mode. JSON được ghi qua file tạm rồi thay thế nguyên tử; cache không hợp lệ được bỏ qua. Kết quả fallback sau khi API thất bại không được lưu thành một lần enrichment API thành công, và offline không đọc cache API như một kết quả cục bộ. Khi tái sử dụng cache, metadata nguồn mới vẫn được ghép lại để tránh giữ nhầm `parent_id` hoặc số trang cũ. Chưa có số liệu latency và chi phí đủ để định lượng mức tiết kiệm trong reflection này.

### Hạn chế của ground truth và bộ chấm

Theo [failure analysis](../failure_analysis.md), bottom-5 được xếp theo trung bình bốn metric: tạm ứng quá hạn, hoàn chi đào tạo, MFA, thâm niên và chu kỳ đổi mật khẩu. Với MFA, thâm niên và chu kỳ mật khẩu, Production đạt faithfulness `1.0` nhưng context recall `0.5`. Ground truth có thông tin chính sách cũ, trong khi câu hỏi không yêu cầu so sánh và context đã ưu tiên hiện hành. Đây là khác biệt phạm vi quan sát được, có thể giải thích recall thấp; không có trace từng mệnh đề để khẳng định chính xác cách bộ chấm đưa ra điểm. Không thể chỉ từ recall kết luận tài liệu hiện hành bị truy xuất sai.

Với tạm ứng 15 triệu thanh toán sau 20 ngày, faithfulness là `0.25` và có **lỗi thực sự trong đầu ra**: câu trả lời nhận đúng 5 ngày quá hạn nhưng tính `20/30` rồi kết luận khoảng `200.000 VNĐ`. Nguồn đã có hạn 15 ngày và phí `2%/tháng`, nên thiếu nguồn không phải nguyên nhân chính được quan sát. Nếu minh họa theo tháng 30 ngày thì phải dùng `5/30`, ra `50.000 VNĐ`; tuy nhiên tài liệu chưa quy định cách tính phần tháng, nên phải ghi rõ giả định. Prompt đã được làm rõ sau đánh giá, còn bước kiểm tra phép tính độc lập là công việc tiếp theo. Đáp án tham chiếu không phải bằng chứng bổ sung cho tài liệu.

Câu hoàn chi đào tạo có faithfulness `0.3333` dù trả đúng 100% của 25 triệu, vì 8 tháng ngắn hơn cam kết 1 năm. Câu Junior trả đúng 17 triệu có faithfulness `0.5`; câu nghỉ không lương 20 ngày trả đúng CEO cũng có faithfulness `0.5`. Các kết luận này có căn cứ trong câu hỏi và context. Nhãn tự động “LLM hallucinating” chưa chứng minh chúng sai; cần rà soát từng claim và độ ổn định của bộ chấm. Giả thuyết về nhiễu hoặc cách đánh giá suy luận chưa được xác nhận bằng trace hay lần chạy lặp.

Diagnostics answer relevancy lưu câu hỏi tái dựng, cosine và nhãn `noncommittal`; công thức có thể đưa điểm về 0 khi xuất hiện nhãn thiếu cam kết. Trong báo cáo mới, câu malware có answer relevancy `0.8967`, faithfulness `1.0`, ba nhãn `noncommittal = 0`; câu MFA có answer relevancy `0.6295` cũng có ba nhãn bằng 0. Hai trường hợp hiện tại không bị gate đưa về 0. Cần tiếp tục phân biệt phủ định theo chính sách với việc không đưa ra câu trả lời trong các kiểm tra bổ sung. Không tự ý sửa điểm hoặc loại câu hỏi để nâng kết quả; lưu diagnostics và nhận xét độc lập, vì một lần chấm bằng LLM chưa xác nhận mọi nguyên nhân lỗi.

## Phần 3: Kế hoạch đề xuất cho chính project Production RAG

Project hiện có pipeline hoàn chỉnh, đường chạy cục bộ, hai báo cáo RAGAS hợp lệ và hai sửa đổi sau đánh giá đã qua kiểm tra cục bộ. Những việc dưới đây là **đề xuất tiếp theo chưa thực hiện**, bổ sung cho prompt và bước lọc nguồn đã sửa.

| Ưu tiên | Công việc đề xuất | Tiêu chí kiểm tra |
|---|---|---|
| 1 | Rà soát bộ câu hỏi theo ý định hiện hành, lịch sử và so sánh; chú thích ground truth có yêu cầu về bản cũ hoặc giả định tính toán. Giữ nguyên bộ gốc khi so sánh với báo cáo hiện tại. | Có bản ghi giải thích cho các trường hợp MFA, thâm niên và pro-rata; không đưa đáp án tham chiếu vào retrieval hoặc prompt trả lời. |
| 2 | Bổ sung cơ chế kiểm tra phép tính xác định, hỗ trợ yêu cầu về đơn vị, thời hạn và giả định đã có trong prompt mới. Kiểm tra mốc 15/16/20 ngày và giữ phân biệt phí tháng với phí minh họa phần tháng. | Không lấy thời gian đã qua thay cho ngày quá hạn; không trình bày quy ước chưa được nguồn xác nhận như chính sách đã ban hành. |
| 3 | So sánh hierarchical với structure-aware trên tài liệu có bảng và ngoại lệ; thử kích thước cha/con bằng một tập kiểm tra tách biệt. | Đo khả năng giữ đủ điều kiện của chính sách, số context và lượng token; không chỉ tối ưu điểm trên 20 câu hiện có. |
| 4 | Khi có tài nguyên và quyền tải model, tạo cache BGE embedding/reranker và so sánh với fallback đang dùng. | Ghi rõ backend từng lần chạy; đo bốn chỉ số chất lượng cùng latency và tài nguyên. Chỉ kết luận về neural sau phép đo thực tế. |
| 5 | Đánh giá tách các thành phần query planning, enrichment và reranking; thu thập cache hit, số yêu cầu và chi phí. | Mỗi lần chỉ thay một yếu tố, giữ corpus và cấu hình bộ chấm tương đương; xác định phần tăng chất lượng và phần tăng chi phí. |
| 6 | Rà soát thủ công các điểm thấp và chạy lặp đánh giá với cùng cấu hình tiếng Việt; bổ sung kiểm tra câu phủ định và phạm vi phiên bản. | Lưu kết quả từng lần và diagnostics, báo mức biến thiên; không thay công thức RAGAS hoặc chọn bỏ kết quả bất lợi. |

**Lộ trình dự kiến:** tuần 1 dành cho rà soát ground truth, bằng chứng phiên bản và giả định số; tuần 2 dành cho thí nghiệm chunking, backend neural nếu đủ điều kiện, và đo chất lượng/latency/chi phí. Đây là phân bổ công việc đề xuất, không phải thời gian hoàn thành đã được xác nhận. Việc OCR các PDF scan cần một hạng mục dữ liệu riêng, kèm kiểm tra chất lượng text trước khi đưa vào chỉ mục.
