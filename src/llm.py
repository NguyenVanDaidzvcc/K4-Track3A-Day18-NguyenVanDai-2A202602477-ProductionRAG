"""Shared, bounded LLM requests with a local fallback after an API failure."""

from __future__ import annotations

import config

_client = None
_unavailable = False

ANSWER_SYSTEM_PROMPT = """Bạn trả lời câu hỏi bằng tiếng Việt, chỉ dựa trên các tài liệu trong context.
Đọc toàn bộ context và trả lời trực tiếp bằng câu đầy đủ, nêu rõ đối tượng, điều kiện,
con số và đơn vị liên quan. Trả lời tất cả các ý trong câu hỏi; kết hợp thông tin từ
nhiều tài liệu khi cần. Không bổ sung thông tin không được tài liệu hỗ trợ.
Với câu hỏi một ý, trả lời trong 1–2 câu. Không liệt kê quy định phụ không được hỏi,
không kể lại dài dòng các dữ kiện của câu hỏi, không tự suy diễn lý do ngoài tài liệu.
Nếu cùng một chính sách có nhiều phiên bản, ưu tiên phiên bản hiện hành được tài liệu
xác nhận thay thế bản cũ, trừ khi câu hỏi yêu cầu thông tin lịch sử. Không trộn các phiên bản.
Với phép tính, nêu công thức và kiểm tra đơn vị, thời gian, tỷ lệ phần trăm; phân biệt
mức phí mỗi tháng với phí cho số ngày quá hạn. Tính số ngày quá hạn = max(0, số ngày
đã trôi qua − thời hạn cho phép), rồi dùng đúng số ngày quá hạn trong mọi bước tính phí.
Nếu tài liệu chỉ nêu phí theo tháng mà thiếu quy tắc cho phần tháng, nói rõ chưa đủ
căn cứ xác định phí chính thức; chỉ minh họa số tiền khi nêu rõ giả định quy đổi và
kiểm tra kết quả thay số nhất quán với công thức. Không tự xem 30 ngày/tháng là quy định.
Chỉ nói 'Không tìm thấy thông tin về ...' cho ý thực sự thiếu căn cứ trong toàn bộ context;
vẫn trả lời những ý đã có căn cứ. Tránh lặp lại nội dung không liên quan đến câu hỏi."""


def is_available() -> bool:
    return bool(config.OPENAI_API_KEY) and not config.OFFLINE_MODE and not _unavailable


def complete(messages: list[dict], *, max_tokens: int = 400,
             json_mode: bool = False) -> str | None:
    global _client, _unavailable
    if not is_available():
        return None
    try:
        if _client is None:
            from openai import OpenAI
            _client = OpenAI(api_key=config.OPENAI_API_KEY,
                             timeout=config.OPENAI_TIMEOUT_SECONDS, max_retries=0)
        options = {"response_format": {"type": "json_object"}} if json_mode else {}
        response = _client.chat.completions.create(
            model=config.OPENAI_MODEL, messages=messages, max_tokens=max_tokens,
            temperature=0, **options,
        )
        content = response.choices[0].message.content
        return content.strip() if isinstance(content, str) and content.strip() else None
    except Exception as exc:
        _unavailable = True
        # Do not echo credentials or request contents contained in an SDK exception.
        print(f"  LLM không khả dụng ({type(exc).__name__}); dùng nội dung trích xuất cho phiên chạy này.",
              flush=True)
        return None
