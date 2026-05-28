"""
config.py
---------
Đọc cấu hình và danh sách API key từ file .env.
Các module khác import từ đây để dùng chung một nguồn cấu hình.
"""

import os
from dotenv import load_dotenv

# Nạp biến môi trường từ file .env nằm cùng thư mục
load_dotenv()


def get_api_keys(max_keys=50):
    """
    Lấy danh sách API key từ .env (GEMINI_API_KEY_1, _2, _3, ...).
    Tự động quét nên thêm bao nhiêu key cũng được — chỉ cần khai báo trong .env.
    Chỉ giữ những key đã được điền (bỏ qua slot rỗng).
    """
    keys = []
    for i in range(1, max_keys + 1):
        val = os.getenv(f"GEMINI_API_KEY_{i}", "")
        if val and val.strip():
            keys.append(val.strip())
    return keys


# --- Cấu hình chung ---
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
INPUT_CSV = os.getenv("INPUT_CSV", "Nhan_data.csv").strip()
INPUT_COLUMN = os.getenv("INPUT_COLUMN", "input_text").strip()
OUTPUT_COLUMN = os.getenv("OUTPUT_COLUMN", "target_reason").strip()

# Lời nhắc hệ thống (system instruction) — giữ nguyên như notebook gốc
SYSTEM_INSTRUCTION = (
    "Đóng vai một chuyên gia ngôn ngữ học. Hãy đọc câu bình luận mạng xã hội sau "
    "và giải thích tại sao nó lại là ngôn từ thù địch hoặc xúc phạm.\n"
    "Quy tắc bắt buộc:\n"
    "1. Trả lời cực kỳ ngắn gọn, tuyệt đối KHÔNG VƯỢT QUÁ 30 từ.\n"
    "2. Bắt buộc trích xuất từ khóa chửi bới / thù địch trong câu gốc và bỏ vào dấu ngoặc kép ('...').\n"
    "3. Trực diện, đi thẳng vào lý do, tuyệt đối không lặp lại câu hỏi, không thêm lời dẫn."
)
