"""
gemini_client.py
----------------
Client Gemini có cơ chế XOAY VÒNG nhiều API key, tối ưu cho free tier.

Phân loại lỗi:
- "dead"   : 403 / PERMISSION_DENIED / key sai  -> bỏ HẲN key này.
- "rate"   : 429 / quota / rate limit           -> key NGHỈ tạm theo retryDelay,
             sau đó tự được dùng lại (free tier giới hạn ~5 request/phút).
- "server" : 503 / 500 / overloaded / timeout   -> lỗi server, thử lại key đó.

Quy tắc dừng:
- Chỉ ném AllKeysExhausted khi TẤT CẢ key đều "dead".
- Nếu tất cả key đang "nghỉ" (rate) -> CHỜ đến khi key sớm nhất hồi lại rồi chạy tiếp
  (KHÔNG crash chương trình).
"""

import re
import time
from google import genai
from google.genai import types

from config import SYSTEM_INSTRUCTION, MODEL


class AllKeysExhausted(Exception):
    """Ném ra khi TẤT CẢ API key đều đã bị chặn vĩnh viễn (dead)."""
    pass


def _classify_error(err: Exception) -> str:
    """Phân loại lỗi -> 'dead' | 'rate' | 'server' | 'other'."""
    msg = str(err).lower()
    if (
        "403" in msg
        or "permission_denied" in msg
        or "denied access" in msg
        or "api key not valid" in msg
        or "api_key_invalid" in msg
    ):
        return "dead"
    if (
        "429" in msg
        or "resource_exhausted" in msg
        or "quota" in msg
        or "rate limit" in msg
        or "exceeded" in msg
    ):
        return "rate"
    if (
        "503" in msg
        or "unavailable" in msg
        or "overloaded" in msg
        or "500" in msg
        or "internal" in msg
        or "timeout" in msg
        or "deadline" in msg
    ):
        return "server"
    return "other"


def _parse_retry_delay(err: Exception, default=60.0) -> float:
    """Đọc thời gian cần nghỉ từ thông báo lỗi 429 (giây)."""
    msg = str(err).lower()
    m = re.search(r"retry in ([\d.]+)s", msg)
    if m:
        return float(m.group(1)) + 2
    m = re.search(r"retrydelay['\"]?\s*[:=]\s*['\"]?(\d+)s", msg)
    if m:
        return float(m.group(1)) + 2
    return default


class RotatingGeminiClient:
    def __init__(self, api_keys, model=MODEL, max_server_retries=6):
        if not api_keys:
            raise ValueError(
                "Không tìm thấy API key nào. Hãy điền key vào file .env "
                "(GEMINI_API_KEY_1, _2, ...)."
            )
        self.api_keys = api_keys
        self.model = model
        self.max_server_retries = max_server_retries
        self.n = len(api_keys)

        self.dead = [False] * self.n            # 403 -> bỏ hẳn
        self.cooldown_until = [0.0] * self.n    # 429 -> nghỉ đến mốc thời gian này
        self.current_index = 0

        self.config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.0,
        )
        self._clients = {}  # cache client theo index để khỏi tạo lại
        print(f"🔑 Tổng cộng {self.n} API key.")

    def _client_for(self, idx):
        if idx not in self._clients:
            self._clients[idx] = genai.Client(api_key=self.api_keys[idx])
        return self._clients[idx]

    def _acquire_key_index(self) -> int:
        """
        Trả về index của một key sẵn sàng dùng (không dead, hết thời gian nghỉ).
        - Nếu mọi key đang nghỉ -> chờ đến khi key sớm nhất hồi lại.
        - Nếu mọi key đều dead -> ném AllKeysExhausted.
        """
        while True:
            now = time.time()
            # Tìm key sẵn sàng, ưu tiên xoay vòng bắt đầu từ current_index
            for offset in range(self.n):
                idx = (self.current_index + offset) % self.n
                if not self.dead[idx] and self.cooldown_until[idx] <= now:
                    self.current_index = idx
                    return idx

            # Không có key nào sẵn sàng -> còn key nào chưa chết không?
            alive = [i for i in range(self.n) if not self.dead[i]]
            if not alive:
                raise AllKeysExhausted(
                    f"Cả {self.n} API key đều bị chặn (403). "
                    "Cần thay key từ tài khoản/project khác."
                )

            # Tất cả key sống đang nghỉ -> chờ đến key hồi sớm nhất
            wait = min(self.cooldown_until[i] for i in alive) - now
            wait = max(1.0, min(wait, 65.0))
            print(f"😴 Tất cả key đang nghỉ (hết hạn mức/phút). Chờ {wait:.0f}s rồi chạy tiếp...")
            time.sleep(wait)

    def generate(self, prompt: str) -> str:
        """Sinh câu trả lời cho 1 prompt, tự xoay/đợi key khi cần."""
        server_retries = 0
        while True:
            idx = self._acquire_key_index()
            try:
                response = self._client_for(idx).models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=self.config,
                )
                return (response.text or "").strip()

            except Exception as err:
                kind = _classify_error(err)

                if kind == "dead":
                    self.dead[idx] = True
                    print(f"⛔ Key #{idx + 1} bị chặn (403) -> bỏ hẳn key này.")
                    continue  # lấy key khác

                if kind == "rate":
                    delay = _parse_retry_delay(err)
                    self.cooldown_until[idx] = time.time() + delay
                    print(f"⏸️  Key #{idx + 1} hết hạn mức -> nghỉ {delay:.0f}s, thử key khác.")
                    continue  # lấy key khác (hoặc đợi nếu hết key)

                if kind == "server":
                    server_retries += 1
                    if server_retries > self.max_server_retries:
                        # Server lỗi dai dẳng -> cho key này nghỉ ngắn rồi đổi key
                        self.cooldown_until[idx] = time.time() + 30
                        server_retries = 0
                        print(f"🔁 Key #{idx + 1}: server lỗi nhiều lần -> nghỉ 30s, đổi key.")
                        continue
                    wait = min(2 ** server_retries, 30)
                    print(f"⏳ Server quá tải (503), thử lại sau {wait}s "
                          f"(lần {server_retries}/{self.max_server_retries})...")
                    time.sleep(wait)
                    continue  # thử lại cùng key

                # Lỗi lạ khác -> ném ra để dừng và xem xét
                raise
