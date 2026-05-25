# src/remote_inference.py
#
# Client gọi ViHateT5 đang host trên Kaggle (qua ngrok). Endpoint chính là
# /predict_all (chạy 4 task: HSD + CTSD + HOS + QA cho cùng 1 câu).

import requests

from src.config import (
    REMOTE_API_URL,
    REMOTE_API_TOKEN,
    REMOTE_API_TIMEOUT,
)


class RemoteModelError(Exception):
    """Lỗi khi gọi model remote (chưa cấu hình URL, timeout, server sập...)."""


class RemoteViHateT5Model:
    def __init__(self, base_url=REMOTE_API_URL, token=REMOTE_API_TOKEN,
                 timeout=REMOTE_API_TIMEOUT):
        self.base_url = (base_url or "").rstrip("/")
        self.token = token
        self.timeout = timeout

    @property
    def is_configured(self) -> bool:
        return bool(self.base_url)

    def _headers(self):
        return {
            "Content-Type": "application/json",
            "X-API-Token": self.token,
            "ngrok-skip-browser-warning": "1",   # tránh trang cảnh báo ngrok free
        }

    def _require_url(self):
        if not self.is_configured:
            raise RemoteModelError(
                "REMOTE_API_URL chưa được đặt. Dán URL ngrok từ Kaggle vào .env."
            )

    def _post(self, path, payload):
        self._require_url()
        try:
            r = requests.post(f"{self.base_url}{path}", json=payload,
                              headers=self._headers(), timeout=self.timeout)
        except requests.Timeout as exc:
            raise RemoteModelError("Server Kaggle phản hồi quá chậm (timeout).") from exc
        except requests.RequestException as exc:
            raise RemoteModelError(f"Không kết nối được server Kaggle: {exc}") from exc
        if r.status_code == 401:
            raise RemoteModelError("Token sai (REMOTE_API_TOKEN không khớp Kaggle).")
        if r.status_code == 503:
            try:
                d = r.json().get("detail", {})
                msg = d.get("message") if isinstance(d, dict) else None
                raise RemoteModelError(msg or "Service unavailable")
            except (ValueError, AttributeError):
                raise RemoteModelError(f"Service unavailable: {r.text[:200]}")
        if r.status_code != 200:
            raise RemoteModelError(f"Server trả lỗi {r.status_code}: {r.text[:200]}")
        return r.json()

    def health(self):
        self._require_url()
        try:
            r = requests.get(f"{self.base_url}/health",
                             headers=self._headers(), timeout=10)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            raise RemoteModelError(f"Không gọi được /health: {exc}") from exc

    def predict_all(self, text, qa_max_length=128):
        return self._post("/predict_all",
                          {"text": text, "qa_max_length": qa_max_length})

    def qa(self, text, max_length=128):
        return self._post("/qa", {"text": text, "max_length": max_length})
