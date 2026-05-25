# src/mock.py
#
# Mock model — trả dữ liệu giả có cấu trúc giống hệt server Kaggle thật,
# để demo/show frontend khi chưa kịp host (REMOTE_API_URL còn trống) hoặc
# khi MOCK_MODE=1.

import random
import time
import unicodedata


# ----- Câu mẫu chính xác cho 3 nút "Ví dụ 1/2/3" -----
_PRESETS = {
    "bài hát hay quá, tôi thích lắm!": {
        "label_hsd": "CLEAN", "raw_hsd": "clean",
        "label_ctsd": "NONE", "raw_ctsd": "none",
        "spans_q": [],
        "qa": "Câu bình luận này là bình thường, không chứa ngôn từ thù địch hay xúc phạm.",
    },
    "mày ngu vãi lồn, chỉ biết phá": {
        "label_hsd": "OFFENSIVE", "raw_hsd": "offensive",
        "label_ctsd": "TOXIC", "raw_ctsd": "toxic",
        "spans_q": ["ngu vãi lồn"],
        "qa": "Câu này chứa ngôn từ xúc phạm trực tiếp ('ngu vãi lồn') tấn công người khác bằng từ ngữ thô tục, mang tính hạ thấp đối phương.",
    },
    "lũ bán nước đáng bị xử bắn hết đi": {
        "label_hsd": "HATE", "raw_hsd": "hate",
        "label_ctsd": "TOXIC", "raw_ctsd": "toxic",
        "spans_q": ["Lũ bán nước", "xử bắn"],
        "qa": "Câu này mang tính thù hằn cao: gán nhãn 'bán nước' cho một nhóm người và kêu gọi bạo lực ('xử bắn hết'). Đây là dấu hiệu hate speech rõ ràng.",
    },
}

_HATE_KW = ["bán nước", "xử bắn", "diệt", "tử hình", "lũ", "bọn phản",
            "đáng chết", "cút"]
_OFF_KW = ["ngu", "vãi", "lồn", "đm", "đụ", "cc", "vcl", "đéo", "thằng",
           "óc", "đần", "phế", "rác"]


def _norm(s):
    return unicodedata.normalize("NFC", str(s)).strip().lower()


def _light_preprocess(text):
    import re
    text = unicodedata.normalize("NFC", str(text)).strip()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or "empty"


def _heuristic(text):
    """Nhận diện thô bằng keyword khi input không khớp preset."""
    low = text.lower()
    found_hate = [k for k in _HATE_KW if k in low]
    found_off = [k for k in _OFF_KW if k in low]

    if found_hate:
        return {
            "label_hsd": "HATE", "raw_hsd": "hate",
            "label_ctsd": "TOXIC", "raw_ctsd": "toxic",
            "spans_q": found_hate[:2],
            "qa": "Câu này có dấu hiệu thù ghét: sử dụng cụm "
                  + " / ".join(f"'{w}'" for w in found_hate[:2])
                  + " mang tính kích động hoặc kêu gọi bạo lực.",
        }
    if found_off:
        return {
            "label_hsd": "OFFENSIVE", "raw_hsd": "offensive",
            "label_ctsd": "TOXIC", "raw_ctsd": "toxic",
            "spans_q": found_off[:2],
            "qa": "Câu này chứa ngôn từ xúc phạm: "
                  + ", ".join(f"'{w}'" for w in found_off[:2])
                  + ". Tuy không cực đoan như thù ghét nhưng vẫn mang tính hạ thấp.",
        }
    return {
        "label_hsd": "CLEAN", "raw_hsd": "clean",
        "label_ctsd": "NONE", "raw_ctsd": "none",
        "spans_q": [],
        "qa": "Câu bình luận này là bình thường, không chứa ngôn từ thù địch hay xúc phạm.",
    }


def _spans_from_text(text, hits):
    spans = []
    low = text.lower()
    for h in hits:
        pos = low.find(h.lower())
        if pos != -1:
            spans.append({"text": text[pos:pos + len(h)],
                          "start": pos, "end": pos + len(h)})
    return spans


class MockModel:
    """Cùng interface với RemoteViHateT5Model (predict_all + qa + health)."""

    is_configured = True
    base_url = "(mock)"

    def health(self):
        return {"status": "ok", "mock": True, "qa_available": True, "device": "mock"}

    def predict_all(self, text, qa_max_length=128):
        pre = _light_preprocess(text)
        key = _norm(pre)
        rule = _PRESETS.get(key) or _heuristic(pre)
        spans = _spans_from_text(pre, rule["spans_q"])

        # giả lập thời gian xử lý cho ra cảm giác "đang chạy"
        time.sleep(random.uniform(0.6, 1.1))

        # Fake label_probs nhưng nhãn được chọn vẫn áp đảo (0.78-0.95).
        def _fake_probs(labels, chosen):
            p = round(random.uniform(0.78, 0.95), 3)
            rest = round((1 - p) / (len(labels) - 1), 3)
            return {lab: (p if lab == chosen else rest) for lab in labels}

        hsd_probs = _fake_probs(["CLEAN", "OFFENSIVE", "HATE"], rule["label_hsd"])
        ctsd_probs = _fake_probs(["NONE", "TOXIC"], rule["label_ctsd"])

        return {
            "raw_input": text,
            "preprocessed": pre,
            "hsd": {
                "label": rule["label_hsd"],
                "raw_output": rule["raw_hsd"],
                "label_probs": hsd_probs,
                "confidence": hsd_probs[rule["label_hsd"]],
                "prefixed_input": f"hate-speech-detection: {pre}",
                "timing_ms": round(random.uniform(120, 220), 1),
            },
            "ctsd": {
                "label": rule["label_ctsd"],
                "raw_output": rule["raw_ctsd"],
                "label_probs": ctsd_probs,
                "confidence": ctsd_probs[rule["label_ctsd"]],
                "prefixed_input": f"toxic-speech-detection: {pre}",
                "timing_ms": round(random.uniform(120, 220), 1),
            },
            "hos": {
                "spans": spans,
                "raw_output": (
                    " ".join(f"[hate]{w}[hate]" for w in rule["spans_q"]) or pre
                ),
                "prefixed_input": f"hate-spans-detection: {pre}",
                "timing_ms": round(random.uniform(150, 280), 1),
            },
            "qa": {
                "available": True,
                "explanation": rule["qa"],
                "prefixed_input": f"hate-speech-qa: Ngữ cảnh: '{pre}' Câu hỏi: Tại sao câu này độc hại?",
                "timing_ms": round(random.uniform(220, 380), 1),
            },
            "timing_ms": {"total": round(random.uniform(700, 1100), 1)},
            "_mock": True,
        }

    def qa(self, text, max_length=128):
        time.sleep(random.uniform(0.3, 0.6))
        pre = _light_preprocess(text)
        rule = _PRESETS.get(_norm(pre)) or _heuristic(pre)
        return {
            "available": True,
            "explanation": rule["qa"],
            "prefixed_input": f"hate-speech-qa: Ngữ cảnh: '{pre}' Câu hỏi: Tại sao câu này độc hại?",
            "timing_ms": round(random.uniform(220, 380), 1),
            "_mock": True,
        }
