# =============================================================================
# ViHateT5 multi-task inference server  (chạy trên Kaggle GPU + ngrok)
# -----------------------------------------------------------------------------
# Model chính : tarudesu/ViHateT5-base-HSD  (text-to-text, đa nhiệm)
# Model QA    : fine-tune từ ViHateT5-base-HSD, nạp lazy từ QA_MODEL_PATH.
#
# Mỗi request /predict_all chạy 4 task song song trên cùng câu input:
#   - HSD  : hate-speech-detection   -> CLEAN / OFFENSIVE / HATE
#   - CTSD : toxic-speech-detection  -> NONE / TOXIC
#   - HOS  : hate-spans-detection    -> các cụm thù ghét + offset
#   - QA   : hate-speech-qa          -> câu giải thích (nếu QA model đã nạp)
#
# Frontend coi mỗi task là blackbox, chỉ hiện kết quả (badge / highlight / text).
# =============================================================================

import os
import re
import time
import unicodedata

import torch
import torch.nn.functional as F
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --------------------------------------------------------------------------- #
# Cấu hình
# --------------------------------------------------------------------------- #
MODEL_NAME = "tarudesu/ViHateT5-base-HSD"

HSD_PREFIX = "hate-speech-detection"
CTSD_PREFIX = "toxic-speech-detection"
HOS_PREFIX = "hate-spans-detection"

# Chuỗi nhãn THẬT mà model sinh ra (đều CHỮ THƯỜNG — verify được từ generate output).
HSD_LABELS = {"clean": "CLEAN", "offensive": "OFFENSIVE", "hate": "HATE"}
CTSD_LABELS = {"none": "NONE", "toxic": "TOXIC"}

MAX_INPUT_LEN = 256

API_TOKEN = os.environ.get("VIHATET5_API_TOKEN", "change-me-please")
QA_MODEL_PATH = os.environ.get(
    "QA_MODEL_PATH", "/kaggle/input/hate-qa-model/hate_qa_model_final"
)
QA_PREFIX_TEMPLATE = "hate-speech-qa: Ngữ cảnh: '{text}' Câu hỏi: Tại sao câu này độc hại?"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[init] device = {DEVICE}")
print(f"[init] loading {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE).eval()
print("[init] HSD/CTSD/HOS model ready.")


# --------------------------------------------------------------------------- #
# Tiền xử lý nhẹ (theo paper §3.1 / §4.2): bỏ link + @username, GIỮ emoji,
# bỏ quote bao quanh, gộp khoảng trắng. KHÔNG normalize teencode/tách từ.
# --------------------------------------------------------------------------- #
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")


def light_preprocess(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text)).strip()
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = text.strip().strip('"').strip("'")
    text = re.sub(r"\s+", " ", text).strip()
    return text or "empty"


# --------------------------------------------------------------------------- #
# Helper sinh nhãn cho task phân loại (HSD / CTSD)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _generate_label(text: str, prefix: str, label_map: dict, default_label: str):
    prefixed = f"{prefix}: {text}"
    enc = tokenizer(prefixed, return_tensors="pt", truncation=True,
                    max_length=MAX_INPUT_LEN).to(DEVICE)
    out_ids = model.generate(**enc, max_length=16)
    raw = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
    low = raw.lower()
    for key, label in label_map.items():
        if key in low:
            return {"label": label, "raw_output": raw,
                    "prefixed_input": prefixed, "_enc": enc}
    return {"label": default_label, "raw_output": raw,
            "prefixed_input": prefixed, "_enc": enc}


# --------------------------------------------------------------------------- #
# Label-probability scoring: chấm điểm mean log-prob của từng nhãn -> softmax.
# Trả về dict {label_canonical: prob}. Dùng cho confidence bar trên UI.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _score_labels(enc, label_map: dict):
    input_ids = enc["input_ids"].to(DEVICE)
    attn = enc["attention_mask"].to(DEVICE)
    logprobs, labels = [], []
    for key, lab in label_map.items():
        dec = tokenizer(key, return_tensors="pt").to(DEVICE)
        out = model(input_ids=input_ids, attention_mask=attn, labels=dec["input_ids"])
        logprobs.append(-out.loss.item())
        labels.append(lab)
    probs = F.softmax(torch.tensor(logprobs), dim=-1).tolist()
    return {lab: float(p) for lab, p in zip(labels, probs)}


# --------------------------------------------------------------------------- #
# HOS: hate-spans-detection — model nhả lại câu với tag [hate]...[hate]
# --------------------------------------------------------------------------- #
_SPAN_RE = re.compile(r"\[hate\](.*?)\[hate\]", re.DOTALL | re.IGNORECASE)


@torch.no_grad()
def _run_hos(text: str):
    prefixed = f"{HOS_PREFIX}: {text}"
    enc = tokenizer(prefixed, return_tensors="pt", truncation=True,
                    max_length=MAX_INPUT_LEN).to(DEVICE)
    out_ids = model.generate(**enc, max_length=MAX_INPUT_LEN)
    decoded = tokenizer.decode(out_ids[0], skip_special_tokens=True)

    spans = []
    for m in _SPAN_RE.finditer(decoded):
        frag = m.group(1).strip()
        if not frag:
            continue
        pos = text.lower().find(frag.lower())
        if pos != -1:
            spans.append({"text": text[pos:pos + len(frag)],
                          "start": pos, "end": pos + len(frag)})
    return {"spans": spans, "raw_output": decoded, "prefixed_input": prefixed}


# --------------------------------------------------------------------------- #
# QA model — lazy load. Sau khi fine-tune trong vihatet5-qa.ipynb,
# trainer.save_model("./hate_qa_model_final")  -> /kaggle/working/hate_qa_model_final
# Save Version + đóng gói thành Dataset 'hate-qa-model' -> mount tự động.
# --------------------------------------------------------------------------- #
_qa = {"model": None, "tokenizer": None, "loaded": False, "error": None}


def get_qa_model():
    if _qa["loaded"]:
        return _qa["model"], _qa["tokenizer"]
    try:
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        print(f"[qa] loading from {QA_MODEL_PATH} ...")
        tok = T5Tokenizer.from_pretrained(QA_MODEL_PATH)
        mdl = T5ForConditionalGeneration.from_pretrained(QA_MODEL_PATH).to(DEVICE).eval()
        _qa.update(model=mdl, tokenizer=tok, loaded=True)
        print("[qa] ready.")
    except Exception as exc:           # noqa: BLE001
        _qa.update(loaded=True, error=repr(exc))
        print(f"[qa] failed: {exc!r}")
    return _qa["model"], _qa["tokenizer"]


@torch.no_grad()
def _run_qa(raw_text: str, max_length: int = 128):
    qmodel, qtok = get_qa_model()
    if qmodel is None:
        return {
            "available": False,
            "explanation": None,
            "message": _qa.get("error") or f"QA model chưa có ở {QA_MODEL_PATH}",
        }
    prefixed = QA_PREFIX_TEMPLATE.format(text=raw_text)
    enc = qtok(prefixed, return_tensors="pt", truncation=True,
               max_length=256).to(DEVICE)
    out_ids = qmodel.generate(**enc, max_length=max_length, do_sample=False)
    explanation = qtok.decode(out_ids[0], skip_special_tokens=True).strip()
    # Sửa đầu câu thường bị lệch ký tự đầu (gặp trong notebook QA gốc).
    if explanation and explanation[0].islower():
        explanation = explanation[0].upper() + explanation[1:]
    return {
        "available": True,
        "explanation": explanation,
        "prefixed_input": prefixed,
    }


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
app = FastAPI(title="ViHateT5 Multi-task Inference API")


class PredictAllRequest(BaseModel):
    text: str
    qa_max_length: int = 128


class QARequest(BaseModel):
    text: str
    max_length: int = 128


def _check_token(token):
    expected = os.environ.get("VIHATET5_API_TOKEN", API_TOKEN)
    if expected and token != expected:
        raise HTTPException(status_code=401, detail="Invalid API token")


@app.get("/")
def root():
    return {
        "service": "ViHateT5 Multi-task Inference API",
        "model": MODEL_NAME,
        "device": DEVICE,
        "endpoints": ["GET /health", "POST /predict_all", "POST /qa"],
        "hint": "Giao diện demo chạy ở máy local (http://127.0.0.1:8501), không phải URL này.",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "device": DEVICE,
        "qa_path": QA_MODEL_PATH,
        "qa_available": os.path.isdir(QA_MODEL_PATH),
    }


@app.post("/predict_all")
def predict_all(req: PredictAllRequest, x_api_token: str = Header(default="")):
    _check_token(x_api_token)
    t0 = time.time()

    raw = req.text
    pre = light_preprocess(raw)

    # --- HSD --- (label theo argmax label_probs để confidence luôn KHỚP badge)
    t = time.time()
    hsd = _generate_label(pre, HSD_PREFIX, HSD_LABELS, default_label="CLEAN")
    hsd["label_probs"] = _score_labels(hsd.pop("_enc"), HSD_LABELS)
    hsd["generated_label"] = hsd["label"]                        # giữ lại "model raw"
    hsd["label"] = max(hsd["label_probs"], key=hsd["label_probs"].get)
    hsd["confidence"] = float(hsd["label_probs"][hsd["label"]])
    hsd["timing_ms"] = round((time.time() - t) * 1000, 1)

    # --- CTSD ---
    t = time.time()
    ctsd = _generate_label(pre, CTSD_PREFIX, CTSD_LABELS, default_label="NONE")
    ctsd["label_probs"] = _score_labels(ctsd.pop("_enc"), CTSD_LABELS)
    ctsd["generated_label"] = ctsd["label"]
    ctsd["label"] = max(ctsd["label_probs"], key=ctsd["label_probs"].get)
    ctsd["confidence"] = float(ctsd["label_probs"][ctsd["label"]])
    ctsd["timing_ms"] = round((time.time() - t) * 1000, 1)

    # --- HOS ---
    t = time.time()
    hos = _run_hos(pre)
    hos["timing_ms"] = round((time.time() - t) * 1000, 1)

    # --- QA ---
    t = time.time()
    qa = _run_qa(raw, max_length=req.qa_max_length)
    qa["timing_ms"] = round((time.time() - t) * 1000, 1)

    return {
        "raw_input": raw,
        "preprocessed": pre,
        "hsd": hsd,
        "ctsd": ctsd,
        "hos": hos,
        "qa": qa,
        "timing_ms": {"total": round((time.time() - t0) * 1000, 1)},
    }


@app.post("/qa")
def qa(req: QARequest, x_api_token: str = Header(default="")):
    """Standalone QA endpoint (cho phép hỏi lại độc lập với /predict_all)."""
    _check_token(x_api_token)
    t0 = time.time()
    result = _run_qa(req.text, max_length=req.max_length)
    if not result.get("available"):
        raise HTTPException(status_code=503, detail=result.get("message", "QA model unavailable"))
    result["timing_ms"] = round((time.time() - t0) * 1000, 1)
    return result
