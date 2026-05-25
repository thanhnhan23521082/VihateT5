# Vietnamese Hate Speech Detector (ViHS)

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/Model-ViHateT5--base--HSD-059669" alt="Model">
  <img src="https://img.shields.io/badge/Kaggle-GPU%20T4-20BEFF?logo=kaggle" alt="Kaggle">
  <img src="https://img.shields.io/badge/ngrok-tunnel-1F1E37?logo=ngrok" alt="ngrok">
</p>

Đồ án môn **CS222 — Xử lý ngôn ngữ tự nhiên nâng cao** (UIT). Demo hệ thống phát hiện ngôn từ thù ghét tiếng Việt **đa nhiệm** dựa trên ViHateT5, chạy 4 task song song trên cùng 1 câu input:

| Task | Prefix | Output |
|---|---|---|
| **HSD** — Hate Speech Detection | `hate-speech-detection` | CLEAN / OFFENSIVE / HATE |
| **CTSD** — Constructive/Toxic | `toxic-speech-detection` | NONE / TOXIC |
| **HOS** — Hate Spans | `hate-spans-detection` | tô vùng `[hate]...[hate]` |
| **QA** — Giải thích lý do | `hate-speech-qa` *(fine-tune riêng)* | câu giải thích "vì sao thù ghét" |

Frontend là **SPA 3 trang**: Single Test, Batch Moderation (upload CSV → thống kê + charts), Đồ án & Datasets.

---

## Kiến trúc

```
   Browser ──► uvicorn (local, port 8501) ──HTTP──► Kaggle Notebook (GPU T4)
                  │                                     │
                  ▼                                     ▼
            src/mock.py (fallback)               ViHateT5-base-HSD (223M)
                                                 + QA fine-tune
                                                 + FastAPI + ngrok
```

- **Local FastAPI** = web UI + proxy `/api/*`, tự fallback **Mock data** khi chưa cấu hình remote.
- **Kaggle GPU T4** host model thật, expose qua **ngrok**.
- Đổi giữa Mock ↔ Real chỉ bằng `.env`.

---

## Thực nghiệm đã làm ([experiment/](experiment/))

Trong quá trình làm đồ án, nhóm đã so sánh **3 hướng tiếp cận** trước khi chốt ViHateT5 cho demo:

| Notebook | Mô hình | Tác vụ | Dataset | Ghi chú |
|---|---|---|---|---|
| [`cs222-qwen-vihsd-victsd.ipynb`](experiment/cs222-qwen-vihsd-victsd.ipynb) | **Qwen2.5-7B** (LoRA, 4-bit) | HSD + CTSD | ViHSD, ViCTSD | Decoder-only LLM, prompt + fine-tune |
| [`cs222-qwen-vihos.ipynb`](experiment/cs222-qwen-vihos.ipynb) | **Qwen2.5-7B** (LoRA, 4-bit) | HOS (span tagging) | ViHOS | Sequence labeling, đánh giá bằng `seqeval` |
| [`cs222-ura-vihsd-victsd.ipynb`](experiment/cs222-ura-vihsd-victsd.ipynb) | **URA-LLaMa** (VN-pretrained) | HSD + CTSD | ViHSD, ViCTSD | Baseline LLaMa Việt hoá |
| [`cs222-ura-vihos.ipynb`](experiment/cs222-ura-vihos.ipynb) | **URA-LLaMa** | HOS | ViHOS | — |
| [`vihatet5-qa.ipynb`](experiment/vihatet5-qa.ipynb) | **ViHateT5** fine-tune | QA giải thích | [`vihsd_qa_gemini_GT_Final.csv`](experiment/vihsd_qa_gemini_GT_Final.csv) (1000 cặp Gemini-annotated) | Sinh câu giải thích lý do thù ghét |

**Kết luận**: ViHateT5 (T5-base, 223M params, đã pretrain trên hate-speech corpus tiếng Việt) cho **F1 cạnh tranh** với Qwen-7B nhưng inference nhanh hơn ~10× và chạy ổn trên 1 GPU T4 → chọn làm backbone cho demo + fine-tune thêm task QA. Chi tiết số liệu xem [`CS222-Draft.pdf`](CS222-Draft.pdf).

---

## Cấu trúc project

```
Vietnamese-Hate-Speech-Detector/
├── app.py                         # FastAPI local: serve UI + proxy /api/*
├── src/
│   ├── config.py                  # Đọc .env (REMOTE_API_URL, TOKEN, MOCK_MODE)
│   ├── remote_inference.py        # HTTP client gọi server Kaggle
│   └── mock.py                    # Mock data fallback (3 preset + heuristic)
├── static/
│   ├── index.html                 # SPA 3 trang + sidebar nav
│   └── app.js                     # Router + 4 task block + batch + charts
├── kaggle/
│   ├── vihatet5_kaggle_server.py  # Server inference (FastAPI + ViHateT5 + ngrok)
│   ├── vihatet5_kaggle_server.ipynb
│   └── README.md                  # Hướng dẫn deploy Kaggle chi tiết
├── experiment/                    # Notebook thực nghiệm (Qwen, URA, QA fine-tune)
├── data/                          # Raw + processed datasets
├── outputs/                       # Eval figures, confusion matrices
├── 2024.findings-acl.355.pdf      # Paper ViHateT5 (ACL 2024 Findings)
├── CS222-Draft.pdf                # Báo cáo đồ án
├── .env.example
└── requirements.txt
```

---

## Chạy lại project khi clone về máy mới

### 1. Cài đặt

```powershell
git clone <repo-url>
cd Vietnamese-Hate-Speech-Detector

python -m venv venv
venv\Scripts\activate              # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt

copy .env.example .env             # macOS/Linux: cp .env.example .env
```

> `requirements.txt` chỉ có dependency cho **local app**: `fastapi`, `uvicorn`, `requests`, `python-dotenv`. Không cần `torch`/`transformers` ở máy local — phần đó chạy trên Kaggle.

### 2. Chọn 1 trong 2 cách chạy

#### Cách A — MOCK mode (không cần Kaggle, demo nhanh)

Bỏ trống `REMOTE_API_URL` trong `.env` (hoặc set `MOCK_MODE=1`):

```powershell
uvicorn app:app --port 8501
```

Mở **http://127.0.0.1:8501** → bấm "Ví dụ 1/2/3" → 4 block hiện ngay. Sidebar status dot **vàng "Mock mode"**.

Mock dùng 3 câu preset + heuristic keyword → đủ để demo UI, dev frontend, screen-record khi chưa có GPU.

#### Cách B — REAL mode (Kaggle GPU T4 + ngrok)

**B.1. Lấy 2 token (1 lần duy nhất)**
- **ngrok authtoken**: đăng ký free tại https://dashboard.ngrok.com → copy authtoken.
- **API token**: tự nghĩ 1 chuỗi bí mật (vd `vihs-2026-x7k9qm`), dùng cho cả Kaggle lẫn `.env` local.

**B.2. (Tuỳ chọn) Fine-tune QA model**

Bỏ qua nếu chỉ cần 3 task HSD/CTSD/HOS — block QA sẽ hiển thị "chưa nạp".

1. Kaggle → **New Dataset** → upload [`experiment/vihsd_qa_gemini_GT_Final.csv`](experiment/vihsd_qa_gemini_GT_Final.csv) → đặt tên `qa-ground-truth`.
2. Kaggle → **New Notebook → Import** [`experiment/vihatet5-qa.ipynb`](experiment/vihatet5-qa.ipynb) → **Accelerator = GPU T4** → Add Data `qa-ground-truth`.
3. **Run All** (~20–40 phút) → **Save Version (Commit)**.
4. Vào Output → **New Dataset from Output** → đặt tên đúng **`hate-qa-model`** (server tự nhận).

**B.3. Host server inference trên Kaggle**

1. Kaggle → **New Notebook → Import** [`kaggle/vihatet5_kaggle_server.ipynb`](kaggle/vihatet5_kaggle_server.ipynb).
2. **Settings → Accelerator = GPU T4**.
3. **Add Data → `hate-qa-model`** (nếu đã làm B.2).
4. Mở **Cell 2 (CẤU HÌNH)**, dán 2 token:
   ```python
   NGROK_AUTHTOKEN = "DAN_NGROK_TOKEN_CUA_BAN"
   os.environ["VIHATET5_API_TOKEN"] = "vihs-2026-x7k9qm"   # phải khớp .env local
   ```
5. **Run All** → Cell 4 in `>>> PUBLIC URL: https://xxx.ngrok-free.dev` rồi quay vô hạn — đó là server đang phục vụ, **đừng tắt**.

**B.4. Trỏ local sang server Kaggle**

Mở `.env`:
```env
REMOTE_API_URL=https://xxx.ngrok-free.dev
REMOTE_API_TOKEN=vihs-2026-x7k9qm
REMOTE_API_TIMEOUT=60
MOCK_MODE=
```
Restart local:
```powershell
uvicorn app:app --port 8501
```
Sidebar status dot **xanh "Remote OK"** = đã kết nối.

> URL ngrok đổi mỗi lần chạy lại Cell 4 → cập nhật `.env` rồi restart local.

---

## API local (port 8501)

| Method | Path | Mô tả |
|---|---|---|
| GET | `/` | Web UI |
| GET | `/api/health` | Trạng thái remote/mock |
| POST | `/api/predict` | 4 task song song cho 1 câu |
| POST | `/api/qa` | Standalone QA cho 1 câu |

**POST `/api/predict`** body:
```json
{ "text": "mày ngu vãi lồn" }
```
Trả về:
```jsonc
{
  "hsd":  { "label": "OFFENSIVE", "confidence": 0.89, "label_probs": {...}, "raw_output": "offensive" },
  "ctsd": { "label": "TOXIC",     "confidence": 0.91, "label_probs": {...}, "raw_output": "toxic" },
  "hos":  { "spans": [{"text":"ngu vãi lồn","start":5,"end":16}], "raw_output": "..." },
  "qa":   { "available": true, "explanation": "Câu này chứa..." },
  "timing_ms": { "total": 820.5 }
}
```

---

## Cơ chế confidence score

ViHateT5 là model **sinh chữ** (text-to-text), không có classifier head. Để có confidence cho 3 nhãn, dùng **label-conditional scoring**:

1. Với mỗi nhãn ứng viên (`clean`, `offensive`, `hate`) → forward pass với `labels=` là token IDs của nhãn đó.
2. Lấy `-loss` = mean log P(nhãn | input).
3. Softmax 3 số → phân phối xác suất trên 3 nhãn.

Không đổi kiến trúc, không train thêm — chỉ "hỏi khôn" model. Chi tiết xem [`kaggle/vihatet5_kaggle_server.py`](kaggle/vihatet5_kaggle_server.py#L97-L108).

---

## Troubleshooting

| Triệu chứng | Nguyên nhân | Cách sửa |
|---|---|---|
| Status đỏ "Remote offline" | `.env` thiếu URL hoặc Cell 4 Kaggle đã dừng | Kiểm tra `.env`, chạy lại Cell 4 |
| 401 Unauthorized | Token Kaggle ≠ Token `.env` | Khớp `VIHATET5_API_TOKEN` ↔ `REMOTE_API_TOKEN` |
| Block QA "chưa nạp" | Chưa Add Data `hate-qa-model` | Xem bước B.2–B.3 |
| Mở URL ngrok ra 404 | Bình thường — server không có route `/` | Test `<URL>/health` thay vì `/` |
| `.env` mới sửa không có hiệu lực | uvicorn cache lúc start | Ctrl+C + restart |
| Trình duyệt không thấy thay đổi | Browser cache | Ctrl+F5 (hard refresh) |
| CSV upload garbled chữ Việt | File save sai encoding | Re-save UTF-8 (Excel: "CSV UTF-8") |

---

## Tham khảo

- **Paper backbone**: [ViHateT5 — ACL 2024 Findings](https://aclanthology.org/2024.findings-acl.355.pdf) ([PDF local](2024.findings-acl.355.pdf))
- **Model HF**: [tarudesu/ViHateT5-base-HSD](https://huggingface.co/tarudesu/ViHateT5-base-HSD)
- **Datasets**:
  - [uitnlp/vihsd](https://huggingface.co/datasets/uitnlp/vihsd) — Vietnamese Hate Speech Detection
  - [tarudesu/ViCTSD](https://huggingface.co/datasets/tarudesu/ViCTSD) — Constructive/Toxic Speech Detection
  - [phusroyal/ViHOS](https://github.com/phusroyal/ViHOS) — Hate Spans
- **Báo cáo đồ án**: [`CS222-Draft.pdf`](CS222-Draft.pdf)

## License

MIT — môn học CS222, UIT.
