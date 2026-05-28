# Vietnamese Hate Speech Detector (ViHS)

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/Model-ViHateT5--base--HSD-059669" alt="Model">
  <img src="https://img.shields.io/badge/Kaggle-GPU%20T4-20BEFF?logo=kaggle" alt="Kaggle">
  <img src="https://img.shields.io/badge/ngrok-tunnel-1F1E37?logo=ngrok" alt="ngrok">
</p>

Đồ án môn **CS222 — Xử lý ngôn ngữ tự nhiên nâng cao** (UIT). Hệ thống demo phát hiện ngôn từ thù ghét tiếng Việt **đa nhiệm** dựa trên **ViHateT5** (Nguyen, ACL 2024 Findings), kèm theo các thực nghiệm so sánh với các LLM quy mô lớn (Qwen2.5-7B, URA-LLaMa) trên 3 dataset benchmark tiếng Việt.

---

## Mục lục

1. [Giới thiệu](#1-giới-thiệu)
2. [Demo: 4 task song song](#2-demo-4-task-song-song)
3. [Quy trình nhóm đã thực hiện](#3-quy-trình-nhóm-đã-thực-hiện)
4. [Thực nghiệm (folder `experiment/`)](#4-thực-nghiệm-folder-experiment)
5. [Kiến trúc hệ thống](#5-kiến-trúc-hệ-thống)
6. [Cấu trúc project](#6-cấu-trúc-project)
7. [Cài đặt local](#7-cài-đặt-local)
8. [Cách A — Chạy MOCK (không cần Kaggle)](#8-cách-a--chạy-mock-không-cần-kaggle)
9. [Cách B — Chạy REAL trên Kaggle GPU + ngrok](#9-cách-b--chạy-real-trên-kaggle-gpu--ngrok)
10. [Xoay khoá (rotate keys)](#10-xoay-khoá-rotate-keys)
11. [API reference](#11-api-reference)
12. [Cơ chế confidence score](#12-cơ-chế-confidence-score)
13. [Troubleshooting](#13-troubleshooting)
14. [Tham khảo](#14-tham-khảo)

---

## 1. Giới thiệu

### Bài toán
**Hate Speech Detection (HSD) tiếng Việt** trên mạng xã hội — phát hiện và phân loại các bình luận chứa ngôn từ thù ghét, xúc phạm, hoặc kêu gọi bạo lực. Tiếng Việt là **ngôn ngữ ít tài nguyên** (low-resource), thiếu corpus chất lượng cao và mô hình pre-trained chuyên biệt cho domain hate-speech.

### Backbone: ViHateT5
Sử dụng [`tarudesu/ViHateT5-base-HSD`](https://huggingface.co/tarudesu/ViHateT5-base-HSD) — T5-base **223M params**, pre-train trên **VOZ-HSD** (~10M comments crawl từ voz.vn, gán nhãn bằng ViSoBERT classifier), sau đó fine-tune đa nhiệm trên 3 dataset benchmark. Mô hình **text-to-text** xử lý 3 task chỉ bằng cách đổi **prefix** ở input.

Theo paper ([`2024.findings-acl.355.pdf`](2024.findings-acl.355.pdf)), ViHateT5 đạt **state-of-the-art** trên cả ViHSD, ViCTSD, ViHOS — vượt các baseline BERT-based như PhoBERT, ViSoBERT, mBERT.

### Mở rộng của nhóm
1. **Task QA mới**: fine-tune thêm 1 prefix `hate-speech-qa` để model **sinh câu giải thích** lý do vì sao câu đó là hate speech — không có sẵn trong paper gốc.
2. **Demo web tương tác**: SPA 3 trang (Single Test, Batch Moderation, Đồ án & Datasets) với confidence bar, span highlight, batch CSV processing.
3. **So sánh thực nghiệm**: chạy thêm Qwen2.5-7B và URA-LLaMa trên cùng 3 dataset để định lượng trade-off **chất lượng ↔ inference cost**.

---

## 2. Demo: 4 task song song

Trên cùng 1 câu input, hệ thống chạy đồng thời:

| Task | Prefix | Output | Mô tả |
|---|---|---|---|
| **HSD** — Hate Speech Detection | `hate-speech-detection` | CLEAN / OFFENSIVE / HATE | Phân loại 3 cấp độ |
| **CTSD** — Constructive/Toxic | `toxic-speech-detection` | NONE / TOXIC | Phát hiện độc hại 2 cấp |
| **HOS** — Hate Spans | `hate-spans-detection` | text có tag `[hate]...[hate]` | Trích xuất cụm từ thù ghét |
| **QA** — Giải thích (fine-tune riêng) | `hate-speech-qa` | câu giải thích | "Tại sao câu này độc hại?" |

Mỗi task hiển thị: **nhãn**, **confidence** (3 thanh xác suất cho 3 nhãn ứng viên), **prefixed input** (debug), **raw output** (chuỗi gốc model sinh), **timing**.

---

## 3. Quy trình nhóm đã thực hiện

Tóm tắt các bước nhóm đã làm, theo thứ tự thời gian:

### Bước 1 — Nghiên cứu paper ViHateT5
- Đọc paper ACL 2024 Findings, hiểu kiến trúc T5 text-to-text, cơ chế prefix multitask, dataset VOZ-HSD và pre-training pipeline.
- Test model trên HuggingFace Inference API với câu mẫu → xác nhận chất lượng tốt với cả 3 task chuẩn (HSD/CTSD/HOS).

### Bước 2 — Thực nghiệm so sánh với các LLM khác
- Chạy **Qwen2.5-7B** (LoRA 4-bit) và **URA-LLaMa** trên 3 dataset ViHSD, ViCTSD, ViHOS để có baseline so sánh.
- Đánh giá bằng `seqeval` (span tagging) và macro-F1 (classification).
- → Kết luận: ViHateT5 cho F1 ngang ngửa Qwen-7B nhưng nhỏ hơn ~30×, inference ~10× nhanh hơn → chọn làm backbone cho demo.

### Bước 3 — Sinh dataset QA bằng Gemini
- Mỗi câu HATE/OFFENSIVE trong ViHSD chưa có annotation "vì sao thù ghét".
- Xây tool riêng [`tool_gemini/`](tool_gemini/) với cơ chế **xoay vòng nhiều API key** (free tier Gemini ~5 req/phút, cần nhiều key chạy song song):
  - Đọc CSV `Nhan_data.csv` (cột `input_text`).
  - Gọi `gemini-2.5-flash` với system instruction giới hạn ≤30 từ, ép bỏ keyword chửi bới vào `'...'`.
  - Tự xoay key khi gặp 429 (quota), bỏ hẳn key 403 (invalid), retry 503 (server overload).
  - Ghi từng dòng ngay → `Ctrl+C` an toàn, chạy lại tiếp từ dòng dở.
- Output ~1000 cặp `(câu thù ghét, câu giải thích)` → review thủ công lọc Gemini sai/lan man → file [`vihsd_qa_gemini_GT_Final.csv`](experiment/vihsd_qa_gemini_GT_Final.csv).

### Bước 4 — Fine-tune QA head trên ViHateT5
- Notebook [`vihatet5-qa.ipynb`](experiment/vihatet5-qa.ipynb): tiếp tục fine-tune `ViHateT5-base-HSD` với prefix mới `hate-speech-qa: ...`.
- Train 5 epoch trên Kaggle GPU T4 (~30 phút) → checkpoint `hate_qa_model_final/`.
- Đóng gói thành Kaggle Dataset tên **`hate-qa-model`** để server tự mount.

### Bước 5 — Hosting trên Kaggle + ngrok
- Vì máy local không đủ GPU để chạy T5 inference nhanh, chuyển sang **host model trên Kaggle GPU T4 free**.
- Notebook server [`kaggle/vihatet5_kaggle_server.ipynb`](kaggle/vihatet5_kaggle_server.ipynb) khởi tạo FastAPI + ViHateT5 + QA model, expose qua **ngrok** (URL public, free tier).
- Máy local chạy uvicorn proxy `/api/*` → ngrok URL.

### Bước 6 — Xây frontend SPA
- 3 trang: **Single Test** (1 câu, 4 block kết quả), **Batch Moderation** (upload CSV, kiểm duyệt hàng loạt với charts), **Đồ án & Datasets** (giới thiệu nhóm + 4 dataset).
- Hash router, vanilla JS + Chart.js, không build pipeline.
- Mock mode fallback ([`src/mock.py`](src/mock.py)) để demo UI khi chưa host Kaggle.

### Bước 7 — Polish
- Multi-encoding CSV parser (UTF-8, Windows-1258, Windows-1252) để upload không lỗi tiếng Việt.
- Sửa label-confidence consistency: lấy `argmax(label_probs)` làm nhãn chính thay vì `generate()` output → badge luôn khớp thanh confidence cao nhất.
- Confidence scoring bằng **label-conditional likelihood** (xem [§12](#12-cơ-chế-confidence-score)).
- Fix QA output mất ký tự đầu câu ở frontend (`fixQAOutput`).

---

## 4. Thực nghiệm (folder [`experiment/`](experiment/))

| Notebook | Mô hình | Tác vụ | Dataset | Setup |
|---|---|---|---|---|
| [`cs222-qwen-vihsd-victsd.ipynb`](experiment/cs222-qwen-vihsd-victsd.ipynb) | **Qwen2.5-7B-Instruct** | HSD + CTSD (classification) | ViHSD + ViCTSD | LoRA r=16, 4-bit (bitsandbytes), 3 epoch |
| [`cs222-qwen-vihos.ipynb`](experiment/cs222-qwen-vihos.ipynb) | **Qwen2.5-7B-Instruct** | HOS (span tagging, BIO) | ViHOS | LoRA r=16, 4-bit, đánh giá bằng `seqeval` (precision/recall/F1) |
| [`cs222-ura-vihsd-victsd.ipynb`](experiment/cs222-ura-vihsd-victsd.ipynb) | **URA-LLaMa** (LLaMa pretrained tiếng Việt) | HSD + CTSD | ViHSD + ViCTSD | Fine-tune full, baseline để so sánh |
| [`cs222-ura-vihos.ipynb`](experiment/cs222-ura-vihos.ipynb) | **URA-LLaMa** | HOS | ViHOS | Fine-tune full, sequence labeling |
| [`vihatet5-qa.ipynb`](experiment/vihatet5-qa.ipynb) | **ViHateT5-base-HSD** + LoRA | QA giải thích | [`vihsd_qa_gemini_GT_Final.csv`](experiment/vihsd_qa_gemini_GT_Final.csv) (1000 cặp Gemini annotated) | Fine-tune 5 epoch trên T4, lr=1e-4 |

### Tóm tắt kết quả

- **HSD/CTSD**: ViHateT5 (223M) ≈ Qwen2.5-7B (7B) về macro-F1, vượt URA-LLaMa rõ rệt. Ưu thế của ViHateT5: nhỏ + nhanh + có pretrained chuyên domain hate-speech.
- **HOS (span)**: Qwen-7B + LoRA cho F1 cao nhất trên ViHOS nhưng inference chậm ~10× và cần ≥16GB VRAM. ViHateT5 đạt 80–85% F1 với tốc độ thực dụng cho demo realtime.
- **QA**: ViHateT5 fine-tune cho output mạch lạc, đôi khi bị **repetition degeneration** (lặp cụm từ) do greedy decoding — đặc tính của T5-base, không phải bug code.

Chi tiết số liệu + plot trong [`CS222-Draft.pdf`](CS222-Draft.pdf) và [`outputs/figures/`](outputs/figures/).

---

## 5. Kiến trúc hệ thống

```
   ┌──────────────────┐
   │   Browser (SPA)  │  static/index.html + app.js
   └────────┬─────────┘
            │  fetch /api/*
            ▼
   ┌──────────────────┐
   │  uvicorn local   │  app.py  (FastAPI, port 8501)
   │  + proxy /api/*  │
   └────────┬─────────┘
            │
       ┌────┴────┐
       │ MOCK?   │  src/config.py đọc .env
       └────┬────┘
            │
   ┌────────┴─────────────────────────────┐
   │ no                                    │ yes / chưa có URL
   ▼                                       ▼
HTTP POST                            src/mock.py
X-API-Token + ngrok-skip header      (3 preset + heuristic)
   │
   ▼
┌────────────────────────────────────┐
│  Kaggle Notebook  (GPU T4 free)    │
│                                    │
│   FastAPI server                   │
│    + ViHateT5-base-HSD             │
│    + hate_qa_model_final (QA)      │
│    + pyngrok tunnel                │
│                                    │
│  URL: https://xxx.ngrok-free.dev   │
└────────────────────────────────────┘
```

- **Local** = UI + proxy + (tuỳ chọn) fallback mock.
- **Kaggle** = inference thật.
- Mọi config đổi qua **`.env`**, không hardcode.

---

## 6. Cấu trúc project

```
Vietnamese-Hate-Speech-Detector/
├── app.py                          # FastAPI local: UI + proxy /api/*
├── requirements.txt                # Chỉ cần fastapi/uvicorn/requests/dotenv ở local
├── .env.example                    # Mẫu cấu hình
│
├── src/
│   ├── config.py                   # Đọc .env (URL, token, mock flag)
│   ├── remote_inference.py         # HTTP client gọi server Kaggle
│   └── mock.py                     # MockModel fallback
│
├── static/
│   ├── index.html                  # SPA 3 trang
│   └── app.js                      # Router + render + batch + Chart.js
│
├── kaggle/
│   ├── vihatet5_kaggle_server.py   # FastAPI + ViHateT5 + ngrok (mã nguồn)
│   ├── vihatet5_kaggle_server.ipynb# Notebook để upload Kaggle
│   └── README.md                   # Hướng dẫn host Kaggle chi tiết
│
├── experiment/                     # 5 notebook thực nghiệm (xem §4)
│   ├── cs222-qwen-vihsd-victsd.ipynb
│   ├── cs222-qwen-vihos.ipynb
│   ├── cs222-ura-vihsd-victsd.ipynb
│   ├── cs222-ura-vihos.ipynb
│   ├── vihatet5-qa.ipynb
│   └── vihsd_qa_gemini_GT_Final.csv
│
├── tool_gemini/                    # Tool sinh QA ground-truth bằng Gemini (xem §3 Bước 3 + §10.4)
│   ├── config.py                   # Đọc .env + system prompt
│   ├── gemini_client.py            # RotatingGeminiClient — xoay key 429/403/503
│   ├── generate_gt.py              # Script chính: CSV → Gemini → CSV
│   ├── Generate_GT.ipynb           # Bản notebook tương đương
│   ├── Nhan_data.csv               # Input data
│   └── README.md                   # Hướng dẫn chi tiết tool
│
├── data/
│   ├── raw/                        # Dataset gốc tải về
│   └── processed/                  # Sau khi clean + split
│
├── outputs/
│   ├── figures/                    # Confusion matrix, F1 plots
│   └── results_training/           # Training logs
│
├── 2024.findings-acl.355.pdf       # Paper ViHateT5
└── CS222-Draft.pdf                 # Báo cáo đồ án
```

---

## 7. Cài đặt local

### Yêu cầu
- **Python 3.10+**
- Trình duyệt hiện đại (Chrome/Edge/Firefox)
- **Không cần GPU** ở máy local — phần inference chạy trên Kaggle.

### Bước cài đặt

```powershell
# 1. Clone
git clone <repo-url>
cd Vietnamese-Hate-Speech-Detector

# 2. Tạo venv (Windows PowerShell)
python -m venv venv
venv\Scripts\Activate.ps1
# macOS/Linux: python -m venv venv && source venv/bin/activate

# 3. Cài deps (chỉ ~5 package, nhỏ)
pip install -r requirements.txt

# 4. Copy file env mẫu
Copy-Item .env.example .env
# macOS/Linux: cp .env.example .env
```

`requirements.txt` chỉ chứa: `fastapi`, `uvicorn`, `pydantic`, `requests`, `python-dotenv`. **Không cài** `torch`/`transformers` ở local.

---

## 8. Cách A — Chạy MOCK (không cần Kaggle)

Dùng khi: demo nhanh, dev frontend, máy không có Internet ra Kaggle, screen-record.

### B.1 Cấu hình
Để **trống** `REMOTE_API_URL` trong `.env`, hoặc set `MOCK_MODE=1`:
```env
REMOTE_API_URL=
MOCK_MODE=1
```

### B.2 Chạy
```powershell
uvicorn app:app --port 8501
# hoặc: python app.py
```

Mở **http://127.0.0.1:8501** → sidebar status dot **vàng "Mock mode"** → bấm "Ví dụ 1/2/3" để test.

Mock dùng:
- **3 preset** cứng (clean / offensive / hate) khớp 100% với câu Ví dụ.
- **Heuristic keyword** (vd `bán nước`, `xử bắn` → HATE; `ngu`, `vãi`, `lồn` → OFFENSIVE) cho câu tự nhập.
- Fake confidence ~0.78–0.95 luôn khớp với nhãn chosen.

---

## 9. Cách B — Chạy REAL trên Kaggle GPU + ngrok

Dùng khi: demo thật, test chất lượng model thật.

### B.0 Chuẩn bị 1 lần — lấy 2 token

**B.0.1 ngrok authtoken** (free, không cần thẻ):
1. Đăng ký https://dashboard.ngrok.com
2. Sidebar → **Your Authtoken** → copy chuỗi 40+ ký tự.

**B.0.2 API token** (tự nghĩ):
- Nghĩ 1 chuỗi bí mật, vd `vihs-2026-x7k9qm` để làm mật khẩu kết nối.
- Sẽ dùng ở **2 chỗ**: Kaggle Cell 2 (`VIHATET5_API_TOKEN`) và file `.env` local (`REMOTE_API_TOKEN`). **Phải khớp** — nếu không sẽ 401.

### B.1 (Tuỳ chọn) Fine-tune QA model

Bỏ qua nếu chỉ cần 3 task HSD/CTSD/HOS — block QA sẽ hiển thị "QA model chưa nạp".

1. Kaggle → **Datasets → New Dataset** → upload [`experiment/vihsd_qa_gemini_GT_Final.csv`](experiment/vihsd_qa_gemini_GT_Final.csv) → đặt tên **`qa-ground-truth`**.
2. Kaggle → **Notebooks → New → Import** → upload [`experiment/vihatet5-qa.ipynb`](experiment/vihatet5-qa.ipynb).
3. **Settings → Accelerator = GPU T4 x1**.
4. **Add Data → qa-ground-truth**.
5. Sửa Cell 14 đường dẫn:
   ```python
   file_path = '/kaggle/input/qa-ground-truth/vihsd_qa_gemini_GT_Final.csv'
   ```
6. **Run All** (~30 phút trên T4) → kết thúc có folder `/kaggle/working/hate_qa_model_final/`.
7. **Save Version (Commit)** → đợi commit xong.
8. Vào tab **Output** của version → **New Dataset from Output** → đặt tên đúng **`hate-qa-model`** (server hard-code tên này).

### B.2 Host server inference

1. Kaggle → **Notebooks → New → Import** → upload [`kaggle/vihatet5_kaggle_server.ipynb`](kaggle/vihatet5_kaggle_server.ipynb).
2. **Settings → Accelerator = GPU T4 x1**, **Internet = ON**.
3. **Add Data → `hate-qa-model`** (nếu đã fine-tune QA ở B.1).
4. Mở **Cell 2 (CẤU HÌNH)** dán 2 token:
   ```python
   NGROK_AUTHTOKEN = "2abc...xyz"                          # token ngrok B.0.1
   os.environ["VIHATET5_API_TOKEN"] = "vihs-2026-x7k9qm"   # token B.0.2
   ```
5. **Run All** từ Cell 1 → Cell 4.
   - Cell 1: check GPU (`True | Tesla T4`).
   - Cell 3: tải `ViHateT5-base-HSD` (~1–2 phút).
   - Cell 4: in
     ```
     >>> PUBLIC URL: https://abcd-1234.ngrok-free.dev
     ```
     Rồi cell **quay liên tục** — đó là server đang phục vụ, **không tắt notebook** hoặc save version nếu sợ tắt nhầm hoặc muốn chạy ngầm.

### B.3 Test server từ trình duyệt
Mở `<PUBLIC_URL>/health`. Phải thấy JSON kiểu:
```json
{"status": "ok", "model": "tarudesu/ViHateT5-base-HSD", "device": "cuda",
 "qa_path": "/kaggle/input/hate-qa-model/hate_qa_model_final",
 "qa_available": true}
```
Nếu `qa_available: false` → chưa add `hate-qa-model` Dataset (3 task khác vẫn chạy).

### B.4 Trỏ local sang Kaggle
Mở `.env`:
```env
REMOTE_API_URL=https://abcd-1234.ngrok-free.dev
REMOTE_API_TOKEN=vihs-2026-x7k9qm
REMOTE_API_TIMEOUT=60
MOCK_MODE=
```
Restart local app:
```powershell
# Ctrl+C nếu uvicorn đang chạy, rồi chạy lại
uvicorn app:app --port 8501
```
Status dot sidebar đổi **xanh "Remote OK"** → đã kết nối.

> ⚠ URL ngrok đổi **mỗi lần Run All Cell 4**. Mỗi phiên Kaggle mới phải update `.env` rồi restart uvicorn.

### B.5 (Tuỳ chọn) Save Version để dùng cho bảo vệ
Khi server đã chạy ổn:
- Notebook server → **Save Version → Save & Run All (Commit)**.
- Bản commit chạy tới giới hạn phiên (~9–12h GPU). URL ngrok in trong **log của version đó**.
- Trước bảo vệ: chạy version, copy URL mới, update `.env`, restart local.

---

## 10. Xoay khoá (key rotation) trong các trường hợp: 

### 10.1 Khi nào cần xoay
- Nghi token bị lộ (commit nhầm vào git, share screen, push lên repo public).
- Sau bảo vệ → muốn đóng hẳn endpoint, không cho ai gọi lại.
- ngrok authtoken bị flag (vd dùng quá rate limit free).

### 10.2 Xoay `VIHATET5_API_TOKEN` (token API server Kaggle)
1. Nghĩ token mới, vd `vihs-2027-newSecret`.
2. **Kaggle**: mở Cell 2 → sửa `os.environ["VIHATET5_API_TOKEN"] = "vihs-2027-newSecret"` → **Restart Kernel & Run All** (token cũ chỉ tồn tại trong process — restart là chết).
3. **Local**: mở `.env` → sửa `REMOTE_API_TOKEN=vihs-2027-newSecret` → **Ctrl+C + restart `uvicorn`** (dotenv chỉ load lúc start).
4. Verify: gọi `/api/predict` từ web. Nếu 401 → 2 nơi chưa khớp.

> ⚠ Đừng commit `.env` vào git. `.gitignore` đã loại sẵn.

### 10.3 Xoay ngrok authtoken
1. https://dashboard.ngrok.com → **Your Authtoken** → **Regenerate**.
2. Copy token mới, dán vào Cell 2 (`NGROK_AUTHTOKEN = "..."`).
3. Restart Kernel & Run All notebook server → URL mới sẽ in ra.
4. Update `REMOTE_API_URL` trong `.env` local + restart uvicorn.

> Token cũ tự động vô hiệu sau khi regenerate. Tunnel đang mở với token cũ sẽ rớt.

### 10.4 Xoay Gemini API key (cho [`tool_gemini/`](tool_gemini/))

Tool sinh QA ground-truth dùng **Gemini free tier** với cơ chế xoay vòng nhiều key trong [`tool_gemini/gemini_client.py`](tool_gemini/gemini_client.py). Free tier giới hạn **~5 request/phút/key**, nên cần nhiều key chạy song song.

**10.4.1 Lấy key mới**
1. Vào https://aistudio.google.com/apikey (đăng nhập Google account khác nhau nếu muốn nhiều key).
2. **Create API key** → chọn project (hoặc tạo project mới) → copy chuỗi `AIza...`.
3. Lặp lại với account khác để có nhiều key (mỗi account = 1 quota riêng).

**10.4.2 Cấu hình `.env`**
File [`tool_gemini/.env`](tool_gemini/) khai báo theo slot:
```env
GEMINI_API_KEY_1=AIzaSy...key_thu_nhat
GEMINI_API_KEY_2=AIzaSy...key_thu_hai
GEMINI_API_KEY_3=AIzaSy...key_thu_ba
# ... thêm bao nhiêu cũng được, slot rỗng tự bỏ qua

GEMINI_MODEL=gemini-2.5-flash
INPUT_CSV=Nhan_data.csv
INPUT_COLUMN=input_text
OUTPUT_COLUMN=target_reason
```

`config.get_api_keys()` quét đến slot 50, chỉ giữ slot có giá trị → muốn xoá key cũ thì để trống slot đó.

**10.4.3 Cơ chế xoay vòng (tự động)**
- **429 / quota / rate limit** → key đi "nghỉ" theo `retryDelay` từ error message, code tự chuyển sang key kế tiếp.
- **403 / PERMISSION_DENIED / API_KEY_INVALID** → key bị đánh dấu `dead`, không dùng lại trong session.
- **503 / overloaded / timeout** → retry cùng key với exponential backoff (2s → 4s → 8s → ... → 30s), sau 6 lần thất bại cho key nghỉ 30s và đổi.
- **Tất cả key đều `dead`** → ném `AllKeysExhausted` → script dừng, lưu tiến trình.

**10.4.4 Xoay key thủ công**
- **Khi nào**: 1 key bị Google block vĩnh viễn, hoặc muốn thay key sau khi đã chia sẻ source.
- Mở [`tool_gemini/.env`](tool_gemini/), thay value của slot tương ứng bằng key mới.
- Chạy lại `python generate_gt.py` — script tự đọc lại `.env` lúc start, tự **skip** các dòng đã có `target_reason` (resume mặc định).

**10.4.5 Best practice**
- 1 key/1 Google account để tránh bị block dây chuyền.
- **Không commit `.env`** vào git ([`tool_gemini/.gitignore`](tool_gemini/) đã loại).
- Sau khi sinh xong dataset → xoá toàn bộ key trong `.env` hoặc revoke trên AI Studio để khỏi bị abuse nếu repo public.
- Free tier resets daily, không cần xoay nếu chỉ vượt quota theo phút — code tự đợi.

### 10.5 Đóng hoàn toàn endpoint (post-demo)
- Kaggle: **Stop Session** notebook server → process chết → ngrok tunnel rớt → URL trả 502/không kết nối được.
- Local: `Ctrl+C` uvicorn.
- (Optional) Regenerate ngrok authtoken cho chắc.
- (Optional) Revoke tất cả Gemini key trong AI Studio.

### 10.6 Best practice
- **Mỗi phiên dùng token khác**: `vihs-<ngày>-<random>`. Lộ token nào thì biết đến từ phiên nào.
- **Đừng dùng `change-me-please`** (default trong code) cho demo thật.
- Token nên dài ≥16 ký tự, chứa cả chữ-số-ký hiệu.

---

## 11. API reference

### Local (port 8501)

| Method | Path | Mô tả |
|---|---|---|
| GET | `/` | Web UI (SPA) |
| GET | `/api/health` | Trạng thái remote/mock |
| POST | `/api/predict` | 4 task song song cho 1 câu |
| POST | `/api/qa` | Standalone QA cho 1 câu |

### Kaggle server (qua ngrok)

| Method | Path | Mô tả |
|---|---|---|
| GET | `/` | Info JSON |
| GET | `/health` | Trạng thái + QA có nạp hay không |
| POST | `/predict_all` | 4 task |
| POST | `/qa` | QA standalone |

**Headers bắt buộc cho POST tới Kaggle**:
- `Content-Type: application/json`
- `X-API-Token: <token>`
- `ngrok-skip-browser-warning: 1` (tránh trang interstitial của ngrok free)

### Ví dụ curl

```bash
curl -X POST https://abcd-1234.ngrok-free.dev/predict_all \
  -H "Content-Type: application/json" \
  -H "X-API-Token: vihs-2026-x7k9qm" \
  -H "ngrok-skip-browser-warning: 1" \
  -d '{"text":"mày ngu vãi lồn"}'
```

### Response `/predict_all`
```jsonc
{
  "raw_input": "mày ngu vãi lồn",
  "preprocessed": "mày ngu vãi lồn",
  "hsd":  {
    "label": "OFFENSIVE",
    "confidence": 0.89,
    "label_probs": {"CLEAN": 0.03, "OFFENSIVE": 0.89, "HATE": 0.08},
    "generated_label": "OFFENSIVE",        // raw từ generate()
    "raw_output": "offensive",
    "prefixed_input": "hate-speech-detection: mày ngu vãi lồn",
    "timing_ms": 145.3
  },
  "ctsd": { "label": "TOXIC", "confidence": 0.91, ... },
  "hos":  {
    "spans": [{"text":"ngu vãi lồn","start":5,"end":16}],
    "raw_output": "mày [hate]ngu vãi lồn[hate]"
  },
  "qa":   {
    "available": true,
    "explanation": "Câu này chứa ngôn từ xúc phạm trực tiếp..."
  },
  "timing_ms": { "total": 820.5 }
}
```

---

## 12. Cơ chế confidence score

ViHateT5 là model **text-to-text**, output là **chuỗi từ tự do** (vd `"hate"`), không có classifier head. Để có xác suất cho 3 nhãn `CLEAN/OFFENSIVE/HATE`, dùng **label-conditional likelihood scoring**:

### Cách làm
1. Với mỗi nhãn ứng viên (`clean`, `offensive`, `hate`):
   - Tokenize nhãn → chuỗi token IDs.
   - Forward pass với `labels=` là chuỗi đó (**teacher forcing**).
   - `out.loss` = mean negative log P(token | input) trên các token của nhãn.
   - `score = -out.loss` = mean log-likelihood của nhãn.
2. Softmax 3 scores → phân phối xác suất chuẩn hoá trên 3 nhãn.
3. **Nhãn chính = `argmax(label_probs)`** (đảm bảo badge luôn khớp thanh confidence cao nhất).
4. Output của `model.generate()` được lưu riêng vào `generated_label` cho debug.

### Vì sao không đổi kiến trúc thành softmax classifier?
- Lớp cuối T5 (`lm_head`) **đã có softmax sẵn** — nhưng trên 32.000 từ vựng, không phải 3 class.
- Trick này = "ép" model trả lời như câu trắc nghiệm, lọc ra prob của 3 nhãn cần quan tâm.
- Không cần train lại, không thêm head mới.

Implementation: [`kaggle/vihatet5_kaggle_server.py:97-108`](kaggle/vihatet5_kaggle_server.py#L97-L108).

---

## 13. Troubleshooting

| Triệu chứng | Nguyên nhân | Cách sửa |
|---|---|---|
| Status đỏ "Remote offline" | `.env` thiếu URL, hoặc Cell 4 Kaggle đã dừng | Kiểm tra `REMOTE_API_URL`, chạy lại Cell 4 |
| Status đỏ "401 Unauthorized" | Token Kaggle ≠ Token `.env` | Khớp `VIHATET5_API_TOKEN` ↔ `REMOTE_API_TOKEN` |
| Block QA "QA model chưa nạp" | Chưa Add Data `hate-qa-model` | Làm bước B.1 |
| Mở URL ngrok ra 404 | Bình thường — server không có route `/` | Test `<URL>/health` thay vì `/` |
| ngrok trang "ERR_NGROK_xxxx" | Authtoken sai hoặc Cell 4 chết | Check Cell 4 log, regenerate authtoken |
| `.env` sửa xong không có hiệu lực | uvicorn cache lúc start | Ctrl+C + restart |
| Browser không thấy thay đổi static | Browser cache | Ctrl+F5 (hard refresh) |
| CSV upload garbled chữ Việt | File save sai encoding | Excel: Save As → "CSV UTF-8" |
| QA output bị lặp cụm từ | Greedy decoding T5-base | Đặc tính model, không phải bug. Có thể thêm `repetition_penalty=1.3` vào `generate()` ở server |
| `RuntimeError: asyncio.run() cannot be called from a running event loop` | uvicorn chạy trực tiếp trong Jupyter | Notebook đã dùng `nest_asyncio + asyncio.get_event_loop().run_until_complete()` — không dùng `uvicorn.run()` |
| Confidence < nhãn khác (vd HATE 36% mà badge HATE) | Code cũ lấy nhãn từ `generate()` thay vì `argmax(probs)` | Đã fix trong [`kaggle/vihatet5_kaggle_server.py`](kaggle/vihatet5_kaggle_server.py) — re-upload + restart kernel |

---

## 14. Tham khảo

### Paper & Model
- **Backbone paper**: Luan Thanh Nguyen, "ViHateT5: Enhancing Hate Speech Detection in Vietnamese With a Unified Text-to-Text Transformer Model", *ACL 2024 Findings* — [PDF local](2024.findings-acl.355.pdf) · [aclanthology](https://aclanthology.org/2024.findings-acl.355.pdf)
- **Pre-trained model**: [tarudesu/ViHateT5-base-HSD](https://huggingface.co/tarudesu/ViHateT5-base-HSD)
- **Pre-training corpus**: [tarudesu/VOZ-HSD](https://huggingface.co/datasets/tarudesu/VOZ-HSD) (~10M comments từ voz.vn)

### Datasets benchmark
- **ViHSD** — [uitnlp/vihsd](https://huggingface.co/datasets/uitnlp/vihsd) (Luu et al. 2021): 33k comments, 3 nhãn CLEAN/OFFENSIVE/HATE.
- **ViCTSD** — [tarudesu/ViCTSD](https://huggingface.co/datasets/tarudesu/ViCTSD) (Nguyen et al. 2021): 10k comments, 2 nhãn TOXIC/NONE + 1 nhãn constructive.
- **ViHOS** — [phusroyal/ViHOS](https://github.com/phusroyal/ViHOS) (Hoang et al. 2023): annotate cụm từ thù ghét ở mức span (BIO tagging).

### Mô hình so sánh trong thực nghiệm
- **Qwen2.5-7B-Instruct** — [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- **URA-LLaMa** — [ura-hcmut/ura-llama-7b](https://huggingface.co/ura-hcmut/ura-llama-7b)

### Báo cáo nhóm
- [`CS222-Draft.pdf`](CS222-Draft.pdf)

---

## License

MIT — phục vụ mục đích học thuật cho môn CS222, UIT.
