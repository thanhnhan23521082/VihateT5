# Setup từ đầu — ViHateT5 multi-task demo

Host model `ViHateT5-base-HSD` + QA fine-tune trên **Kaggle GPU T4**, expose qua
**ngrok**, máy local làm proxy + UI.

Có **2 notebook** trên Kaggle:
| Notebook | Vai trò | File |
|---|---|---|
| **QA fine-tune** | Train 1 lần để có model giải thích | `vihatet5-qa.ipynb` (ở root repo) |
| **Server** | Host 4 task + ngrok, luôn chạy | `kaggle/vihatet5_kaggle_server.ipynb` |

---

## A. Chuẩn bị 1 lần — đăng ký ngrok + tự đặt API token

1. Tạo tài khoản ngrok free: <https://dashboard.ngrok.com> → copy **Authtoken**.
2. Tự nghĩ một chuỗi **API token** bí mật (vd `vihs-2026-x7k9qm`). Sẽ dùng cả 2 phía.
3. (Tuỳ chọn) tạo tài khoản Kaggle nếu chưa có.

---

## B. Notebook QA fine-tune — chỉ làm khi muốn task QA hoạt động

> Bỏ qua phần này nếu bạn chỉ cần 3 task HSD/CTSD/HOS. Block QA trên web sẽ hiện
> "QA model chưa được nạp" — không sập app.

### B1 — Upload CSV training thành Kaggle Dataset
File `vihsd_qa_gemini_GT_Final.csv` đã có sẵn ở root repo.
- Kaggle → **Datasets → New Dataset** → upload CSV → đặt tên `qa-ground-truth`.

### B2 — Upload notebook QA
- **Notebooks → New Notebook → File → Import Notebook** → `vihatet5-qa.ipynb`.
- **Settings → Accelerator = GPU T4**.
- **Add Data → qa-ground-truth**.

### B3 — Sửa 1 dòng path trong Cell 14
```python
file_path = '/kaggle/input/qa-ground-truth/vihsd_qa_gemini_GT_Final.csv'
```

### B4 — Chạy & lưu
- Chỉ cần chạy **Cell 1 (cài), Cell 3 (load base), Cell 14 (train + save)**. Bỏ qua cell evaluation.
- Train ~5 epoch trên T4: 20–40 phút. Xong sẽ có `/kaggle/working/hate_qa_model_final/`.
- **Save Version → Save & Run All (Commit)** — đợi commit xong.

### B5 — Đóng gói thành Dataset `hate-qa-model`
- Vào Output của Version vừa commit → **New Dataset** → tên: **`hate-qa-model`** (đúng tên này để server tự nhận).
- Bên trong dataset phải có folder `hate_qa_model_final/` (chứa config.json + weights + tokenizer).

---

## C. Notebook Server — chạy mỗi phiên demo

### C1 — Upload notebook server
- **Notebooks → New Notebook → File → Import Notebook** → `kaggle/vihatet5_kaggle_server.ipynb`.
- **Settings → Accelerator = GPU T4**.
- **Add Data**:
  - `hate-qa-model` (nếu đã làm phần B; nếu chưa thì bỏ qua, QA sẽ hiện "chưa nạp").

### C2 — Dán 2 token (Cell 2 của notebook)
```python
NGROK_AUTHTOKEN = "<authtoken ngrok của bạn>"
os.environ["VIHATET5_API_TOKEN"] = "vihs-2026-x7k9qm"   # khớp REMOTE_API_TOKEN ở .env
```

### C3 — Run All
- **Run All** từ Cell 1 → Cell 4.
- Cell 1: kiểm tra GPU (`True | Tesla T4`).
- Cell 3: tải model HSD (~1–2 phút). Nếu có Dataset QA, log cũng sẽ in `[qa] ready.` khi `/predict_all` gọi lần đầu.
- Cell 4: in `>>> PUBLIC URL: https://xxxx.ngrok-free.dev` rồi **quay liên tục** (server đang phục vụ — đừng tắt).

### C4 — Test nhanh từ trình duyệt
Mở `<PUBLIC_URL>/health` — thấy JSON kèm `"qa_available": true/false` là server sống.

---

## D. Máy local — kết nối vào server Kaggle

### D1 — Điền `.env`
```env
REMOTE_API_URL=https://xxxx.ngrok-free.dev
REMOTE_API_TOKEN=vihs-2026-x7k9qm
REMOTE_API_TIMEOUT=60
```

### D2 — Chạy local app
```powershell
pip install -r requirements.txt
uvicorn app:app --port 8501
# hoặc: python app.py
```

Mở **http://127.0.0.1:8501**:
- Nhập câu (có dấu!) hoặc bấm 3 nút "Ví dụ 1/2/3" → bấm **Chạy Phân tích Toàn diện**.
- 4 block hiện song song: HSD badge, CTSD badge, HOS highlight, QA explanation.

---

## E. Save Version để demo trên bảo vệ (tuỳ chọn)

Khi server đã chạy ổn:
- Kaggle notebook server → **Save Version → Save & Run All (Commit)**.
- Bản version đó chạy nền tới giới hạn phiên (~9–12h GPU). URL ngrok in trong **log của version**.
- Lưu ý: ngrok free đổi URL mỗi lần `connect()` → cập nhật `.env` trước khi demo.

---

## F. Khi có lỗi

| Triệu chứng | Nguyên nhân thường gặp | Sửa |
|---|---|---|
| Web hiện "Remote offline" | `.env` thiếu URL/token, hoặc Cell 4 đã dừng | Kiểm tra `.env` + chạy lại Cell 4 |
| 401 Unauthorized | `VIHATET5_API_TOKEN` ở Kaggle ≠ `REMOTE_API_TOKEN` ở `.env` | Khớp đúng cả 2 |
| Block QA hiện "chưa nạp" | Chưa Add Data `hate-qa-model` hoặc QA_MODEL_PATH sai | Add Data hoặc set env |
| Mở URL ngrok ra 404 ở `/` | Bình thường — server không có route `/` | Truy cập `/health` |
| Trang ngrok "ERR_NGROK..." | Cell 4 chết | Run All lại notebook server |
| Tokenization "mày" → `<unk>` | Bản chất tokenizer (không phải bug) | Bỏ qua, model vẫn dự đoán được |

---

## G. Endpoint server (tham khảo nhanh)

| Method | Path | Mô tả |
|---|---|---|
| GET | `/` | Info JSON |
| GET | `/health` | Trạng thái + có QA model hay không |
| POST | `/predict_all` | Chạy đồng thời 4 task cho 1 câu |
| POST | `/qa` | Standalone QA (sinh giải thích) |

Header bắt buộc cho POST: `X-API-Token: <token>`.

```bash
curl -X POST <PUBLIC_URL>/predict_all \
  -H "X-API-Token: vihs-2026-x7k9qm" \
  -H "Content-Type: application/json" \
  -H "ngrok-skip-browser-warning: 1" \
  -d '{"text":"mày ngu vãi lồn"}'
```
