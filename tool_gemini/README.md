# Tạo Ground Truth bằng Gemini API (xoay vòng nhiều key)

## Các file
| File | Vai trò |
|------|---------|
| `.env` | Nơi bạn **dán các API key** (hiện 8 slot) và chỉnh cấu hình |
| `config.py` | Đọc cấu hình + key từ `.env` |
| `gemini_client.py` | Client Gemini có cơ chế **xoay vòng key** khi hết quota |
| `generate_gt.py` | Script chính: đọc CSV, gọi Gemini, lưu kết quả |
| `Nhan_data.csv` | Dữ liệu: cột `input_text` (đầu vào) + `target_reason` (kết quả) |

## Cách dùng
1. Mở file `.env`, dán key vào `GEMINI_API_KEY_1`, `_2`, ... `_8` (thêm slot tuỳ ý, tool tự quét).
2. Chạy:
   ```powershell
   .\venv\Scripts\python.exe generate_gt.py
   ```

## Cơ chế
- **Xoay key:** dùng key #1; khi key hết hạn mức (lỗi 429/quota) tự chuyển sang #2 → #3 → ... đến key cuối.
- **Hết tất cả key:** lưu tiến trình và dừng. Nạp key mới vào `.env` rồi chạy lại.
- **Lưu & chạy tiếp:** kết quả ghi vào CSV **ngay sau mỗi dòng**. Chạy lại sẽ tự bỏ qua
  các dòng đã có kết quả và tiếp tục từ dòng còn dở. Có thể `Ctrl+C` để dừng an toàn.
