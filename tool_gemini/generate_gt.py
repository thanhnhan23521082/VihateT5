"""
generate_gt.py
--------------
Tạo Ground Truth tự động cho từng dòng trong file CSV bằng Gemini API.

Cách hoạt động:
- Đọc file CSV (mặc định Nhan_data.csv) gồm cột input_text và target_reason.
- Với MỖI dòng còn TRỐNG ở cột target_reason -> gọi Gemini sinh câu trả lời.
- LƯU LẠI CSV ngay sau mỗi dòng -> có thể dừng/chạy lại bất cứ lúc nào,
  lần sau sẽ TỰ ĐỘNG bỏ qua các dòng đã có kết quả và chạy tiếp từ dòng cuối.
- Nếu cả 3 key hết hạn mức -> lưu tiến trình và DỪNG chương trình.

Chạy:  python generate_gt.py
"""

import csv
import sys

# Ép console in được tiếng Việt + emoji trên Windows (tránh lỗi cp1252)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from config import (
    get_api_keys,
    INPUT_CSV,
    INPUT_COLUMN,
    OUTPUT_COLUMN,
)
from gemini_client import RotatingGeminiClient, AllKeysExhausted


def read_rows(path):
    """Đọc toàn bộ CSV thành danh sách dict + giữ lại thứ tự cột."""
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    return fieldnames, rows


def write_rows(path, fieldnames, rows):
    """Ghi lại toàn bộ CSV (encoding utf-8-sig để Excel hiển thị tiếng Việt đúng)."""
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    api_keys = get_api_keys()
    if not api_keys:
        print(
            "❌ Chưa có API key. Mở file .env và dán key vào "
            "GEMINI_API_KEY_1, _2, _3, ... rồi chạy lại."
        )
        sys.exit(1)

    print(f"📂 File dữ liệu: {INPUT_CSV}")
    print(f"🔑 Số API key tìm thấy: {len(api_keys)}")

    fieldnames, rows = read_rows(INPUT_CSV)

    # Kiểm tra cột tồn tại
    if INPUT_COLUMN not in fieldnames:
        print(f"❌ Không thấy cột '{INPUT_COLUMN}' trong file. Có các cột: {fieldnames}")
        sys.exit(1)
    if OUTPUT_COLUMN not in fieldnames:
        # Nếu chưa có cột kết quả thì thêm vào
        fieldnames = list(fieldnames) + [OUTPUT_COLUMN]
        for r in rows:
            r.setdefault(OUTPUT_COLUMN, "")

    total = len(rows)
    done = sum(1 for r in rows if (r.get(OUTPUT_COLUMN) or "").strip())
    print(f"📊 Tổng {total} dòng — đã có sẵn {done} dòng, cần xử lý {total - done} dòng.\n")
    print("=" * 60)

    client = RotatingGeminiClient(api_keys)

    processed_now = 0
    try:
        for i, row in enumerate(rows):
            prompt = (row.get(INPUT_COLUMN) or "").strip()
            existing = (row.get(OUTPUT_COLUMN) or "").strip()

            # Bỏ qua dòng đã có kết quả hoặc dòng input rỗng
            if existing:
                continue
            if not prompt:
                continue

            print(f"[{i + 1}/{total}] ⏳ Đang phân tích...")
            answer = client.generate(prompt)  # có thể ném AllKeysExhausted

            row[OUTPUT_COLUMN] = answer
            write_rows(INPUT_CSV, fieldnames, rows)  # lưu ngay sau mỗi dòng
            processed_now += 1
            print(f"[{i + 1}/{total}] 🎯 {answer}")
            print("-" * 60)

    except AllKeysExhausted as e:
        write_rows(INPUT_CSV, fieldnames, rows)
        print("\n" + "=" * 60)
        print(f"🛑 {e} (đã thử xoay vòng hết tất cả key)")
        print(f"💾 Đã lưu tiến trình. Vừa xử lý thêm {processed_now} dòng trong lần này.")
        print("👉 Nạp thêm API key mới vào .env rồi chạy lại để tiếp tục từ dòng đang dở.")
        sys.exit(2)

    except KeyboardInterrupt:
        write_rows(INPUT_CSV, fieldnames, rows)
        print("\n🟡 Bạn đã dừng thủ công. Tiến trình đã được lưu lại.")
        sys.exit(0)

    # Hoàn tất toàn bộ
    write_rows(INPUT_CSV, fieldnames, rows)
    remaining = sum(1 for r in rows if not (r.get(OUTPUT_COLUMN) or "").strip()
                    and (r.get(INPUT_COLUMN) or "").strip())
    print("\n" + "=" * 60)
    print(f"✅ Hoàn tất! Lần này xử lý thêm {processed_now} dòng.")
    if remaining:
        print(f"ℹ️  Còn {remaining} dòng chưa có kết quả (input rỗng?).")
    else:
        print("🎉 Tất cả các dòng đã có Ground Truth.")


if __name__ == "__main__":
    main()
