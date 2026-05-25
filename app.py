from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import os
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from src.remote_inference import RemoteViHateT5Model, RemoteModelError
from src.mock import MockModel

app = FastAPI(title="ViHateT5 multi-task demo")

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Chọn model:
#   - MOCK_MODE=1 trong env  -> ép dùng mock dù đã có URL
#   - REMOTE_API_URL trống   -> tự động dùng mock (đỡ phải chạy Kaggle)
#   - còn lại                -> gọi server thật trên Kaggle
_remote = RemoteViHateT5Model()
_force_mock = os.environ.get("MOCK_MODE", "").strip() in ("1", "true", "True")
if _force_mock or not _remote.is_configured:
    model = MockModel()
    print("⚠ Đang dùng MOCK MODEL (dữ liệu giả). "
          "Điền REMOTE_API_URL trong .env để chuyển sang model thật.")
else:
    model = _remote
    print(f"Remote model client ready -> {_remote.base_url}")


class PredictRequest(BaseModel):
    text: str
    qa_max_length: int = 128


class QARequest(BaseModel):
    text: str
    max_length: int = 128


@app.get("/", response_class=HTMLResponse)
async def get_index():
    return FileResponse("static/index.html")


@app.get("/api/health")
async def health():
    try:
        return {"ok": True, "remote": model.health()}
    except RemoteModelError as exc:
        return JSONResponse(status_code=503, content={"ok": False, "message": str(exc)})


@app.post("/api/predict")
async def predict(request: PredictRequest):
    """Chạy đồng thời 4 task: HSD, CTSD, HOS, QA — trả kết quả gom block."""
    try:
        return model.predict_all(request.text, qa_max_length=request.qa_max_length)
    except RemoteModelError as exc:
        return JSONResponse(
            status_code=503,
            content={"error": True, "message": str(exc)},
        )


@app.post("/api/qa")
async def qa(request: QARequest):
    try:
        return model.qa(request.text, max_length=request.max_length)
    except RemoteModelError as exc:
        return JSONResponse(
            status_code=503,
            content={"explanation": None, "message": str(exc)},
        )


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8501, reload=True)
