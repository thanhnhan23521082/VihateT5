from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import os
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from src.inference import ViHSDModel
from src.config import DEFAULT_THRESHOLD_HATE, DEFAULT_THRESHOLD_OFF

app = FastAPI()

# Đảm bảo thư mục static tồn tại
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

try:
    model = ViHSDModel()
    print("Model loaded successfully.")
except Exception as e:
    print("Warning: Model could not be loaded:", repr(e))
    model = None

class PredictRequest(BaseModel):
    text: str
    threshold: float = 0.5

@app.get("/", response_class=HTMLResponse)
async def get_index():
    return FileResponse("static/index.html")

@app.post("/api/predict")
async def predict(request: PredictRequest):
    if model is None:
        return {"label": "ERROR", "confidence": 0.0, "message": "Model not loaded"}
    
    result = model.predict(request.text, t_hate=request.threshold, t_off=DEFAULT_THRESHOLD_OFF)
    
    return result

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8501, reload=True)

