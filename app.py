# app.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import base64

from model import analyze_face_from_frame

app = FastAPI(
    title="Face Guide API",
    description="웹캠 가이드용 얼굴 위치 분석 서비스",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# 0) 헬스체크
# ==========================
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# ==========================
# 1) 파일 업로드 방식
# ==========================
@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    content = await file.read()
    np_arr = np.frombuffer(content, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "이미지 디코딩 실패"}

    return analyze_face_from_frame(frame)

# ==========================
# 2) base64 방식
# ==========================
@app.post("/analyze_base64")
async def analyze_base64(data: dict):
    if "image" not in data:
        return {"error": "image(base64) 필드 필요"}

    try:
        img_bytes = base64.b64decode(data["image"])
    except:
        return {"error": "base64 디코딩 실패"}

    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "이미지 디코딩 실패"}

    return analyze_face_from_frame(frame)

# ==========================
# 로컬 실행용
# ==========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
