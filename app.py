# app.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import base64

from model import analyze_face_from_frame

app = FastAPI(
    title="Face Guide API",
    description="얼굴 위치 분석 및 상태 반환 API",
    version="1.0.0"
)

# CORS 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------
# 헬스체크
# --------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# --------------------------------
# Multipart 이미지 업로드 방식
# --------------------------------
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    content = await file.read()
    np_arr = np.frombuffer(content, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "이미지 디코딩 실패"}

    return analyze_face_from_frame(frame)


# --------------------------------
# base64 이미지 처리 방식
# --------------------------------
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


# 로컬 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
