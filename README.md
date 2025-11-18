# README.md

# 📘 Thumbnail Selector AI  
# Google Vision API + Custom Smile/Eye Scoring 기반 영상 썸네일 자동 생성기  
# (Flask API 서버 포함)

---

## 🌐 Overview

Thumbnail Selector AI는 비디오에서 일정 간격으로 프레임을 추출한 후  
Google Cloud Vision API 얼굴 분석 + 스마일 곡률/입 벌림 기반 감정 점수 등을 결합하여  
AI가 가장 좋은 썸네일을 자동으로 선정하는 시스템입니다.

이 프로젝트는 다음 기능을 제공합니다:

- Google Vision AI 얼굴 분석
- 스마일 곡률(score), 입 벌림도(lip distance)
- 흐림/노출/얼굴 각도 기반 품질 점수
- 0.25초 간격 프레임 추출
- 최고 점수 프레임을 Base64 JPEG로 반환
- Flask REST API 제공

---

## 🧱 Project Structure

thumbnail-ai/
 ├─ model.py               # AI 썸네일 분석 로직  
 ├─ app.py                 # Flask API 서버  
 ├─ requirements.txt       # Python dependencies  
 ├─ service-account.json   # Google Vision API 자격증명(필수)  
 └─ README.md  

---

## ⚙️ Features

- Vision AI 얼굴 landmark 분석
- 커스텀 감정/품질 점수화 알고리즘
- 420점 만점 scoring system
- 프레임 단위 분석하여 최적 썸네일 자동 선택
- Flask 기반 /analyze API 제공

---

## 🔧 How It Works

### 1) 프레임 추출  
0.25초 간격으로 자동 추출

### 2) Vision AI 얼굴 분석  
- Blur / Exposure  
- Joy likelihood  
- Roll / Pan / Tilt  
- Landmark(입/입꼬리) 좌표 기반 Smile Index

### 3) 점수 계산  
- 품질 점수 (0 ~ 120)  
- 감정 점수 (0 ~ 300)  
- 총점 = 420점 만점

### 4) 최고 점수 프레임 선택 후 Base64 반환

---

## 🧪 Python Example

from model import find_best_thumbnail

result = find_best_thumbnail("video.mov")

print(result["time_sec"])
print(result["score"])
print(result["image_base64"])

---

## 🚀 Flask API (app.py)

### ▶ Endpoint  
POST /analyze  
(비디오 파일 업로드: multipart/form-data)

### ▶ JavaScript Example

const form = new FormData();
form.append("video", file);

fetch("http://YOUR-EC2-IP:8000/analyze", {
  method: "POST",
  body: form
})
  .then(res => res.json())
  .then(console.log);

### ▶ Response Example

{
  "message": "Analysis successful",
  "best_time_sec": 1.25,
  "score": 312,
  "image_base64": "<base64 JPEG>"
}

---

## 📦 requirements.txt

opencv-python  
numpy  
google-cloud-vision  
Pillow  
flask  
flask-cors  

(EC2·Docker 환경 추천: opencv-python-headless)

---

## 🔐 Google Vision API Credentials

service-account.json 파일 필요  
Google Vision 콘솔에서 발급 → 프로젝트 디렉토리에 저장

model.py 내부에서 자동 설정됨:

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service-account.json"

---

## 🖥 Local Run

pip install -r requirements.txt
python app.py

---

