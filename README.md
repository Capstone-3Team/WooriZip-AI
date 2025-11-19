# 🐶 Pet Detector + Shorts Creator AI  
Google Cloud Vision + OpenCV + FFmpeg 기반 반려동물 출현 구간 탐지 및 숏츠 생성 API

---

## 📌 Overview

이 프로젝트는 업로드된 영상에서 **개/고양이/반려동물 등장 구간을 자동 탐지**하고,  
탐지된 구간을 FFmpeg으로 이어붙여 **최종 숏츠 영상(.mp4)** 로 생성하는 서버입니다.

구성:

```
model.py        # Vision 분석 + FFmpeg 숏츠 생성 로직
app.py          # Flask API 서버 (detect / compile)
requirements.txt
README.md
```

---

## 🚀 API Endpoints

### ✔ POST `/detect`
반려동물 등장 시간 구간 반환

#### Request
`multipart/form-data`  
필드명: **video**

#### Response Example
```json
{
  "message": "success",
  "segments": [
    [3.0, 8.0],
    [15.0, 22.0]
  ]
}
```

---

### ✔ POST `/compile`
탐지된 구간으로 숏츠 생성

#### Request
```json
{
  "video_path": "original.mp4",
  "segments": [[3,8],[15,22]]
}
```

#### Response
```json
{
  "message": "success",
  "output": "pet_shorts.mp4"
}
```

---

## 🛠 환경변수

Google Cloud Vision API에서 사용할 **프로젝트 ID** 필요:

```
export GCP_PROJECT_ID="YOUR_PROJECT_ID"
```

서비스 계정 키는:

```
export GOOGLE_APPLICATION_CREDENTIALS="service-account.json"
```

---

## 📦 설치 & 실행

```
pip install -r requirements.txt
python app.py
```

---

## 🧱 Internals (How It Works)

1. OpenCV로 1초 간격 프레임 추출  
2. Vision API → Label Detection (`dog`, `cat`, `pet` 등)  
3. 등장/사라짐 구간을 segment로 기록  
4. FFmpeg filter_complex로 각 segment 클립 이어붙임  

---

## 🎯 Summary

- Vision AI 기반 반려동물 구간 자동 탐지  
- 0.5초 이하 무효 클립 제거  
- FFmpeg으로 숏츠 자동 생성  
- Flask 기반 REST API 제공  

🎉 서버/앱/웹 어디서나 반려동물 숏츠 생성 자동화 가능!

