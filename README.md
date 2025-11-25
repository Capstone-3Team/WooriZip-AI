## 🐶 Pet Detector AI  (일상피드 인식용)
Google Cloud Vision + OpenCV 기반 반려동물 자동 분류 API

---

## 📌 Overview
이 프로젝트는 업로드된 **사진(Image)** 또는 **영상(Video)** 에서  
개/고양이/반려동물 등장 여부를 자동 탐지하는 서버입니다.

❗ FFmpeg 기반 편집 기능 없음  
❗ Vision 기반의 **분류(Classification)** 기능만 수행

구성:

```
model.py        # Vision API 기반 반려동물 분석 로직
app.py          # Flask API 서버 (/classify)
requirements.txt
README.md
```

---

## 🚀 API Endpoints

### ✔ POST `/classify`
업로드된 사진/영상에서 반려동물 등장 여부 반환

#### Request
`multipart/form-data`  
필드명: file

#### Response Example (Image)
```json
{
  "message": "success",
  "data": {
    "file_type": "image",
    "is_pet_present": true
  }
}
```

#### Response Example (Video)
```json
{
  "message": "success",
  "data": {
    "file_type": "video",
    "is_pet_present": true,
    "timestamps": [1.0, 3.0, 5.0]
  }
}
```

---

## 🛠 환경변수
Google Cloud Vision API에서 사용할 프로젝트 ID:

```bash
export GCP_PROJECT_ID="YOUR_PROJECT_ID"
```

서비스 계정 키(JSON):

```bash
export GOOGLE_APPLICATION_CREDENTIALS="service-account.json"
```

---

## 📦 설치 & 실행

```bash
pip install -r requirements.txt
python app.py
```

---

## 🧱 Internals (How It Works)

1. 업로드된 파일의 MIME 타입 검사 → 이미지/비디오 자동 분류  
2. 이미지(image): Vision API Label Detection  
3. 비디오(video):  
   - OpenCV로 1초 간격 프레임 추출  
   - 각 프레임을 Vision API로 분석  
   - 반려동물 등장 timestamp 기록  
4. 결과를 JSON 형태로 반환  

---

## 🎯 Summary

- Vision 기반 반려동물 등장 자동 분류  
- 이미지/비디오 모두 지원  
- 영상의 경우 timestamp 제공  
- Flask 기반 REST API

🎉 어떤 서비스에도 반려동물 포함 미디어 자동 분류 기능을 쉽게 통합 가능!
