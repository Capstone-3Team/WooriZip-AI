# WooriZip-AI
##🐶 Pet Detector AI
Google Cloud Vision + OpenCV 기반 반려동물 자동 분류 API
###📌 Overview
이 프로젝트는 업로드된 사진(Image) 또는 영상(Video) 파일에서
개/고양이/반려동물 등장 여부를 자동 탐지하는 서버입니다.
###⚠️ 영상 편집(FFmpeg) 기능은 없습니다.
오직 Vision 기반의 반려동물 포함 여부 분류만 수행합니다.
구성:
model.py        # Vision API 기반 분석 로직 (사진/영상 공통)
app.py          # Flask API 서버 (/classify)
requirements.txt
README.md
###🚀 API Endpoints
✔ POST /classify
업로드된 파일이 사진인지 영상인지 자동 판별하고,
반려동물 등장 여부를 반환합니다.
Request
multipart/form-data
필드명: file
###📸 Response Example (Image)
{
  "message": "success",
  "data": {
    "file_type": "image",
    "is_pet_present": true
  }
}
###🎥 Response Example (Video)
{
  "message": "success",
  "data": {
    "file_type": "video",
    "is_pet_present": true,
    "timestamps": [1.0, 3.0, 4.0]
  }
}
###🛠 환경변수
Google Cloud Vision API 프로젝트 ID:
export GCP_PROJECT_ID="YOUR_PROJECT_ID"
서비스 계정 키(JSON) 설정:
export GOOGLE_APPLICATION_CREDENTIALS="service-account.json"
###📦 설치 & 실행
pip install -r requirements.txt
python app.py
###🧱 Internals (How It Works)
업로드된 파일의 MIME 타입 검사 → image / video 자동 감지
Image 처리
Vision API Label Detection 호출
dog, cat, pet, animal, puppy, kitten 등 키워드 기반 판별
Video 처리
OpenCV로 1초 간격 프레임 추출
각 프레임을 Vision API로 분석하여 반려동물 감지
등장한 초(timestamp)를 배열로 저장
결과는 JSON 형태로 백엔드 DB 저장 가능
###🎯 Summary
Vision AI 기반 반려동물 자동 분류
이미지/영상 모두 지원
프레임 추출 최적화
Flask REST API 제공
###🎉 서비스 내 반려동물 콘텐츠만 따로 모아보는 기능을 매우 쉽게 구현할 수 있습니다!
