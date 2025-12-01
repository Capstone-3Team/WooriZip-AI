# 🧠 WooriZip AI Server  
AI 기능 4종을 통합한 Flask 기반 백엔드 서버입니다.

본 서버는 다음 기능을 제공합니다:

1. **Thumbnail Generator**  
   - 얼굴 인식 + 웃는 표정 우선 점수화 → 최적 썸네일 자동 추출  

2. **STT → 요약 + 제목 생성**  
   - Gemini Flash 모델로 오디오 내용을 1줄 요약 + 영상 주제 기반 제목 자동 생성  

3. **Pet Shorts Generator**  
   - 반려동물 등장 구간 자동 탐지 + 구간만 모아 숏츠 파일 생성  

4. **Pet Daily Classifier**  
   - 업로드된 사진/영상에서 반려동물 등장 여부 분류  

5. **Face Arrangement (Face Guide)**  
   - 촬영 시 사용자 얼굴 위치 분석  
   - "come_in / move_back / perfect" 상태 반환  

---

## 📁 Project Structure

```
WooriZip-AI/
│
├── app.py
├── requirements.txt
│── shorts/generated
└── models/
    ├── thumb_stt.py
    ├── pet_shorts.py
    ├── pet_daily.py
    ├── face_arrange.py
    ├── __init__.py
```

---

## 🚀 How to Run

### 1) Install dependencies
```
pip install -r requirements.txt
```

### 2) Run Flask server
```
python app.py
```

서버는 다음 주소에서 실행됩니다:

```
http://localhost:8000
```

---

# 📌 API Endpoints

아래 모든 요청은 **multipart/form-data** 또는 **JSON** 형식을 사용합니다.

---

# 🎯 1) Thumbnail API

### **POST /thumbnail**

**Request**
```
form-data:
  video: <mp4 file>
```

**Response**
```json
{
  "message": "Thumbnail analysis successful",
  "time_sec": 1.35,
  "score": 422.1,
  "image_base64": "..."
}
```

---

# 📝 2) STT + Summary + Title API

### **POST /stt**

**Request**
```
form-data:
  api_key: <Gemini API key>
  video: <mp4>
```

**Response**
```json
{
  "message": "summary + title generation successful",
  "summary": "영상 내용을 한 문장으로 요약…",
  "title": "자동 생성된 제목"
}
```

---

# 🐶 3) Pet Detect API (반려동물 등장 구간)

### **POST /detect**

**Request**
```
form-data:
  video: <mp4>
```

**Response**
```json
{
  "message": "success",
  "segments": [
    [1.0, 3.0],
    [8.0, 11.0]
  ]
}
```

---

# ✂️ 4) Pet Shorts Generate API

### **POST /compile**

**Request (JSON)**
```json
{
  "video_path": "uploads/video1.mp4",
  "segments": [[1.0, 3.5], [8.0, 11.2]]
}
```

**Response**
```json
{
  "message": "success",
  "output": "shorts/3f1d9a53.mp4"
}
```

---

# 📷 5) Pet Daily Classifier API

### **POST /classify**

업로드된 파일이 **사진인지 영상인지 자동 분류**하여  
반려동물 등장 여부를 판별합니다.

**Request**
```
form-data:
  file: <image or video>
```

**Response**
```json
{
  "message": "success",
  "data": {
    "file_type": "video",
    "is_pet_present": true,
    "timestamps": [0.0, 1.0, 2.0]
  }
}
```

---

# 🙂 6) Face Arrangement API

### **POST /face_arrange**

이미지 한 장을 분석하여 다음 상태 중 하나 반환:

- `perfect`
- `move_back`
- `come_in`

**Request (form-data or base64 JSON)**  
1) Multipart 이미지 업로드
```
form-data:
  file: <image>
```

2) Base64 JSON
```json
{
  "image": "<base64_string>"
}
```

**Response**
```json
{
  "message": "success",
  "data": {
    "state": "move_back",
    "message": "조금 뒤로 물러나세요",
    "is_good": false
  }
}
```

---

# 🔧 Environment Variables

`.env` 또는 서버 환경 변수에서 설정:

```
GOOGLE_APPLICATION_CREDENTIALS=service-account.json
GCP_PROJECT_ID=your_project_id
```

---


# 📢 Notes

- ffmpeg는 시스템에 설치되어 있어야 합니다  
  macOS → `brew install ffmpeg`  
  Ubuntu → `sudo apt install ffmpeg`  

- pet_shorts 출력 파일은 자동으로 UUID 기반 파일명으로 저장됩니다.


