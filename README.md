# Family Video AI Server 🎥

가족 영상 통화·일기 영상을 분석하는 AI 서버입니다.

- 😃 웃는 얼굴 썸네일 추출 (Google Cloud Vision)
- 🗣️ STT + 요약 + 제목 생성 (Gemini 2.5 Flash)
- 두 기능 모두 Flask 하나의 서버(app.py)에서 제공

---

## ✨ Features

### 1. 😃 웃는 얼굴 썸네일 추출 (`POST /thumbnail`)
업로드된 영상에서 프레임을 추출한 후  
Vision API로 품질·감정·웃음 여부를 분석해 가장 좋은 썸네일을 선택합니다.

### 2. 🗣️ STT + 요약 + 제목 생성 (`POST /stt`)
업로드된 영상에서 오디오를 추출하고  
Gemini 모델을 통해 STT → 요약 → 제목을 생성합니다.

---

## 📁 Project Structure

.
├── app.py
├── model.py
├── requirements.txt
└── README.md

---

## 📦 Installation

git clone <repo-url>
cd <project-folder>
python -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt


---

## 🔐 Environment Variables

export GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
export GOOGLE_APPLICATION_CREDENTIALS="/home/ubuntu/service-account.json"

---

## 🏃 Run (Local)

export GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
python app.py


서버 URL:

```
http://0.0.0.0:8000
```

---

## 🌐 API Usage

### 1) POST /thumbnail  
curl -X POST http://localhost:8000/thumbnail
-F "video=@/path/to/video.mp4"

Response 예시:
{
"message": "Thumbnail analysis successful",
"best_time_sec": 3.5,
"score": 742,
"image_base64": "<base64>"
}

---

### 2) POST /stt  
curl -X POST http://localhost:8000/stt
-F "video=@/path/to/video.mp4"

Response 예시:
{
"message": "success",
"transcript": "...",
"summary": "...",
"title": "..."
}

---

## ☁️ EC2 Deployment

### 1. 설치
sudo apt update
sudo apt install -y python3 python3-venv python3-pip ffmpeg git

### 2. 프로젝트 세팅
git clone <repo-url> family-video-ai
cd family-video-ai
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt


### 3. 환경 변수 등록
echo 'export GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"' >> ~/.bashrc
echo 'export GOOGLE_APPLICATION_CREDENTIALS="/home/ubuntu/service-account.json"' >> ~/.bashrc
source ~/.bashrc

### 4. 서버 실행
source venv/bin/activate
python app.py

접속:
```
http://<EC2-PUBLIC-IP>:8000
```

---

## 📝 Notes

- 프론트엔드 업로드 필드명: `"video"`
- 두 기능은 독립적:
  - `/thumbnail` → find_best_thumbnail()
  - `/stt` → analyze_video_content()
- model.py는 endpoint 변경과 무관
```

---
