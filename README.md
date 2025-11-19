# 🎙 Video STT + Summary AI
Google Gemini 2.5 Flash 기반 영상 → 오디오 추출 → STT → 요약 API

---

## 📌 Overview

이 프로젝트는 업로드된 **영상 파일에서 오디오를 추출**하고  
**Google Gemini AI로 STT(받아쓰기) + 한 문장 요약**을 생성하는 AI 백엔드입니다.

구성 파일:

```
model.py         # AI 분석(STT+요약) 로직
app.py           # Flask API 서버
requirements.txt
service-account.json (X) → Gemini는 API Key 기반
```

---

## 🚀 API Endpoint

### `POST /analyze`

#### 요청 형식
`multipart/form-data`  
필드명: **video**

#### Response
```json
{
  "message": "success",
  "transcript": "오디오 내용 전체",
  "summary": "요약 1문장"
}
```

---

## 🧪 JS 예시

```js
const form = new FormData();
form.append("video", file);

fetch("http://<EC2-IP>:8000/analyze", {
  method: "POST",
  body: form
})
  .then(r => r.json())
  .then(console.log);
```

---

## ⚙ 환경변수 설정

Flask 서버 실행 전에 반드시 설정:

```
export GOOGLE_API_KEY="YOUR_API_KEY"
```

Windows:

```
set GOOGLE_API_KEY=YOUR_API_KEY
```

---

## 🖥 서버 실행

```
pip install -r requirements.txt
python app.py
```

EC2 접속:

```
http://<EC2-IP>:8000/analyze
```

---

## 🔧 FFmpeg 설치 (필수)

pydub으로 오디오를 추출하려면 반드시 필요:

```
sudo apt-get update
sudo apt-get install ffmpeg
```

---

## 🎯 Summary

- 영상 → 오디오 추출  
- Gemini AI → STT + 요약  
- Flask API로 프론트/앱 어디서든 사용 가능  
- AWS 배포 가능  

---
