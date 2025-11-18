# 📘 Face Guide Service  
### _FastAPI 기반 얼굴 위치 실시간 분석 서버 (AWS EC2 배포)_  

---

## 🌐 API Base URL (AWS EC2)

### ⭐ 현재 API 서버 주소
```
http://3.226.76.135:8000
```
```

---

## 📌 Overview

Face Guide Service는 Mediapipe FaceMesh를 사용하여  
이미지(프레임) 단위로 얼굴 위치를 분석하는 AI API 서버입니다.

서버는 아래 3개의 상태 중 하나를 JSON으로 반환합니다:

- `come_in` – 화면 안으로 들어오세요  
- `move_back` – 가까우니 뒤로 물러나세요  
- `perfect` – 적정 위치  

프론트엔드는 웹캠에서 캡처한 프레임을 EC2 서버로 보내 실시간처럼 사용합니다.

---

## 🧱 Project Structure

```
face-guide-service/
 ├─ app.py               # FastAPI 엔트리포인트
 ├─ model.py             # Mediapipe 얼굴 분석 로직
 ├─ requirements.txt     # Python dependency 목록
 ├─ Dockerfile           # Docker 이미지 빌드 파일
 ├─ nginx.conf           # (선택) reverse-proxy + SSL
 ├─ docker-compose.yml   # backend + nginx 구성
 └─ README.md
```

---

## ⚙️ Technology Stack

- Python 3.10+
- FastAPI
- Mediapipe FaceMesh
- OpenCV
- Docker / Nginx (optional)
- AWS EC2

---

# 🚀 API Endpoints

## ✔ POST /analyze  
이미지 파일 업로드 방식 (`multipart/form-data`)

### JavaScript Example
```js
const form = new FormData();
form.append("file", imageBlob);

fetch("http://3.226.76.135:8000/analyze", {
  method: "POST",
  body: form
})
  .then(res => res.json())
  .then(console.log);
```

### Response Example
```json
{
  "message": "완벽합니다!",
  "state": "perfect",
  "is_good": true
}
```

---

## ✔ POST /analyze_base64  
Base64 이미지 업로드 방식

### Request Example
```json
{
  "image": "<base64-string>"
}
```

---

## ✔ GET /health  
서버 동작 여부 확인

```
http://3.226.76.135:8000/health
```

---

# 🧠 Model Logic Summary

Face landmark 기반으로 아래 규칙 적용:

1. 얼굴 없음 → `come_in`  
2. 얼굴이 너무 큼 (bw or bh > 0.70) → `move_back`  
3. 얼굴의 절반 이상이 프레임 밖 → `come_in`  
4. 눈 위치가 너무 위쪽 (avg_eye_y < 0.15) → `come_in`  
5. 나머지 → `perfect`  

---

# 🖥 Local Development (Optional)

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Run FastAPI (port 8000)
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Swagger 문서:
```
http://localhost:8000/docs
```

---

# 🐳 Docker Deployment

### Build Image
```bash
docker build -t face-guide-backend .
```

### Run Container
```bash
docker run -d -p 8000:8000 face-guide-backend
```

---

# ☁️ AWS EC2 Deployment Guide

## ✔ 1) EC2 보안그룹 인바운드 규칙 설정

| Port | Protocol | Source |
|------|----------|--------|
| 8000 | TCP | 0.0.0.0/0 |
| 22 | TCP | Your IP |

## ✔ 2) FastAPI 실행 (서버 내부)
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

외부 접속:
```
http://3.226.76.135:8000/docs
```

---

# 🌐 docker-compose (Optional)

```
docker-compose up -d
```

서비스 구조:
```
Client → Nginx (80/443) → FastAPI (8000)
```
---

# 🎯 Summary

- API 서버: `http://3.226.76.135:8000`
- 프론트는 프레임을 반복 업로드하여 실시간처럼 사용
- EC2 보안그룹에서 8000 포트 Must Open!

