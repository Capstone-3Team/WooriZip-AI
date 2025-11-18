# WooriZip-AI
FastAPI 기반 얼굴 위치 실시간 분석 서버 (AWS EC2: 3.226.76.135:8000)
📌 Overview
Face Guide Service는 Mediapipe FaceMesh 기반 얼굴 위치 분석 API입니다.
웹캠 또는 앱에서 캡처한 이미지(프레임)를 서버로 보내면,
come_in
move_back
perfect
세 가지 상태 중 하나를 반환하며,
신분증/얼굴 촬영 UX 개선, 촬영 가이드, 실시간 얼굴 안내 등에서 활용할 수 있습니다.
🧱 Project Structure
face-guide-service/
 ├─ app.py               # FastAPI 엔트리포인트
 ├─ model.py             # Mediapipe 얼굴 분석 로직
 ├─ requirements.txt     # Python dependencies
 ├─ Dockerfile           # Docker 이미지 빌드
 ├─ nginx.conf           # (옵션) HTTPS + Reverse Proxy
 ├─ docker-compose.yml   # backend + nginx
 └─ README.md
🌐 API Base URL (AWS EC2 서버)
⭐ 현재 배포된 서버 URL:
http://3.226.76.135:8000
⭐ Swagger UI (API 문서):
http://3.226.76.135:8000/docs
👉 여기 접속해서 프론트 없이 바로 테스트 가능함.
⚙️ How It Works (System Architecture)
프론트엔드 또는 앱:
웹캠에서 현재 프레임을 캡처
서버(3.226.76.135:8000)에 이미지 업로드
서버가 얼굴 분석
JSON 형태로 상태 반환
프론트에서 UI(자막/안내음성 등) 업데이트
즉, 실시간 분석도 프레임 단위 이미지 업로드 반복으로 구현됨.
🧪 API Endpoints
1) POST /analyze
이미지 파일 업로드 방식 (multipart/form-data)
Request (JavaScript 예시)
const form = new FormData();
form.append("file", imageBlob);  // webcam frame blob

fetch("http://3.226.76.135:8000/analyze", {
  method: "POST",
  body: form
})
  .then(res => res.json())
  .then(console.log);
Response
{
  "message": "완벽합니다!",
  "state": "perfect",
  "is_good": true
}
2) POST /analyze_base64
Base64 문자열 업로드 방식
Request
{
  "image": "<base64-encoded-frame>"
}
3) GET /health
서버 헬스 체크.
http://3.226.76.135:8000/health
🚀 Local Development (선택)
AWS EC2에 올리기 전에 로컬에서 실행할 수도 있음.
1) Install
pip install -r requirements.txt
2) Run
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
🌐 Deployment on AWS EC2 (현재 배포된 환경 기준)
EC2 OS: Ubuntu 가정
🟦 1) 필수: 보안그룹 인바운드 규칙
아래가 열려 있어야 서버 접속 가능함:
Port	Protocol	Source
8000	TCP	0.0.0.0/0
22 (SSH)	TCP	내 IP
🟦 2) 서버 실행
EC2 안에서:
uvicorn app:app --host 0.0.0.0 --port 8000
로그:
Uvicorn running on http://0.0.0.0:8000
🐳 Docker Deployment (Optional)
Build
docker build -t face-guide-backend .
Run
docker run -d -p 8000:8000 face-guide-backend
📘 Model Logic (요약)
얼굴 없음 → come_in
얼굴이 너무 큼(bw/bh > 0.7) → move_back
얼굴의 절반 이상이 프레임 밖 → come_in
눈 위치가 너무 위쪽 (avg_eye_y < 0.15) → come_in
나머지 → perfect
👥 Who Should Read This?
프론트엔드
이미지 전송 방식
실시간 분석 호출 구조
백엔드
배포(EC2), 포트, API 구조
팀 전체
서비스 전체 흐름 이해
🎯 Summary
FastAPI 서버 주소는 http://3.226.76.135:8000
Swagger UI는 http://3.226.76.135:8000/docs
프론트는 이미지(blob/base64)를 서버로 보내 분석 결과를 받는다
실시간 분석도 결국 “프레임 반복 호출”
AWS EC2 보안그룹에 반드시 포트 8000 열어야 한다
