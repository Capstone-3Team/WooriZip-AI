# model.py
"""
웃는 얼굴 썸네일 + 요약/제목 생성 통합 AI 모듈

- find_best_thumbnail(video_path): Google Cloud Vision 기반 웃는 얼굴 썸네일 선정
- analyze_video_content(video_path, api_key): Gemini 2.5 Flash 기반 내용 요약 + 제목 생성
"""

import os
import cv2
import base64
import json
import numpy as np
from pydub import AudioSegment
from google.cloud import vision
import google.generativeai as genai
import mediapipe as mp

# ============================================================
# 0. Vision API 초기화
# ============================================================
if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service-account.json"

vision_client = vision.ImageAnnotatorClient()

# Mediapipe 초기화
mp_face = mp.solutions.face_detection
mp_facedetector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.45)

# Likelihood 매핑 (문자열 기준)
LIKELIHOOD_SCORE = {
    "UNKNOWN": 0,
    "VERY_UNLIKELY": 0,
    "UNLIKELY": 1,
    "POSSIBLE": 2,
    "LIKELY": 4,
    "VERY_LIKELY": 5,
}


# ============================================================
# 1. Mediapipe 1차 필터링 (빠른 얼굴/웃음 후보 탐지)
# ============================================================
def is_smile_candidate(frame):
    """
    Mediapipe로 빠르게 웃을 가능성 있는 프레임인지 판단
    - 얼굴이 있는지
    - 입 주변 변화량(variance)으로 대략적인 표정 변화 체크
    """
    results = mp_facedetector.process(frame)
    if not results.detections:
        return False

    det = results.detections[0]

    box = det.location_data.relative_bounding_box
    h, w, _ = frame.shape
    x1, y1 = int(box.xmin * w), int(box.ymin * h)
    x2, y2 = x1 + int(box.width * w), y1 + int(box.height * h)

    face_roi = frame[y1:y2, x1:x2]
    if face_roi.size == 0:
        return False

    roi_h = face_roi.shape[0]
    mouth_region = face_roi[int(roi_h * 0.55): int(roi_h * 0.85), :]

    if mouth_region.size == 0:
        return False

    gray = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY)
    variance = gray.var()

    return variance > 40  # 경험적 threshold


# ============================================================
# 2. Vision API Batch 얼굴 분석 (정확한 감정/웃음 판별)
#    - 요청당 최대 16장 제한 때문에 chunk 처리
#    - Likelihood ENUM → .name 으로 변환해서 점수 매핑
# ============================================================
def analyze_batch(frames):
    """
    frames: [{ "time_sec", "image_bytes", "image_cv2" }, ...]
    각 frame에 "score" 필드를 추가해서 반환
    """
    MAX_BATCH = 16
    all_results = []

    for i in range(0, len(frames), MAX_BATCH):
        chunk = frames[i:i + MAX_BATCH]

        requests = []
        for f in chunk:
            image = vision.Image(content=f["image_bytes"])
            req = vision.AnnotateImageRequest(
                image=image,
                features=[vision.Feature(type_=vision.Feature.Type.FACE_DETECTION)]
            )
            requests.append(req)

        response = vision_client.batch_annotate_images(requests=requests)

        for frame, res in zip(chunk, response.responses):
            faces = res.face_annotations

            if not faces:
                frame["score"] = 0
                all_results.append(frame)
                continue

            total_score = 0
            for face in faces:
                # ENUM → 문자열(.name) 변환 후 매핑
                blur_val = LIKELIHOOD_SCORE.get(face.blurred_likelihood.name, 0)
                under_val = LIKELIHOOD_SCORE.get(face.under_exposed_likelihood.name, 0)
                joy_val = LIKELIHOOD_SCORE.get(face.joy_likelihood.name, 0)

                base_quality = 0
                if blur_val < 3:
                    base_quality += 40
                if under_val < 3:
                    base_quality += 20
                if abs(face.roll_angle) < 20 and abs(face.pan_angle) < 20:
                    base_quality += 20

                joy_score = joy_val / 5.0 * 300
                total_score += base_quality + joy_score

            frame["score"] = total_score
            all_results.append(frame)

    return all_results


# ============================================================
# 3. 프레임 추출 + 1차 필터링
# ============================================================
def extract_candidate_frames(video_path, sec_interval=0.25):
    """
    모든 프레임을 보지 않고,
    sec_interval 간격으로 샘플링 + Mediapipe로 웃는 후보만 남김
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(fps * sec_interval), 1)

    frames = []
    frame_idx = 0
    total = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total += 1

        if frame_idx % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if is_smile_candidate(rgb):
                ok, buffer = cv2.imencode(".jpg", frame)
                if ok:
                    frames.append({
                        "time_sec": frame_idx / fps,
                        "image_bytes": buffer.tobytes(),
                        "image_cv2": frame,
                    })

        frame_idx += 1

    cap.release()

    if len(frames) == 0:
        print(f"😢 전체 프레임 {total}개 중 웃는 후보 없음")
    else:
        print(f"⚡ 전체 프레임 {total} → 후보 {len(frames)}개")

    return frames


# ============================================================
# 4. 메인: 최종 썸네일 찾기
# ============================================================
def find_best_thumbnail(video_path):
    candidates = extract_candidate_frames(video_path)

    if len(candidates) == 0:
        return None

    # 비용 절약용 상한선
    if len(candidates) > 30:
        candidates = candidates[:30]

    scored = analyze_batch(candidates)
    scored.sort(key=lambda x: x["score"], reverse=True)
    best = scored[0]

    print(f"🎉 최종 썸네일 (score={best['score']:.1f}, time={best['time_sec']:.2f}s)")

    ok, buffer = cv2.imencode(".jpg", best["image_cv2"])
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "time_sec": best["time_sec"],
        "score": best["score"],
        "image_base64": img_base64,
    }


# ============================================================
# 5. 비디오 → 오디오 추출
# ============================================================
def extract_audio(video_path, audio_path="temp_audio.mp3"):
    """
    비디오 파일에서 오디오를 추출하여 mp3로 저장
    """
    try:
        video = AudioSegment.from_file(video_path)
        video.export(audio_path, format="mp3")
        return audio_path
    except Exception as e:
        raise RuntimeError(f"Audio extraction failed: {e}")


# ============================================================
# 6. Gemini 2.5 Flash: 요약 + 제목 생성
# ============================================================
def analyze_video_content(video_path, api_key):
    """
    비디오 → 오디오 추출 → Gemini Flash로 요약 + 제목 생성
    transcript(전체 STT)는 포함하지 않고 summary/title만 반환
    """
    if api_key is None or api_key.strip() == "":
        raise ValueError("유효한 Google API Key가 필요합니다.")

    genai.configure(api_key=api_key)

    audio_file_path = extract_audio(video_path)

    try:
        with open(audio_file_path, "rb") as f:
            audio_bytes = f.read()

        model = genai.GenerativeModel("models/gemini-2.5-flash")

        prompt = """
        이 오디오 내용을 한국어로 한 문장으로 요약하고,
        영상의 주제를 반영한 간결한 제목을 생성하세요.

        반드시 아래 JSON 형식으로만 응답하세요:
        {
          "summary": "...",
          "title": "..."
        }
        """

        response = model.generate_content(
            [
                {"mime_type": "audio/mpeg", "data": audio_bytes},
                prompt
            ]
        )

        text = response.text.strip()
        clean = text.lstrip("```json").lstrip("```").rstrip("```").strip()
        data = json.loads(clean)

        summary = data.get("summary", "")
        title = data.get("title", "")

        return {
            "summary": summary,
            "title": title
        }

    except Exception as e:
        raise RuntimeError(f"Gemini 분석 오류: {e}")

    finally:
        if os.path.exists(audio_file_path):
            os.remove(audio_file_path)
