# model.py
"""
웃는 얼굴 썸네일 + STT/요약/제목 생성 통합 AI 모듈

- find_best_thumbnail(video_path): Google Cloud Vision 기반 웃는 얼굴 썸네일 선정
- analyze_video_content(video_path, api_key): Gemini 2.5 Flash 기반 STT + 요약 + 제목 생성
"""

import os
import cv2
import base64
import numpy as np
from google.cloud import vision
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


# ============================================================
# 1. Mediapipe 1차 필터링 (빠른 얼굴/웃음 후보 탐지)
# ============================================================

def is_smile_candidate(frame):
    """
    Mediapipe로 빠르게 웃을 가능성 있는 프레임인지 판단
    - 입이 크게 벌어졌는지
    - 입꼬리가 올라갔는지
    """

    results = mp_facedetector.process(frame)
    if not results.detections:
        return False  # 얼굴 없음 → 제거

    det = results.detections[0]

    # Bounding box
    box = det.location_data.relative_bounding_box
    h, w, _ = frame.shape
    x1, y1 = int(box.xmin * w), int(box.ymin * h)
    x2, y2 = x1 + int(box.width * w), y1 + int(box.height * h)

    face_roi = frame[y1:y2, x1:x2]
    if face_roi.size == 0:
        return False

    # 단순 입색역(하단 40%)에서 입 벌어짐 체크 → 매우 빠름
    roi_h = face_roi.shape[0]
    mouth_region = face_roi[int(roi_h*0.55): int(roi_h*0.85), :]

    if mouth_region.size == 0:
        return False

    # 입 주변 대비 증가 → 입 벌렸을 확률 ↑
    gray = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY)
    variance = gray.var()  # 표정 변화(입 모양 변화)로 variance가 증가함

    return variance > 40   # 경험적 threshold (조절 가능)


# ============================================================
# 2. Vision API Batch 얼굴 분석 (정확한 감정/웃음 판별)
# ============================================================

LIKELIHOOD_SCORE = {
    "UNKNOWN": 0, "VERY_UNLIKELY": 0, "UNLIKELY": 1,
    "POSSIBLE": 2, "LIKELY": 4, "VERY_LIKELY": 5
}

def analyze_batch(frames):
    """
    Vision API BatchAnnotateImages로 여러 프레임을 한번에 처리
    """

    requests = []
    for f in frames:
        image = vision.Image(content=f["image_bytes"])
        requests.append(vision.AnnotateImageRequest(image=image, features=[
            vision.Feature(type_=vision.Feature.Type.FACE_DETECTION)
        ]))

    response = vision_client.batch_annotate_images(requests=requests)

    results = []
    for frame, res in zip(frames, response.responses):
        faces = res.face_annotations

        if not faces:
            frame["score"] = 0
            results.append(frame)
            continue

        total_score = 0

        for face in faces:
            base_quality = 0
            if LIKELIHOOD_SCORE.get(face.blurred_likelihood, 0) < 3: base_quality += 40
            if LIKELIHOOD_SCORE.get(face.under_exposed_likelihood, 0) < 3: base_quality += 20
            if abs(face.roll_angle) < 20 and abs(face.pan_angle) < 20: base_quality += 20

            joy_score = LIKELIHOOD_SCORE.get(face.joy_likelihood, 0) / 5.0 * 300

            total_score += base_quality + joy_score

        frame["score"] = total_score
        results.append(frame)

    return results


# ============================================================
# 3. 프레임 추출 + 1차 필터링
# ============================================================

def extract_candidate_frames(video_path, sec_interval=0.25):
    """
    모든 프레임을 추출하지만,
    Mediapipe로 ‘웃음 후보’만 반환 (80~95% 제거됨)
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = int(fps * sec_interval)

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

    print(f"⚡ 전체 프레임: {total} → 후보 프레임: {len(frames)}개 (속도 {total/len(frames):.1f}배 향상 예상)")
    return frames


# ============================================================
# 4. 메인: 최종 썸네일 찾기
# ============================================================

def find_best_thumbnail(video_path):
    # --- 1차 필터링 ---
    candidates = extract_candidate_frames(video_path)

    if len(candidates) == 0:
        print("😢 웃는 얼굴 후보 없음. 영상 중간 썸네일 반환")
        return None

    # 너무 많으면 30장만 Vision API로 분석
    if len(candidates) > 30:
        candidates = candidates[:30]

    # --- Vision API batch 분석 ---
    scored = analyze_batch(candidates)

    # 최고 점수 프레임 선택
    scored.sort(key=lambda x: x["score"], reverse=True)
    best = scored[0]

    print(f"🎉 최종 썸네일 결정! (score={best['score']:.1f}, time={best['time_sec']:.2f}s)")

    ok, buffer = cv2.imencode(".jpg", best["image_cv2"])
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "time_sec": best["time_sec"],
        "score": best["score"],
        "image_base64": img_base64,
    }


import os
import json
from pydub import AudioSegment
import google.generativeai as genai

# =======================================
# 2. 요약 + 제목 생성 (Gemini 2.5 Flash)
# =======================================

def extract_audio(video_path, audio_path="temp_audio.mp3"):
    """
    비디오에서 오디오만 MP3로 추출
    """
    try:
        video = AudioSegment.from_file(video_path)
        video.export(audio_path, format="mp3")
        return audio_path
    except Exception as e:
        raise RuntimeError(f"Audio extraction failed: {e}")


def analyze_video_content(video_path, api_key):
    """
    비디오 → 오디오 추출 → Gemini로 요약 + 제목 생성
    (upload_file 제거 / 바이너리 직접 입력)
    """
    if api_key is None or api_key.strip() == "":
        raise ValueError("유효한 Google API Key가 필요합니다.")

    # Gemini API 설정
    genai.configure(api_key=api_key)

    # 1. 오디오 파일 추출
    audio_file_path = extract_audio(video_path)

    try:
        # 2. 오디오 바이너리 직접 읽기
        with open(audio_file_path, "rb") as f:
            audio_bytes = f.read()

        # 3. 빠른 모델
        model = genai.GenerativeModel("models/gemini-2.5-flash")

        # 4. 간결하고 속도 빠른 프롬프트
        prompt = """
        이 오디오 내용을 한국어로 한 문장 요약하고,
        영상의 주제를 반영한 간결한 제목을 생성하세요.

        반드시 JSON:
        {
          "summary": "...",
          "title": "..."
        }
        """

        # 5. 파일 업로드 대신 바이너리 직접 전달
        response = model.generate_content(
            [ 
                {"mime_type": "audio/mpeg", "data": audio_bytes}, 
                prompt
            ]
        )

        clean_text = response.text.strip().lstrip("```json").rstrip("```").strip()
        results = json.loads(clean_text)

        summary = results.get("summary", "")
        title = results.get("title", "")

    except Exception as e:
        raise RuntimeError(f"Gemini 분석 오류: {e}")

    finally:
        # 임시 오디오 삭제
        if os.path.exists(audio_file_path):
            os.remove(audio_file_path)

    return {
        "summary": summary,
        "title": title
    }
