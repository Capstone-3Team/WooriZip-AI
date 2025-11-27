"""
웃는 얼굴 썸네일 + 요약/제목 생성 (초고속 최적화 버전)
"""

import os
import cv2
import base64
import json
import numpy as np
from pydub import AudioSegment
from pydub.effects import speedup
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
mp_facedetector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.50)

# Likelihood 매핑
LIKELIHOOD_SCORE = {
    "UNKNOWN": 0,
    "VERY_UNLIKELY": 0,
    "UNLIKELY": 1,
    "POSSIBLE": 2,
    "LIKELY": 4,
    "VERY_LIKELY": 5,
}


# ============================================================
# 1. Mediapipe 후보 필터링 (강화 버전)
# ============================================================
def is_smile_candidate(frame):
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

    # ⭐ NEW: percentile 기반 threshold (필터 더 강함)
    threshold = np.percentile(gray, 75)

    return variance > threshold


# ============================================================
# 2. Vision API Batch 분석 (16개 단위)
# ============================================================
def analyze_batch(frames):
    MAX_BATCH = 16
    all_results = []

    for i in range(0, len(frames), MAX_BATCH):
        chunk = frames[i:i + MAX_BATCH]

        requests = [
            vision.AnnotateImageRequest(
                image=vision.Image(content=f["image_bytes"]),
                features=[vision.Feature(type_=vision.Feature.Type.FACE_DETECTION)]
            )
            for f in chunk
        ]

        response = vision_client.batch_annotate_images(requests=requests)

        for frame, res in zip(chunk, response.responses):
            faces = res.face_annotations
            if not faces:
                frame["score"] = 0
                all_results.append(frame)
                continue

            total_score = 0

            for face in faces:
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
# 3. 빠른 후보 프레임 추출
# ============================================================
def extract_candidate_frames(video_path, sec_interval=0.35):
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

    print(f"⚡ 전체 {total}프레임 → 후보 {len(frames)}개")
    return frames


# ============================================================
# 4. 최종 썸네일 선택
# ============================================================
def find_best_thumbnail(video_path):
    candidates = extract_candidate_frames(video_path)

    if len(candidates) == 0:
        return None

    # 비용절감: 16장까지만 Vision 호출
    if len(candidates) > 16:
        candidates = candidates[:16]

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
# 5. 요약 + 제목 생성 (Flash Lite + 1.2x 오디오)
# ============================================================
def extract_audio(video_path, audio_path="temp_audio.mp3"):
    try:
        audio = AudioSegment.from_file(video_path)

        # ⭐ NEW: 1.2x 속도 증가 (pitch 변화 거의 없음)
        audio = speedup(audio, playback_speed=1.2, chunk_size=60, crossfade=40)

        audio.export(audio_path, format="mp3")
        return audio_path
    except Exception as e:
        raise RuntimeError(f"Audio extraction failed: {e}")


def analyze_video_content(video_path, api_key):
    if not api_key:
        raise ValueError("유효한 Google API Key 필요")

    genai.configure(api_key=api_key)

    audio_file_path = extract_audio(video_path)

    try:
        with open(audio_file_path, "rb") as f:
            audio_bytes = f.read()

        model = genai.GenerativeModel("models/gemini-2.5-flash-lite")

        prompt = """
        이 오디오 내용을 한국어로 한 문장 요약하고,
        영상의 주제를 반영한 간결한 제목을 생성하세요.
        반드시 JSON:
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

        clean = response.text.strip().lstrip("```json").rstrip("```").strip()
        result = json.loads(clean)

        return {
            "summary": result.get("summary", ""),
            "title": result.get("title", "")
        }

    except Exception as e:
        raise RuntimeError(f"Gemini 분석 오류: {e}")

    finally:
        if os.path.exists(audio_file_path):
            os.remove(audio_file_path)
