"""
웃는 얼굴 썸네일 + 요약/제목 생성 통합 AI 모듈
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
# 1. Mediapipe 1차 필터링 — 웃음 후보
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

    return variance > 40


# ============================================================
# 2. Vision API — Batch 얼굴 분석 (16장씩)
# ============================================================
def analyze_batch(frames):
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
                blur = LIKELIHOOD_SCORE.get(face.blurred_likelihood.name, 0)
                under = LIKELIHOOD_SCORE.get(face.under_exposed_likelihood.name, 0)
                joy = LIKELIHOOD_SCORE.get(face.joy_likelihood.name, 0)

                base_q = 0
                if blur < 3: base_q += 40
                if under < 3: base_q += 20
                if abs(face.roll_angle) < 20 and abs(face.pan_angle) < 20:
                    base_q += 20

                joy_score = joy / 5.0 * 300
                total_score += base_q + joy_score

            frame["score"] = total_score
            all_results.append(frame)

    return all_results


# ============================================================
# 3. 프레임 추출 + 1차 필터링
# ============================================================
def extract_candidate_frames(video_path, sec_interval=0.25):
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
                ok, buf = cv2.imencode(".jpg", frame)
                if ok:
                    frames.append({
                        "time_sec": frame_idx / fps,
                        "image_bytes": buf.tobytes(),
                        "image_cv2": frame,
                    })

        frame_idx += 1

    cap.release()
    print(f"⚡ 전체 {total}프레임 → 후보 {len(frames)}개")

    return frames


# ============================================================
# 4. 최종 썸네일
# ============================================================
def find_best_thumbnail(video_path):
    candidates = extract_candidate_frames(video_path)
    if len(candidates) == 0:
        return None

    if len(candidates) > 30:
        candidates = candidates[:30]

    scored = analyze_batch(candidates)
    scored.sort(key=lambda x: x["score"], reverse=True)
    best = scored[0]

    print(f"🎉 최종 썸네일 선택 (score={best['score']:.1f}, {best['time_sec']:.2f}s)")

    ok, buf = cv2.imencode(".jpg", best["image_cv2"])
    img_b64 = base64.b64encode(buf).decode("utf-8")

    return {
        "time_sec": best["time_sec"],
        "score": best["score"],
        "image_base64": img_b64,
    }


# ============================================================
# 5. STT — 요약 + 제목 생성 (안전 버전)
# ============================================================
def extract_audio(video_path, audio_path="temp_audio.mp3"):
    try:
        audio = AudioSegment.from_file(video_path)
        audio.export(audio_path, format="mp3")
        return audio_path
    except Exception as e:
        raise RuntimeError(f"Audio extraction failed: {e}")


def analyze_video_content(video_path, api_key):
    if not api_key or api_key.strip() == "":
        raise ValueError("유효한 Google API Key가 필요합니다.")

    genai.configure(api_key=api_key)

    audio_path = extract_audio(video_path)

    try:
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        model = genai.GenerativeModel("models/gemini-2.5-flash")

        prompt = """
JSON만 출력하세요. 절대 설명 금지.
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

        # 응답 text 자체가 없는 경우
        if not response.text:
            raise RuntimeError("Gemini 응답이 없습니다. (response.text is None)")

        clean = (
            response.text
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

        try:
            data = json.loads(clean)
        except Exception:
            print("❌ Gemini JSON 파싱 실패 — 응답 원본:")
            print(response.text)
            raise RuntimeError("Gemini JSON 형식 오류")

        return {
            "summary": data.get("summary", ""),
            "title": data.get("title", "")
        }

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
