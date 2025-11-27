# model.py
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
# Google Vision API 초기화
# ============================================================
if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service-account.json"

vision_client = vision.ImageAnnotatorClient()

mp_face = mp.solutions.face_detection
mp_facedetector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.45)


# ============================================================
# 1. Mediapipe 1차 후보 필터링
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
# 2. Vision API Batch 처리
# ============================================================
LIKELIHOOD_SCORE = {
    "UNKNOWN": 0, "VERY_UNLIKELY": 0, "UNLIKELY": 1,
    "POSSIBLE": 2, "LIKELY": 4, "VERY_LIKELY": 5
}

def analyze_batch(frames):
    """
    Vision API는 요청당 최대 16장의 이미지만 허용.
    따라서 16개씩 쪼개서 여러 번 호출한 뒤 결과를 합쳐야 한다.
    """
    MAX_BATCH = 16
    all_results = []

    # 16장씩 chunking
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

        # chunk 결과 처리
        for frame, res in zip(chunk, response.responses):
            faces = res.face_annotations

            if not faces:
                frame["score"] = 0
                all_results.append(frame)
                continue

            total_score = 0
            for face in faces:
                base_quality = 0
                if LIKELIHOOD_SCORE[face.blurred_likelihood] < 3:
                    base_quality += 40
                if LIKELIHOOD_SCORE[face.under_exposed_likelihood] < 3:
                    base_quality += 20
                if abs(face.roll_angle) < 20 and abs(face.pan_angle) < 20:
                    base_quality += 20

                joy_score = LIKELIHOOD_SCORE[face.joy_likelihood] / 5 * 300
                total_score += base_quality + joy_score

            frame["score"] = total_score
            all_results.append(frame)

    return all_results



# ============================================================
# 3. 후보 프레임 추출
# ============================================================
def extract_candidate_frames(video_path, sec_interval=0.25):
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
    return frames


# ============================================================
# 4. BEST 썸네일 찾기
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

    ok, buffer = cv2.imencode(".jpg", best["image_cv2"])
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "time_sec": best["time_sec"],
        "score": best["score"],
        "image_base64": img_base64
    }


# ============================================================
# 5. 비디오→오디오 추출
# ============================================================
def extract_audio(video_path, audio_path="temp_audio.mp3"):
    try:
        video = AudioSegment.from_file(video_path)
        video.export(audio_path, format="mp3")
        return audio_path
    except Exception as e:
        raise RuntimeError(f"Audio extraction failed: {e}")


# ============================================================
# 6. Gemini 요약 + 제목 생성 (정식 포맷)
# ============================================================
def analyze_video_content(video_path, api_key):
    if api_key is None or api_key.strip() == "":
        raise ValueError("유효한 Google API Key가 필요합니다.")

    genai.configure(api_key=api_key)

    audio_file_path = extract_audio(video_path)

    try:
        with open(audio_file_path, "rb") as f:
            audio_bytes = f.read()

        model = genai.GenerativeModel("models/gemini-2.5-flash")

        prompt = """
        이 오디오 내용을 한국어로 한 문장 요약하고,
        영상의 주제를 반영한 간결한 제목을 생성하세요.

        JSON ONLY:
        {
          "summary": "...",
          "title": "..."
        }
        """

        response = model.generate_content(
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {"mime_type": "audio/mpeg", "data": audio_bytes},
                        {"text": prompt}
                    ]
                }
            ]
        )

        clean = response.text.strip().lstrip("```json").rstrip("```").strip()
        data = json.loads(clean)

        return {
            "summary": data.get("summary", ""),
            "title": data.get("title", "")
        }

    finally:
        if os.path.exists(audio_file_path):
            os.remove(audio_file_path)
