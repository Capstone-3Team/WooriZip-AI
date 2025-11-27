"""
웃는 얼굴 썸네일 + 요약/제목 생성
세로 영상 자동 회전 보정 + FaceMesh + Flash Lite (최적화 버전)
"""

import os
import cv2
import base64
import json
import numpy as np
import subprocess
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


# ============================================================
# 1. ffprobe로 회전 정보 읽기
# ============================================================
def get_rotation(video_path):
    """
    ffprobe로 영상 metadata에서 회전 정보 읽기
    90 / 180 / 270 / 없으면 0
    """
    try:
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream_tags=rotate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        output = subprocess.check_output(cmd).decode().strip()
        return int(output) if output else 0
    except:
        return 0


def correct_rotation(frame, rotation):
    """
    회전 metadata에 따라 프레임 회전 보정
    """
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


# ============================================================
# 2. FaceMesh 기반 필터링
# ============================================================
mp_face_mesh = mp.solutions.face_mesh
mesh_detector = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

UPPER_LIP = 13
LOWER_LIP = 14
LEFT_MOUTH = 61
RIGHT_MOUTH = 291


def is_smile_candidate(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = mesh_detector.process(rgb)

    if not result.multi_face_landmarks:
        return False

    lm = result.multi_face_landmarks[0].landmark
    h, w, _ = frame.shape

    def pos(idx):
        return np.array([lm[idx].x * w, lm[idx].y * h])

    upper = pos(UPPER_LIP)
    lower = pos(LOWER_LIP)
    left = pos(LEFT_MOUTH)
    right = pos(RIGHT_MOUTH)

    # 1) 입 벌어짐
    lip_distance = np.linalg.norm(upper - lower)

    # 2) 입꼬리 곡률
    center = (upper + lower) / 2
    curvature = (center[1] - left[1]) + (center[1] - right[1])

    # 기본 smile score
    smile_score = curvature * 0.6 + lip_distance * 0.4

    # threshold 강화 제거 → 기본값만
    return smile_score > 6


# ============================================================
# 3. Vision API Batch 분석
# ============================================================
LIKELIHOOD_SCORE = {
    "UNKNOWN": 0,
    "VERY_UNLIKELY": 0,
    "UNLIKELY": 1,
    "POSSIBLE": 2,
    "LIKELY": 4,
    "VERY_LIKELY": 5,
}

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

                base = 0
                if blur_val < 3: base += 40
                if under_val < 3: base += 20
                if abs(face.roll_angle) < 20 and abs(face.pan_angle) < 20:
                    base += 20

                joy_score = joy_val / 5.0 * 300
                total_score += base + joy_score

            frame["score"] = total_score
            all_results.append(frame)

    return all_results


# ============================================================
# 4. 회전 보정 포함 후보 프레임 추출
# ============================================================
def extract_candidate_frames(video_path, sec_interval=0.35):
    rotation = get_rotation(video_path)

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

        # ⭐ 회전 보정
        frame = correct_rotation(frame, rotation)

        total += 1
        if frame_idx % step == 0:
            if is_smile_candidate(frame):
                ok, buffer = cv2.imencode(".jpg", frame)
                if ok:
                    frames.append({
                        "time_sec": frame_idx / fps,
                        "image_cv2": frame,
                        "image_bytes": buffer.tobytes(),
                    })

        frame_idx += 1

    cap.release()
    print(f"⚡ 전체 {total}프레임 → 후보 {len(frames)}개")
    return frames


# ============================================================
# 5. 최종 썸네일
# ============================================================
def find_best_thumbnail(video_path):
    candidates = extract_candidate_frames(video_path)

    if len(candidates) == 0:
        return None

    if len(candidates) > 12:
        candidates = candidates[:12]

    scored = analyze_batch(candidates)
    scored.sort(key=lambda x: x["score"], reverse=True)
    best = scored[0]

    ok, buffer = cv2.imencode(".jpg", best["image_cv2"])
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    print(f"🎉 최종 썸네일 (score={best['score']:.1f}, time={best['time_sec']:.2f}s)")

    return {
        "time_sec": best["time_sec"],
        "score": best["score"],
        "image_base64": img_base64,
    }


# ============================================================
# 6. 오디오 → 1.2x → Gemini Flash Lite
# ============================================================
def extract_audio(video_path, audio_path="temp_audio.mp3"):
    try:
        audio = AudioSegment.from_file(video_path)

        # 오디오 길이 감소 (1.2x)
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
