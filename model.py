# model.py
"""
웃는 얼굴 썸네일 + STT/요약/제목 생성 통합 AI 모듈

- find_best_thumbnail(video_path): Google Cloud Vision 기반 웃는 얼굴 썸네일 선정
- analyze_video_content(video_path, api_key): Gemini 2.5 Flash 기반 STT + 요약 + 제목 생성
"""

import os
import json
import base64

import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from google.cloud import vision
import google.generativeai as genai
from pydub import AudioSegment


# =======================================
# 0. Google Vision 클라이언트 초기화
# =======================================

try:
    # 환경변수에 서비스 계정 키 경로가 없으면 기본 파일명 사용
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service-account.json"

    vision_client = vision.ImageAnnotatorClient()
    print("✅ (model.py) Vision AI 클라이언트가 초기화되었습니다.")
except Exception as e:
    print(f"❌ (model.py) Vision AI 클라이언트 초기화 오류: {e}")
    print("   'service-account.json' 파일이 올바른지 확인하세요.")


# Vision API의 likelihood 값을 정량화한 스코어 맵
LIKELIHOOD_SCORE = {
    "UNKNOWN": 0,
    "VERY_UNLIKELY": 0,
    "UNLIKELY": 1,
    "POSSIBLE": 2,
    "LIKELY": 4,
    "VERY_LIKELY": 5,
}


# =======================================
# 1. 웃는 얼굴 썸네일 분석 로직
# =======================================

def _analyze_frame_for_thumbnail(image_bytes):
    """
    한 프레임(이미지)에 대해:
    - 여러 얼굴이 등장하면 각 얼굴의 점수를 합산하여
    - '프레임 전체 점수'를 반환
    """
    image = vision.Image(content=image_bytes)
    response = vision_client.face_detection(image=image)
    faces = response.face_annotations

    if not faces:
        return 0, "얼굴 없음", "N/A"

    total_score = 0
    mouth_info_summary = []

    for face in faces:
        # -------- 1. 기본 품질 점수 --------
        base_quality_score = 0

        # 흐림 정도가 심하지 않으면 가산점
        if LIKELIHOOD_SCORE.get(face.blurred_likelihood, 0) < 3:
            base_quality_score += 50
        # 노출 부족이 심하지 않으면 가산점
        if LIKELIHOOD_SCORE.get(face.under_exposed_likelihood, 0) < 3:
            base_quality_score += 20
        # 기울기(roll/pan)가 심하지 않으면 가산점
        if abs(face.roll_angle) < 20 and abs(face.pan_angle) < 20:
            base_quality_score += 30
        # 얼굴 검출 신뢰도가 높은 경우 가산점
        if face.detection_confidence > 0.7:
            base_quality_score += 20

        # -------- 2. 감정(웃음) 점수 --------
        api_score_norm = LIKELIHOOD_SCORE.get(face.joy_likelihood, 0) / 5.0

        landmarks = {lm.type_: lm.position for lm in face.landmarks}
        required_lm = [
            vision.FaceAnnotation.Landmark.Type.UPPER_LIP,
            vision.FaceAnnotation.Landmark.Type.LOWER_LIP,
            vision.FaceAnnotation.Landmark.Type.MOUTH_CENTER,
            vision.FaceAnnotation.Landmark.Type.MOUTH_LEFT,
            vision.FaceAnnotation.Landmark.Type.MOUTH_RIGHT,
        ]

        if not all(lm_type in landmarks for lm_type in required_lm):
            landmark_score_norm = 0.0
            mouth_info = f"Joy:{api_score_norm * 5:.0f}, Landmark:FAIL"
        else:
            # 입 벌어진 정도
            lip_distance = abs(
                landmarks[vision.FaceAnnotation.Landmark.Type.UPPER_LIP].y
                - landmarks[vision.FaceAnnotation.Landmark.Type.LOWER_LIP].y
            )

            # 입꼬리 올라간 정도 (중앙 - 좌/우 높이 차이)
            center_y = landmarks[vision.FaceAnnotation.Landmark.Type.MOUTH_CENTER].y
            left_y = landmarks[vision.FaceAnnotation.Landmark.Type.MOUTH_LEFT].y
            right_y = landmarks[vision.FaceAnnotation.Landmark.Type.MOUTH_RIGHT].y

            curvature = (center_y - left_y) + (center_y - right_y)

            curvature_norm = np.clip(curvature / 15.0, 0, 1)
            lip_norm = np.clip(lip_distance / 10.0, 0, 1)

            landmark_score_norm = curvature_norm * 0.7 + lip_norm * 0.3
            mouth_info = (
                f"Joy:{api_score_norm * 5:.0f}, "
                f"Pull:{curvature:.2f}, Open:{lip_distance:.2f}"
            )

        # -------- 3. 감정 종합 점수 --------
        emotion_norm = api_score_norm * 0.8 + landmark_score_norm * 0.2
        emotion_score = emotion_norm * 300

        face_score = base_quality_score + emotion_score
        total_score += face_score
        mouth_info_summary.append(mouth_info)

    return total_score, "여러 얼굴", "; ".join(mouth_info_summary)


def _extract_frames_by_interval(video_path, sec_per_frame=0.25):
    """
    비디오 파일에서 일정 간격(sec_per_frame)으로 프레임을 추출
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"오류: 비디오 파일을 열 수 없습니다: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * sec_per_frame)
    if frame_interval == 0:
        frame_interval = 1

    frames_data = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            ret_enc, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if ret_enc:
                current_time_sec = frame_count / fps
                frames_data.append(
                    {
                        "time_sec": current_time_sec,
                        "image_bytes": buffer.tobytes(),
                        "image_cv2": frame,
                    }
                )

        frame_count += 1

    cap.release()
    print(f"✅ 총 {len(frames_data)}개의 프레임이 추출되었습니다.")
    return frames_data


def _process_frame(frame_data):
    """
    병렬 처리를 위한 프레임 분석 래퍼
    """
    try:
        score, status, mouth_info_str = _analyze_frame_for_thumbnail(
            frame_data["image_bytes"]
        )
        frame_data["score"] = score
        frame_data["status"] = status
        frame_data["mouth"] = mouth_info_str
        return frame_data
    except Exception as e:
        print(f"Frame 처리 중 오류: {e}")
        return None


def find_best_thumbnail(video_path):
    """
    비디오 파일을 분석하여 가장 좋은 썸네일을 반환하는 함수 (병렬 처리)
    반환값:
    {
        "time_sec": <초 단위 프레임 위치>,
        "score": <썸네일 점수>,
        "image_base64": <JPG 이미지의 base64 문자열>
    }
    """
    frames = _extract_frames_by_interval(video_path, sec_per_frame=0.25)
    if not frames:
        return None

    print(f"\nGoogle Vision AI 병렬 분석 시작 (총 {len(frames)}개)...\n")

    scored_frames = []

    max_workers = min(10, len(frames))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_process_frame, f) for f in frames]

        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result:
                scored_frames.append(result)

            if i % 10 == 0:
                print(f"  진행률: {i + 1}/{len(frames)} 프레임 완료")

    if not scored_frames:
        return None

    scored_frames.sort(key=lambda x: x["score"], reverse=True)
    best_thumbnail = scored_frames[0]

    # 점수가 0 이하이면, 의미 있는 얼굴이 없다 판단하고 중간 프레임 반환
    if best_thumbnail["score"] <= 0:
        print("결과: 🌄 유의미한 얼굴 없음. 50% 지점 프레임 반환")
        best_thumbnail = frames[len(frames) // 2]
    else:
        print(f"결과: 😃 베스트 썸네일 선정! (점수: {int(best_thumbnail['score'])})")

    ret, buffer = cv2.imencode(".jpg", best_thumbnail["image_cv2"])
    if not ret:
        return None

    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "time_sec": best_thumbnail["time_sec"],
        "score": best_thumbnail["score"],
        "image_base64": img_base64,
    }


# =======================================
# 2. STT + 요약 + 제목 생성 (Gemini 2.5 Flash)
# =======================================

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


def analyze_video_content(video_path, api_key):
    """
    비디오 → 오디오 추출 → Gemini STT + 요약 + 제목 생성
    transcript / summary / title 반환
    """
    if api_key is None or api_key.strip() == "" or api_key == "YOUR_API_KEY_HERE":
        raise ValueError("유효한 Google API Key가 필요합니다.")

    # Gemini API 키 설정
    genai.configure(api_key=api_key)

    # 1. 오디오 파일 생성
    audio_file_path = extract_audio(video_path)

    # 2. Gemini 파일 업로드
    try:
        audio_file = genai.upload_file(path=audio_file_path)
    except Exception as e:
        raise RuntimeError(f"Gemini audio upload failed: {e}")

    # 3. Gemini 모델
    model = genai.GenerativeModel("models/gemini-2.5-flash-preview-09-2025")

    # 4. 프롬프트
    prompt = """
    이 오디오 파일은 가족 일기입니다. 다음 작업을 수행하세요:

    1. [STT]: 오디오의 내용을 한국어 텍스트로 모두 받아쓰기
    2. [요약]: 중요한 내용만 한 문장으로 요약
       (대화체 말투 금지 — 사실 기반 요약)
    3. [제목]: 이 영상의 주요 주제를 반영한 매우 간결한 제목 생성
       (예: “오늘의 가족 여행”, “아이의 학교 생활 이야기” 같은 형식)

    반드시 JSON 형태로만 응답:
    {
      "transcript": "...",
      "summary": "...",
      "title": "..."
    }
    """

    try:
        response = model.generate_content([audio_file, prompt])
        text = response.text.strip()

        # ```json ... ``` 형식으로 오는 경우를 대비해 양쪽 정리
        clean_text = text.lstrip("```json").rstrip("```").strip()
        results = json.loads(clean_text)

        transcript = results.get("transcript", "")
        summary = results.get("summary", "")
        title = results.get("title", "")

    except Exception as e:
        raise RuntimeError(f"Gemini 분석 중 오류: {e}")

    finally:
        # 로컬 임시 오디오 삭제
        if os.path.exists(audio_file_path):
            os.remove(audio_file_path)

        # Gemini 서버에 업로드된 파일 삭제
        try:
            genai.delete_file(audio_file.name)
        except Exception:
            pass

    return {
        "transcript": transcript,
        "summary": summary,
        "title": title,
    }
