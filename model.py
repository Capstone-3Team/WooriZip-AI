import os
import cv2
import numpy as np
import math
import io
import base64
from google.cloud import vision
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 1. Google Vision 클라이언트 초기화 ---
try:
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service-account.json"
        
    vision_client = vision.ImageAnnotatorClient()
    print("✅ (model.py) Vision AI 클라이언트가 초기화되었습니다.")
except Exception as e:
    print(f"❌ (model.py) Vision AI 클라이언트 초기화 오류: {e}")
    print("   'service-account.json' 파일이 올바른지 확인하세요.")


# --- 2. AI 점수화 로직 (여러 얼굴 합산 + GPT 피드백 반영) ---

LIKELIHOOD_SCORE = {
    'UNKNOWN': 0, 'VERY_UNLIKELY': 0, 'UNLIKELY': 1,
    'POSSIBLE': 2, 'LIKELY': 4, 'VERY_LIKELY': 5
}


def _analyze_frame_for_thumbnail(image_bytes):
    """
    여러 얼굴이 등장할 경우 모든 얼굴의 점수를 합산하여
    '해당 프레임 전체 점수'를 반환한다.
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
        if LIKELIHOOD_SCORE.get(face.blurred_likelihood, 0) < 3: base_quality_score += 50
        if LIKELIHOOD_SCORE.get(face.under_exposed_likelihood, 0) < 3: base_quality_score += 20
        if abs(face.roll_angle) < 20 and abs(face.pan_angle) < 20: base_quality_score += 30
        if face.detection_confidence > 0.7: base_quality_score += 20

        # -------- 2. 감정 점수 --------
        api_score_norm = LIKELIHOOD_SCORE.get(face.joy_likelihood, 0) / 5.0
        landmarks = {lm.type_: lm.position for lm in face.landmarks}

        required_lm = [
            vision.FaceAnnotation.Landmark.Type.UPPER_LIP,
            vision.FaceAnnotation.Landmark.Type.LOWER_LIP,
            vision.FaceAnnotation.Landmark.Type.MOUTH_CENTER,
            vision.FaceAnnotation.Landmark.Type.MOUTH_LEFT,
            vision.FaceAnnotation.Landmark.Type.MOUTH_RIGHT
        ]

        if not all(lm_type in landmarks for lm_type in required_lm):
            landmark_score_norm = 0.0
            mouth_info = f"Joy:{api_score_norm*5:.0f}, Landmark:FAIL"
        else:
            lip_distance = abs(
                landmarks[vision.FaceAnnotation.Landmark.Type.UPPER_LIP].y -
                landmarks[vision.FaceAnnotation.Landmark.Type.LOWER_LIP].y
            )

            center_y = landmarks[vision.FaceAnnotation.Landmark.Type.MOUTH_CENTER].y
            left_y = landmarks[vision.FaceAnnotation.Landmark.Type.MOUTH_LEFT].y
            right_y = landmarks[vision.FaceAnnotation.Landmark.Type.MOUTH_RIGHT].y

            curvature = (center_y - left_y) + (center_y - right_y)
            curvature_norm = np.clip(curvature / 15.0, 0, 1)
            lip_norm = np.clip(lip_distance / 10.0, 0, 1)

            landmark_score_norm = (curvature_norm * 0.7) + (lip_norm * 0.3)
            mouth_info = (
                f"Joy:{api_score_norm*5:.0f}, "
                f"Pull:{curvature:.2f}, Open:{lip_distance:.2f}"
            )

        # -------- 3. 감정 종합 점수 --------
        emotion_norm = (api_score_norm * 0.8) + (landmark_score_norm * 0.2)
        emotion_score = emotion_norm * 300

        face_score = base_quality_score + emotion_score
        total_score += face_score

        mouth_info_summary.append(mouth_info)

    return total_score, "여러 얼굴", "; ".join(mouth_info_summary)


# ----------------------------------------------------
# 3. 프레임 추출 함수
# ----------------------------------------------------

def _extract_frames_by_interval(video_path, sec_per_frame=0.25):
    """
    (내부 함수)
    비디오 파일 경로에서 프레임을 추출합니다.
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
            ret_enc, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if ret_enc:
                current_time_sec = frame_count / fps
                frames_data.append({
                    'time_sec': current_time_sec,
                    'image_bytes': buffer.tobytes(),
                    'image_cv2': frame
                })

        frame_count += 1

    cap.release()
    print(f"✅ 총 {len(frames_data)}개의 프레임이 추출되었습니다.")
    return frames_data


# ----------------------------------------------------
# 4. 메인 함수 — 병렬 처리 + 여러 얼굴 합산 + Vision API
# ----------------------------------------------------

def _process_frame(frame_data):
    """ 병렬 처리를 위한 프레임 분석 래퍼 """
    try:
        score, status, mouth_info_str = _analyze_frame_for_thumbnail(frame_data['image_bytes'])
        frame_data['score'] = score
        frame_data['status'] = status
        frame_data['mouth'] = mouth_info_str
        return frame_data
    except Exception as e:
        print(f"Frame 처리 중 오류: {e}")
        return None


def find_best_thumbnail(video_path):
    """
    비디오 파일을 분석하여 가장 좋은 썸네일을 반환하는 함수 (병렬 처리 개선)
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
                print(f"  진행률: {i+1}/{len(frames)} 프레임 완료")

    scored_frames.sort(key=lambda x: x['score'], reverse=True)
    best_thumbnail = scored_frames[0]

    if best_thumbnail['score'] <= 0:
        print("결과: 🌄 유의미한 얼굴 없음. 50% 지점 프레임 반환")
        best_thumbnail = frames[len(frames) // 2]
    else:
        print(f"결과: 😃 베스트 썸네일 선정! (점수: {int(best_thumbnail['score'])})")

    ret, buffer = cv2.imencode('.jpg', best_thumbnail['image_cv2'])
    if not ret:
        return None

    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "time_sec": best_thumbnail['time_sec'],
        "score": best_thumbnail['score'],
        "image_base64": img_base64
    }
