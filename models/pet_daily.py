"""
반려동물 탐지 (사진/영상) - 편집 없이 분류만 하는 모듈
Google Cloud Vision + OpenCV 기반
(병렬 Vision API 속도 최적화 버전)
"""

import os
import cv2
import mimetypes
from google.cloud import vision
from concurrent.futures import ThreadPoolExecutor, as_completed


# -------------------------------
# Vision API 초기화
# -------------------------------
def init_vision(project_id=None):
    if project_id:
        client_options = {"quota_project_id": project_id}
        return vision.ImageAnnotatorClient(client_options=client_options)
    return vision.ImageAnnotatorClient()


# =====================================================
# 1. IMAGE 분석 - 반려동물 여부 
# =====================================================
def detect_pet_in_image(image_path, project_id=None):
    client = init_vision(project_id)

    with open(image_path, "rb") as f:
        content = f.read()

    image = vision.Image(content=content)
    res = client.label_detection(image=image)

    pet_keywords = {"dog", "cat", "pet", "animal", "puppy", "kitten", "canidae"}

    has_pet = False
    for label in res.label_annotations:
        if label.description.lower() in pet_keywords and label.score >= 0.70:
            has_pet = True
            break

    return {
        "file_type": "image",
        "is_pet_present": has_pet
    }


# =====================================================
# 2. VIDEO 분석 - 프레임 추출
# =====================================================
def extract_frames(video_path, sec_per_frame=1.0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"비디오 파일을 열 수 없습니다: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps * sec_per_frame)
    if interval < 1:
        interval = 1

    frames = []
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % interval == 0:
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                frames.append({
                    "time_sec": idx / fps,
                    "image_bytes": buf.tobytes(),
                })

        idx += 1

    cap.release()
    return frames


# =====================================================
# 3. 단일 프레임 반려동물 탐지
# =====================================================
def detect_pet_in_frame(image_bytes, vision_client):
    image = vision.Image(content=image_bytes)
    res = vision_client.label_detection(image=image)

    pet_keywords = {"dog", "cat", "pet", "animal", "puppy", "kitten", "canidae"}

    for label in res.label_annotations:
        if label.description.lower() in pet_keywords and label.score >= 0.70:
            return True
    return False


# =====================================================
# 3-1. 병렬 처리용 래퍼
# =====================================================
def _process_frame(frame, client):
    try:
        found = detect_pet_in_frame(frame["image_bytes"], client)
        return {
            "time_sec": frame["time_sec"],
            "found": found
        }
    except Exception as e:
        print("Frame 처리 오류:", e)
        return {
            "time_sec": frame["time_sec"],
            "found": False
        }


# =====================================================
# 4. VIDEO - 전체 영상 분류 (병렬 Vision API 적용)
# =====================================================
def detect_pet_in_video(video_path, project_id=None):
    client = init_vision(project_id)

    frames = extract_frames(video_path, sec_per_frame=1.0)

    results = []

    # 병렬 처리 시작
    max_workers = min(10, len(frames))
    print(f"병렬 Vision API 처리 시작 (workers={max_workers})")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_process_frame, f, client)
            for f in frames
        ]

        for future in as_completed(futures):
            results.append(future.result())

    # 시간순 정렬
    results.sort(key=lambda x: x["time_sec"])

    pet_timestamps = [
        r["time_sec"] for r in results if r["found"]
    ]

    return {
        "file_type": "video",
        "is_pet_present": len(pet_timestamps) > 0,
        "timestamps": pet_timestamps
    }


# =====================================================
# 5. 통합 함수 - 사진/영상 자동 판별
# =====================================================
def classify_media(file_path, project_id=None):
    mime, _ = mimetypes.guess_type(file_path)

    if mime and mime.startswith("image"):
        return detect_pet_in_image(file_path, project_id)

    if mime and mime.startswith("video"):
        return detect_pet_in_video(file_path, project_id)

    return {"error": "지원하지 않는 파일 형식입니다."}
