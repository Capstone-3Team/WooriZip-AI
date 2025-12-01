"""
반려동물 탐지 (사진/영상) - 병렬 Vision API 최적화 버전
"""

import os
import cv2
import mimetypes
from google.cloud import vision
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------------------------------------------------
# Vision API 초기화
# ------------------------------------------------------
def init_vision(project_id=None):
    if project_id:
        client_options = {"quota_project_id": project_id}
        return vision.ImageAnnotatorClient(client_options=client_options)
    return vision.ImageAnnotatorClient()

# ------------------------------------------------------
# 사진 분석
# ------------------------------------------------------
def detect_pet_in_image(image_path, project_id=None):
    client = init_vision(project_id)

    with open(image_path, "rb") as f:
        content = f.read()

    image = vision.Image(content=content)
    res = client.label_detection(image=image)

    pet_keywords = {"dog", "cat", "pet", "animal", "puppy", "kitten", "canidae"}
    has_pet = any(label.description.lower() in pet_keywords and label.score >= 0.70
                  for label in res.label_annotations)

    return {"file_type": "image", "is_pet_present": has_pet}

# ------------------------------------------------------
# 영상 → 1초당 1프레임 추출
# ------------------------------------------------------
def extract_frames(video_path, sec_per_frame=1.0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"비디오 파일을 열 수 없습니다: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(int(fps * sec_per_frame), 1)

    frames = []
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                frames.append({"time_sec": idx / fps, "image_bytes": buf.tobytes()})
        idx += 1

    cap.release()
    return frames

# ------------------------------------------------------
# 단일 프레임 반려동물 탐지
# ------------------------------------------------------
def detect_pet_in_frame(image_bytes, client):
    image = vision.Image(content=image_bytes)
    res = client.label_detection(image=image)

    pet_keywords = {"dog", "cat", "pet", "animal", "puppy", "kitten", "canidae"}
    return any(label.description.lower() in pet_keywords and label.score >= 0.70
               for label in res.label_annotations)

# ------------------------------------------------------
# 영상 전체 분석 (병렬 처리)
# ------------------------------------------------------
def detect_pet_in_video(video_path, project_id=None):
    client = init_vision(project_id)
    frames = extract_frames(video_path, sec_per_frame=1.0)

    results = []
    max_workers = min(10, len(frames))

    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = [exe.submit(detect_pet_in_frame, f["image_bytes"], client)
                   for f in frames]

        for f, frame in zip(as_completed(futures), frames):
            results.append({
                "time_sec": frame["time_sec"],
                "found": f.result()
            })

    pet_times = [r["time_sec"] for r in results if r["found"]]
    return {"file_type": "video", "is_pet_present": len(pet_times) > 0, "timestamps": pet_times}

# ------------------------------------------------------
# 파일 타입 자동 분기
# ------------------------------------------------------
def classify_media(file_path, project_id=None):
    mime, _ = mimetypes.guess_type(file_path)

    if mime and mime.startswith("image"):
        return detect_pet_in_image(file_path, project_id)

    if mime and mime.startswith("video"):
        return detect_pet_in_video(file_path, project_id)

    return {"error": "지원하지 않는 파일 형식입니다."}
