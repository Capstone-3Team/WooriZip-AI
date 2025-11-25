"""
반려동물 탐지 & 반려동물 등장 구간 숏츠 생성 AI 모듈
Google Cloud Vision + OpenCV + FFmpeg 기반
(병렬 Vision API 적용 버전)
"""

import os
import cv2
import numpy as np
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


# -------------------------------
# 1. 프레임 추출 함수 (동일)
# -------------------------------
def extract_frames(video_path, sec_per_frame=1.0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"비디오 파일을 열 수 없습니다: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps * sec_per_frame)
    if interval < 1:
        interval = 1

    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0:
            ret_jpg, buffer = cv2.imencode(".jpg", frame)
            if ret_jpg:
                frames.append({
                    "time_sec": frame_idx / fps,
                    "image_bytes": buffer.tobytes(),
                })

        frame_idx += 1

    cap.release()
    return frames, fps


# -------------------------------
# 2. 단일 프레임 반려동물 탐지
# -------------------------------
def detect_pet_in_frame(image_bytes, vision_client):
    image = vision.Image(content=image_bytes)
    response = vision_client.label_detection(image=image)

    labels = response.label_annotations
    pet_keywords = {"dog", "cat", "pet", "puppy", "kitten", "animal", "canidae"}

    if labels:
        for label in labels:
            if label.description.lower() in pet_keywords and label.score >= 0.70:
                return True
    return False


# -------------------------------
# 2-1. 병렬 처리용 래퍼
# -------------------------------
def _process_frame(frame, vision_client):
    try:
        has_pet = detect_pet_in_frame(frame["image_bytes"], vision_client)
        return {
            "time_sec": frame["time_sec"],
            "has_pet": has_pet
        }
    except Exception as e:
        print("프레임 분석 오류:", e)
        return {
            "time_sec": frame["time_sec"],
            "has_pet": False
        }


# -------------------------------
# 3. 전체 영상 분석 (병렬 Vision API)
# -------------------------------
def find_pet_segments(video_path, project_id=None):
    vision_client = init_vision(project_id)

    frames, fps = extract_frames(video_path, sec_per_frame=1.0)

    results = []

    # 병렬 Vision API
    max_workers = min(10, len(frames))
    print(f"병렬 Vision API 처리 시작 (workers={max_workers})")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_process_frame, f, vision_client)
            for f in frames
        ]

        for future in as_completed(futures):
            results.append(future.result())

    # 시간 순 정렬
    results.sort(key=lambda x: x["time_sec"])

    # -------------------------------
    # 반려동물 등장 구간 분석
    # -------------------------------
    segments = []
    is_pet = False
    start_t = 0

    for r in results:
        t = r["time_sec"]
        found = r["has_pet"]

        # 등장 시작
        if found and not is_pet:
            is_pet = True
            start_t = t

        # 사라짐
        elif not found and is_pet:
            if t - start_t > 0.5:
                segments.append((start_t, t))
            is_pet = False

    # 영상 끝까지 등장 중이었다면
    if is_pet:
        end_t = results[-1]["time_sec"]
        if end_t - start_t > 0.5:
            segments.append((start_t, end_t))

    return segments


# -------------------------------
# 4. FFmpeg로 클립 이어붙이기 (동일)
# -------------------------------
def compile_pet_shorts(video_path, segments, output_path):
    if not segments:
        raise ValueError("반려동물 클립이 없습니다.")

    selects = []
    for s, e in segments:
        selects.append(f"between(t,{s},{e})")

    select_str = "+".join(selects)

    filter_complex = (
        f"[0:v]select='{select_str}',setpts=N/FR/TB[v];"
        f"[0:a]aselect='{select_str}',asetpts=N/FR/TB[a]"
    )

    command = (
        f'ffmpeg -i "{video_path}" -filter_complex "{filter_complex}" '
        f'-map "[v]" -map "[a]" -y "{output_path}"'
    )

    os.system(command)
    return output_path
