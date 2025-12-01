"""
반려동물 탐지 & 반려동물 등장 구간 숏츠 생성 AI 모듈
Google Cloud Vision + OpenCV + FFmpeg 기반
(병렬 Vision API 적용 버전, S3 업로드용 구조 정리)
"""

import os
import cv2
import uuid
import numpy as np
from google.cloud import vision
from concurrent.futures import ThreadPoolExecutor, as_completed


# ------------------------------------------------------
# 1) Vision API 초기화
# ------------------------------------------------------
def init_vision(project_id=None):
    if project_id:
        client_options = {"quota_project_id": project_id}
        return vision.ImageAnnotatorClient(client_options=client_options)
    return vision.ImageAnnotatorClient()


# ------------------------------------------------------
# 2) FPS 기반 프레임 추출
# ------------------------------------------------------
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
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                frames.append({
                    "time_sec": frame_idx / fps,
                    "image_bytes": buf.tobytes(),
                })

        frame_idx += 1

    cap.release()
    return frames


# ------------------------------------------------------
# 3) Vision API: 단일 프레임 분석
# ------------------------------------------------------
def detect_pet_in_frame(image_bytes, client):
    image = vision.Image(content=image_bytes)
    response = client.label_detection(image=image)

    pet_keywords = {"dog", "cat", "pet", "puppy", "kitten", "animal", "canidae"}

    for label in response.label_annotations:
        if label.description.lower() in pet_keywords and label.score >= 0.70:
            return True
    return False


# ------------------------------------------------------
# 4) 병렬 프레임 처리
# ------------------------------------------------------
def _process_frame(frame, client):
    try:
        has_pet = detect_pet_in_frame(frame["image_bytes"], client)
        return {
            "time_sec": frame["time_sec"],
            "has_pet": has_pet
        }
    except Exception as e:
        print("[Frame ERROR]", e, flush=True)
        return {
            "time_sec": frame["time_sec"],
            "has_pet": False
        }


# ------------------------------------------------------
# 5) 전체 영상에서 반려동물 등장 구간 탐지
# ------------------------------------------------------
def find_pet_segments(video_path, project_id=None):
    client = init_vision(project_id)
    frames = extract_frames(video_path, sec_per_frame=1.0)

    print(f"병렬 Vision API 처리 시작 (workers={min(10, len(frames))})", flush=True)

    results = []
    with ThreadPoolExecutor(max_workers=min(10, len(frames))) as exe:
        futures = [exe.submit(_process_frame, f, client) for f in frames]

        for f in as_completed(futures):
            results.append(f.result())

    results.sort(key=lambda x: x["time_sec"])

    # --- 구간 분석 ---
    segments = []
    in_seg = False
    start_t = 0

    for r in results:
        t = r["time_sec"]
        pet = r["has_pet"]

        if pet and not in_seg:
            in_seg = True
            start_t = t

        elif not pet and in_seg:
            if t - start_t >= 0.5:
                segments.append((start_t, t))
            in_seg = False

    # 끝까지 등장했으면 마지막 구간 추가
    if in_seg:
        end_t = results[-1]["time_sec"]
        if end_t - start_t >= 0.5:
            segments.append((start_t, end_t))

    print("반려동물 감지 결과:", segments, flush=True)
    return segments


# ------------------------------------------------------
# 6) 반려동물 숏츠 생성 + 로컬 저장 (S3 업로드는 app.py에서)
# ------------------------------------------------------
def compile_pet_shorts(video_path, segments, output_path=None):
    if not segments:
        raise ValueError("반려동물 클립이 없습니다.")

    # 1) UUID 기반 output path 생성
    if output_path is None:
        os.makedirs("shorts/generated", exist_ok=True)
        short_id = uuid.uuid4().hex[:12]
        output_path = f"shorts/generated/pet_shorts_{short_id}.mp4"

    # 2) FFmpeg concat list 생성
    list_path = f"{output_path}.txt"

    with open(list_path, "w") as f:
        for s, e in segments:
            f.write(f"file '{video_path}'\n")
            f.write(f"inpoint {s}\n")
            f.write(f"outpoint {e}\n")

    # 3) FFmpeg 실행
    cmd = (
        f"ffmpeg -y -f concat -safe 0 -i {list_path} "
        f"-c:v libx264 -preset veryfast -c:a aac {output_path}"
    )

    print("[FFmpeg 실행]", cmd, flush=True)
    os.system(cmd)

    os.remove(list_path)
    return output_path
