import os
import cv2
import uuid
import subprocess
import boto3
from google.cloud import vision
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# AWS S3 설정
# ============================================================
S3_ACCESS_KEY = "AKIAZGFBE7RXDDF3I357"
S3_SECRET_KEY = "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"
S3_BUCKET = "woorizip-local-files"
S3_REGION = "ap-northeast-2"

s3_client = boto3.client(
    "s3",
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    region_name=S3_REGION
)

# ============================================================
# Google Vision 초기화
# ============================================================
def init_vision(project_id=None):
    if project_id:
        return vision.ImageAnnotatorClient(client_options={"quota_project_id": project_id})
    return vision.ImageAnnotatorClient()


# ============================================================
# 프레임 추출
# ============================================================
def extract_frames(video_path, sec_per_frame=1.0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("비디오를 열 수 없습니다: " + video_path)

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
                frames.append({
                    "time_sec": idx / fps,
                    "image_bytes": buf.tobytes(),
                })
        idx += 1

    cap.release()
    return frames


# ============================================================
# 프레임별 반려동물 존재 감지
# ============================================================
def detect_pet_in_frame(image_bytes, client):
    image = vision.Image(content=image_bytes)
    res = client.label_detection(image=image)

    keywords = {"dog", "cat", "pet", "puppy", "kitten", "animal", "canidae"}
    for label in res.label_annotations:
        if label.description.lower() in keywords and label.score >= 0.70:
            return True
    return False


# ============================================================
# 반려동물 구간 자동 탐색
# ============================================================
def find_pet_segments(video_path, project_id=None):
    client = init_vision(project_id)
    frames = extract_frames(video_path)

    results = []
    with ThreadPoolExecutor(max_workers=min(10, len(frames))) as exe:
        futures = [exe.submit(detect_pet_in_frame, f["image_bytes"], client) for f in frames]
        for f, frame in zip(as_completed(futures), frames):
            results.append({"time_sec": frame["time_sec"], "has_pet": f.result()})

    results.sort(key=lambda x: x["time_sec"])

    segments = []
    in_seg = False
    start = 0

    for r in results:
        if r["has_pet"] and not in_seg:
            in_seg = True
            start = r["time_sec"]
        elif not r["has_pet"] and in_seg:
            end = r["time_time"]
            if end - start >= 0.5:
                segments.append((start, end))
            in_seg = False

    if in_seg:
        end = results[-1]["time_sec"]
        if end - start >= 0.5:
            segments.append((start, end))

    return segments


# ============================================================
# ❗ 최종 숏츠 생성 + S3 업로드
# ============================================================
def compile_pet_shorts(video_path, segments):
    if not segments:
        raise ValueError("반려동물 구간이 없습니다.")

    # 로컬 임시 파일명 (ffmpeg가 생성)
    local_out_name = f"pet_shorts_{uuid.uuid4().hex[:10]}.mp4"
    local_output_path = os.path.join("/tmp", local_out_name)

    # ffmpeg에서 사용할 filter_complex 생성
    filter_parts = []
    idx = 0

    for s, e in segments:
        filter_parts.append(
            f"[0:v]trim=start={s}:end={e},setpts=PTS-STARTPTS[v{idx}];"
        )
        idx += 1

    concat_inputs = "".join(f"[v{i}]" for i in range(len(segments)))

    filter_complex = (
        "".join(filter_parts) +
        f"{concat_inputs}concat=n={len(segments)}:v=1:a=0[out]"
    )

    # ffmpeg 실행
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        local_output_path
    ]

    subprocess.run(cmd, check=False)

    # =========================================================
    # S3 업로드
    # =========================================================
    s3_key = f"shorts/{local_out_name}"

    s3_client.upload_file(
        local_output_path,
        S3_BUCKET,
        s3_key,
        ExtraArgs={"ContentType": "video/mp4"}
    )

    # 업로드된 URL 반환
    s3_url = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{s3_key}"

    # 로컬 임시 파일 삭제
    if os.path.exists(local_output_path):
        os.remove(local_output_path)

    return s3_url
