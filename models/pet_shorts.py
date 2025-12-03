import os
import cv2
import uuid
import subprocess
from google.cloud import vision
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # models 폴더
GENERATED_DIR = os.path.join(BASE_DIR, "../shorts/generated")
os.makedirs(GENERATED_DIR, exist_ok=True)


def init_vision(project_id=None):
    if project_id:
        return vision.ImageAnnotatorClient(client_options={"quota_project_id": project_id})
    return vision.ImageAnnotatorClient()


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


def detect_pet_in_frame(image_bytes, client):
    image = vision.Image(content=image_bytes)
    res = client.label_detection(image=image)

    keywords = {"dog", "cat", "pet", "puppy", "kitten", "animal", "canidae"}
    for label in res.label_annotations:
        if label.description.lower() in keywords and label.score >= 0.70:
            return True
    return False


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
            end = r["time_sec"]
            if end - start >= 0.5:
                segments.append((start, end))
            in_seg = False

    if in_seg:
        end = results[-1]["time_sec"]
        if end - start >= 0.5:
            segments.append((start, end))

    return segments


def compile_pet_shorts(video_path, segments):
    """
    trim + concat filter 방식 (정확하고 안정적)
    """
    if not segments:
        raise ValueError("반려동물 구간이 없습니다.")

    out_name = f"pet_shorts_{uuid.uuid4().hex[:10]}.mp4"
    output_path = os.path.join(GENERATED_DIR, out_name)

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

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        output_path
    ]

    subprocess.run(cmd, check=False)

    return output_path
