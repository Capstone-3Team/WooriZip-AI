import os
import cv2
import uuid
from google.cloud import vision
from concurrent.futures import ThreadPoolExecutor, as_completed

def init_vision(project_id=None):
    if project_id:
        return vision.ImageAnnotatorClient(client_options={"quota_project_id": project_id})
    return vision.ImageAnnotatorClient()


def extract_frames(video_path, sec_per_frame=1.0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("비디오를 열 수 없습니다.")

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

    # 연속 구간 찾기
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
    if not segments:
        raise ValueError("반려동물 구간이 없습니다.")

    os.makedirs("shorts/generated", exist_ok=True)
    out_name = f"pet_shorts_{uuid.uuid4().hex[:10]}.mp4"
    output_path = f"shorts/generated/{out_name}"

    list_path = f"{output_path}.txt"

    with open(list_path, "w") as f:
        for s, e in segments:
            f.write(f"file '{video_path}'\n")
            f.write(f"inpoint {s}\n")
            f.write(f"outpoint {e}\n")

    cmd = f"ffmpeg -y -f concat -safe 0 -i {list_path} -c:v libx264 -preset fast -c:a aac {output_path}"
    os.system(cmd)

    os.remove(list_path)
    return output_path
