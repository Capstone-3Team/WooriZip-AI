import os
import cv2
import base64
import numpy as np
from uuid import uuid4
from multiprocessing import Process, Queue
from flask import Flask, request, jsonify
from flask_cors import CORS

# 모델 import
from models.thumb_stt import find_best_thumbnail, analyze_video_content
from models.face_arrange import analyze_face_from_frame
from models.pet_daily import classify_media
from models.pet_detect import find_pet_segments, compile_pet_shorts

# Worker queues
stt_q = Queue()
stt_res_q = Queue()
pet_q = Queue()
pet_res_q = Queue()

app = Flask(__name__)
CORS(app)

# ============================================================
# 1) 얼굴 정렬 (실시간) - Worker NO
# ============================================================
@app.route("/face_arrange", methods=["POST"])
def face_arrange_api():
    if "file" in request.files:
        img_bytes = request.files["file"].read()
    else:
        data = request.get_json()
        img_bytes = base64.b64decode(data["image"])

    frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "image decode failed"}), 400

    result = analyze_face_from_frame(frame)
    return jsonify({"message": "success", "data": result})


# ============================================================
# 2) 썸네일 추출 (빠름 → Worker NO)
# ============================================================
@app.route("/thumbnail", methods=["POST"])
def thumbnail_api():
    if "video" not in request.files:
        return jsonify({"error": "No video provided"}), 400

    temp_path = f"temp_{uuid4().hex}.mp4"
    request.files["video"].save(temp_path)

    try:
        result = find_best_thumbnail(temp_path)
        os.remove(temp_path)
        if not result:
            return jsonify({"error": "No valid thumbnail"}), 500
        return jsonify(result)
    except Exception as e:
        os.remove(temp_path)
        return jsonify({"error": str(e)}), 500


# ============================================================
# 3) STT + 요약 + 제목 생성 → Worker
# ============================================================
@app.route("/stt", methods=["POST"])
def stt_api():
    if "video" not in request.files:
        return jsonify({"error": "No video provided"}), 400

    api_key = request.form.get("api_key")
    if not api_key:
        return jsonify({"error": "Missing API Key"}), 400

    task_id = uuid4().hex
    temp_path = f"temp_{task_id}.mp4"
    request.files["video"].save(temp_path)

    stt_q.put({"id": task_id, "path": temp_path, "api_key": api_key})
    result = stt_res_q.get()

    os.remove(temp_path)
    return jsonify(result)


# ============================================================
# 4) 반려동물 DAILY → Worker
# ============================================================
@app.route("/pet_daily", methods=["POST"])
def pet_daily_api():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    ext = file.filename.split(".")[-1]
    temp_path = f"temp_{uuid4().hex}.{ext}"
    file.save(temp_path)

    pet_q.put({"mode": "daily", "path": temp_path})
    result = pet_res_q.get()

    os.remove(temp_path)
    return jsonify(result)


# ============================================================
# 5) 반려동물 숏츠 → Worker
# ============================================================
@app.route("/detect", methods=["POST"])
def detect_api():
    if "video" not in request.files:
        return jsonify({"error": "No video provided"}), 400

    temp_path = f"temp_{uuid4().hex}.mp4"
    request.files["video"].save(temp_path)

    pet_q.put({"mode": "shorts", "path": temp_path})
    result = pet_res_q.get()

    os.remove(temp_path)
    return jsonify(result)


# ============================================================
# Worker start
# ============================================================
def start_workers():
    from workers.stt_worker import run_stt_worker
    from workers.pet_worker import run_pet_worker

    Process(target=run_stt_worker, args=(stt_q, stt_res_q)).start()
    Process(target=run_pet_worker, args=(pet_q, pet_res_q)).start()


if __name__ == "__main__":
    start_workers()
    app.run(host="0.0.0.0", port=8000)
