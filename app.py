import os
from uuid import uuid4
from flask import Flask, request, jsonify
from flask_cors import CORS
from multiprocessing import Process, Queue

# Worker Queues
thumbnail_q = Queue()
thumbnail_res_q = Queue()

stt_q = Queue()
stt_res_q = Queue()

pet_q = Queue()
pet_res_q = Queue()

app = Flask(__name__)
CORS(app)

# ------------------------------------------------------
# 1) Thumbnail API → Thumbnail Worker로 전달
# ------------------------------------------------------
@app.route("/thumbnail", methods=["POST"])
def thumbnail_api():
    if "video" not in request.files:
        return jsonify({"error": "No video provided"}), 400

    task_id = uuid4().hex
    temp_path = f"temp_{task_id}.mp4"
    request.files["video"].save(temp_path)

    thumbnail_q.put({"id": task_id, "path": temp_path})
    result = thumbnail_res_q.get()

    os.remove(temp_path)
    return jsonify(result)


# ------------------------------------------------------
# 2) STT(API) → STT Worker로 전달
# ------------------------------------------------------
@app.route("/stt", methods=["POST"])
def stt_api():
    if "video" not in request.files:
        return jsonify({"error": "No video provided"}), 400

    api_key = request.form.get("api_key", "")
    if not api_key:
        return jsonify({"error": "Missing API Key"}), 400

    task_id = uuid4().hex
    temp_path = f"temp_{task_id}.mp4"
    request.files["video"].save(temp_path)

    stt_q.put({"id": task_id, "path": temp_path, "api_key": api_key})
    result = stt_res_q.get()

    os.remove(temp_path)
    return jsonify(result)


# ------------------------------------------------------
# 3) Pet 구간 (shorts) 탐지
# ------------------------------------------------------
@app.route("/detect", methods=["POST"])
def detect_api():
    if "video" not in request.files:
        return jsonify({"error": "No video provided"}), 400

    task_id = uuid4().hex
    temp_path = f"temp_{task_id}.mp4"
    request.files["video"].save(temp_path)

    pet_q.put({"id": task_id, "path": temp_path, "mode": "shorts"})
    result = pet_res_q.get()

    os.remove(temp_path)
    return jsonify(result)


# ------------------------------------------------------
# 4) Pet Daily (사진/영상 분류)
# ------------------------------------------------------
@app.route("/pet_daily", methods=["POST"])
def pet_daily_api():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    task_id = uuid4().hex
    file = request.files["file"]
    ext = file.filename.split(".")[-1]
    temp_path = f"temp_{task_id}.{ext}"
    file.save(temp_path)

    pet_q.put({"id": task_id, "path": temp_path, "mode": "daily"})
    result = pet_res_q.get()

    os.remove(temp_path)
    return jsonify(result)


# ------------------------------------------------------
# Worker 실행
# ------------------------------------------------------
def start_workers():
    from workers.thumbnail_worker import run_thumbnail_worker
    from workers.stt_worker import run_stt_worker
    from workers.pet_worker import run_pet_worker

    Process(target=run_thumbnail_worker, args=(thumbnail_q, thumbnail_res_q)).start()
    Process(target=run_stt_worker, args=(stt_q, stt_res_q)).start()
    Process(target=run_pet_worker, args=(pet_q, pet_res_q)).start()


if __name__ == "__main__":
    start_workers()
    app.run(host="0.0.0.0", port=8000)
