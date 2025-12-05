import traceback
import os
import cv2
import base64
import numpy as np
from uuid import uuid4
from multiprocessing import Process, Queue
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
load_dotenv()

# 모델 import
from models.thumb_stt import find_best_thumbnail, analyze_video_content
from models.face_arrange import analyze_face_from_frame
from models.pet_daily import classify_media
from models.pet_shorts import find_pet_segments, compile_pet_shorts

# Worker queues
stt_q = Queue()
stt_res_q = Queue()
pet_q = Queue()
pet_res_q = Queue()

app = Flask(__name__)
CORS(app)


# ============================================================
# 1) 얼굴 정렬 (실시간)
# ============================================================
@app.route("/face_arrange", methods=["POST"])
def face_arrange_api():
    print("\n📌 [DEBUG] /face_arrange 호출됨")

    try:
        # 이미지 읽기
        if "file" in request.files:
            img_bytes = request.files["file"].read()
        else:
            data = request.get_json()
            if not data or "image" not in data:
                print("❌ [ERROR] image(base64) 또는 file 없음")
                return jsonify({"error": "image(base64) or file required"}), 400

            try:
                img_bytes = base64.b64decode(data["image"])
            except:
                print("❌ [ERROR] base64 decode 실패")
                return jsonify({"error": "base64 decode failed"}), 400

        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            print("❌ [ERROR] frame decode 실패")
            return jsonify({"error": "image decode failed"}), 400

        # 얼굴 분석
        result = analyze_face_from_frame(frame)
        print(f"📌 [DEBUG] 분석 결과: {result}")

        return jsonify(result)

    except Exception as e:
        print("\n🔥🔥🔥 [EXCEPTION in /face_arrange]")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ============================================================
# 2) 썸네일 추출
# ============================================================
@app.route("/thumbnail", methods=["POST"])
def thumbnail_api():
    print("\n📌 [DEBUG] /thumbnail 호출됨")

    if "video" not in request.files:
        print("❌ [ERROR] video 없음")
        return jsonify({"error": "No video provided"}), 400

    temp_path = f"temp_{uuid4().hex}.mp4"
    request.files["video"].save(temp_path)
    print(f"📌 [DEBUG] 저장된 파일: {temp_path}")

    try:
        result = find_best_thumbnail(temp_path)
        print(f"📌 [DEBUG] 썸네일 분석 결과: {result}")

        os.remove(temp_path)

        if not result:
            print("❌ [ERROR] find_best_thumbnail() 결과 없음")
            return jsonify({"error": "No valid thumbnail"}), 500

        return jsonify(result)

    except Exception as e:
        print("\n🔥🔥🔥 [EXCEPTION in /thumbnail]")
        traceback.print_exc()
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"error": str(e)}), 500


# ============================================================
# 3) STT + 요약 + 제목 생성 → Worker
# ============================================================
@app.route("/stt", methods=["POST"])
def stt_api():
    print("\n📌 [DEBUG] /stt 호출됨")

    if "video" not in request.files:
        print("❌ [ERROR] video 없음")
        return jsonify({"error": "No video provided"}), 400

    api_key = request.form.get("api_key")
    if not api_key:
        print("❌ [ERROR] API Key 없음")
        return jsonify({"error": "Missing API Key"}), 400

    file = request.files["video"]
    filename = file.filename or "upload.webm"

    # 확장자 추출
    if "." in filename:
        ext = filename.rsplit(".", 1)[-1].lower()
    else:
        ext = "webm"

    task_id = uuid4().hex
    temp_path = f"temp_{task_id}.{ext}"
    file.save(temp_path)
    print(f"📌 [DEBUG] STT 파일 저장: {temp_path}")

    try:
        stt_q.put({"id": task_id, "path": temp_path, "api_key": api_key})
        print("📌 [DEBUG] STT 작업 큐에 전달 완료")

        result = stt_res_q.get()
        print(f"📌 [DEBUG] STT 결과: {result}")

        return jsonify(result)

    except Exception as e:
        print("\n🔥🔥🔥 [EXCEPTION in /stt]")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ============================================================
# 4) 반려동물 DAILY
# ============================================================
@app.route("/pet_daily", methods=["POST"])
def pet_daily_api():
    print("\n📌 [DEBUG] /pet_daily 호출됨")

    if "file" not in request.files:
        print("❌ [ERROR] file 없음")
        return jsonify({"error": "No file provided"}), 400

    try:
        file = request.files["file"]
        ext = file.filename.split(".")[-1]
        temp_path = f"temp_{uuid4().hex}.{ext}"
        file.save(temp_path)

        pet_q.put({"mode": "daily", "path": temp_path})
        print("📌 [DEBUG] daily worker 전달 완료")

        result = pet_res_q.get()
        print(f"📌 [DEBUG] daily 결과: {result}")

        os.remove(temp_path)
        return jsonify(result)

    except Exception as e:
        print("\n🔥🔥🔥 [EXCEPTION in /pet_daily]")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ============================================================
# 5) 반려동물 숏츠
# ============================================================
@app.route("/detect", methods=["POST"])
def detect_api():
    print("\n📌 [DEBUG] /detect 호출됨")

    if "video" not in request.files:
        print("❌ [ERROR] video 없음")
        return jsonify({"error": "No video provided"}), 400

    try:
        temp_path = f"temp_{uuid4().hex}.mp4"
        request.files["video"].save(temp_path)

        pet_q.put({"mode": "shorts", "path": temp_path})
        print("📌 [DEBUG] shorts worker 전달 완료")

        result = pet_res_q.get()
        print(f"📌 [DEBUG] shorts 결과: {result}")

        os.remove(temp_path)
        return jsonify(result)

    except Exception as e:
        print("\n🔥🔥🔥 [EXCEPTION in /detect]")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ============================================================
# Worker 시작
# ============================================================
def start_workers():
    from workers.stt_worker import run_stt_worker
    from workers.pet_worker import run_pet_worker

    print("🔥 STT Worker started.")
    Process(target=run_stt_worker, args=(stt_q, stt_res_q)).start()

    print("🔥 Pet Worker started.")
    Process(target=run_pet_worker, args=(pet_q, pet_res_q)).start()


if __name__ == "__main__":
    start_workers()
    print("🚀 App Started on port 8000")
    app.run(host="0.0.0.0", port=8000)
