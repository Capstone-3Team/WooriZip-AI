import os
import base64
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS

# -------------------------------
# 모델 import
# -------------------------------
from models.thumb_stt import find_best_thumbnail, analyze_video_content
from models.pet_shorts import find_pet_segments, compile_pet_shorts
from models.pet_daily import classify_media
from models.face_arrange import analyze_face_from_frame
from utils.s3_upload import upload_to_s3

app = Flask(__name__)
CORS(app)

# =========================================================
# 1) 썸네일 추출 API
# =========================================================
@app.route("/thumbnail", methods=["POST"])
def thumbnail():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files["video"]
    temp_path = f"temp_thumbnail_{os.getpid()}.mp4"
    video_file.save(temp_path)

    try:
        result = find_best_thumbnail(temp_path)

        if result is None:
            return jsonify({"error": "Failed to detect valid thumbnail"}), 500

        return jsonify({
            "message": "Thumbnail analysis successful",
            "time_sec": result["time_sec"],
            "score": result["score"],
            "image_base64": result["image_base64"]
        })

    except Exception as e:
        print("[Thumbnail ERROR]", e)
        return jsonify({"error": str(e)}), 500

    finally:
        os.remove(temp_path) if os.path.exists(temp_path) else None


# =========================================================
# 2) STT + 요약 + 제목 생성 API
# =========================================================
@app.route("/stt", methods=["POST"])
def stt():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    api_key = request.form.get("api_key", "")
    if not api_key:
        return jsonify({"error": "Missing API Key"}), 400

    video_file = request.files["video"]
    temp_path = f"temp_stt_{os.getpid()}.mp4"
    video_file.save(temp_path)

    try:
        result = analyze_video_content(temp_path, api_key)

        return jsonify({
            "message": "summary + title generation successful",
            "summary": result["summary"],
            "title": result["title"]
        })

    except Exception as e:
        print("[STT ERROR]", e)
        return jsonify({"error": str(e)}), 500

    finally:
        os.remove(temp_path) if os.path.exists(temp_path) else None


# =========================================================
# 3) 반려동물 출현 구간 탐지 API
# =========================================================
@app.route("/detect", methods=["POST"])
def detect():
    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    video_file = request.files["video"]
    temp_path = f"temp_pet_{os.getpid()}.mp4"
    video_file.save(temp_path)

    try:
        PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
        segments = find_pet_segments(temp_path, project_id=PROJECT_ID)

        return jsonify({
            "message": "success",
            "segments": segments
        })

    except Exception as e:
        print("[Pet Detect ERROR]", e)
        return jsonify({"error": str(e)}), 500

    finally:
        os.remove(temp_path) if os.path.exists(temp_path) else None


# =========================================================
# 4) 반려동물 숏츠 생성 API (compile → shorts)
# =========================================================
@app.route("/shorts", methods=["POST"])
def shorts():
    data = request.json

    if not data or "segments" not in data or "video_path" not in data:
        return jsonify({"error": "video_path + segments required"}), 400

    segments = data["segments"]
    video_path = data["video_path"]

    try:
        # 1) 로컬 숏츠 생성
        output_path = compile_pet_shorts(video_path, segments)

        # 2) S3 업로드
        s3_url = upload_to_s3(output_path, "pet-shorts")

        # 3) 로컬 파일 삭제
        os.remove(output_path)

        return jsonify({
            "message": "success",
            "shortsUrl": s3_url
        })

    except Exception as e:
        print("[Pet Shorts ERROR]", e)
        return jsonify({"error": str(e)}), 500


# =========================================================
# 5) 반려동물 단일 분류 API (pet_daily)
# =========================================================
@app.route("/pet_daily", methods=["POST"])
def pet_daily():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    media_file = request.files["file"]
    filename = media_file.filename

    ext = filename.split(".")[-1] if "." in filename else "tmp"
    temp_path = f"temp_daily_{os.getpid()}.{ext}"
    media_file.save(temp_path)

    try:
        PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
        result = classify_media(temp_path, project_id=PROJECT_ID)

        return jsonify({
            "message": "success",
            "result": result
        })

    except Exception as e:
        print("[Pet Daily ERROR]", e)
        return jsonify({"error": str(e)}), 500

    finally:
        os.remove(temp_path) if os.path.exists(temp_path) else None


# =========================================================
# 6) 얼굴 위치 분석 API
# =========================================================
@app.route("/face_arrange", methods=["POST"])
def face_arrange_api():
    if "file" in request.files:
        img_bytes = request.files["file"].read()
    else:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "image(base64) or file required"}), 400
        try:
            img_bytes = base64.b64decode(data["image"])
        except:
            return jsonify({"error": "base64 decode failed"}), 400

    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "image decode failed"}), 400

    try:
        result = analyze_face_from_frame(frame)
        return jsonify({"message": "success", "data": result})
    except Exception as e:
        print("[Face Arrange ERROR]", e)
        return jsonify({"error": str(e)}), 500


# =========================================================
# Run Flask (local only)
# =========================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
