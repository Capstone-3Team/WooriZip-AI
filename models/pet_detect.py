# app.py
import os
from flask import Flask, request, jsonify
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

PROJECT_ID = os.environ.get("GCP_PROJECT_ID")  # Vision API 프로젝트 ID


# -------------------------------
# POST /detect → 반려동물 출현 구간 반환
# -------------------------------
@app.route("/detect", methods=["POST"])
def detect():
    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    file = request.files["video"]
    temp_path = f"temp_video_{os.getpid()}.mp4"
    file.save(temp_path)

    try:
        segments = model.find_pet_segments(temp_path, project_id=PROJECT_ID)
        return jsonify({
            "message": "success",
            "segments": segments
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# -------------------------------
# POST /compile → 숏츠 영상 생성
# -------------------------------
@app.route("/compile", methods=["POST"])
def compile():
    data = request.json

    if not data or "segments" not in data or "video_path" not in data:
        return jsonify({"error": "video_path + segments required"}), 400

    segments = data["segments"]
    video_path = data["video_path"]

    output_path = "pet_shorts.mp4"

    try:
        result = model.compile_pet_shorts(video_path, segments, output_path)
        return jsonify({
            "message": "success",
            "output": result
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
