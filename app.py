import os
from flask import Flask, request, jsonify
from flask_cors import CORS

# 변경된 파일명 반영
from models.thumb_stt import find_best_thumbnail, analyze_video_content
from models.pet_detect import find_pet_segments, compile_pet_shorts

app = Flask(__name__)
CORS(app)


# ---------------------------------------------------------
# 1) 썸네일 API
# ---------------------------------------------------------
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
        print(f"[Thumbnail ERROR] {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ---------------------------------------------------------
# 2) STT + 요약 + 제목 API
# ---------------------------------------------------------
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
            "message": "STT + summary + title generation successful",
            "summary": result["summary"],
            "title": result["title"]
        })

    except Exception as e:
        print(f"[STT ERROR] {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ---------------------------------------------------------
# 3) 반려동물 출현 구간 탐지 API
# ---------------------------------------------------------
@app.route("/detect", methods=["POST"])
def detect():
    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    file = request.files["video"]
    temp_path = f"temp_pet_{os.getpid()}.mp4"
    file.save(temp_path)

    try:
        PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
        segments = find_pet_segments(temp_path, project_id=PROJECT_ID)

        return jsonify({
            "message": "success",
            "segments": segments
        })

    except Exception as e:
        print(f"[Pet Detect ERROR] {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ---------------------------------------------------------
# 4) 반려동물 숏츠 생성 API
# ---------------------------------------------------------
@app.route("/compile", methods=["POST"])
def compile():
    data = request.json

    if not data or "segments" not in data or "video_path" not in data:
        return jsonify({"error": "video_path + segments required"}), 400

    segments = data["segments"]
    video_path = data["video_path"]
    output_path = "pet_shorts.mp4"

    try:
        output = compile_pet_shorts(video_path, segments, output_path)

        return jsonify({
            "message": "success",
            "output": output
        })

    except Exception as e:
        print(f"[Pet Compile ERROR] {e}")
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------
# Run server
# ---------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
