import os
from flask import Flask, request, jsonify
from flask_cors import CORS

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
        result = model.find_best_thumbnail(temp_path)

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

    # 프론트에서 전달한 Gemini API KEY
    api_key = request.form.get("api_key", "")
    if not api_key:
        return jsonify({"error": "Missing API Key"}), 400

    video_file = request.files["video"]
    temp_path = f"temp_stt_{os.getpid()}.mp4"
    video_file.save(temp_path)

    try:
        result = model.analyze_video_content(temp_path, api_key)

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
# Run server
# ---------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
