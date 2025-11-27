# app.py
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import model

app = Flask(__name__)
CORS(app)


@app.route("/thumbnail", methods=["POST"])
def thumbnail():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    temp_path = f"temp_thumbnail_{os.getpid()}.mp4"
    request.files["video"].save(temp_path)

    try:
        result = model.find_best_thumbnail(temp_path)
        if result is None:
            return jsonify({"error": "No smiling face detected"}), 500

        return jsonify({
            "message": "Thumbnail analysis successful",
            "time_sec": result["time_sec"],
            "score": int(result["score"]),
            "image_base64": result["image_base64"]
        })

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.route("/stt", methods=["POST"])
def stt():
    api_key = os.environ.get("GOOGLE_API_KEY", None)
    if not api_key:
        return jsonify({"error": "GOOGLE_API_KEY is missing"}), 500

    if "video" not in request.files:
        return jsonify({"error": "No video provided"}), 400

    temp_video = f"temp_video_{os.getpid()}.mp4"
    request.files["video"].save(temp_video)

    try:
        result = model.analyze_video_content(temp_video, api_key)
        return jsonify({
            "message": "success",
            "summary": result["summary"],
            "title": result["title"]
        })

    finally:
        if os.path.exists(temp_video):
            os.remove(temp_video)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
