import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback

import model  # 통합된 model.py

app = Flask(__name__)
CORS(app)

# =======================================
# 1. 썸네일 API
# =======================================
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
            return jsonify({"error": "Failed to analyze video"}), 500

        return jsonify({
            "message": "Thumbnail analysis successful",
            "time_sec": result["time_sec"],
            "score": int(result["score"]),
            "image_base64": result["image_base64"]
        })

    except Exception as e:
        print("\n🔥🔥🔥 썸네일 오류 발생 🔥🔥🔥")
        traceback.print_exc()
        print("🔥🔥🔥 END 🔥🔥🔥\n")
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)



# =======================================
# 2. STT + 요약 + 제목 생성 API
# =======================================
@app.route("/stt", methods=["POST"])
def stt():
    api_key = os.environ.get("GOOGLE_API_KEY", None)
    if not api_key:
        return jsonify({"error": "환경변수 GOOGLE_API_KEY 가 필요합니다."}), 500

    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    temp_video_path = f"temp_video_{os.getpid()}.mp4"
    file.save(temp_video_path)

    try:
        result = model.analyze_video_content(temp_video_path, api_key)

        return jsonify({
            "message": "success",
            "summary": result["summary"],
            "title": result["title"]
        })

    except Exception as e:
        print("\n🔥🔥🔥 STT 오류 발생 🔥🔥🔥")
        traceback.print_exc()
        print("🔥🔥🔥 END 🔥🔥🔥\n")
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
