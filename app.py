# app.py
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import model

app = Flask(__name__)
CORS(app)  # 모든 도메인 허용

# -------------------------
# POST /analyze
# -------------------------
@app.route("/analyze", methods=["POST"])
def analyze():
    """
    업로드된 영상에서 오디오 STT + 요약 생성
    """

    # 1) API 키
    api_key = os.environ.get("GOOGLE_API_KEY", None)
    if not api_key:
        return jsonify({"error": "환경변수 GOOGLE_API_KEY 가 필요합니다."}), 500

    # 2) 영상 업로드 체크
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    temp_video_path = f"temp_video_{os.getpid()}.mp4"
    file.save(temp_video_path)

    try:
        result = model.analyze_video_content(temp_video_path, api_key)
        return jsonify({
            "message": "success",
            "transcript": result["transcript"],
            "summary": result["summary"],
            "title": result["title"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)


if __name__ == "__main__":
    # EC2 환경에서는 host=0.0.0.0 필요
    app.run(host="0.0.0.0", port=8000, debug=True)
