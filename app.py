# app.py
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

# 통합된 AI 로직
import model

app = Flask(__name__)
CORS(app)  # 프론트엔드 도메인 여러 개에서 호출할 수 있도록 CORS 허용


# =======================================
# 1. 웃는 얼굴 썸네일 API
# =======================================
@app.route("/thumbnail", methods=["POST"])
def thumbnail():
    """
    업로드된 영상에서 '웃는 얼굴' 기반 베스트 썸네일 추출
    """
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files["video"]
    temp_path = f"temp_thumbnail_{os.getpid()}.mp4"
    video_file.save(temp_path)

    try:
        result = model.find_best_thumbnail(temp_path)

        if result is None:
            return jsonify({"error": "Failed to analyze video"}), 500

        return jsonify(
            {
                "message": "Thumbnail analysis successful",
                "best_time_sec": result["time_sec"],
                "score": int(result["score"]),
                "image_base64": result["image_base64"],
            }
        )

    except Exception as e:
        print(f"썸네일 분석 중 오류 발생: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# =======================================
# 2. STT + 요약 + 제목 생성 API
# =======================================
@app.route("/analyze", methods=["POST"])
def analyze():
    """
    업로드된 영상에서 오디오 STT + 요약 + 제목 생성
    """
    # 1) Gemini API 키
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
        return jsonify(
            {
                "message": "success",
                "transcript": result["transcript"],
                "summary": result["summary"],
                "title": result["title"],
            }
        )
    except Exception as e:
        print(f"STT/요약 분석 중 오류 발생: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)


if __name__ == "__main__":
    # EC2 환경에서는 host=0.0.0.0 필수
    app.run(host="0.0.0.0", port=8000, debug=True)
