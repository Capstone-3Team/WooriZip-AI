# app.py
from werkzeug.utils import secure_filename
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import model

app = Flask(__name__)
CORS(app)

PROJECT_ID = os.environ.get("GCP_PROJECT_ID")


# -------------------------------------
# POST /classify → 사진/영상 반려동물 여부 판별
# -------------------------------------
@app.route("/classify", methods=["POST"])
def classify():
    try:
        file = request.files.get("file")
        if file is None:
            return jsonify({"error": "파일이 필요합니다."}), 400

        # ⭐ 수정 1: 원본 파일명을 포함하여 저장 (확장자 유지)
        filename = secure_filename(file.filename)
        temp_path = f"temp_upload_{filename}"

        file.save(temp_path)

        # Vision AI 분석
        result = model.classify_media(temp_path, PROJECT_ID)

        return jsonify({
            "message": "success",
            "data": result
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # ⭐ 수정 2: finally에서도 안전하게 삭제 처리
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except:
            pass


if __name__ == "__main__":
    app.run(debug=True)
