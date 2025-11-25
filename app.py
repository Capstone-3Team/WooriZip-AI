# app.py
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

        temp_path = "temp_upload"
        file.save(temp_path)

        result = model.classify_media(temp_path, PROJECT_ID)

        return jsonify({
            "message": "success",
            "data": result
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    app.run(debug=True)
