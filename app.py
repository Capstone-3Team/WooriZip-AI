import os
from flask import Flask, request, jsonify
from flask_cors import CORS

# [중요] 우리가 만든 AI 로직을 model.py에서 가져옵니다.
import model 

# --- 1. Flask 앱 초기화 ---
app = Flask(__name__)
CORS(app) # 모든 도메인에서의 요청을 허용 (테스트용)

# --- 2. Flask API 엔드포인트 정의 ---
@app.route('/analyze', methods=['POST'])
def analyze_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files['video']
    
    # 임시 파일 이름 생성 (보안상 안전한 방식 추천)
    temp_path = f"temp_video_{os.getpid()}.mov" 
    video_file.save(temp_path)

    try:
        # [핵심] AI 분석을 model.py의 함수에 맡깁니다.
        result = model.find_best_thumbnail(temp_path)
        
        if result is None:
            return jsonify({"error": "Failed to analyze video"}), 500
        
        # 성공 시, 썸네일(Base64)과 정보 반환
        return jsonify({
            "message": "Analysis successful",
            "best_time_sec": result['time_sec'],
            "score": int(result['score']),
            "image_base64": result['image_base64']
        })

    except Exception as e:
        print(f"분석 중 오류 발생: {e}")
        return jsonify({"error": str(e)}), 500
    
    finally:
        # 임시 파일 삭제
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
