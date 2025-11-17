# WooriZip-AI
openapi: 3.0.0
info:
  title: "WooriZip-AI 썸네일 분석기 API"
  description: "동영상 파일을 업로드하면 AI가 최적의 썸네일 프레임을 분석하여 Base64 이미지로 반환합니다."
  version: "1.0.0"

servers:
  - url: "http://<YOUR_EC2_SERVER_IP>:5000"
    description: "개발/프로덕션 서버 (EC2)"

paths:
  /analyze:
    post:
      summary: "동영상 썸네일 분석"
      description: "동영상 파일을 받아 AI가 '기쁨' 점수 및 품질을 기반으로 베스트 썸네일을 선정합니다."
      operationId: "analyzeVideo"
      
      requestBody:
        description: "분석할 동영상 파일"
        required: true
        content:
          multipart/form-data:
            schema:
              type: "object"
              properties:
                video:
                  type: "string"
                  format: "binary"
                  description: "업로드할 동영상 파일 (mp4, mov 등)"
              required:
                - "video"
                
      responses:
        "200":
          description: "분석 성공"
          content:
            application/json:
              schema:
                type: "object"
                properties:
                  message:
                    type: "string"
                    example: "Analysis successful"
                  best_time_sec:
                    type: "number"
                    format: "float"
                    example: 5.75
                  score:
                    type: "integer"
                    example: 410
                  image_base64:
                    type: "string"
                    format: "base64"
                    description: "추천된 썸네일 이미지의 Base64 인코딩 문자열 (data: prefix 없음)"
                    example: "iVBORw0KGgoAAAANSUhEUgA..."

        "400":
          description: "잘못된 요청 (예: 파일 없음)"
          content:
            application/json:
              schema:
                type: "object"
                properties:
                  error:
                    type: "string"
                    example: "No video file provided"

        "500":
          description: "서버 내부 오류 (예: AI 분석 실패)"
          content:
            application/json:
              schema:
                type: "object"
                properties:
                  error:
                    type: "string"
                    example: "Failed to analyze video"
