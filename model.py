# model.py
"""
STT + 요약 + 키워드 추출 AI 분석 모듈
Google Gemini 2.5 Flash + pydub 기반
"""

import google.generativeai as genai
from pydub import AudioSegment
import os
import json


# -------------------------
# 1) 오디오 추출
# -------------------------
def extract_audio(video_path, audio_path="temp_audio.mp3"):
    """
    비디오 파일에서 오디오를 추출하여 mp3로 저장합니다.
    """
    try:
        video = AudioSegment.from_file(video_path)
        video.export(audio_path, format="mp3")
        return audio_path
    except Exception as e:
        raise RuntimeError(f"Audio extraction failed: {e}")


# -------------------------
# 2) Gemini AI 호출
# -------------------------
def analyze_video_content(video_path, api_key):
    """
    비디오 → 오디오 추출 → Gemini STT + 요약 실행 후
    transcript/summary 반환
    """

    if api_key is None or api_key.strip() == "" or api_key == "YOUR_API_KEY_HERE":
        raise ValueError("유효한 Google API Key가 필요합니다.")

    # API 키 설정
    genai.configure(api_key=api_key)

    # 1. 오디오 파일 생성
    audio_file_path = extract_audio(video_path)

    # 2. Gemini 서버에 오디오 업로드
    try:
        audio_file = genai.upload_file(path=audio_file_path)
    except Exception as e:
        raise RuntimeError(f"Gemini audio upload failed: {e}")

    # 3. Gemini 모델
    model = genai.GenerativeModel("models/gemini-2.5-flash-preview-09-2025")

    # 4. 프롬프트
    prompt = """
    이 오디오 파일은 가족 일기입니다. 다음 작업을 수행하세요:

    1. [STT]: 오디오의 내용을 한국어 텍스트로 모두 받아쓰기
    2. [요약]: 중요한 내용만 한 문장으로 요약
       (대화체 말투 금지 — 사실 기반 요약)
    
    반드시 JSON 형태로만 응답:
    {
      "transcript": "...",
      "summary": "..."
    }
    """

    try:
        response = model.generate_content([audio_file, prompt])
        text = response.text.strip()

        # json 추출
        clean_text = text.lstrip("```json").rstrip("```").strip()
        results = json.loads(clean_text)
        transcript = results.get("transcript", "")
        summary = results.get("summary", "")

    except Exception as e:
        raise RuntimeError(f"Gemini 분석 중 오류: {e}")

    finally:
        # 로컬 임시 오디오 삭제
        if os.path.exists(audio_file_path):
            os.remove(audio_file_path)

        # 서버 업로드 파일 삭제
        try:
            genai.delete_file(audio_file.name)
        except:
            pass

    return {
        "transcript": transcript,
        "summary": summary
    }
