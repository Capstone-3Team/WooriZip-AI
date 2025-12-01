import os
import json
import traceback
from models.thumb_stt import analyze_video_content


def run_stt_worker(stt_q, stt_res_q):
    print("🔥 STT Worker started.", flush=True)

    while True:
        try:
            task = stt_q.get()

            # 종료 신호
            if task is None:
                print("🛑 STT Worker stopped.", flush=True)
                break

            task_id = task.get("id")
            video_path = task.get("path")
            api_key = task.get("api_key")

            print(f"🔍 [STT Worker] Processing ID={task_id}, file={video_path}", flush=True)

            # ============================================================
            # 1) STT + 요약 + 제목 생성
            # ============================================================
            try:
                result = analyze_video_content(video_path, api_key)

                # 정상 결과
                stt_res_q.put({
                    "id": task_id,
                    "summary": result.get("summary", ""),
                    "title": result.get("title", "")
                })
                print(f"✅ [STT Worker] Done ID={task_id}", flush=True)

            except Exception as e:
                print("❌ [STT Worker ERROR] 분석 실패", flush=True)
                traceback.print_exc()

                stt_res_q.put({
                    "id": task_id,
                    "error": str(e)
                })

        except Exception as e:
            # 예상치 못한 전체 루프 에러 방지
            print("🔥 [STT Worker] Fatal Error in loop", flush=True)
            traceback.print_exc()

            stt_res_q.put({
                "id": "unknown",
                "error": f"Fatal Worker Error: {e}"
            })
