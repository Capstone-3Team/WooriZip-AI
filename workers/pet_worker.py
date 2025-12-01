from models.pet_daily import classify_media
from models.pet_shorts import find_pet_segments, compile_pet_shorts

def run_pet_worker(task_q, result_q):
    print("🔥 Pet Worker started.")

    while True:
        task = task_q.get()    # { id, path, mode }
        video_path = task["path"]
        mode = task.get("mode", "daily")

        try:
            # -----------------------
            # DAILY 모드 (사진/영상 분류)
            # -----------------------
            if mode == "daily":
                result = classify_media(video_path)

                result_q.put({
                    "message": "success",
                    "result": result
                })

            # -----------------------
            # SHORTS 모드 (구간 탐지 → 숏츠 생성)
            # -----------------------
            elif mode == "shorts":
                segments = find_pet_segments(video_path)

                # shorts/generated 에 저장됨
                output_path = compile_pet_shorts(video_path, segments)

                result_q.put({
                    "message": "success",
                    "segments": segments,
                    "output_path": output_path
                })

            else:
                result_q.put({"error": f"Unknown mode: {mode}"})

        except Exception as e:
            result_q.put({"error": str(e)})
