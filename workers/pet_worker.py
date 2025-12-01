from models.pet_daily import classify_media
from models.pet_shorts import find_pet_segments, compile_pet_shorts

def run_pet_worker(task_q, result_q):
    print("🔥 Pet Worker started.")

    while True:
        task = task_q.get()
        video_path = task["path"]
        mode = task.get("mode", "daily")

        try:
            # 반려동물 분류(daily)
            if mode == "daily":
                result = classify_media(video_path)
                result_q.put({
                    "message": "success",
                    "result": result
                })

            # 반려동물 숏츠 (구간 탐지)
            elif mode == "shorts":
                segments = find_pet_segments(video_path)

                # shorts/generated 폴더에 저장
                output_path = compile_pet_shorts(video_path, segments)

                result_q.put({
                    "message": "success",
                    "segments": segments,
                    "output_path": output_path
                })

        except Exception as e:
            result_q.put({"error": str(e)})
