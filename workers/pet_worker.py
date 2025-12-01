from models.pet_daily import classify_media
from models.pet_shorts import find_pet_segments, compile_pet_shorts

def run_pet_worker(task_q, result_q):
    print("🔥 Pet Worker started.")

    while True:
        task = task_q.get()
        mode = task["mode"]
        video_path = task["path"]

        try:
            # DAILY 모드
            if mode == "daily":
                res = classify_media(video_path)
                result_q.put({"message": "success", "result": res})
                continue

            # SHORTS 모드
            if mode == "shorts":
                segments = find_pet_segments(video_path)
                output = compile_pet_shorts(video_path, segments)

                result_q.put({
                    "message": "success",
                    "segments": segments,
                    "output_path": output
                })
                continue

            result_q.put({"error": f"Unknown mode: {mode}"})

        except Exception as e:
            result_q.put({"error": str(e)})
