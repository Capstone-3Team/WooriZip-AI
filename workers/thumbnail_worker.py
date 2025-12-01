from models.thumb_stt import find_best_thumbnail

def run_thumbnail_worker(task_q, result_q):
    print("🔥 Thumbnail Worker started.")

    while True:
        task = task_q.get()
        video_path = task["path"]

        try:
            result = find_best_thumbnail(video_path)
            if result is None:
                result_q.put({"error": "No valid thumbnail found"})
            else:
                result_q.put({
                    "message": "success",
                    **result
                })
        except Exception as e:
            result_q.put({"error": str(e)})
