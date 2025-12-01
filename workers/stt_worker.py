from models.thumb_stt import analyze_video_content

def run_stt_worker(task_q, result_q):
    print("🔥 STT Worker started.")

    while True:
        task = task_q.get()
        video_path = task["path"]
        api_key = task["api_key"]

        try:
            result = analyze_video_content(video_path, api_key)
            result_q.put({
                "message": "success",
                **result
            })
        except Exception as e:
            result_q.put({"error": str(e)})
