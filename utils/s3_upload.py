import boto3
import os
from datetime import datetime

s3 = boto3.client("s3")

def upload_to_s3(file_path, key_prefix="shorts"):
    bucket = os.environ.get("AWS_BUCKET_NAME")
    region = os.environ.get("AWS_REGION")

    filename = os.path.basename(file_path)
    key = f"{key_prefix}/{filename}"

    s3.upload_file(file_path, bucket, key, ExtraArgs={'ContentType': 'video/mp4'})

    # 업로드 완료 후 퍼블릭 URL 구성
    return f"https://{bucket}.s3.{region}.amazonaws.com/{key}"
