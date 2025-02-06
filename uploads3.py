import os
from pathlib import Path
from tqdm import tqdm
import boto3
from concurrent.futures import ThreadPoolExecutor, as_completed


class CloudflareR2Uploader:
    def __init__(self, account_id, access_key_id, secret_access_key, max_workers=4):
        self.client = boto3.client(
            "s3",
            endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name="auto",
        )
        self.max_workers = max_workers

    def get_existing_files(self, bucket_name):
        try:
            paginator = self.client.get_paginator("list_objects_v2")
            filelist = []
            for page in paginator.paginate(Bucket=bucket_name):
                if "Contents" in page:
                    for obj in page["Contents"]:
                        filelist.append(obj["Key"])
            return set(filelist)
        except Exception as e:
            print(f"Error: {str(e)}")
            return set()

    def upload_file(self, bucket_name, file_path, s3_key):
        try:
            self.client.upload_file(file_path, bucket_name, s3_key)
            return True, s3_key
        except Exception as e:
            return False, f"Error: {s3_key} - {str(e)}"

    def upload_directory(self, local_dir, bucket_name, prefix=""):
        existing_files = self.get_existing_files(bucket_name)
        all_files = []
        for root, _, files in os.walk(local_dir):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, local_dir)
                s3_key = f"{prefix}/{relative_path}".replace("\\", "/")

                if s3_key not in existing_files:
                    all_files.append((file_path, s3_key))

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for file_path, s3_key in all_files:
                future = executor.submit(
                    self.upload_file, bucket_name, file_path, s3_key
                )
                futures.append(future)

            with tqdm(total=len(futures), desc="上傳檔案") as pbar:
                for future in as_completed(futures):
                    success, result = future.result()
                    if not success:
                        print(result)
                    pbar.update(1)


if __name__ == "__main__":
    ACCOUNT_ID = ""
    ACCESS_KEY_ID = ""
    SECRET_ACCESS_KEY = ""
    BUCKET_NAME = "mygo"
    LOCAL_DIR = "output"
    PREFIX = "animelines"

    uploader = CloudflareR2Uploader(
        ACCOUNT_ID, ACCESS_KEY_ID, SECRET_ACCESS_KEY, max_workers=24
    )
    uploader.upload_directory(LOCAL_DIR, BUCKET_NAME, PREFIX)
