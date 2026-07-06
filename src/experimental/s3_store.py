"""Storing model artifacts in s3."""


def upload_to_s3(local_path: str, bucket: str, key: str):
    try:
        import boto3

        s3 = boto3.client("s3")
        s3.upload_file(local_path, bucket, key)
        return True
    except Exception as e:
        print(f"s3 upload stub: {e}")
        return False
