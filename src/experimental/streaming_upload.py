"""Handling large image uploads with streaming."""
from pathlib import Path
def save_streaming_upload(file_obj, dest: str, chunk_size=8192):
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as out:
        while True:
            chunk = file_obj.read(chunk_size)
            if not chunk: break
            out.write(chunk)
    return dest
def validate_stream_size(file_obj, max_mb=50):
    file_obj.seek(0, 2)
    size = file_obj.tell()
    file_obj.seek(0)
    return size <= max_mb * 1024 * 1024
