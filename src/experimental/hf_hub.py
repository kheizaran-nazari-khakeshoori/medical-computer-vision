"""Releasing model weights via huggingface hub."""

try:

    def push_model(model_path, repo_id):
        from huggingface_hub import HfApi

        api = HfApi()
        api.upload_file(path_or_fileobj=model_path, path_in_repo="model.pth", repo_id=repo_id)

except Exception:

    def push_model(*a, **k):
        print("huggingface_hub not installed")
