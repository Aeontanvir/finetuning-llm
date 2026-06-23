"""Publish a trained adapter or merged model to the Hugging Face Hub.

Set your token first:
    export HF_TOKEN=hf_xxx        # or run `huggingface-cli login`

Example:
    python src/push_to_hub.py --path results/lora-adapter \\
        --repo your-username/my-finetuned-llm --private
"""
from __future__ import annotations

import argparse
import os

from huggingface_hub import HfApi, create_repo


def push(path: str, repo: str, private: bool) -> None:
    token = os.environ.get("HF_TOKEN")
    # Create the repo (no-op if it already exists).
    create_repo(repo, private=private, exist_ok=True, token=token)
    print(f"Uploading {path} -> https://huggingface.co/{repo}")
    HfApi().upload_folder(
        folder_path=path,
        repo_id=repo,
        token=token,
        commit_message="Upload fine-tuned model",
    )
    print("Upload complete.")
    print(f"Load it with: AutoModelForCausalLM.from_pretrained('{repo}')")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Local folder to upload")
    parser.add_argument("--repo", required=True, help="username/repo-name on the Hub")
    parser.add_argument("--private", action="store_true", help="Make the repo private")
    a = parser.parse_args()
    push(a.path, a.repo, a.private)
