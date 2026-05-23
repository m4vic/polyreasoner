"""
push_specialist_space.py
=========================
Uploads the new v2 Gradio Space to HuggingFace.

Creates: neuralchemy/threat-matrix-analyzer-v2

Usage:
  python push_specialist_space.py
"""

import os
from huggingface_hub import HfApi

HF_ORG   = "neuralchemy"
REPO_ID  = f"{HF_ORG}/threat-matrix-analyzer-v2"
SPACE_DIR = r"f:\AI-IN-THE-LOOP\dataset_pipeline\spaces\threat-matrix-analyzer-v2"

api = HfApi()

def main():
    print("=" * 60)
    print(f"  Uploading Space → {REPO_ID}")
    print("=" * 60)

    api.create_repo(repo_id=REPO_ID, repo_type="space", space_sdk="gradio",
                    exist_ok=True, private=False)
    print("  ✅ Repo created/verified")

    api.upload_folder(
        folder_path=SPACE_DIR,
        path_in_repo="",
        repo_id=REPO_ID,
        repo_type="space",
        commit_message="Upload Threat Matrix Analyzer v2 — 5-Dimensional BERT MoE",
    )
    print(f"\n  ✅ Space live at: https://huggingface.co/spaces/{REPO_ID}")

if __name__ == "__main__":
    main()
