"""
Upload the Threat Matrix Analyzer Space to HuggingFace.
Creates: neuralchemy/threat-matrix-analyzer (Gradio Space)
"""

from huggingface_hub import HfApi
import os

HF_ORG = "neuralchemy"
SPACE_NAME = "threat-matrix-analyzer"
REPO_ID = f"{HF_ORG}/{SPACE_NAME}"
SPACE_DIR = r"f:\AI-IN-THE-LOOP\dataset_pipeline\spaces\threat-matrix-analyzer"

def main():
    api = HfApi()

    print(f"Creating Space: {REPO_ID}")
    api.create_repo(
        repo_id=REPO_ID,
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True,
        private=False,
    )

    print("Uploading Space files...")
    api.upload_folder(
        folder_path=SPACE_DIR,
        repo_id=REPO_ID,
        repo_type="space",
        commit_message="Deploy Threat Matrix Analyzer with all models",
    )

    print(f"\nDone! Space will build at:")
    print(f"  https://huggingface.co/spaces/{REPO_ID}")
    print(f"\nNote: First build may take 2-3 minutes to download all models.")

if __name__ == "__main__":
    main()
