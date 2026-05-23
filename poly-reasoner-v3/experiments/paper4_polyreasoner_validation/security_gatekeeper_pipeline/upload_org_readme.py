"""
Upload the NeurAlchemy org README to HuggingFace.
Uses the update_repo_settings approach since .profile naming is restricted.
"""

import os
import tempfile
from huggingface_hub import HfApi

HF_ORG = "neuralchemy"
README_PATH = r"f:\AI-IN-THE-LOOP\dataset_pipeline\hf_org_readme.md"

def main():
    api = HfApi()

    # Upload as org profile via metadata API
    # HF org READMEs go to a special "profile" space
    repo_id = f"{HF_ORG}/neuralchemy-profile"

    print(f"Creating/updating: {repo_id}")
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="static",
        exist_ok=True,
        private=False,
    )

    print("Uploading README.md as index.html content...")
    api.upload_file(
        path_or_fileobj=README_PATH,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="space",
        commit_message="Update org profile with complete research inventory",
    )

    print(f"\nDone! View at: https://huggingface.co/spaces/{repo_id}")
    print(f"\nNote: To set this as the org landing page, go to:")
    print(f"  https://huggingface.co/organizations/{HF_ORG}/settings")
    print(f"  and paste the README content into the org description.")

if __name__ == "__main__":
    main()
