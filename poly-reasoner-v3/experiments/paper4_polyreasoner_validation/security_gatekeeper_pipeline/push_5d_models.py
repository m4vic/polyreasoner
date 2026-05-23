import os
from huggingface_hub import HfApi

HF_ORG = "neuralchemy"
MODELS_DIR = r"f:\AI-IN-THE-LOOP\dataset_pipeline\models"

api = HfApi()

# The 5 Dimensional Models
dimensions = [
    "intent",
    "technique",
    "surface",
    "severity",
    "binary"
]

def main():
    print("="*60)
    print(" NeurAlchemy — Pushing the 5 Dimensional Models to HuggingFace")
    print("="*60)

    for dim in dimensions:
        # Use the EXACT repo names currently on your HF account
        repo_name = f"distilbert-specialist-{dim}-threat-matrix"
        repo_id = f"{HF_ORG}/{repo_name}"
        local_path = os.path.join(MODELS_DIR, f"specialist_{dim}", "final")
        
        if os.path.exists(local_path):
            print(f"\n📦 Uploading: {repo_id}")
            print(f"📂 Source:    {local_path}")
            try:
                # Ensure repo exists
                api.create_repo(repo_id, exist_ok=True, private=False)
                
                # Upload weights (This will OVERWRITE the older models on HF)
                api.upload_folder(
                    folder_path=local_path,
                    repo_id=repo_id,
                    commit_message=f"Update {dim} dimension model (inverse-frequency weighted retrain)"
                )
                print(f"✅ Successfully uploaded {dim}!")
            except Exception as e:
                print(f"❌ Error uploading {dim}: {e}")
        else:
            print(f"❌ Path not found: {local_path}")

if __name__ == "__main__":
    main()
