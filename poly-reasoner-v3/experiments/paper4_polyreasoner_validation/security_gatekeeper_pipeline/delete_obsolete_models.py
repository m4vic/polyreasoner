import os
from huggingface_hub import HfApi

api = HfApi()

old_repos_to_delete = [
    "neuralchemy/distilbert-expert-direct-injection-threat-matrix",
    "neuralchemy/distilbert-expert-indirect-injection-threat-matrix",
    "neuralchemy/distilbert-expert-obfuscation-threat-matrix",
    "neuralchemy/distilbert-expert-role-hijack-threat-matrix",
    "neuralchemy/distilbert-expert-system-extraction-threat-matrix",
    "neuralchemy/distilbert-expert-tool-abuse-threat-matrix",
    "neuralchemy/distilbert-multiclass-threat-matrix",
    "neuralchemy/classical-ml-threat-matrix"
]

def main():
    print("="*60)
    print(" NeurAlchemy — Deleting Obsolete HuggingFace Models")
    print("="*60)
    
    for repo_id in old_repos_to_delete:
        try:
            print(f"🗑️ Deleting: {repo_id}")
            api.delete_repo(repo_id=repo_id, repo_type="model")
            print(f"✅ Deleted successfully.")
        except Exception as e:
            print(f"⚠️ Skipped or error (might already be deleted): {e}")

if __name__ == "__main__":
    main()
