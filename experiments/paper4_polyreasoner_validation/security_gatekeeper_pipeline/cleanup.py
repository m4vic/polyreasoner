import os
import shutil

def main():
    root_dir = r"f:\AI-IN-THE-LOOP\dataset_pipeline"
    archive_dir = os.path.join(root_dir, "_archived_tools")
    
    os.makedirs(archive_dir, exist_ok=True)
    
    # Files to archive (one-off data collection/sanity tools that are successfully completed)
    files_to_archive = [
        "audit_dataset.py",
        "clean_dataset_labels.py",
        "download_hackaprompt.py",
        "fix_hub_dataset.py",
        "label_hackaprompt.py",
        "split_data.py",
        "upload_dataset_card.py",
        "build_core.py",
        "build_advanced.py"
    ]
    
    moved_count = 0
    for filename in files_to_archive:
        src = os.path.join(root_dir, filename)
        dest = os.path.join(archive_dir, filename)
        if os.path.exists(src):
            shutil.move(src, dest)
            print(f"Archived: {filename}")
            moved_count += 1
            
    print(f"\n✅ Successfully moved {moved_count} old scripts to the _archived_tools directory!")
    print("🧹 Your dataset_pipeline folder is now sparkling clean and focused on training models.")

if __name__ == "__main__":
    main()
