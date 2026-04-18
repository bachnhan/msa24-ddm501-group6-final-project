import opendatasets as od
import os
import json
from pathlib import Path
from dotenv import load_dotenv

def download_kaggle_dataset():
    # 1. Load Credentials
    load_dotenv(override=True)
    for secret_file in [".secret", "/etc/secrets/.env", "/etc/secrets/.secret"]:
        if os.path.exists(secret_file):
            load_dotenv(secret_file, override=True)
    
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")
    
    if not username or not key:
        print("❌ Error: KAGGLE_USERNAME or KAGGLE_KEY not found in .secret file.")
        return

    # 2. Create kaggle.json temporarily for opendatasets
    kaggle_creds = {"username": username, "key": key}
    with open("kaggle.json", "w") as f:
        json.dump(kaggle_creds, f)
    
    # 3. Download
    dataset_url = "https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset"
    print(f"🚀 Downloading dataset from {dataset_url}...")
    
    try:
        od.download(dataset_url)
        
        # 4. Move to data/ folder
        repo_root = Path(__file__).parent.parent
        target_data_dir = repo_root / "data"
        target_data_dir.mkdir(exist_ok=True)
        
        # The library creates a folder named 'customer-churn-dataset'
        downloaded_folder = Path("customer-churn-dataset")
        if downloaded_folder.exists():
            for csv_file in downloaded_folder.glob("*.csv"):
                dest = target_data_dir / csv_file.name
                print(f"      Moving {csv_file.name} to {dest}...")
                csv_file.replace(dest)
            
            # Clean up
            import shutil
            shutil.rmtree(downloaded_folder)
            print("✅ Dataset successfully downloaded and moved to data/ folder.")
        else:
            print("⚠️ Warning: Downloaded folder not found. Check local directory.")

    except Exception as e:
        print(f"❌ Failed to download: {e}")
    finally:
        # Final clean up of creds file
        if os.path.exists("kaggle.json"):
            os.remove("kaggle.json")

if __name__ == "__main__":
    download_kaggle_dataset()
