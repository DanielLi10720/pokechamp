import os
import base64
import json
from pathlib import Path
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, HfFolder, create_repo

# === CONFIGURATION ===
TEAM_FOLDER = "/data1/DanielLi/pokechamp/bayesian_dataset"                     # Folder with .txt team files
DATASET_NAME = "pokechamp_vgc_teams"         # Name for your HF dataset repo
FORMAT_TAG = "gen9vgc2025regi"                     # Tag for key prefix (e.g., gen1ou)
OUTPUT_FILE = "/data1/DanielLi/pokechamp/hf_dataset"            # Local dataset export file
PRIVATE = False                           # True = private repo


def encode_team_file(filepath: str) -> str:
    """Read a .txt team file and return its Base64-encoded content."""
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read().strip()
    return base64.b64encode(text.encode("utf-8")).decode("utf-8")


def build_jsonl():
    """Create a Hugging Face–style JSONL dataset file from .txt teams."""
    team_dir = Path(TEAM_FOLDER)
    team_files = sorted(team_dir.glob("*.txt"))
    if not team_files:
        raise FileNotFoundError(f"No .txt files found in {TEAM_FOLDER}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for i, path in enumerate(team_files, start=1):
            encoded = encode_team_file(path)
            key = f"{FORMAT_TAG}/team_{i:06d}"
            entry = {"__key__": key, FORMAT_TAG: encoded}
            out.write(json.dumps(entry) + "\n")

    print(f"✅ Created {OUTPUT_FILE} with {len(team_files)} teams")
    return OUTPUT_FILE


def upload_to_hf(output_file: str):
    """Upload the dataset to Hugging Face."""
    # Authenticate
    token = HfFolder.get_token()
    if not token:
        raise RuntimeError("No Hugging Face token found. Run `huggingface-cli login` first.")

    api = HfApi()

    # Create the repo if it doesn't exist
    username = api.whoami(token=token)["name"]
    repo_id = f"{username}/{DATASET_NAME}"
    create_repo(repo_id, repo_type="dataset", private=PRIVATE, exist_ok=True)
    print(f"📂 Using dataset repo: {repo_id}")

    # Load JSONL into a HF Dataset
    ds = Dataset.from_json(output_file)
    dsdict = DatasetDict({"train": ds})

    # Push to Hugging Face Hub
    dsdict.push_to_hub(repo_id, token=token)
    print(f"🚀 Uploaded dataset successfully to https://huggingface.co/datasets/{repo_id}")


def main():
    file_path = build_jsonl()
    upload_to_hf(file_path)


if __name__ == "__main__":
    main()
