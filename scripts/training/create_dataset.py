import os
from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="/data1/DanielLi/pokechamp/bayesian_dataset",
    repo_id="Daniel10720/pokechamp_vgc_teams",
    repo_type="dataset",
)