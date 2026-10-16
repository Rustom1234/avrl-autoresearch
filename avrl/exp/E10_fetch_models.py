#!/usr/bin/env python
import os
from huggingface_hub import snapshot_download

ROOT = "/homes/rdubash/active-audit"
MODELS_DIR = os.path.join(ROOT, "models")

REPOS = {
    "OLMo-2-0425-1B": "allenai/OLMo-2-0425-1B",
    "OLMo-2-0425-1B-Instruct": "allenai/OLMo-2-0425-1B-Instruct",
}

for name, repo_id in REPOS.items():
    local_dir = os.path.join(MODELS_DIR, name)
    print(f"downloading {repo_id} -> {local_dir}")
    path = snapshot_download(repo_id=repo_id, local_dir=local_dir)
    print(f"done: {path}")
