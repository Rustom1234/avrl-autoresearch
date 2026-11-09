#!/usr/bin/env python
"""E10 model fetch: download the OLMo-2-0425-1B base and Instruct checkpoints used as policies in the E10 code-RL runs.

Input: the Hugging Face hub repos allenai/OLMo-2-0425-1B and allenai/OLMo-2-0425-1B-Instruct (network access).
Output: local copies under models/OLMo-2-0425-1B and models/OLMo-2-0425-1B-Instruct (read by E10p0_rollouts.py and p1_code_grpo.py).

Usage: .venv/bin/python avrl/exp/E10_fetch_models.py   (cwd = repository root)
"""
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
