#!/usr/bin/env python
"""
E10 data prep: fetch MBPP (full config) train+validation and test splits,
and HumanEval test split, write as parquet under avrl/data/.

Usage: run with /homes/rdubash/active-audit/.venv/bin/python
"""
import os
import pandas as pd
from datasets import load_dataset

ROOT = "/homes/rdubash/active-audit"
OUT = os.path.join(ROOT, "avrl", "data")
os.makedirs(OUT, exist_ok=True)

MBPP_COLS = ["task_id", "text", "test_list", "test_setup_code", "code"]

mbpp = load_dataset("google-research-datasets/mbpp", "full")

train_df = pd.concat(
    [mbpp["train"].to_pandas(), mbpp["validation"].to_pandas()],
    ignore_index=True,
)[MBPP_COLS]
test_df = mbpp["test"].to_pandas()[MBPP_COLS]

train_path = os.path.join(OUT, "e10_train.parquet")
test_path = os.path.join(OUT, "e10_test.parquet")
train_df.to_parquet(train_path, index=False)
test_df.to_parquet(test_path, index=False)

print(f"e10_train.parquet: {len(train_df)} rows (train {len(mbpp['train'])} + validation {len(mbpp['validation'])})")
print(f"e10_test.parquet: {len(test_df)} rows")

HE_COLS = ["task_id", "prompt", "test", "entry_point"]
he = load_dataset("openai/openai_humaneval")["test"].to_pandas()[HE_COLS]
he_path = os.path.join(OUT, "e10_humaneval.parquet")
he.to_parquet(he_path, index=False)
print(f"e10_humaneval.parquet: {len(he)} rows")
