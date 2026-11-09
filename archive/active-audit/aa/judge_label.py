"""Expensive detector D: an LLM judge for contextual sensitivity, run offline with vLLM on the local GPU.

Labels a deterministic subsample of the FineWeb shard and writes runs/judge/labels_<n>.parquet with
columns id, url, judge_sensitive (0/1), judge_category, judge_raw. This is the label the active
auditor is actually meant to find; regex is only a cheap feature.

Usage (inside .venv-judge):
  python aa/judge_label.py --model models/Qwen3-4B --n 20000 --seed 0 --out runs/judge/labels_20k.parquet
"""
import argparse, json, os, re, sys, time
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aa import data as D  # noqa: E402

RUBRIC = """You are auditing a document from a web crawl before it is used to train a language model.
Decide whether the document contains SENSITIVE PERSONAL OR SECRET INFORMATION that should be excluded from training.

Mark SENSITIVE if the document contains any of:
- personal contact details of a private individual (home address, personal phone, personal email) in a context that identifies the person;
- credentials, API keys, private keys, passwords, session tokens;
- medical, financial, legal, or other private details about an identifiable private person;
- government identifiers (SSN, passport, national ID numbers), payment card numbers.

Do NOT mark as sensitive: business contact details of organizations, public figures acting in their public role, generic placeholders or examples, fictional content, or aggregate statistics.

Document (truncated):
<<<
{doc}
>>>

Answer with one line of JSON: {{"sensitive": true|false, "category": "<contact|credential|medical|financial|legal|government_id|payment|none>", "reason": "<one short sentence>"}}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_chars", type=int, default=6000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", default=None, help="parquet with an id column: label exactly these docs instead of a random sample")
    a = ap.parse_args()
    from vllm import LLM, SamplingParams

    if a.ids:
        ids = set(pd.read_parquet(a.ids, columns=["id"])["id"])
        df = D.load_fineweb(columns=("id", "url", "text")); df = df[df.id.isin(ids)].reset_index(drop=True)
    else:
        df = D.load_fineweb(n=a.n, seed=a.seed, columns=("id", "url", "text"))
    prompts = [[{"role": "user", "content": RUBRIC.format(doc=t[: a.max_chars])}] for t in df["text"].astype(str)]
    llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=4096, enable_prefix_caching=True)
    sp = SamplingParams(temperature=0.0, max_tokens=120)
    t = time.time()
    outs = llm.chat(prompts, sp, use_tqdm=True, chat_template_kwargs={"enable_thinking": False})
    dt = time.time() - t
    sens, cat, raw = [], [], []
    for o in outs:
        txt = o.outputs[0].text.strip()
        raw.append(txt)
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        try:
            j = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            j = {}
        sens.append(int(bool(j.get("sensitive", False))))
        cat.append(str(j.get("category", "parse_error" if not j else "none")))
    df = df.drop(columns=["text"])
    df["judge_sensitive"], df["judge_category"], df["judge_raw"] = sens, cat, raw
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_parquet(a.out, index=False)
    print(json.dumps({"n": len(df), "seed": a.seed, "model": a.model, "docs_per_s": len(df) / dt, "wall_s": dt,
                      "sensitive_rate": float(df["judge_sensitive"].mean()),
                      "categories": df["judge_category"].value_counts().to_dict()}, indent=1))


if __name__ == "__main__":
    main()
