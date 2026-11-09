"""Synthetic items, call results and frames shared by the P3 tests."""
import numpy as np
import pandas as pd

import call as call_mod


def make_items(n=12, split="burn"):
    kinds = ["control", "N-big", "N-mid"]
    out = []
    for i in range(n):
        kind = kinds[i % 3]
        out.append({"item_id": f"it{i:02d}", "entry_key": f"e{i // 2}", "split": split, "kind": kind,
                    "truth": "clean" if kind == "control" else kind, "text": f"L001| text {i}",
                    "plant": None if kind == "control" else {"lines": [5], "old": "a", "new": "b", "accept_lines": [4, 5, 6]}})
    return out


def good_output(conf=70, lines=()):
    return {"problems": [{"line": ln, "kind": "number", "quote": "q", "explanation": "e"} for ln in lines], "judgment_confidence": conf}


def fake_result(obj=None, served="haiku", ok=True, out_tokens=100, cost=0.002, attempts=1):
    return call_mod.CallResult(obj=obj if obj is not None else good_output(), text="", ok=ok, error=None if ok else "boom",
                               model_name=served, model_id=served, provider="claude",
                               usage={"input": 2, "output": out_tokens, "cache_read": 0, "cache_write": 0},
                               cost_usd_plan=cost, cost_usd_nominal=cost, latency_s=1.5, attempts=attempts,
                               requested_model="haiku", served_model=served, error_class=None if ok else "model")


def make_frame(n=60, seed=0, levels=("low", "high"), informative_high=True):
    """Complete-case frame: entries of 2 items, Haiku accuracy about 0.7, Sonnet about 0.9 and 6x the cost."""
    rng = np.random.default_rng(seed)
    d = {"item_id": [f"i{i}" for i in range(n)], "entry_key": [f"e{i // 2}" for i in range(n)], "split": "test", "kind": "N-big",
         "truth": "N-big", "w": rng.uniform(0.5, 1.5, n), "s_cor": (rng.random(n) < 0.9).astype(float), "s_cost": 0.012 * np.ones(n)}
    for lv in levels:
        cor = (rng.random(n) < 0.7).astype(float)
        informative = informative_high and lv == "high"
        conf = np.where(cor == 1, 80, 60) + rng.integers(-5, 6, n) if informative else rng.integers(55, 85, n)
        d.update({f"cor_{lv}": cor, f"conf_{lv}": conf.astype(float), f"cost_{lv}": (0.002 if lv == "low" else 0.004) * np.ones(n),
                  f"tok_{lv}": (400.0 if lv == "low" else 900.0) * np.ones(n), f"lat_{lv}": 5.0 * np.ones(n)})
    return pd.DataFrame(d)
