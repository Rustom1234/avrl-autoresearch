#!/usr/bin/env python3
"""Curation loop with no model: each slot tries one random change to a small menu of hand-written curation settings.

    python curation/loop/menu.py --run-dir curation/runs/C1/N_01 --pool v2 --slots 12 --seed 1 --sigma 0.0073 \
        [--gpu-host willow] [--resume]

The baseline a routed loop has to beat for its model dollars to mean anything. Same engine, trainer, seeds and z rule
as the model loops; only the proposals differ. The menu is the set of no-model curators of the pilot
(task/baselines/curators.py) made combinable:
  dedup  drop exact duplicates                                  off (start), on
  junk   drop documents caught by the hand heuristic            off (start), on
  rank   order by TF-IDF similarity to the target sample        none (start), word, char (3-grams of characters)
  keep   with a ranking, keep the top k x T tokens              0.3, 0.5, 0.8, 1.0 (start), 1.5
A move changes one setting (keep moves one step). The move of a slot is drawn at random from the loop seed among the
moves not yet rejected on the current best settings; the list of rejected moves is cleared when a move is accepted
or when every move has been tried.
"""
import argparse, json, random, sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine as E
from routed import purge_slot, RemoteResult

START = {"dedup": False, "junk": False, "rank": "none", "keep": 1.0}
RANKS = ("none", "word", "char")
KEEPS = (0.3, 0.5, 0.8, 1.0, 1.5)

TEMPLATE = '''"""Written by the no-model menu search (curation/loop/menu.py)."""
import hashlib, os, random
import numpy as np

CFG = %r
T = 9830400
RNG = random.Random(int(os.environ.get("CURATE_SEED", 0)))


def junk(d):
    if len(d) < 200:
        return True
    lines = [l for l in d.split("\\n") if l.strip()]
    if len(lines) >= 5 and len(set(lines)) / len(lines) < 0.5:
        return True
    if sum(1 for l in lines if len(l) >= 60 and " " not in l.strip()) / max(1, len(lines)) > 0.3:
        return True
    return sum(c.isdigit() for c in d) / len(d) > 0.3


def select(docs, target_sample):
    ids = list(range(len(docs)))
    if CFG["dedup"]:
        seen, keep = set(), []
        for i in ids:
            h = hashlib.md5(docs[i].strip().encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                keep.append(i)
        ids = keep
    if CFG["junk"]:
        ids = [i for i in ids if not junk(docs[i])]
    if CFG["rank"] != "none":
        import tiktoken
        from sklearn.feature_extraction.text import TfidfVectorizer
        if CFG["rank"] == "word":
            vec, cut = TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=50000, stop_words="english",
                                       dtype=np.float32), None
        else:
            vec, cut = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), sublinear_tf=True, min_df=2,
                                       max_features=50000, dtype=np.float32), 1000
        X = vec.fit_transform([docs[i][:cut] for i in ids])
        c = np.asarray(vec.transform([t[:cut] for t in target_sample]).mean(axis=0)).ravel()
        order = np.argsort(-(X @ (c / np.linalg.norm(c))))
        enc = tiktoken.get_encoding("gpt2")
        ntok = np.array([len(t) + 1 for t in enc.encode_ordinary_batch([docs[ids[j]] for j in order], num_threads=8)])
        ids = [ids[j] for j in order[: int(np.searchsorted(np.cumsum(ntok), CFG["keep"] * T)) + 1]]
    out = []
    for _ in range(4):
        e = list(ids)
        RNG.shuffle(e)
        out += e
    return out
'''


def render(cfg):
    return TEMPLATE % (cfg,)


def moves_of(cfg):
    """Every single-setting change of cfg, as [setting, new value]."""
    out = [["dedup", not cfg["dedup"]], ["junk", not cfg["junk"]]]
    out += [["rank", r] for r in RANKS if r != cfg["rank"]]
    if cfg["rank"] != "none":
        i = KEEPS.index(cfg["keep"])
        out += [["keep", KEEPS[j]] for j in (i - 1, i + 1) if 0 <= j < len(KEEPS)]
    return out


class MenuLoop(RemoteResult, E.Loop):
    def __init__(self, cfg, runner=None):
        super().__init__(cfg, models=SimpleNamespace(is_open=lambda name: False), runner=runner)
        self.s.setdefault("menu", {"config": dict(START), "tried": [], "pending": None})

    def ask(self, role, user, schema=None):
        """The engine asks for a plan and a file; both come from the move of this slot, at no cost."""
        key, val = self.s["menu"]["pending"]["move"]
        cfg = dict(self.s["menu"]["config"], **{key: val})
        obj = {"idea": "set %s to %s" % (key, val), "why": "menu"} if role == "planner" else None
        text = "" if role == "planner" else "```python\n%s```" % render(cfg)
        self.slot_calls.append({"role": role, "model": "menu", "ok": True, "usd": 0.0, "error_class": None})
        return SimpleNamespace(ok=True, obj=obj, text=text, error_class=None, cost_usd_nominal=0.0, served_model="menu")

    def settle(self):
        """Apply the outcome of the pending move once its slot is in the ledger (also after a stop between the two)."""
        m, lf = self.s["menu"], self.dir / "ledger.jsonl"
        p = m.get("pending")
        recs = [json.loads(l) for l in lf.read_text().splitlines()] if lf.exists() else []
        rec = next((r for r in recs if p and r["slot"] == p["slot"]), None)
        if rec is None:
            return
        if rec["decision"] == "accept":
            m["config"], m["tried"] = dict(m["config"], **{p["move"][0]: p["move"][1]}), []
        else:
            m["tried"].append(p["move"])
        m["pending"] = None
        self.save()

    def run_slot(self):
        s, m = self.s, self.s["menu"]
        self.settle()
        purge_slot(self.dir, s["slot"] + 1)
        moves = [mv for mv in moves_of(m["config"]) if mv not in m["tried"]]
        if not moves:
            m["tried"], moves = [], moves_of(m["config"])
        m["pending"] = {"slot": s["slot"] + 1, "move": random.Random(self.cfg["seed"] * 6007 + s["slot"] + 1).choice(moves)}
        self.save()
        super().run_slot()
        self.settle()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--pool", default="v2")
    ap.add_argument("--slots", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sigma", type=float, required=True)
    ap.add_argument("--gpu-host", default=None)
    ap.add_argument("--cell", default="N")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)
    run_dir = Path(a.run_dir).resolve()
    cf = run_dir / "config.json"
    if cf.exists():
        if not a.resume:
            raise SystemExit("%s exists; pass --resume to continue that loop" % cf)
        cfg = json.loads(cf.read_text())
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg = {"loop_id": run_dir.name, "run_dir": str(run_dir), "cell": a.cell, "pool": a.pool, "slots": a.slots,
               "seed": a.seed, "sigma": a.sigma, "gpu_host": a.gpu_host, "policy": "menu", "routing": None}
        cf.write_text(json.dumps(cfg, indent=1))
    lp = MenuLoop(cfg)
    if a.resume and lp.s["status"] != "new":
        lp.s["resumes"] = lp.s.get("resumes", 0) + 1
    lp.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
