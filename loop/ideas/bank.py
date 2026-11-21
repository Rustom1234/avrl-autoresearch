#!/usr/bin/env python3
"""B1, the stall bank: which source of ideas restarts a stalled loop?

A stalled loop is the state of a finished 40-slot loop (its incumbent program, ledger, notes and backlog). Six
sources are each asked once for N ideas: planner model in {haiku, sonnet, opus} x context in
  c  continued: the loop's own history in the prompt (results table, notes, backlog titles), as the engine sends it;
  f  fresh:     the incumbent program only, as the first planner call of a new loop would see it.
Same task sentence in both ("Propose N distinct ideas."), same program card. Every proposal is then tested against
the SAME frozen incumbent by the unchanged engine pipeline (coder, static check, job, z rule, replicate, critic)
with the loop's own coder. The coder's context is the same for every idea (the stalled loop's table and notes), the
incumbent never moves and no memo is written, so the outcome of an idea does not depend on the order of testing.
Near duplicates across sources (the engine's own rule: MiniLM cosine at least 0.90) are tested once and credited to
every source that proposed them; near duplicates of an idea the stalled loop already tried are not tested.

  bank.py ask  --root loop/runs/B1 --src loop/runs/BG1_R --loops bbb_HH_1 caco_HH_1 ... [--n 10]
  bank.py pool --root loop/runs/B1 [--shards 2]
  bank.py run  --config loop/runs/B1/<loop>/s0/config.json

Frozen engine and task files are imported, not edited.
"""
import argparse, copy, importlib.util, json, os, random, shutil, sys, threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOOP = HERE.parent
REPO = LOOP.parent
sys.path.insert(0, str(LOOP / "core"))
_spec = importlib.util.spec_from_file_location("bench_task", LOOP / "bench" / "task.py")
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)          # also puts loop/core and loop/cifar on sys.path
import arms, ledger, scheduler           # noqa: E402
from services import Clock, ModelService, Services  # noqa: E402

MODELS = ("haiku", "sonnet", "opus")
CTX = ("c", "f")
CARD_SLOTS = 80      # the card of the E1 continuation ("You have 80 slots in total"), the same for every source
TAU_BGE = 0.824      # "same idea" level of loop/ideas/rough_overlap.py (99th percentile of pairs known to differ)


def text_of(i):
    return "%s. %s" % (i.get("title", ""), i.get("spec", ""))


# ------------------------------------------------------------------ ask

def cmd_ask(a):
    root, src = Path(a.root).resolve(), Path(a.src).resolve()
    if a.registry:                      # a registry file with models the frozen one does not hold
        sys.path.insert(0, str(LOOP / "models"))
        import call as call_mod
        call_mod.REGISTRY_PATH = Path(a.registry).resolve()
    consts = json.loads((LOOP / "bench" / "consts.json").read_text())

    def one_loop(lid):
        s = src / lid
        st, cfg0 = json.loads((s / "state.json").read_text()), json.loads((s / "config.json").read_text())
        assert st["status"] == "done" and st["slot_index"] == st["n_slots"], lid
        task = cfg0["task"]
        P = bench.make_prompts(task, consts[task])
        d = root / lid
        d.mkdir(parents=True, exist_ok=True)
        inc0 = d / "incumbent0.py"
        if not inc0.exists():
            shutil.copy(st["incumbent"]["path"], inc0)
        inc_src = inc0.read_text()
        records = ledger.Ledger(s / "ledger.jsonl").read()
        fz = {"loop": lid, "task": task, "src": str(s), "records": records, "memo": st["memo"],
              "family_stats": st["family_stats"], "incumbent": st["incumbent"], "backlog": st["backlog"],
              "config": {k: cfg0[k] for k in ("arm", "cell", "task", "rep", "consts", "voter_model", "critic_model",
                                              "vote_k", "job_timeout_s", "timeout_s", "routing", "rng_seed")}}
        (d / "frozen.json").write_text(json.dumps(fz, indent=1, default=str))
        out = d / "proposals.jsonl"
        have = {(r["model"], r["ctx"]) for r in map(json.loads, open(out))} if out.exists() else set()
        models = ModelService(str(d), "B1_" + lid, max_tokens=8192)
        table_c = ledger.results_table(records, st["incumbent"])
        table_f = ledger.results_table([], st["incumbent"])
        for m in a.models:
            for ctx in a.ctx:
                if (m, ctx) in have:        # one try per source: a failed call stays recorded and is not repeated
                    continue
                user = (P.planner(inc_src, a.n, table_c, st["memo"], st["family_stats"], st["backlog"]) if ctx == "c"
                        else P.planner(inc_src, a.n, table_f, "", {}, []))
                r = models.call("planner", m, user, system=P.program_card(CARD_SLOTS), schema=P.SCHEMAS["planner"])
                rec = {"loop": lid, "model": m, "ctx": ctx, "ok": r.ok, "error": r.error, "served": r.served_model,
                       "items": (r.obj or {}).get("items", [])[:a.n] if r.ok else [],
                       "cost_usd_nominal": r.cost_usd_nominal, "usage": r.usage, "latency_s": r.latency_s,
                       "prompt_chars": len(user)}
                with open(out, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print("%s %s/%s ok=%s items=%d cost=%.3f %.0fs" % (lid, m, ctx, r.ok, len(rec["items"]),
                                                                    r.cost_usd_nominal, r.latency_s), flush=True)

    ts = [threading.Thread(target=one_loop, args=(lid,)) for lid in a.loops]
    for t in ts:
        t.start()
    for t in ts:
        t.join()


# ------------------------------------------------------------------ pool

def cmd_pool(a):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(8)
    bge = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
    mini = SentenceTransformer("all-MiniLM-L6-v2", device="cpu", local_files_only=True)   # the engine's own detector
    root = Path(a.root).resolve()
    shards = []
    for d in sorted(p.parent for p in root.glob("*/proposals.jsonl")):
        fz = json.loads((d / "frozen.json").read_text())
        props = []
        for r in map(json.loads, open(d / "proposals.jsonl")):
            for k, it in enumerate(r["items"]):
                props.append({"model": r["model"], "ctx": r["ctx"], "rank": k, "title": it["title"],
                              "spec": it["spec"], "family": it.get("family", "other")})
        if not props:
            continue
        hist = fz["backlog"]
        tried = [i for i in hist if i["times_tried"] > 0]
        enc = lambda mod, xs: mod.encode(xs, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        pt, ht = [text_of(p) for p in props], [text_of(i) for i in hist]
        tried_mask = np.array([i["times_tried"] > 0 for i in hist])
        Pb, Hb, Pm, Hm = enc(bge, pt), enc(bge, ht), enc(mini, pt), enc(mini, ht)
        Sb, Sm = Pb @ Hb.T, Pm @ Hm.T
        for k, p in enumerate(props):
            p["sim_hist_bge"] = float(Sb[k].max())
            p["sim_tried_bge"] = float(Sb[k][tried_mask].max()) if tried_mask.any() else 0.0
            p["sim_hist_mini"] = float(Sm[k].max())
            p["dup_of_tried"] = bool(tried_mask.any() and Sm[k][tried_mask].max() >= scheduler.COSINE_MIN)
        # one idea per group of near duplicates; the order of the pass is random and fixed by the loop's seed
        rng = random.Random(fz["config"]["rng_seed"] + 7)
        order = list(range(len(props)))
        rng.shuffle(order)
        ideas, heads = [], []
        for k in order:
            p = props[k]
            if p["dup_of_tried"]:
                p["idea"] = None
                continue
            j = next((n for n, h in enumerate(heads) if float(Pm[k] @ Pm[h]) >= scheduler.COSINE_MIN), None)
            if j is None:
                heads.append(k)
                ideas.append({"id": "b%03d" % (len(ideas) + 1), "title": p["title"], "spec": p["spec"],
                              "family": p["family"], "sim_hist_bge": p["sim_hist_bge"],
                              "sim_tried_bge": p["sim_tried_bge"], "sim_hist_mini": p["sim_hist_mini"], "sources": []})
                j = len(ideas) - 1
            ideas[j]["sources"].append([p["model"], p["ctx"], p["rank"]])
            p["idea"] = ideas[j]["id"]
        (d / "ideas.json").write_text(json.dumps(ideas, indent=1))
        (d / "props.json").write_text(json.dumps(props, indent=1))
        np.save(d / "props_bge.npy", Pb)
        inc = fz["incumbent"]
        for s in range(a.shards):
            items = [i["id"] for n, i in enumerate(ideas) if n % a.shards == s]
            sd = d / ("s%d" % s)
            sd.mkdir(exist_ok=True)
            cfg = dict(fz["config"], loop_id="B1_%s_s%d" % (d.name, s), run_dir=str(sd), n_slots=3 * len(items) + 3,
                       base_train_py=str(d / "incumbent0.py"), anchor={"mean": inc["mean_adjusted"], "n": inc["n"]},
                       memo_every=10 ** 9, frozen=str(d / "frozen.json"), ideas=str(d / "ideas.json"), items=items,
                       card_slots=CARD_SLOTS, stop_file=str(root / "STOP"),
                       rng_seed=fz["config"]["rng_seed"] + 100 + s)
            if not (sd / "config.json").exists():
                (sd / "config.json").write_text(json.dumps(cfg, indent=1))
            shards.append(str(sd / "config.json"))
        print("%s: %d proposals, %d near duplicates of tried ideas dropped, %d ideas to test" % (
            d.name, len(props), sum(p["dup_of_tried"] for p in props), len(ideas)))
    (root / "shards.json").write_text(json.dumps(shards, indent=1))


# ------------------------------------------------------------------ recode: the same ideas with another coder

def cmd_recode(a):
    """Shards that test the SAME ideas with another coder model (and repairer), everything else unchanged."""
    root = Path(a.root).resolve()
    shards = []
    for d in sorted(p.parent for p in root.glob("*/ideas.json")):
        fz = json.loads((d / "frozen.json").read_text())
        ideas = json.loads((d / "ideas.json").read_text())
        inc = fz["incumbent"]
        routing = dict(fz["config"]["routing"], coder=a.coder, repairer=a.coder)
        for s in range(a.shards):
            items = [i["id"] for n, i in enumerate(ideas) if n % a.shards == s]
            sd = d / ("c%s%d" % (a.coder[0], s))        # cs0, cs1 for a Sonnet coder; s0, s1 are the first stage
            sd.mkdir(exist_ok=True)
            cfg = dict(fz["config"], routing=routing, loop_id="B1_%s_c%s%d" % (d.name, a.coder[0], s), run_dir=str(sd),
                       n_slots=3 * len(items) + 3, base_train_py=str(d / "incumbent0.py"),
                       anchor={"mean": inc["mean_adjusted"], "n": inc["n"]}, memo_every=10 ** 9,
                       frozen=str(d / "frozen.json"), ideas=str(d / "ideas.json"), items=items, card_slots=CARD_SLOTS,
                       stop_file=str(root / "STOP"), rng_seed=fz["config"]["rng_seed"] + 100 + s)   # same replicate seeds
            if not (sd / "config.json").exists():
                (sd / "config.json").write_text(json.dumps(cfg, indent=1))
            shards.append(str(sd / "config.json"))
    (root / ("shards_%s.json" % a.coder)).write_text(json.dumps(shards, indent=1))
    print("wrote %d shards with coder %s" % (len(shards), a.coder))


# ------------------------------------------------------------------ run

class BankEngine(arms.LoopEngine):
    """The engine with a fixed list of ideas, a fixed incumbent and a fixed coder context."""

    def __init__(self, config, services):
        super().__init__(config, services)
        self._fz = json.loads(Path(self.cfg["frozen"]).read_text())
        s = self.state
        if not s.get("bank_init"):
            ideas = {i["id"]: i for i in json.loads(Path(self.cfg["ideas"]).read_text())}
            s["backlog"] = [scheduler.make_item(k, ideas[k]["title"], ideas[k]["spec"], ideas[k]["family"])
                            for k in self.cfg["items"]]
            s["backlog_created"], s["memo"], s["bank_init"] = True, self._fz["memo"], True
            s["family_stats"] = copy.deepcopy(self._fz["family_stats"])
            self._save()
        self.system = arms.prompts.program_card(self.cfg["card_slots"])

    def _table(self):
        return ledger.results_table(self._fz["records"], self.state["incumbent"])

    def _pick_item(self):
        for i in self.state["backlog"]:
            if i["status"] == "open" and i["times_tried"] == 0:
                return i
        raise arms.EngineStop("bank_done")

    def _accept(self, cand):
        pass    # the incumbent stays: every idea is judged against the same program and bar

    def _finish(self, cand, accepted):
        super()._finish(cand, accepted)
        self.state["family_stats"] = copy.deepcopy(self._fz["family_stats"])
        self._save()


def cmd_run(a):
    cfg = json.loads(Path(a.config).read_text())
    jobs = bench.install(cfg)
    models = ModelService(cfg["run_dir"], cfg["loop_id"], max_tokens=cfg.get("max_tokens", 8192))
    eng = BankEngine(cfg, Services(models, jobs, Clock()))
    status = eng.run()
    s = eng.state
    left = sum(1 for i in s["backlog"] if i["status"] == "open" and i["times_tried"] == 0)
    print("%s: %s (%s), %d slots used, %d ideas left" % (cfg["loop_id"], status, eng.stop_reason, s["slot_index"], left))
    if left == 0:
        (Path(cfg["run_dir"]) / "done").write_text("done\n")
    return 0 if left == 0 or eng.stop_reason == "stop_file" else 3


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ask")
    p.add_argument("--root", required=True)
    p.add_argument("--src", required=True)
    p.add_argument("--loops", nargs="+", required=True)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--models", nargs="+", default=list(MODELS))
    p.add_argument("--ctx", nargs="+", default=list(CTX), choices=CTX)
    p.add_argument("--registry", help="model registry file to use instead of loop/models/models.json")
    p = sub.add_parser("pool")
    p.add_argument("--root", required=True)
    p.add_argument("--shards", type=int, default=2)
    p = sub.add_parser("recode")
    p.add_argument("--root", required=True)
    p.add_argument("--coder", required=True)
    p.add_argument("--shards", type=int, default=2)
    p = sub.add_parser("run")
    p.add_argument("--config", required=True)
    a = ap.parse_args()
    sys.exit({"ask": cmd_ask, "pool": cmd_pool, "run": cmd_run, "recode": cmd_recode}[a.cmd](a) or 0)
