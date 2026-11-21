#!/usr/bin/env python3
"""Curation loops whose models are chosen slot by slot by a routing policy (engine.py runs one fixed table).

    python curation/loop/routed.py --run-dir curation/runs/C1/F_01 --policy first --strong fable --cheap qwen35-4b \
        --pool v2 --slots 12 --seed 1 --sigma 0.0073 [--gpu-host willow] [--harm 0.5] [--resume]

Policies (each returns the planner and coder model of the coming slot from what the loop has seen so far):
  fixed:P,C    the same two models in every slot (P and C are "strong" or "cheap", or a model name)
  first        strong for both roles in slot 1, cheap afterwards
  escalate:K   cheap by default; after K slots in a row with no accepted change, one slot with strong for both
               roles, then cheap again
  until:K      strong for both roles until K changes have been accepted, cheap afterwards
  random       a fair coin per role and slot, drawn from the loop seed
  table:FILE   a learned router: a JSON table from state to models, written by train_router.py
--harm R replaces the coder's first file of a slot by a copy of the current best with probability R (the damaged
loop used as a validity check; the call is still made and paid for).
The route and the state of every slot are written to routes.jsonl, which is the training data for a learned router.
"""
import argparse, json, random, subprocess, sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine as E

ROLES = ("planner", "coder")


def state_of(history_decisions, slot, n_slots):
    """What a policy may look at before a slot: all of it is known to the loop at that point."""
    stall = fails = 0
    for d in reversed(history_decisions):
        if d == "accept":
            break
        stall += 1
    for d in reversed(history_decisions):
        if d not in ("fail", "no_idea"):
            break
        fails += 1
    return {"slot": slot, "left": n_slots - slot + 1, "stall": stall, "fails": fails,
            "n_accept": history_decisions.count("accept")}


def route(policy, state, cfg, since_escalation=None):
    """Models for the coming slot: {"planner": name, "coder": name}."""
    strong, cheap = cfg["strong"], cfg["cheap"]
    name = lambda x: {"strong": strong, "cheap": cheap}.get(x, x)
    kind, _, arg = policy.partition(":")
    if kind == "fixed":
        p, c = arg.split(",")
        return {"planner": name(p), "coder": name(c)}
    if kind == "first":
        m = strong if state["slot"] == 1 else cheap
        return {"planner": m, "coder": m}
    if kind == "escalate":
        m = strong if min(state["stall"], since_escalation) >= int(arg) else cheap
        return {"planner": m, "coder": m}
    if kind == "until":
        m = strong if state["n_accept"] < int(arg) else cheap
        return {"planner": m, "coder": m}
    if kind == "random":
        rng = random.Random(cfg["seed"] * 7919 + state["slot"])
        return {r: (strong if rng.random() < 0.5 else cheap) for r in ROLES}
    if kind == "table":
        import train_router
        return {r: name(m) for r, m in train_router.decide(json.loads(Path(arg).read_text()), state).items()}
    raise ValueError("unknown policy %r" % policy)


def purge_slot(run_dir, slot):
    """Remove the trainer results of a slot that was cut off. After a resume the models write a new file, and the
    engine would otherwise score it with the stored result of the old one."""
    for p in (Path(run_dir) / "runs").glob("s%02d_*" % slot):
        p.unlink()


def fetch_result(host, out):
    """A trainer result as the GPU host sees it, or None. The home directory is shared over NFS, and a file the host has
    just written can be invisible here for a while; the host that wrote it always sees it."""
    p = subprocess.run(["ssh", "-o", "BatchMode=yes", host, "cat %s" % E.shlex.quote(str(out))], capture_output=True,
                       text=True, timeout=120)
    try:
        return json.loads(p.stdout) if p.returncode == 0 else None
    except ValueError:
        return None


class RemoteResult:
    """A run on another host that reports no result is checked on that host before it counts as failed."""

    def train(self, path, seed, tag):
        res, err = super().train(path, seed, tag)
        host = self.cfg.get("gpu_host")
        if res is None and host:
            res = fetch_result(host, self.dir / "runs" / ("%s_s%d.json" % (tag, seed)))
        return (res, None) if res is not None else (None, err)


class RoutedLoop(RemoteResult, E.Loop):
    def __init__(self, cfg, models=None, runner=None):
        super().__init__(cfg, models, runner)
        ratio = self.pool_tokens / E.T
        if abs(ratio - 2) > 0.1:                  # the task statement names the pool size; keep it true on other pools
            E.SYSTEM = E.SYSTEM.replace("about 2T tokens", "about %.1fT tokens" % ratio)

    def run_slot(self):
        s, cfg = self.s, self.cfg
        purge_slot(self.dir, s["slot"] + 1)
        decisions = [h["decision"] for h in s.setdefault("route_log", [])]
        state = state_of(decisions, s["slot"] + 1, cfg["slots"])
        inc = self.inc_score(E.SEED_A)             # dev gain so far over the starting file, in noise sds
        state["gain_sd"] = (s.setdefault("start_dev", inc) - inc) / cfg["sigma"]
        last_esc = max([h["slot"] for h in s["route_log"] if h.get("escalated")], default=0)
        r = route(cfg["policy"], state, cfg, since_escalation=s["slot"] - last_esc)
        cfg["routing"] = r
        usd0 = s["usd"]
        self._harmed = None
        super().run_slot()
        rec = json.loads((self.dir / "ledger.jsonl").read_text().splitlines()[-1])
        row = dict(state, planner=r["planner"], coder=r["coder"], decision=rec["decision"], usd=s["usd"] - usd0,
                   dev=rec.get("dev"), inc=rec.get("inc"), harmed=self._harmed,
                   escalated=cfg["policy"].startswith("escalate") and r["planner"] == cfg["strong"])
        s["route_log"].append(row)
        self.save()
        with open(self.dir / "routes.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")

    def ask(self, role, user, schema=None):
        r = super().ask(role, user, schema)
        if role == "coder" and self.cfg.get("harm") and self._harmed is None:      # first coder call of the slot
            rng = random.Random(self.cfg["seed"] * 104729 + self.s["slot"])
            self._harmed = rng.random() < self.cfg["harm"]
            if self._harmed and r.ok:
                r = SimpleNamespace(ok=True, obj=None, text="```python\n%s```" % self.inc_src(), error_class=None,
                                    cost_usd_nominal=r.cost_usd_nominal, served_model=r.served_model)
        return r


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--strong", default="fable")
    ap.add_argument("--cheap", default="qwen35-4b")
    ap.add_argument("--harm", type=float, default=0.0)
    ap.add_argument("--pool", default="v2")
    ap.add_argument("--slots", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sigma", type=float, required=True)
    ap.add_argument("--gpu-host", default=None)
    ap.add_argument("--cell", default=None)
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
               "seed": a.seed, "sigma": a.sigma, "gpu_host": a.gpu_host, "policy": a.policy, "strong": a.strong,
               "cheap": a.cheap, "harm": a.harm, "routing": None}
        cf.write_text(json.dumps(cfg, indent=1))
    lp = RoutedLoop(cfg)
    if a.resume and lp.s["status"] != "new":
        lp.s["resumes"] = lp.s.get("resumes", 0) + 1
    lp.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
