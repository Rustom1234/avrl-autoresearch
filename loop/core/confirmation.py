"""Confirmation block (DESIGN.md sections 2 and 12) plus the small file and queue helpers shared with controller.py.

Every final incumbent is run 4 times (seeds 1001 to 1004, identical for all arms) in waves of 4 jobs: 3 confirmation
runs and 1 anchor run (the unmodified Ampere baseline), arms mixed by a seeded shuffle, every job sealed. A wave is
submitted whole and awaited before the next one, so at most 4 jobs are pending or running at once and the anchor of a
wave sees the same load as its confirmation runs.

Files under <root>/confirmation/: waves.json (frozen wave plan), results.jsonl (one row per finished job),
summary.json (per loop and anchor arithmetic). `run_confirmation` can be called again after any interruption: jobs
already in results.jsonl are skipped (jobs submitted but not yet recorded are submitted again).
"""
import json
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]  # directory that holds the `loop` package
BASE_TRAIN = REPO / "loop" / "substrate" / "base" / "train_ampere.py"

SEEDS = (1001, 1002, 1003, 1004)
WAVE_CONFIRMS = 3
JOB_TIMEOUT_S = 900     # per-job timeout handed to the worker (training budget is 300 s, a run is about 450 s)
WAVE_TIMEOUT_S = 2400   # how long we wait for the 4 jobs of a wave before counting the missing ones as failed
POLL_S = 20


# ---- small file helpers (also used by controller.py) ----
def atomic_write(path, text):
    path = Path(path)
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def read_jsonl(path):
    rows = []
    try:
        with open(path) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
    except OSError:
        pass
    return rows


def append_jsonl(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def make_client():
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from loop.substrate.queue_client import QueueClient
    return QueueClient()


# ---- jobs ----
def job_ok(res, sealed):
    return bool(res) and res.get("status") == "ok" and res.get("official_val_bpb") is not None and (
        not sealed or res.get("sealed_val_bpb") is not None)


def run_jobs(client, jobs, poll_s=POLL_S, timeout_s=WAVE_TIMEOUT_S):
    """jobs: dicts with train_py, seed, sealed, tag, meta (at most 4). Submit all, wait for all, then resubmit each
    failed job once (crash, timeout, infra failure, missing result). Returns [(job_id, result or None, attempt)]
    in job order; the last attempt's outcome is the one returned."""
    out = [None] * len(jobs)
    todo = list(range(len(jobs)))
    for attempt in (1, 2):
        ids = {i: client.submit(jobs[i]["train_py"], seed=jobs[i]["seed"], sealed=jobs[i]["sealed"],
                                timeout_s=JOB_TIMEOUT_S, tag=jobs[i]["tag"], meta=jobs[i]["meta"]) for i in todo}
        res = client.wait(list(ids.values()), timeout_s, poll_s)
        todo = []
        for i, jid in ids.items():
            out[i] = (jid, res.get(jid), attempt)
            if not job_ok(res.get(jid), jobs[i]["sealed"]):
                todo.append(i)
        if not todo:
            break
    return out


def collect_incumbents(plan):
    """Final incumbents of every loop whose state.json says done or stopped. Returns (incumbents, not_confirmed ids)."""
    found, skipped = [], []
    for loop in plan["loops"]:
        run_dir = Path(loop["run_dir"])
        ss = read_json(run_dir / "state.json") or {}
        path = (ss.get("incumbent") or {}).get("path")
        if ss.get("status") in ("done", "stopped") and path:
            p = Path(path) if os.path.isabs(path) else run_dir / path
            if not p.is_file():
                raise FileNotFoundError("incumbent of %s not found: %s" % (loop["loop_id"], p))
            found.append({"loop_id": loop["loop_id"], "arm": loop["arm"], "train_py": str(p)})
        else:
            skipped.append(loop["loop_id"])
    return found, skipped


def build_waves(incumbents, seed):
    """4 runs per incumbent shuffled across arms, cut into groups of 3, one anchor job added to each group.
    The anchor of wave w uses seed SEEDS[(w - 1) % 4], so anchor seeds cycle through the confirmation seeds."""
    rng = random.Random(seed)
    jobs = [{"kind": "confirm", "loop_id": i["loop_id"], "arm": i["arm"], "seed": s, "train_py": i["train_py"]}
            for i in incumbents for s in SEEDS]
    rng.shuffle(jobs)
    waves = []
    for w, k in enumerate(range(0, len(jobs), WAVE_CONFIRMS), start=1):
        group = jobs[k:k + WAVE_CONFIRMS] + [{"kind": "anchor", "loop_id": None, "arm": None,
                                              "seed": SEEDS[(w - 1) % len(SEEDS)], "train_py": str(BASE_TRAIN)}]
        rng.shuffle(group)
        waves.append({"wave": w, "jobs": group})
    return waves


def _key(wave, job):
    return (wave, job["kind"], job["loop_id"], job["seed"])


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def _good(r):
    return r["status"] == "ok" and r["official_val_bpb"] is not None and r["sealed_val_bpb"] is not None


def summarize(rows):
    """Per loop: mean official and sealed over its successful runs, improvement = mean over its runs of
    (same-wave anchor sealed minus incumbent sealed) (and the same on the official shard). Overall: the anchor's
    official minus sealed offset (the shard gap baseline)."""
    anchors = {r["wave"]: r for r in rows if r["kind"] == "anchor" and _good(r)}
    runs = {}
    for r in rows:
        if r["kind"] == "confirm":
            runs.setdefault(r["loop_id"], {"arm": r["arm"], "rows": []})["rows"].append(r)
    loops = {}
    for lid, d in runs.items():
        ok = [r for r in d["rows"] if _good(r)]
        paired = [r for r in ok if r["wave"] in anchors]
        loops[lid] = {
            "arm": d["arm"], "n_runs": len(d["rows"]), "n_ok": len(ok), "n_paired": len(paired),
            "mean_official": _mean(r["official_val_bpb"] for r in ok),
            "mean_sealed": _mean(r["sealed_val_bpb"] for r in ok),
            "improvement": _mean(anchors[r["wave"]]["sealed_val_bpb"] - r["sealed_val_bpb"] for r in paired),
            "improvement_official": _mean(anchors[r["wave"]]["official_val_bpb"] - r["official_val_bpb"]
                                          for r in paired)}
    a = list(anchors.values())
    return {"loops": loops,
            "anchor": {"n_ok": len(a), "n_failed": sum(1 for r in rows if r["kind"] == "anchor") - len(a),
                       "mean_official": _mean(r["official_val_bpb"] for r in a),
                       "mean_sealed": _mean(r["sealed_val_bpb"] for r in a),
                       "offset_official_minus_sealed": _mean(r["official_val_bpb"] - r["sealed_val_bpb"] for r in a)},
            "n_rows": len(rows)}


def run_confirmation(root, plan, client=None, poll_s=POLL_S, wave_timeout_s=WAVE_TIMEOUT_S, log=print):
    cdir = Path(root) / "confirmation"
    cdir.mkdir(exist_ok=True)
    wpath, rpath = cdir / "waves.json", cdir / "results.jsonl"
    waves = read_json(wpath)
    if waves is None:
        incumbents, skipped = collect_incumbents(plan)
        if not incumbents:
            raise SystemExit("no loop is done or stopped with an incumbent; nothing to confirm")
        if skipped:
            log("not confirmed (not done or stopped, or no incumbent): " + ", ".join(skipped))
        waves = build_waves(incumbents, plan["seed"])
        atomic_write(wpath, json.dumps(waves, indent=1))
    client = client or make_client()
    done = {_key(r["wave"], r) for r in read_jsonl(rpath)}
    for w in waves:
        todo = [j for j in w["jobs"] if _key(w["wave"], j) not in done]
        if not todo:
            continue
        log("wave %d: submitting %d jobs" % (w["wave"], len(todo)))
        jobs = [dict(j, sealed=True, tag="confirm-anchor" if j["kind"] == "anchor" else "confirm-%s" % j["loop_id"],
                     meta={"wave": w["wave"], "kind": j["kind"], "loop_id": j["loop_id"], "arm": j["arm"]})
                for j in todo]
        for j, (jid, res, attempt) in zip(todo, run_jobs(client, jobs, poll_s, wave_timeout_s)):
            res = res or {}
            append_jsonl(rpath, {"wave": w["wave"], "loop_id": j["loop_id"], "arm": j["arm"], "seed": j["seed"],
                                 "kind": j["kind"], "official_val_bpb": res.get("official_val_bpb"),
                                 "sealed_val_bpb": res.get("sealed_val_bpb"), "gpu": res.get("gpu"),
                                 "status": res.get("status") or "no_result", "job_id": jid, "attempt": attempt})
    summary = summarize(read_jsonl(rpath))
    atomic_write(cdir / "summary.json", json.dumps(summary, indent=1, sort_keys=True))
    return summary
