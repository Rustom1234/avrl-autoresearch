#!/usr/bin/env python3
"""Main curation study C1: write the frozen list of loops, then run it a fixed number at a time per GPU host.

    python curation/loop/launch_c1.py plan --root curation/runs/C1 --sigma 0.007266
    python curation/loop/launch_c1.py run  --root curation/runs/C1 --hosts local willow cherry --per-host 2 [--blocks 1 2]

`plan` writes <root>/plan.json once and refuses to overwrite it. `run` reads only that file. Loops that already
finished are skipped and unfinished ones resume, so the same `run` command can be given again after a stop.
Model calls are made from this machine; only the trainer runs on the other hosts.

Cells (strong and cheap are the two anchors' models):
  SS  strong plans and codes          CC  cheap plans and codes
  SC  strong plans, cheap codes       CS  cheap plans, strong codes
  F1  strong in slot 1 only           E3  cheap, one strong slot after 3 slots without an accepted change
  UA  strong until the first accept   RR  a coin per role and slot (two loops per block; router training data)
  HH  the middle model in both roles  N   no model: random moves on a menu of hand-written curators
  DH  SS with half of the coder's files replaced by a copy of the current best (the validity check)
"""
import argparse, json, queue, random, subprocess, sys, threading
from pathlib import Path

HERE = Path(__file__).resolve().parent

CELLS = [("SS", "fixed:strong,strong", 0.0), ("CC", "fixed:cheap,cheap", 0.0), ("SC", "fixed:strong,cheap", 0.0),
         ("CS", "fixed:cheap,strong", 0.0), ("F1", "first", 0.0), ("E3", "escalate:3", 0.0), ("UA", "until:1", 0.0),
         ("RR", "random", 0.0), ("RR", "random", 0.0), ("HH", "fixed:middle,middle", 0.0), ("N", "menu", 0.0),
         ("DH", "fixed:strong,strong", 0.5)]


def make_plan(a):
    loops = []
    for b in range(1, a.blocks + 1):
        block, n_rr = [], 0
        for cell, policy, harm in CELLS:
            n, seed = b, b
            if cell == "RR":                      # the second random loop of a block gets its own seed and number
                n_rr += 1
                if n_rr == 2:
                    n, seed = a.blocks + b, 50 + b
            block.append({"id": "%s_%02d" % (cell, n), "cell": cell, "block": b, "seed": seed,
                          "kind": "menu" if policy == "menu" else "routed",
                          "policy": policy.replace("middle", a.middle), "harm": harm})
        random.Random(a.plan_seed * 1000 + b).shuffle(block)
        loops += block
    return {"study": "C1", "pool": a.pool, "slots": a.slots, "sigma": a.sigma, "strong": a.strong, "cheap": a.cheap,
            "middle": a.middle, "blocks": a.blocks, "plan_seed": a.plan_seed, "loops": loops}


def command(plan, lp, root, host):
    run_dir = root / lp["id"]
    cmd = [sys.executable, str(HERE / ("menu.py" if lp["kind"] == "menu" else "routed.py")), "--run-dir", str(run_dir)]
    if (run_dir / "config.json").exists():
        return cmd + ["--resume"] + ([] if lp["kind"] == "menu" else ["--policy", lp["policy"]]) + [
            "--sigma", str(plan["sigma"])]            # the stored config (and its GPU host) is used
    cmd += ["--pool", plan["pool"], "--slots", str(plan["slots"]), "--seed", str(lp["seed"]),
            "--sigma", str(plan["sigma"]), "--cell", lp["cell"]]
    if lp["kind"] == "routed":
        cmd += ["--policy", lp["policy"], "--strong", plan["strong"], "--cheap", plan["cheap"], "--harm", str(lp["harm"])]
    if host != "local":
        cmd += ["--gpu-host", host]
    return cmd


def run(a):
    root = Path(a.root).resolve()
    plan = json.loads((root / "plan.json").read_text())
    (root / "logs").mkdir(exist_ok=True)
    jobs = queue.Queue()
    for lp in plan["loops"]:
        if a.blocks is None or lp["block"] in a.blocks:
            jobs.put(lp)

    def worker(host):
        while True:
            try:
                lp = jobs.get_nowait()
            except queue.Empty:
                return
            if (root / lp["id"] / "final.json").exists():
                continue
            with open(root / "logs" / (lp["id"] + ".log"), "a") as log:
                rc = subprocess.run(command(plan, lp, root, host), stdout=log, stderr=subprocess.STDOUT).returncode
            print("%s on %s: exit %d" % (lp["id"], host, rc), flush=True)

    threads = [threading.Thread(target=worker, args=(h,)) for h in a.hosts for _ in range(a.per_host)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("all loops finished", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--root", required=True)
    p.add_argument("--pool", default="v2")
    p.add_argument("--sigma", type=float, required=True)
    p.add_argument("--slots", type=int, default=12)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--strong", default="fable")
    p.add_argument("--cheap", default="gemma4-12b")
    p.add_argument("--middle", default="haiku")
    p.add_argument("--plan-seed", type=int, default=20261110)
    r = sub.add_parser("run")
    r.add_argument("--root", required=True)
    r.add_argument("--hosts", nargs="+", default=["local"], help='"local" is this machine')
    r.add_argument("--per-host", type=int, default=2)
    r.add_argument("--blocks", type=int, nargs="+", default=None, help="only these blocks (default all)")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        return run(a)
    root = Path(a.root).resolve()
    pf = root / "plan.json"
    if pf.exists():
        raise SystemExit("%s exists; the plan is frozen" % pf)
    root.mkdir(parents=True, exist_ok=True)
    plan = make_plan(a)
    pf.write_text(json.dumps(plan, indent=1))
    for b in range(1, a.blocks + 1):
        print("block %d: %s" % (b, " ".join(l["id"] for l in plan["loops"] if l["block"] == b)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
