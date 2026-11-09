import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import harness_tail  # noqa: E402
import worker  # noqa: E402

PYKEY = "%d.%d" % sys.version_info[:2]
LN2 = 0.6931471805599453
EVAL_TOKENS = 40 * 524288


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def is_dead(pid):
    try:
        stat = Path("/proc/%d/stat" % pid).read_text()
    except OSError:
        return True
    return stat.rsplit(")", 1)[1].split()[0] == "Z"


def expected_bpb(v):
    """bpb of the fake model on a shard holding the number v, by pure-Python math: logits_k = softcap(v*k/8),
    y cycles over 0..7, token k has k bytes (token 0 is special)."""
    ls = [15 * math.tanh(v * k / 8 / 15) for k in range(8)]
    lse = math.log(sum(math.exp(x) for x in ls))
    return sum(lse - ls[y] for y in range(1, 8)) / (LN2 * 28)


def approx(v):
    return pytest.approx(v, rel=1e-4)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A fake BASE (fake prepare.py + its own pinned hashes), a fake cache (data, tokenizer, sealed) to be
    staged to a node-local stage root, and a queue ROOT. Needs torch: the fake models are nn.Modules."""
    torch = pytest.importorskip("torch")
    base, root, alt = (tmp_path / n for n in ("base", "root", "alt"))
    src = tmp_path / "src_home" / ".cache" / "autoresearch"  # stands in for /home/rustom/.cache/autoresearch
    stage = tmp_path / "stage"  # stands in for /tmp/arcache_<uid>
    for d in (base, alt, src / "data", src / "tokenizer", src / "sealed"):
        d.mkdir(parents=True)
    shutil.copy(HERE / "fake_prepare.py", base / "prepare.py")
    (src / "data" / "shard_00000.parquet").write_text("1.5")
    (src / "data" / "shard_06542.parquet").write_text("1.25")  # official val shard: v = 1.25
    (src / "sealed" / "shard_06541.parquet").write_text("2.5")  # sealed shard: v = 2.5
    (alt / "shard_06542.parquet").write_text("0.1")  # where a tampered DATA_DIR points
    torch.save(torch.arange(8, dtype=torch.int32), src / "tokenizer" / "token_bytes.pt")
    (src / "tokenizer" / "tokenizer.pkl").write_bytes(b"fake tokenizer")
    shutil.copy(src / "tokenizer" / "token_bytes.pt", alt / "token_bytes.pt")  # a tampered TOKENIZER_DIR still works
    shutil.copy(src / "tokenizer" / "tokenizer.pkl", alt / "tokenizer.pkl")
    text = (base / "prepare.py").read_text()
    hashes = harness_tail.prepare_code_hashes(text)
    pins = {"prepare_py_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "evaluate_bpb_code_sha256": {PYKEY: hashes.pop("evaluate_bpb")},
            "pipeline_code_sha256": {PYKEY: hashes}}
    (base / "pinned_hashes.json").write_text(json.dumps(pins))
    monkeypatch.setenv("FAKE_ALT_DIR", str(alt))
    monkeypatch.setenv("FAKE_PIDFILE", str(tmp_path / "child.pid"))
    for k in ("FAKE_MODE", "FAKE_NOTORCH", "FAKE_IGNORE_TERM", "FAKE_OOM_ABOVE", "FAKE_OOM_KIND", "AR_SEALED_DIR"):
        monkeypatch.delenv(k, raising=False)
    # no real GPU looked at by default: tests that care replace nvidia_smi themselves
    monkeypatch.setattr(worker.Worker, "nvidia_smi", lambda self, *a: None)
    ns = SimpleNamespace(tmp=tmp_path, base=base, alt=alt, src=src, stage=stage, root=root, pins=pins,
                         stage_data=stage / ".cache" / "autoresearch" / "data",
                         stage_tok=stage / ".cache" / "autoresearch" / "tokenizer",
                         stage_sealed=stage / "sealed")

    def make_worker(gpu=0, **kw):
        return worker.Worker(gpu, root, base, sys.executable, poll=kw.pop("poll", 0.1),
                             sealed_dir=str(src / "sealed"), cache_src=str(src), stage_root=str(stage), **kw)

    def put_job(job_id, mtime=None, train_py=HERE / "fake_train.py", **fields):
        job = {"job_id": job_id, "kind": "train", "train_py": str(train_py), "seed": None,
               "sealed": False, "timeout_s": 120, "priority": 0, "tag": "t", "meta": {}}
        job.update(fields)
        pend = root / "queue" / "pending"
        pend.mkdir(parents=True, exist_ok=True)
        p = pend / (job_id + ".json")
        p.write_text(json.dumps(job))
        if mtime is not None:
            os.utime(p, (mtime, mtime))
        return p

    def run_job(job_id="j1", worker_obj=None, **fields):
        """Queue one job, run a worker with --once semantics, return its result.json."""
        put_job(job_id, **fields)
        (worker_obj or make_worker()).run(once=True)
        return json.loads((root / "results" / job_id / "result.json").read_text())

    def write_heartbeat(gpu, job=None, age=0.0, pid=None, child_pgid=None):
        d = root / "workers"
        d.mkdir(parents=True, exist_ok=True)
        (d / ("gpu%d.json" % gpu)).write_text(json.dumps({
            "ts": time.time() - age, "gpu": gpu, "host": "h", "current_job": job,
            "pid": pid, "child_pgid": child_pgid}))

    def put_running(job_id, age=0.0, **fields):
        """A job file in queue/running whose claim time (mtime) is `age` seconds ago."""
        job = {"job_id": job_id, "kind": "train", "train_py": "/x", "timeout_s": 100, "priority": 0}
        job.update(fields)
        run = root / "queue" / "running"
        run.mkdir(parents=True, exist_ok=True)
        p = run / (job_id + ".json")
        p.write_text(json.dumps(job))
        os.utime(p, (time.time() - age, time.time() - age))
        for d in ("pending", "done"):
            (root / "queue" / d).mkdir(parents=True, exist_ok=True)
        return p

    ns.make_worker, ns.put_job, ns.run_job = make_worker, put_job, run_job
    ns.write_heartbeat, ns.put_running = write_heartbeat, put_running
    return ns
