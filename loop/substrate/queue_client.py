"""Client for the autoresearch GPU job queue. Pure stdlib.

Runs on the machine that hosts the research loop (spruce) and talks to the cluster login node
(hannah) over ssh. The queue itself is a directory tree on the cluster's shared filesystem, served by
one worker.py per GPU (see worker.py and README.md for the protocol). The loop uses this class to
hand a candidate train.py to a GPU and get its score back:

    c = QueueClient()
    job_id = c.submit("train.py", seed=1, tag="A_01_slot3")   # copy the file, enqueue the job
    result = c.wait([job_id], timeout_s=900)[job_id]          # result.json as a dict, or None
    result["status"], result["official_val_bpb"]              # val_bpb = validation bits per byte, lower is better

Every remote action is one ssh (or scp) call; all subprocess use goes through QueueClient._run.
That single choke point is what the tests replace, either with a recorder or with a loopback that
runs the same shell scripts locally. There is no persistent connection, so a dropped network costs
at most one retried call.
"""
import json
import os
import re
import shlex
import subprocess
import time
import uuid


class QueueError(RuntimeError):
    """A remote command still failed after all retries (network down, ssh refused, script error)."""
    pass


def _clean(s):
    """s with every run of characters outside [A-Za-z0-9._-] replaced by "_", so that a job id is safe
    as a file name and inside a shell command."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))


class QueueClient:
    """Handle on one remote queue. host is the ssh target, root the queue directory on the cluster.
    Each call is tried up to `attempts` times with waits of backoff_s, 2 * backoff_s, ... in between;
    each ssh or scp is cut off after cmd_timeout_s. BatchMode=yes makes ssh fail instead of prompting for a password."""
    def __init__(self, host="rustom@hannah", root="/home/rustom/arloop",
                 ssh_opts=("-o", "BatchMode=yes", "-o", "ConnectTimeout=15"),
                 attempts=3, backoff_s=2.0, cmd_timeout_s=120,
                 python="/home/rustom/autoresearch_base/.venv/bin/python"):
        self.python = python  # interpreter used on hannah for worker.py maintenance modes
        self.host, self.root, self.ssh_opts = host, root.rstrip("/"), tuple(ssh_opts)
        self.attempts, self.backoff_s, self.cmd_timeout_s = attempts, backoff_s, cmd_timeout_s

    # ---- transport ----
    def _run(self, cmd):
        """The only place a subprocess is started (tests monkeypatch this method)."""
        return subprocess.run(cmd, capture_output=True, text=True, timeout=self.cmd_timeout_s)

    def _call(self, cmd):
        """Run cmd with exponential backoff; return stdout or raise QueueError."""
        last = ""
        for i in range(self.attempts):
            try:
                r = self._run(cmd)
                if r.returncode == 0:
                    return r.stdout
                last = "rc=%s %s" % (r.returncode, (r.stderr or "").strip()[-300:])
            except (subprocess.TimeoutExpired, OSError) as e:
                last = repr(e)
            if i < self.attempts - 1:
                time.sleep(self.backoff_s * 2 ** i)
        raise QueueError("%s failed after %d attempts: %s" % (cmd[0], self.attempts, last))

    def _ssh(self, script):
        """Run a shell script on the remote host; returns its stdout."""
        return self._call(["ssh", *self.ssh_opts, self.host, script])

    def _scp(self, local, remote):
        """Copy a local file to a path on the remote host."""
        return self._call(["scp", *self.ssh_opts, local, "%s:%s" % (self.host, remote)])

    def _cd(self):
        """Shell snippet that enters the queue root (all remote paths below are relative to it)."""
        return "cd %s" % shlex.quote(self.root)

    def _heredoc(self, dest, text):
        """Shell snippet writing text to dest atomically (tmp file in the same dir, then mv)."""
        eof = "AR_EOF_" + uuid.uuid4().hex
        tmp = os.path.join(os.path.dirname(dest), "." + os.path.basename(dest) + ".tmp")
        return "cat > %s <<'%s'\n%s\n%s\nmv %s %s\n" % (
            shlex.quote(tmp), eof, text, eof, shlex.quote(tmp), shlex.quote(dest))

    @staticmethod
    def _check_ids(job_ids):
        """Reject ids that are not plain file-name characters before they reach a remote shell."""
        for j in job_ids:
            if not re.fullmatch(r"[A-Za-z0-9._-]+", j):
                raise ValueError("bad job_id %r" % (j,))

    # ---- API ----
    def submit(self, train_py_local_path, *, seed=None, sealed=False, timeout_s=600, priority=0,
               tag="", meta=None):
        """Copy a candidate train.py to the remote inbox and enqueue one training job for it. Returns
        the job id (unix time, cleaned tag, random suffix). seed: if set, the worker patches the script's
        seed lines to it. sealed: also score on the held-out shard. timeout_s: wall-clock limit of the
        run. priority: higher is claimed first. tag and meta are echoed back in the result.
        The script is copied first and the job file appears last (atomic mv), so a worker can never
        claim a job whose train.py is not there yet."""
        if not os.path.isfile(train_py_local_path):
            raise FileNotFoundError(train_py_local_path)
        job_id = _clean("%d_%s_%s" % (int(time.time()), _clean(tag)[:40], uuid.uuid4().hex[:8]))
        remote_train = "%s/inbox/%s.train.py" % (self.root, job_id)
        job = {"job_id": job_id, "kind": "train", "train_py": remote_train, "seed": seed,
               "sealed": bool(sealed), "timeout_s": int(timeout_s), "priority": int(priority),
               "tag": tag, "meta": meta or {}}
        body = json.dumps(job)  # before any transfer, so unserializable meta fails early
        self._scp(str(train_py_local_path), remote_train)
        self._ssh("set -e\n%s\nmkdir -p queue/pending\n%s" % (
            self._cd(), self._heredoc("queue/pending/%s.json" % job_id, body)))
        return job_id

    def poll(self, job_ids):
        """job_id -> result dict, or None if its result.json does not exist yet. One ssh call."""
        ids = list(job_ids)
        self._check_ids(ids)
        if not ids:
            return {}
        # Each existing result is printed between @@AR_BEGIN <id> and @@AR_END marker lines, so many
        # results come back in one round trip and are split apart again below.
        script = (
            "cd %s/results 2>/dev/null || exit 0\n"
            "for id in %s; do\n"
            "  if [ -f \"$id/result.json\" ]; then\n"
            "    printf '@@AR_BEGIN %%s\\n' \"$id\"; cat \"$id/result.json\"; printf '\\n@@AR_END\\n'\n"
            "  fi\n"
            "done\n" % (shlex.quote(self.root), " ".join(shlex.quote(i) for i in ids)))
        out = self._ssh(script)
        results = {i: None for i in ids}
        for jid, body in re.findall(r"@@AR_BEGIN (\S+)\n(.*?)\n@@AR_END", out, re.S):
            try:
                if jid in results:
                    results[jid] = json.loads(body)
            except ValueError:
                pass
        return results

    def wait(self, job_ids, timeout_s, poll_interval=20, max_poll_failures=3):
        """Poll until every job has a result or timeout_s elapses. Returns job_id -> result or None.
        A failed poll is tolerated (the network may blip during a long wait); only max_poll_failures
        failures in a row raise QueueError."""
        ids = list(job_ids)
        results = {i: None for i in ids}
        deadline = time.time() + timeout_s
        failures = 0
        while True:
            todo = [i for i in ids if results[i] is None]
            try:
                results.update({k: v for k, v in self.poll(todo).items() if v is not None})
                failures = 0
            except QueueError:
                failures += 1
                if failures >= max_poll_failures:
                    raise
            if all(results[i] is not None for i in ids) or time.time() >= deadline:
                return results
            time.sleep(max(0, min(poll_interval, deadline - time.time())))

    def status(self):
        """Snapshot of the queue in one ssh call: counts of pending / running / done jobs, the running
        job ids, whether STOP is set, and each worker's heartbeat with age_s added. Ages use the remote
        clock (@@NOW), so clock skew between the two machines does not make a worker look stale."""
        script = (
            "%s || exit 1\n"
            "echo \"@@NOW $(date +%%s)\"\n"
            "for d in pending running done; do\n"
            "  echo \"@@COUNT $d $(ls queue/$d 2>/dev/null | grep -c '[.]json$')\"\n"
            "done\n"
            "echo \"@@RUNNING $(ls queue/running 2>/dev/null | grep '[.]json$' | sed 's/[.]json$//' | tr '\\n' ' ')\"\n"
            "if [ -e STOP ]; then echo '@@STOP 1'; else echo '@@STOP 0'; fi\n"
            "for f in workers/gpu*.json; do\n"
            "  if [ -f \"$f\" ]; then echo '@@HB'; cat \"$f\"; echo; fi\n"
            "done\n" % self._cd())
        out = self._ssh(script)
        st = {"pending": 0, "running": 0, "done": 0, "running_jobs": [], "stop": False, "workers": {}}
        now = time.time()
        lines = out.splitlines()
        for n, line in enumerate(lines):
            if line.startswith("@@NOW "):
                now = float(line.split()[1])
            elif line.startswith("@@COUNT "):
                _, name, cnt = line.split()
                st[name] = int(cnt)
            elif line.startswith("@@RUNNING"):
                st["running_jobs"] = line.split()[1:]
            elif line.startswith("@@STOP "):
                st["stop"] = line.split()[1] == "1"
            elif line == "@@HB" and n + 1 < len(lines):
                try:
                    hb = json.loads(lines[n + 1])
                    st["workers"][hb["gpu"]] = hb
                except (ValueError, KeyError):
                    pass
        for hb in st["workers"].values():
            hb["age_s"] = max(0.0, now - hb.get("ts", now))
        return st

    def stop_workers(self):
        """Touch ROOT/STOP: workers finish their current job and exit."""
        self._ssh("%s && touch STOP" % self._cd())

    def cancel(self, job_id):
        """Move a still-pending job to done with a cancelled result. False if a worker already claimed it."""
        self._check_ids([job_id])
        res = json.dumps({"job_id": job_id, "status": "cancelled", "official_val_bpb": None,
                          "sealed_val_bpb": None, "finished": time.time()})
        q = shlex.quote(job_id)
        script = ("%s\nif mv queue/pending/%s.json queue/done/%s.json 2>/dev/null; then\n"
                  "mkdir -p results/%s\n%secho CANCELLED\nelse echo NOT_PENDING; fi\n" % (
                      self._cd(), q, q, q, self._heredoc("results/%s/result.json" % job_id, res)))
        return "CANCELLED" in self._ssh(script)

    def requeue_stale(self):
        """Run worker.py --requeue-stale on hannah: running jobs with no fresh worker heartbeat, or older
        than timeout_s + 120 s, go back to pending with retries+1 (after 2 retries: done, infra_fail).
        Returns {"requeued": [job_id...], "infra_fail": [job_id...]}."""
        out = self._ssh("%s && %s code/worker.py --requeue-stale --root %s" % (
            self._cd(), shlex.quote(self.python), shlex.quote(self.root)))
        return json.loads(out.strip().splitlines()[-1])
