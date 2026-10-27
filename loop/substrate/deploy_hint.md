# Deploy (run these yourself; nothing here was executed)

All commands run from spruce, in `/homes/rdubash/active-audit/loop/substrate`.

1. Create the directory tree on hannah (NFS home, so n7 sees it). The `logs` dir must exist
   before `sbatch`, Slurm will not create the `--output` directory.

```
ssh -o BatchMode=yes rustom@hannah 'bash -s' <<'EOF2'
mkdir -p /home/rustom/arloop/{code,logs,inbox,queue/pending,queue/running,queue/done,results,workers}
EOF2
```

2. Copy the worker code (worker.py imports harness_tail.py, ship both) and the sbatch script.

```
scp -o BatchMode=yes worker.py harness_tail.py rustom@hannah:/home/rustom/arloop/code/
scp -o BatchMode=yes start_workers.sbatch rustom@hannah:/home/rustom/arloop/
```

3. Check the base install matches the pin (must print 4f2ba9cbb8ba8c4a3d35be405a913e2f3be3af9aea103ed52ef7b2a662058150).
   Workers refuse to start otherwise.

```
ssh -o BatchMode=yes rustom@hannah 'sha256sum /home/rustom/autoresearch_base/prepare.py; /home/rustom/autoresearch_base/.venv/bin/python -V; ls /home/rustom/.cache/autoresearch/data /home/rustom/.cache/autoresearch/tokenizer /home/rustom/.cache/autoresearch/sealed; which taskset nvidia-smi'
```

   If the Python version is not one of 3.10, 3.12, 3.13, 3.14, `harness.prepare_code_hash_ok` and
   `harness.pipeline_code_hash_ok` will be `null` (unpinned). To pin it, copy harness_tail.py to hannah and run

```
scp -o BatchMode=yes harness_tail.py rustom@hannah:/tmp/harness_tail.py
ssh -o BatchMode=yes rustom@hannah '/home/rustom/autoresearch_base/.venv/bin/python /tmp/harness_tail.py /home/rustom/autoresearch_base/prepare.py'
```

   It prints a JSON fragment for that interpreter; save it as `/home/rustom/autoresearch_base/pinned_hashes.json`
   (the worker reads BASE first and each key replaces the built-in table, so the fragment alone is enough for that Python).

4. Start the four workers (add `-w n7` after `sbatch` if the gpu partition has other nodes). The sbatch script
   requeues leftover `queue/running` jobs, pins worker g to cores 8g..8g+7 (`taskset`), copies data, tokenizer and the
   sealed shard once to `/tmp/arcache_<uid>` on the node (the trainer runs with `HOME` there; Triton and torch caches
   follow it, off NFS), and then supervises every 60 s (stale workers are killed and restarted, stuck jobs requeued,
   see `logs/supervisor.log`). The first start copies the whole data dir, so expect a delay before the first heartbeat.

```
ssh -o BatchMode=yes rustom@hannah 'cd /home/rustom/arloop && sbatch start_workers.sbatch'
ssh -o BatchMode=yes rustom@hannah 'squeue -u rustom; tail -n 5 /home/rustom/arloop/logs/worker_gpu*.log'
```

5. Smoke test from spruce with the unmodified baseline train.py. Expect `status == "ok"`,
   `harness.prepare_code_hash_ok`, `harness.pipeline_code_hash_ok` and `harness.constants_ok` all `True`,
   `mismatch_flag is False`, `target_leak_flag is False` (expected: `target_leak_loss_a` and `target_leak_loss_b` nearly
   equal; `null` plus a `target_leak_error` means the check could not run), `eval_batch_size_used == 64`, and
   `harness.val_shard_sha256` set (keep it as the reference value for the val shard). Note the wall time: the
   harness adds eval and possibly `torch.compile` recompiles on top of `train.py`'s own run.

```
cd /homes/rdubash/active-audit/loop/substrate
python - <<'EOF2'
from queue_client import QueueClient
c = QueueClient()
jid = c.submit("/path/to/baseline/train.py", tag="baseline", timeout_s=900, sealed=True)
print(c.wait([jid], 1500)[jid])
print(c.status())
EOF2
```

6. Pin the data files. The smoke job's `harness.data_files_sha256` has the sha256 of `token_bytes.pt`, `tokenizer.pkl`
   and the val shard. Once you trust them, save `{"data_files": {...that dict...}}` as
   `/home/rustom/autoresearch_base/pinned_hashes.json` (merge with the repin fragment above if you made one; each
   top-level key replaces the built-in value) and restart the workers. Until then `data_files_ok` is `null`.

7. If workers or the node died, from spruce: `python -c "from queue_client import QueueClient; print(QueueClient().requeue_stale())"`
   moves orphaned running jobs back to pending (retries + 1, `infra_fail` after 2). The sbatch supervisor does the same
   every minute while it is alive.

8. Stop gracefully (workers finish their current job, then exit; `start_workers.sbatch` removes STOP on the next launch):

```
python -c "from queue_client import QueueClient; QueueClient().stop_workers()"
```
