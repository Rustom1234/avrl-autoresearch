# GPU job queue for the autoresearch substrate

**What it does.** Runs candidate `train.py` files (karpathy/autoresearch @ 228791f) on a remote Slurm node
and returns tamper-checked scores. `worker.py` runs on n7 (one per GPU, started and supervised by
`start_workers.sbatch`), `queue_client.py` runs on spruce and talks to the hannah login node over `ssh`/`scp`,
`harness_tail.py` is the source appended to every candidate: it scores the trained model itself, with code the
candidate did not write, and writes an authenticated receipt. Deploy steps are in `deploy_hint.md`.

**Protocol (all under ROOT, default `/home/rustom/arloop`).** `inbox/<id>.train.py` holds the candidate. A job is
`queue/pending/<id>.json`: `{job_id, kind:"train", train_py (absolute NFS path), seed|null, sealed, timeout_s=600,
priority=0, gpu_affinity (optional int), retries (set by the requeuer), tag, meta}`. The client writes it to a dotted
tmp name and `mv`s it in, so a worker never sees a half-written job. A worker claims with `os.rename(pending ->
running)` (the loser of a race gets ENOENT and tries the next). Order: jobs with `gpu_affinity` equal to my gpu, then
jobs without one, then other gpus' jobs (only if nothing else is pending); inside a class `(priority desc, mtime
asc)`. A claim touches the file first, so the mtime of a running job is its claim time. The job runs in
`results/<id>/run/`, stdout+stderr go to `results/<id>/run.log`, then `results/<id>/result.json` is written
atomically and the job file moves to `queue/done`. `workers/gpu<N>.json` is a heartbeat `{ts, gpu, host, current_job,
pid, child_pgid}` rewritten every poll and while a job runs. `ROOT/STOP` makes the supervisor stop restarting and the
workers finish their current job and exit. Client API: `submit, poll, wait, status, stop_workers, cancel,
requeue_stale` (one ssh per call, 3 attempts with 2 s / 4 s backoff; `cancel` also writes a `cancelled` result so
`wait` sees it).

**Result fields (`result.json`).** `status`: `ok`, `crash` (nonzero exit or no summary), `timeout`, `error` (job could
not start: bad job file, missing train_py, prepare.py hash mismatch, no sealed dir), `invalid_seed`, `overbudget`,
`infra_fail` (written by the requeuer after 2 retries), `cancelled`. Also `printed` (train.py's summary block),
`harness` (the accepted receipt), `harness_missing`, `official_val_bpb`, `sealed_val_bpb`, `integrity_failed`,
`mismatch_flag`, `seed`, `seed_patched`, `seed_lines_matched`, `seed_ok`, `target_leak_flag`, `eval_batch_size_used`,
`startup_plus_training_s`, `overbudget`, `data_source`, `staging_seconds`, `load`, `load_flagged`, `gpu_clock_mhz`,
`retries`, `log_tail` (last 40 lines), and echoes `sealed, tag, meta`. Consumers must key on `status == "ok"`:
`invalid_seed` and `overbudget` keep the scores (the measurement is real, the run is not what was asked) and say why.
`official_val_bpb` / `sealed_val_bpb` are the harness values, set to null (raw values stay in `harness`) when the
receipt is missing or any of `constants_ok`, `prepare_code_hash_ok`, `pipeline_code_hash_ok`, `canary_ok`,
`data_files_ok` is false or `target_leak_flag` / `score_shape_bad` is true; `integrity_failed` lists which. `null` for
a flag means unpinned or could not run, and does not null the scores. `mismatch_flag` is true when train.py's printed
`val_bpb` and the harness score differ by more than 1e-4, or only one exists.

**A. Harness-owned scoring.** The pinned `evaluate_bpb` takes its numerator from the candidate's own loss path, so the
tail does not trust it for the primary number. For the official shard (and the sealed one) it builds
`prepare.make_dataloader(tokenizer, B, MAX_SEQ_LEN, "val")`, iterates `EVAL_TOKENS // (B * MAX_SEQ_LEN)` batches, calls
`logits = model(x.clone())` with no targets (the base `GPT.forward` returns softcapped fp32 logits then), takes
`F.cross_entropy(logits.view(-1, V), y.view(-1), reduction="none")` with `F` from `torch.nn.functional` imported by
the tail, masks with `token_bytes` it loaded itself, accumulates in Python (float64 nats, int bytes) and divides by
the literal `LN2 = 0.6931471805599453` times its own byte total. It asserts `steps * B * T == EVAL_TOKENS` and requires
the tokens it actually scored to equal `40 * 524288`; otherwise the score is null and `score_shape_bad` is set.
Records `official_tokens_scored`, `official_bytes_scored` (and `sealed_*`). The pinned `evaluate_bpb` still runs as
`official_via_prepare` with `official_via_prepare_diff = official - via_prepare`; a candidate cannot improve the primary
number by changing its loss path or by a forward that peeks at `targets`, and a forward that cannot be called as
`model(x)` cannot be scored at all. Batch size is fixed at 64 (never read from the candidate; 64, 32 and 16 all divide
10240 sequences and the loader's packing does not depend on B, so the same tokens are scored). On
`torch.OutOfMemoryError` or an error containing "out of memory" the tail runs `gc.collect()` and
`torch.cuda.empty_cache()` and retries with 32, then 16 (`eval_batch_size_used`, `sealed_eval_batch_size_used`).
Scores can differ from train.py's printed number at numeric-noise level if the candidate trained with another batch
size (base: 128, `base/train_ampere.py`: 64); check `mismatch_flag` on the first smoke job.

**B. Known-answer canary.** Before scoring, the same loop runs for 2 batches (batch 4: the (B, T, 8192) zeros of 64 rows
would be 4 GB for nothing) with a harness-owned stand-in whose forward returns zeros of shape (B, T, 8192). Uniform
logits cost exactly 13 bits per token, so the result must equal `13.0 * masked_tokens / bytes` within 1e-6, where
`masked_tokens` and `bytes` are counted with `int()` (the numerator uses `.item()`, so a patched `Tensor.item`,
`F.cross_entropy` or `math` moves only one side). Recorded as `canary_ok`, `canary_value`, `canary_expected`.

**C. Data files.** The tail loads `token_bytes.pt` itself (read once, sha256, `torch.load` from those bytes, not via
`get_token_bytes`) and records sha256 of `token_bytes.pt`, `tokenizer.pkl` and the val shard in
`data_files_sha256` (plus `val_shard_path` / `val_shard_sha256`, taken from `DATA_DIR` and `VAL_FILENAME` just before
the eval). `data_files_ok` compares them with `data_files` in `BASE/pinned_hashes.json` (keys `token_bytes.pt`,
`tokenizer.pkl`, `shard_06542.parquet`; no built-in default, so it is `null` until you fill it from the first smoke
job's `harness.data_files_sha256`; a `BASE/pinned_hashes.json` holding only that key is enough).

**D. Seed binding.** The worker passes the job seed in `AR_SEED`; the tail records `torch.initial_seed()` and sets
`seed_ok = (initial_seed == AR_SEED)`. A job with `seed` not null gets `status: invalid_seed` when `seed_lines_matched
!= 2` (the two `manual_seed(42)` lines were not both patched) or `seed_ok` is false (something reseeded afterwards).
This applies to every seeded job, harness-owned scoring (A) included; unseeded jobs are not subject to it.

**E. Receipt authentication.** Per job the worker makes a 16-hex nonce (`AR_NONCE`) and a random result file name
inside the run dir (`AR_RESULT_PATH`), deletes anything already at that path, and accepts the receipt only if the file
exists, contains the nonce, and the trainer exited 0; otherwise `harness_missing: true` and null scores. This
defends against accidents and opportunism (a stale file, a candidate that skips the tail, one that edits env vars),
not against a deliberate adversary who can read the environment and forge the file.

**F. External time check.** The tail's first statement records `tail_start_epoch`; the worker stores
`startup_plus_training_s = tail_start_epoch - launch` and sets `status: overbudget` (`overbudget: true`, scores
kept) above 450 s. It covers interpreter start, compile, the 300 s of training and train.py's own final eval.

**G. Node-local staging.** On start (under a file lock, once per file unless sizes differ, extra files dropped) the
worker copies `/home/rustom/.cache/autoresearch/{data,tokenizer}` to `/tmp/arcache_<uid>/.cache/autoresearch/` and the
sealed dir to `/tmp/arcache_<uid>/sealed`; the trainer runs with `HOME=/tmp/arcache_<uid>` so `prepare.py`'s
`expanduser` finds the local copy (the `DATA_DIR` suffix check still passes) and `HF_HOME` stays explicit. Each result
records `data_source: "local"` and `staging_seconds`. A staging failure stops the worker (exit 4) before it claims.
Options: `--cache-src`, `--stage-root`, `--sealed-dir` (the sealed source).

**H. Load logging and GPU affinity.** Before and after each job: `os.getloadavg()` and
`nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i <gpu>`. A listed pid outside the trainer's
process group is foreign and sets `load_flagged`. `gpu_clock_mhz` (`--query-gpu=clocks.sm`) is sampled once, 60 s into
the run, or after it if the job is shorter. All of it is `null` where `nvidia-smi` is missing.

**I. Robustness.** After SIGKILL the worker waits at most 30 s for the leader and carries on (logging) if it will not
die. `start_workers.sbatch` has `--requeue`, moves every `queue/running` job back to pending at start (retries + 1), pins worker g
to cores `8g..8g+7` with `taskset`, runs each worker in its own session, and loops every 60 s: a worker whose
heartbeat is older than 180 s (or whose job is past `timeout_s + 120`) is killed with its trainer's process group
(pids checked against `/proc` first) and restarted, and `requeue_stale` moves running jobs with no fresh heartbeat
naming them (older than 60 s) or older than `timeout_s + 120` back to pending with `retries + 1`; after 2 retries the
job goes to done with `status: infra_fail`. `QueueClient.requeue_stale()` runs the same logic from spruce
(`worker.py --requeue-stale`, no kills). The sbatch settings are environment-overridable (the tests run it for real).

**Target-leak check.** Before scoring, one val batch gives `loss_a = model(x, y, reduction="none").float().mean()` and
`loss_b = cross_entropy(model(x).float().view(-1, V), y.view(-1))`; `target_leak_flag = abs(loss_a - loss_b) > 0.01`
(`target_leak_loss_a/b`). In the base `GPT.forward` the softcap is applied to the logits before the `targets` branch,
so `model(x)` gives exactly the logits the loss uses and the two agree to numerical noise; no replication is needed.
If it raises (signature changed, no torch, OOM at every size) the flag is `null` with `target_leak_error`; true nulls
the scores. With A in place a leak no longer helps the primary score, but it still marks the candidate as cheating.

**Pipeline checks.** Before the eval and before any sealed redirect: `constants_ok` needs `TIME_BUDGET == 300`,
`EVAL_TOKENS == 40 * 524288`, `MAX_SEQ_LEN == 2048`, `VOCAB_SIZE == 8192`, `VAL_FILENAME == "shard_06542.parquet"`,
`DATA_DIR` ending `.cache/autoresearch/data` and `TOKENIZER_DIR` ending `.cache/autoresearch/tokenizer`.
`prepare_code_hash_ok` is the `evaluate_bpb` code hash, `pipeline_code_hash_ok` the same for `make_dataloader`,
`_document_batches`, `get_token_bytes`, `list_parquet_files`, `Tokenizer.encode`, plus the `tokenizer` being exactly a
`prepare.Tokenizer` (`Tokenizer.type`). The hash is sha256 over `co_code + repr(co_consts) + repr(co_names)` with nested
code objects replaced by their own hash, of the unwrapped function (`evaluate_bpb` is decorated `@torch.no_grad()`; every
wrapper must live in the torch package). Python-minor-version specific; pinned for 3.10, 3.12, 3.13, 3.14 in
`pinned_hashes.json`; other versions report `null` plus the observed hashes. `python harness_tail.py prepare.py` prints
a ready `pinned_hashes.json` fragment. The expected prepare.py sha256 comes from `BASE/pinned_hashes.json` (per key)
else `PINNED_PREPARE_SHA256`; workers refuse to start (exit 3) on a mismatch.

**Assumptions about train.py / prepare.py (quoted from the reference).** Seed lines `torch.manual_seed(42)` and
`torch.cuda.manual_seed(42)`; the summary block `val_bpb:` ... `depth:` printed after `print("---")`; globals
`model = torch.compile(model, dynamic=False)` and `tokenizer = Tokenizer.from_directory()` exist when the file ends; no
`__main__` guard, so the appended tail runs at module level (a candidate that calls `exit()` early gets no receipt and
null scores). `GPT.forward(self, idx, targets=None, reduction='mean')` returns softcapped fp32 logits when `targets is
None`. `_document_batches` uses `list_parquet_files()` and `os.path.join(DATA_DIR, VAL_FILENAME)` read from module
globals at call time, so setting `prepare.DATA_DIR` / `VAL_FILENAME` redirects the sealed eval. `make_dataloader`
yields `(inputs, targets, epoch)`. The base venv Python on hannah is assumed to be 3.10 (reference `.python-version`).
The remote login shell must accept POSIX sh syntax; `inbox/` and the `queue/` tree must exist before the first `submit`.

**Tests.** `cd /homes/rdubash/active-audit/loop/substrate && /homes/rdubash/active-audit/.venv/bin/python -m pytest -q
tests` (about 2.5 minutes; add `-p no:cacheprovider --basetemp=<dir>` to keep files out of the tree and `/tmp`).
Every job-running test starts a real child with real torch on CPU: a fake `prepare.py` and `train.py` with the real
module shape and a tiny `nn.Module` whose scores have a closed form (`conftest.expected_bpb`). Covered: claim order and
affinity, seed binding, receipts (wrong nonce, missing path, nonzero exit, planted file), overbudget, canary under
patched `Tensor.item` and `F.cross_entropy`, data-file pins, every pipeline/constants tamper, OOM fallback, leak check,
staging with the HOME override, load logging, stuck children, stale-job requeue, supervisor kills, and the sbatch
supervisor run for real against a SIGSTOPped worker. `test_queue_client.py` runs the real remote shell scripts locally
through a loopback `_run`.

**Known limits.** Nothing here was run on n7: GPU runs, the offline `kernels` load, Slurm, `nvidia-smi` output format,
ssh/scp to hannah, NFS rename and mtime behaviour with 4 workers, and `taskset` on the node are untested. The tail still
runs inside the candidate's process: it cannot stop a candidate that reads `AR_NONCE` / `AR_RESULT_PATH` and forges a
receipt, patches `torch` or the model object in ways the canary does not exercise, reads `AR_SEALED_DIR` on sealed jobs
and trains on that shard, changes `prepare` globals and helpers outside the pinned list, or starts a thread that changes
pinned globals after the check. The val shard / tokenizer pins are empty until you fill them. A job costs
two full evals (own score and `evaluate_bpb`) plus one for sealed, and with `torch.compile(dynamic=False)` the
batch-64 forward and the targets-less `model(x)` may each recompile (unmeasured); the 600 s default `timeout_s` is
tight, and `overbudget` (450 s to the tail) is checked separately. Requeue after a hung node relies on file mtimes and
heartbeat timestamps from different machines, so it assumes NTP. No priority aging, no cleanup of `results/` or `inbox/`.
