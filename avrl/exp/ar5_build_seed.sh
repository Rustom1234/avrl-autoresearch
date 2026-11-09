#!/bin/bash
# AR5 step 3: build the allow-list seed /homes/rdubash/avrl_live/_seed_day1/ (day-one view of the project).
# Reads only /homes/rdubash/active-audit. Writes only /homes/rdubash/avrl_live/_seed_day1/ and its sibling
# /homes/rdubash/avrl_live/_seed_day1_meta/ (SEED_MANIFEST.sha256, SEED_EDITS.log, SEED_ALLOWLIST.txt).
# The meta files live outside the seed tree on purpose: SEED_EDITS.log quotes the very strings the seed
# must not contain, and cp -a of the seed into each run would otherwise leak them to the agents.
# Layout mirrors the original repo so ROOT = dirname(dirname(__file__)) in the p0 scripts works.
set -euo pipefail

SRC=/homes/rdubash/active-audit
LIVE=/homes/rdubash/avrl_live
OUT=$LIVE/_seed_day1
META=$LIVE/_seed_day1_meta
PY=/homes/rdubash/yes/bin/python
GATE='AR1|AR2|AR3|AR5|S1 |Where the claims stand|M0[1-9]|L9[a-d]|bootstrap\.json|KILL_REVERSED|goldberg|hannah|5090|Priority note|settled'
# Known benign leak-gate hits (extended regexes matched against grep -n output lines). Expected: none.
ALLOW_RE=''

[ -d "$SRC/avrl" ] || { echo "missing $SRC/avrl" >&2; exit 1; }
mkdir -p "$LIVE"
rm -rf "$OUT" "$META"
mkdir -p "$OUT/avrl/data" "$OUT/avrl/runs/p0" "$OUT/lit/avrl" "$OUT/runs" "$OUT/models" "$META"

# ---- plain copies ----
for f in p0_rollouts.py p0_analyze.py allocators.py; do cp -p "$SRC/avrl/$f" "$OUT/avrl/$f"; done
for f in gsm8k_train.parquet gsm8k_test.parquet; do cp -p "$SRC/avrl/data/$f" "$OUT/avrl/data/$f"; done
for f in rollouts.parquet groups.parquet feats.parquet summary.json result.json \
         sim_cell30_cheap.json sim_cell30_cheap+emb.json sim_cell100_cheap.json sim_cell100_cheap+emb.json \
         sim_alloc.py analyze.log run.log done.flag; do
  if [ -f "$SRC/avrl/runs/p0/$f" ]; then cp -p "$SRC/avrl/runs/p0/$f" "$OUT/avrl/runs/p0/$f"
  else echo "note: avrl/runs/p0/$f does not exist, skipped"; fi
done
cp -p "$SRC/lit/avrl/SYNTHESIS.md" "$OUT/lit/avrl/SYNTHESIS.md"
cp -p "$SRC/runs/sim_generic.py" "$OUT/runs/sim_generic.py"
for m in Qwen2.5-0.5B-Instruct Qwen2.5-1.5B-Instruct Qwen3-0.6B-Base Qwen3-4B; do
  if [ -e "$SRC/models/$m" ]; then ln -s "$SRC/models/$m" "$OUT/models/$m"
  else echo "note: models/$m does not exist, skipped"; fi
done

# ---- edited files (ledger cut, PROJECT.md, backlog) ----
SRC="$SRC" OUT="$OUT" META="$META" "$PY" - <<'PYEOF'
import os, re
SRC, OUT, META = os.environ["SRC"], os.environ["OUT"], os.environ["META"]
log = open(os.path.join(META, "SEED_EDITS.log"), "w")
def note(s): log.write(s + "\n")

def rep(text, old, new, fname, need=1):
    n = text.count(old)
    if n < need:
        raise SystemExit(f"edit not applied ({n} < {need}): {fname}: {old!r}")
    note(f"{fname} | x{n} | {old!r} -> {new!r}")
    return text.replace(old, new)

def read(p): return open(p, encoding="utf-8").read().split("\n")

# 1. RESEARCH_LOG.md: header plus E01 only. Cut located by content, line 25 must be the E03 header.
L = read(os.path.join(SRC, "avrl/RESEARCH_LOG.md"))
hdr = [i for i, l in enumerate(L) if l.startswith("## E03 2026-09-30 kill")]
assert hdr and hdr[0] == 24, f"ledger line 25 is not the E03 header: {hdr[:3]}"
assert L[24].startswith("## E03 2026-09-30 kill")
open(os.path.join(OUT, "avrl/RESEARCH_LOG.md"), "w", encoding="utf-8").write("\n".join(L[:24]).rstrip("\n") + "\n")
note(f"avrl/RESEARCH_LOG.md | kept lines 1-{hdr[0]} (line {hdr[0]+1} is the E03 header, dropped with everything after)")

# 2. PROJECT.md: lines 1 to 26 (up to the line before "## Findings to date"), then substitutions.
L = read(os.path.join(SRC, "avrl/PROJECT.md"))
f2 = [i for i, l in enumerate(L) if l.startswith("## Findings to date")]
assert len(f2) == 1 and f2[0] == 26, f"PROJECT.md cut moved: {f2}"
t = "\n".join(L[:f2[0]]).rstrip("\n") + "\n"
note(f"avrl/PROJECT.md | kept lines 1-{f2[0]} (line {f2[0]+1} is '## Findings to date', dropped with everything after)")
t = rep(t, "Compute: goldberg (RTX 5090) for phase 0, hannah (4x A100 40GB) for phase 1.", "Compute: one RTX A4000 16 GB workstation.", "PROJECT.md")
t = rep(t, "(goldberg, today)", "(this machine)", "PROJECT.md")
t = rep(t, "(hannah, launched by workflow)", "(deferred: needs a larger GPU, not available in this run)", "PROJECT.md")
bad = re.findall(r"goldberg|hannah|5090", t)
assert not bad, f"PROJECT.md still mentions {bad}"
open(os.path.join(OUT, "avrl/PROJECT.md"), "w", encoding="utf-8").write(t)

# 3. EXPERIMENT_BACKLOG.md
L = read(os.path.join(SRC, "lit/avrl/EXPERIMENT_BACKLOG.md"))
def find(prefix):
    h = [i for i, l in enumerate(L) if l.startswith(prefix)]
    assert len(h) == 1, f"{prefix!r} found {len(h)} times"
    return h[0]
def section(prefix):  # header through the line before the next '## ' header
    s = find(prefix)
    e = next((i for i in range(s + 1, len(L)) if L[i].startswith("## ")), len(L))
    return L[s:e]
prop = find("## Proposed by the loop")
src_i = find("## Sources")
assert src_i < prop
main = L[:prop]
while main and main[-1].strip() == "":
    main.pop()
note(f"EXPERIMENT_BACKLOG.md | kept lines 1-{len(main)} (through '## Sources', line {src_i+1}); '## Proposed by the loop' at line {prop+1} and after dropped")
harness = section("## Harness status note")
compute = section("## Compute correction")
note(f"EXPERIMENT_BACKLOG.md | appended '## Harness status note' (source lines {find('## Harness status note')+1}-{find('## Harness status note')+len(harness)}) and '## Compute correction' (source lines {find('## Compute correction')+1}-{find('## Compute correction')+len(compute)})")
feas = ["## Feasibility note (this run)",
        "Items marked A100, needing a larger cluster, a second machine, or the phase-1 GRPO harness (E15 to E20, G04, anything needing p1_grpo.py; about 89 s per step on this card, so 100 steps take 2.5 hours) are not runnable under the 45-minute cap and must not be planned.",
        ""]
t = "\n".join(main + [""] + harness + compute + feas)
fn = "EXPERIMENT_BACKLOG.md"
t = rep(t, "/homes/rdubash/active-audit/avrl", "{BASE}/avrl", fn)
t = rep(t, "/homes/rdubash/active-audit/lit", "{BASE}/lit", fn)
# machine mentions: single-machine rewording (feasibility note is appended after, so it is exempt from these rules)
body, tail = t.rsplit("## Feasibility note (this run)", 1)
rules = [
    ("the hannah cluster (4x A100 40GB) for phase 1, not a single A100", "a multi-GPU cluster for phase 1, not a single GPU", 1),
    ("on the hannah cluster per PROJECT.md", "on a multi-GPU cluster per PROJECT.md", 1),
    ("on the hannah cluster, estimate only", "on a multi-GPU cluster, estimate only", 1),
    ("a vLLM-generation upgrade is being tested and the hannah job (four arms, 150 steps, job queued behind another user) is the parallel path.", "a vLLM-generation upgrade is being tested.", 1),
    ("passed a smoke test on goldberg:", "passed a smoke test on this machine:", 1),
    ("runnable on goldberg at reduced steps", "runnable on this machine at reduced steps", 1),
    ("goldberg's GPU is an NVIDIA", "This machine's GPU is an NVIDIA", 1),
    (", or run them on hannah's A100 40GB nodes.", ".", 1),
    ("on the RTX 5090", "on this machine", 1),
    ("one RTX 5090", "this machine", 1),
    ("the RTX 5090", "this machine", 0),
    ("RTX 5090", "this machine", 1),
    ("the 5090 budget", "this machine's budget", 1),
    ("on the 5090", "on this machine", 1),
]
for old, new, need in rules:
    body = rep(body, old, new, fn, need)
t = body + "## Feasibility note (this run)" + tail
bad = re.findall(r"goldberg|hannah|5090", body)
assert not bad, f"backlog still mentions {set(bad)}"
for m in re.findall(r"/homes/rdubash/active-audit\S*", t):
    note(f"EXPERIMENT_BACKLOG.md | remaining absolute path kept (not avrl/ or lit/): {m}")
open(os.path.join(OUT, "lit/avrl/EXPERIMENT_BACKLOG.md"), "w", encoding="utf-8").write(t)
log.close()
PYEOF

# ---- manifest (regular files only; symlinks are listed separately), edits log is already in META ----
( cd "$OUT" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum ) > "$META/SEED_MANIFEST.sha256"
( cd "$OUT" && find . -type l -printf '%p -> %l\n' | LC_ALL=C sort ) > "$META/SEED_LINKS.txt"

# ---- leak gate ----
hits=$(grep -rInE "$GATE" "$OUT" || true)
if [ -n "$ALLOW_RE" ]; then
  printf '%s\n' "$hits" | grep -E "$ALLOW_RE" > "$META/SEED_ALLOWLIST.txt" || true
  hits=$(printf '%s\n' "$hits" | grep -vE "$ALLOW_RE" || true)
else
  : > "$META/SEED_ALLOWLIST.txt"
fi
echo "== seed manifest: $(wc -l < "$META/SEED_MANIFEST.sha256") files, $(du -sh "$OUT" | cut -f1) on disk, sha256 of manifest $(sha256sum < "$META/SEED_MANIFEST.sha256" | cut -c1-16)"
echo "== SEED_EDITS.log ($META/SEED_EDITS.log)"
cat "$META/SEED_EDITS.log"
echo "== leak gate: allow-listed hits $(wc -l < "$META/SEED_ALLOWLIST.txt"), unexpected hits $(printf '%s' "$hits" | grep -c . || true)"
if [ -n "$hits" ]; then echo "$hits"; echo "LEAK GATE FAILED" >&2; exit 1; fi
echo "LEAK GATE CLEAN"
