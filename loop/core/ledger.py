"""Append-only ledger.jsonl records and the code-generated results table for prompts (DESIGN.md section 10).

Where it sits. Each loop run keeps one ledger.jsonl in its run directory. arms.py appends one record per
event of a slot (slot = one GPU training run of a candidate train.py), and the same records are rendered
as a markdown table that the planner, coder and analyst models see as "Results so far". The analysis
scripts read the ledger afterwards, so the field names in FIELDS are a shared schema (ENGINE.md lists them).

Every number here comes from worker result dicts via the engine; nothing is parsed from model text.
Model-written text (descriptions) is flattened to one line and truncated before it reaches a prompt.

Terms: val_bpb = validation bits per byte, lower is better. official = scored on the public validation
shard, sealed = scored on a held-out shard the loop never sees. adjusted = official minus the per-GPU
offset. incumbent = the best train.py accepted so far. z1, z2 = z scores of the candidate against the
incumbent after one run and after its replicate or audit run (acceptance.py).

Entry points: new_record (build and validate a record), Ledger (append and read), results_table (prompt
view), claimed_best (the loop's own final claim, before confirmation).
"""
import json
import os
from statistics import fmean

# Record schema. Every record carries every key (None when not applicable) so readers never need .get().
FIELDS = ("slot", "attempt", "kind", "idea_id", "description", "edit_class", "model_calls", "job_id", "gpu",
          "status", "official_val_bpb", "adjusted_val_bpb", "sealed_val_bpb", "flags", "z1", "z2", "decision",
          "incumbent_before", "incumbent_after", "lint", "critic", "charged", "ts")
# kind: why the run was made. candidate = a new edit, replicate = second run of a borderline candidate,
# audit = extra run forced by a tripwire, crash_fix = a repaired version of a candidate that crashed.
KINDS = ("candidate", "replicate", "audit", "crash_fix")
# decision: outcome of the record. "replicate" and "audit" mean "not decided yet, run again"; "infra" marks
# an infrastructure failure (model call or GPU job) that is not the candidate's fault: it is not charged as a
# slot and is left out of the results table.
DECISIONS = ("accept", "reject", "replicate", "audit", "invalid", "crash", "lint_fail", "infra", "abort")
DESC_CHARS = 90        # description width in the prompt table


def new_record(**kw):
    """A full record with every schema key present. Unknown keys are an error (field names are shared)."""
    bad = set(kw) - set(FIELDS)
    if bad:
        raise KeyError("unknown ledger fields: %s" % sorted(bad))
    rec = {k: None for k in FIELDS}
    rec.update(model_calls=[], flags=[], charged=True)
    rec.update(kw)
    if rec["kind"] not in KINDS or rec["decision"] not in DECISIONS:
        raise ValueError("bad kind or decision: %r %r" % (rec["kind"], rec["decision"]))
    return rec


class Ledger:
    """JSONL file, one record per line, opened for append on every write so a crash never truncates earlier lines."""

    def __init__(self, path):
        self.path = str(path)

    def append(self, rec):
        """Write one record as one line and force it to disk before returning (fsync), so a record the
        loop has acted on survives a node failure and a resumed run sees the same history."""
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def read(self):
        """All records in file order ([] if the file does not exist yet)."""
        if not os.path.exists(self.path):
            return []
        with open(self.path) as f:
            return [json.loads(line) for line in f if line.strip()]


def one_line(text, n=DESC_CHARS):
    """Model-written text made safe for a table cell: whitespace collapsed to single spaces, "|" replaced
    (it would break the markdown row), cut to n characters. Keeps a description from injecting extra rows
    or long instructions into later prompts."""
    s = " ".join(str(text or "").replace("|", "/").split())
    return s if len(s) <= n else s[:n - 3] + "..."


def groups(records):
    """Group records into experiments: a new group starts at kind candidate or crash_fix, replicate and audit
    records join the group before them. Records for infra events are left out."""
    out = []
    for r in records:
        if r["decision"] == "infra":
            continue
        if r["kind"] in ("candidate", "crash_fix") or not out:
            out.append([r])
        else:
            out[-1].append(r)
    return out


def _fmt(x):
    """A val_bpb for the table: four decimals, "-" when missing."""
    return "-" if x is None else "%.4f" % x


def results_table(records, incumbent):
    """Markdown table, one row per experiment. delta = mean adjusted val_bpb of its runs minus the incumbent mean
    before the experiment (negative is better). `incumbent` is the current state incumbent (header line only)."""
    lines = ["incumbent now: mean_adjusted %.4f over %d run(s)" % (incumbent["mean_adjusted"], incumbent["n"]), "",
             "| slot | description | official | delta vs incumbent | decision | runs | flags |",
             "|---|---|---|---|---|---|---|"]
    for g in groups(records):
        # A record without job_id never reached a GPU (lint failure, abort, unusable reply): it has no
        # numbers, and its status is shown in the flags column instead.
        runs = [r for r in g if r["job_id"] is not None and r["adjusted_val_bpb"] is not None]
        before = g[0]["incumbent_before"]
        delta = (fmean(r["adjusted_val_bpb"] for r in runs) - before["mean"]) if runs and before else None
        official = "/".join(_fmt(r["official_val_bpb"]) for r in g if r["job_id"] is not None) or "-"
        flags = sorted({f for r in g for f in r["flags"]} | {r["status"] for r in g if r["job_id"] is None and r["status"]})
        lines.append("| %d | %s | %s | %s | %s | %d | %s |" % (
            g[0]["slot"], one_line(g[0]["description"]), official,
            "-" if delta is None else "%+.4f" % delta, g[-1]["decision"],
            sum(1 for r in g if r["job_id"] is not None), ", ".join(flags) or "none"))
    return "\n".join(lines)


def claimed_best(records):
    """The loop's own claim: mean raw official val_bpb of the runs behind the final incumbent (the last experiment
    that ended in accept), or None if nothing was accepted. This is the "claimed official" of the selection gap."""
    last = None
    for g in groups(records):
        if g[-1]["decision"] == "accept":
            last = g
    if last is None:
        return None
    vals = [r["official_val_bpb"] for r in last if r["official_val_bpb"] is not None]
    return fmean(vals) if vals else None
