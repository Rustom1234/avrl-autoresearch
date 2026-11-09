"""Run-log view for the model prompt (DESIGN.md section 10).

Where it sits. After each training run arms.py calls sanitize_tail(log_text, result) and stores the
returned text as the "Last run" section of the next prompt (and as the crash log shown to the repairer).
Stdlib only, no I/O. val_bpb = validation bits per byte, lower is better.

Rationale. run.log is text produced by code a model wrote (karpathy/autoresearch issue 64), so it
can carry anything, including instructions aimed at the next model call (prompt injection) or a
made-up score. Two facts decide what the next prompt may contain:

1. Harness numbers are authoritative. When the worker's result dict is given, the summary block
   is rebuilt from result fields (official_val_bpb, the numeric printed.* fields, status, flags)
   and no val_bpb line is taken from the log, because a printed val_bpb is spoofable and the
   official one is recomputed by the harness tail. Free text from the log cannot ride on a
   rebuilt block: every value is a number or an identifier-shaped flag.
2. A fix attempt needs the traceback. For status crash or timeout the model gets the last
   max_lines raw log lines (control characters stripped, lines cut to 300 chars), frames
   included. No filter on imperative phrases is applied: such a filter drops legitimate error
   text ("Please install ...", "ignore" inside a warnings call) and any paraphrase beats it.

Without a result dict (no worker, arm A) an allowlist is used: the nine summary lines and
the harness line in the form `key: <one number>`, error lines, and the 15 lines before the
first Traceback.

The output is shown to the model only. No decision may use it; decisions come from
harness_result.json.
"""
import re

# The nine `key: value` lines train.py prints after its "---" line at the end of a run.
SUMMARY_KEYS = ("val_bpb", "training_seconds", "total_seconds", "peak_vram_mb", "mfu_percent",
                "total_tokens_M", "num_steps", "num_params_M", "depth")
HARNESS_KEY = "harness_official_val_bpb"   # the line the harness tail prints with its recomputed value
# Printed values that are safe to pass on from the result dict. val_bpb is deliberately absent: the only
# score the model is shown is the official one.
PRINTED_FIELDS = ("training_seconds", "total_seconds", "peak_vram_mb", "num_steps", "num_params_M")
FLAG_KEYS = ("mismatch_flag", "target_leak_flag", "harness_missing", "overbudget", "load_flagged")
RAW_TAIL_STATUSES = ("crash", "timeout")   # statuses where the raw tail is needed to write a fix
MAX_LINE_CHARS = 300
CONTEXT_BEFORE_TRACEBACK = 15

# A single number (also nan, inf, none), optionally signed, with thousands commas or an exponent. Anchored
# in _SUMMARY so that nothing may follow the number on an allowlisted line.
_NUM = r"[-+]?(?:\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?|(?i:nan|inf|none))"
_SUMMARY = re.compile(r"^(%s)\s*:\s*%s\s*$" % ("|".join(SUMMARY_KEYS + (HARNESS_KEY,)), _NUM))
_ERROR = re.compile(r"(?i:traceback|error|out of memory)|\bOOM\b")
_CTRL = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|[\x00-\x08\x0b-\x1f\x7f]")   # ANSI escapes and control chars (tab kept)
_IDENT = re.compile(r"[\w.-]{1,60}")   # shape of a flag name: one short token, no spaces, so no prose


def _clean(line):
    """line without control characters or trailing whitespace, cut to MAX_LINE_CHARS."""
    line = _CTRL.sub("", line).rstrip()
    return line if len(line) <= MAX_LINE_CHARS else line[:MAX_LINE_CHARS - 3] + "..."


def _num(x):
    """x if it is an int or float, else None. bool is excluded (it is an int subclass in Python), and a
    string is never passed through, which is what keeps free text out of the rebuilt summary."""
    return None if isinstance(x, bool) or not isinstance(x, (int, float)) else x


def _flags(result):
    """Flag names set on the result, in a fixed order and without duplicates. Extra entries under
    result["flags"] are kept only if they look like identifiers."""
    out = [k for k in FLAG_KEYS if result.get(k)]
    if result.get("canary_ok") is False:
        out.append("canary_failed")
    extra = result.get("flags")
    extra = [extra] if isinstance(extra, str) else extra if isinstance(extra, (list, tuple)) else []
    out += [f for f in extra if isinstance(f, str) and _IDENT.fullmatch(f)]
    return list(dict.fromkeys(out))


def _summary_from_result(result):
    """Summary block built only from result fields: status, official val_bpb, the numeric printed
    fields, flags. No line of the log is copied."""
    printed = result.get("printed") if isinstance(result.get("printed"), dict) else {}
    status = result.get("status")
    official = _num(result.get("official_val_bpb"))
    lines = ["status: %s" % (_clean(status) if isinstance(status, str) and status else "unknown"),
             "official_val_bpb: %s" % ("none" if official is None else official)]
    lines += ["%s: %s" % (k, printed[k]) for k in PRINTED_FIELDS if _num(printed.get(k)) is not None]
    lines.append("flags: %s" % (", ".join(_flags(result)) or "none"))
    return "\n".join(lines)


def _allowlist_tail(log_text, max_lines):
    """Fallback when no result dict exists: keep at most max_lines log lines chosen by priority, then
    print them in their original order."""
    raw = [_CTRL.sub("", x).rstrip() for x in log_text.splitlines()]
    # prio maps line index to priority: 0 = the last occurrence of each summary line, 1 = error lines,
    # 2 = context just before the first traceback. Lines not in prio are dropped.
    prio, last_summary, first_tb = {}, {}, None
    for i, line in enumerate(raw):
        m = _SUMMARY.match(line)
        if m:
            last_summary[m.group(1)] = i
        if first_tb is None and "Traceback" in line:
            first_tb = i
        if _ERROR.search(line):
            prio[i] = 1
    for i in last_summary.values():
        prio[i] = 0
    if first_tb is not None:
        for i in range(max(0, first_tb - CONTEXT_BEFORE_TRACEBACK), first_tb):
            prio.setdefault(i, 2)
    keep = [i for i in prio if raw[i]]
    keep.sort(key=lambda i: (prio[i], -i))  # summary first, then errors, then context; newest first
    keep = sorted(keep[:max(0, max_lines)])
    return "\n".join(_clean(raw[i]) for i in keep)


def sanitize_tail(log_text, result=None, max_lines=25):
    """Text for the model's next prompt. `result` is the worker's result.json dict (or None).

    Three cases: no result dict -> allowlisted log lines; status crash or timeout -> the last max_lines
    raw lines (cleaned), because a fix needs the traceback; any other status -> a summary rebuilt from
    result fields, with nothing copied from the log."""
    if result is None:
        return _allowlist_tail(log_text, max_lines)
    if result.get("status") in RAW_TAIL_STATUSES:
        lines = log_text.splitlines()
        return "\n".join(_clean(x) for x in lines[max(0, len(lines) - max_lines):])
    return _summary_from_result(result)
