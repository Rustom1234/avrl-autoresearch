"""Run-log view for the model prompt (DESIGN.md section 10).

Rationale. run.log is text produced by code the agent wrote (karpathy/autoresearch issue 64),
and two facts decide what the next prompt may contain:

1. Harness numbers are authoritative. When the worker's result dict is given, the summary block
   is rebuilt from result fields (official_val_bpb, the numeric printed.* fields, status, flags)
   and no val_bpb line is taken from the log, because a printed val_bpb is spoofable and the
   official one is recomputed by the harness tail. Free text from the log cannot ride on a
   rebuilt block: every value is a number or an identifier-shaped flag.
2. A fix attempt needs the traceback. For status crash or timeout the model gets the last
   max_lines raw log lines (control characters stripped, lines cut to 300 chars), frames
   included. The old imperative-phrase filter is gone: it dropped legitimate error text
   ("Please install ...", "ignore" inside a warnings call) and any paraphrase beat it.

Without a result dict (no worker, arm A) the old allowlist is used: the nine summary lines and
the harness line in the form `key: <one number>`, error lines, and the 15 lines before the
first Traceback.

The output is for the agent's eyes only. No decision may use it; decisions come from
harness_result.json.
"""
import re

SUMMARY_KEYS = ("val_bpb", "training_seconds", "total_seconds", "peak_vram_mb", "mfu_percent",
                "total_tokens_M", "num_steps", "num_params_M", "depth")
HARNESS_KEY = "harness_official_val_bpb"
PRINTED_FIELDS = ("training_seconds", "total_seconds", "peak_vram_mb", "num_steps", "num_params_M")
FLAG_KEYS = ("mismatch_flag", "target_leak_flag", "harness_missing", "overbudget", "load_flagged")
RAW_TAIL_STATUSES = ("crash", "timeout")
MAX_LINE_CHARS = 300
CONTEXT_BEFORE_TRACEBACK = 15

_NUM = r"[-+]?(?:\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?|(?i:nan|inf|none))"
_SUMMARY = re.compile(r"^(%s)\s*:\s*%s\s*$" % ("|".join(SUMMARY_KEYS + (HARNESS_KEY,)), _NUM))
_ERROR = re.compile(r"(?i:traceback|error|out of memory)|\bOOM\b")
_CTRL = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|[\x00-\x08\x0b-\x1f\x7f]")
_IDENT = re.compile(r"[\w.-]{1,60}")


def _clean(line):
    line = _CTRL.sub("", line).rstrip()
    return line if len(line) <= MAX_LINE_CHARS else line[:MAX_LINE_CHARS - 3] + "..."


def _num(x):
    return None if isinstance(x, bool) or not isinstance(x, (int, float)) else x


def _flags(result):
    out = [k for k in FLAG_KEYS if result.get(k)]
    if result.get("canary_ok") is False:
        out.append("canary_failed")
    extra = result.get("flags")
    extra = [extra] if isinstance(extra, str) else extra if isinstance(extra, (list, tuple)) else []
    out += [f for f in extra if isinstance(f, str) and _IDENT.fullmatch(f)]
    return list(dict.fromkeys(out))


def _summary_from_result(result):
    printed = result.get("printed") if isinstance(result.get("printed"), dict) else {}
    status = result.get("status")
    official = _num(result.get("official_val_bpb"))
    lines = ["status: %s" % (_clean(status) if isinstance(status, str) and status else "unknown"),
             "official_val_bpb: %s" % ("none" if official is None else official)]
    lines += ["%s: %s" % (k, printed[k]) for k in PRINTED_FIELDS if _num(printed.get(k)) is not None]
    lines.append("flags: %s" % (", ".join(_flags(result)) or "none"))
    return "\n".join(lines)


def _allowlist_tail(log_text, max_lines):
    raw = [_CTRL.sub("", x).rstrip() for x in log_text.splitlines()]
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
    """Text for the model's next prompt. `result` is the worker's result.json dict (or None)."""
    if result is None:
        return _allowlist_tail(log_text, max_lines)
    if result.get("status") in RAW_TAIL_STATUSES:
        lines = log_text.splitlines()
        return "\n".join(_clean(x) for x in lines[max(0, len(lines) - max_lines):])
    return _summary_from_result(result)
