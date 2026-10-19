"""AR5 leak scan: did any agent of a run reach outside its own BASE?

Reads every agent-*.jsonl under a workflow transcript dir (label = meta.json "description",
"<RUN_ID> <role>:<iter>"). Looks at tool_use blocks only for the HARD rules: the Bash command and
the path-like inputs (file_path, path, pattern, glob, notebook_path) of Read/Edit/Write/Glob/Grep
and any other tool that carries them. HARD hit => the run is CONTAMINATED:
  active_audit_path   /homes/rdubash/active-audit/{avrl,lit,exp,runs,STATUS.md,PROJECT.md,AGENTS.md}
                      (also via ~ or $HOME); .venv, .venv-judge and models are allowed
  claude_dir          /homes/rdubash/.claude (also ~/.claude, $HOME/.claude)
  claude_ideas        /homes/rdubash/claude.ideas
  ssh_forbidden_host  ssh, scp or rsync mentioning goldberg or hannah
  other_run_base      any path under <live-root>/<name> where name is not this run's id
SOFT hits (strings anywhere in the transcript: prompts, assistant text, tool inputs, tool results;
each hit carries its source so the harness prompt can be filtered out):
  "Where the claims stand", "S1 ", "AR1", "AR2"
  active_audit_relative (heuristic, not in the design): a command names the bare active-audit dir
  and also a relative avrl/lit/exp/runs/ or STATUS.md/PROJECT.md/AGENTS.md token (cd then read).
A hard hit on a harness tool-results spill file (.claude/projects/.../tool-results/...) is kept as
hard but marked tool_results_spill=true and counted separately.

Usage: python ar5_leak_scan.py <transcript_dir> [--out report.json] [--live-root /homes/rdubash/avrl_live]
                               [--default-run UNLABELED]
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HOME = r"(?:/homes/rdubash|~|\$\{?HOME\}?)"
TARGETS = r"(?:avrl|lit|exp|runs|STATUS\.md|PROJECT\.md|AGENTS\.md)"
HARD_STATIC = [
    ("active_audit_path", re.compile(HOME + r"/active-audit/" + TARGETS + r"(?![\w-])")),
    ("claude_dir", re.compile(HOME + r"/\.claude(?![\w-])")),
    ("claude_ideas", re.compile(HOME + r"/claude\.ideas(?![\w-])")),
    ("ssh_forbidden_host", re.compile(r"\b(?:ssh|scp|rsync)\b[^\n]*?\b(?:goldberg|hannah)\b")),
]
SOFT_STRINGS = [
    ("Where the claims stand", re.compile(r"Where the claims stand")),
    ("S1 ", re.compile(r"(?<![A-Za-z0-9])S1 ")),
    ("AR1", re.compile(r"(?<![A-Za-z0-9])AR1(?![0-9])")),
    ("AR2", re.compile(r"(?<![A-Za-z0-9])AR2(?![0-9])")),
]
BARE_AA = re.compile(HOME + r"/active-audit(?![\w/-])|\bcd active-audit\b")
REL_TOKEN = re.compile(r"(?<![\w/.-])(?:(?:avrl|lit|exp|runs)/|(?:STATUS|PROJECT|AGENTS)\.md)")
PATH_KEYS = ("file_path", "path", "pattern", "glob", "notebook_path")
LABEL_RUN = re.compile(r"^([A-Z][A-Z0-9_]*) (?:gate|plan|run|analyze|critic|screen[01]):")
SPILL = re.compile(r"/\.claude/projects/[^ ]*/tool-results/")
CAP = 50  # stored examples per (run, kind, rule[, source]); counts stay exact


def snippet(text, m, w=60):
    s = text[max(0, m.start() - w): m.end() + w].replace("\n", " ")
    return s


def tool_strings(block):
    """Strings of a tool_use block that name a command or a path."""
    inp = block.get("input") or {}
    out = []
    if isinstance(inp.get("command"), str):
        out.append(("command", inp["command"]))
    for k in PATH_KEYS:
        v = inp.get(k)
        if isinstance(v, str):
            if v.startswith("/"):
                v = os.path.normpath(v)
            out.append((k, v))
    # absolute path tokens in a command that climb with /../ are also checked in normalized form
    cmd = inp.get("command")
    if isinstance(cmd, str) and "/../" in cmd:
        for tok in re.split(r"[\s\"'`;|&()<>=]+", cmd):
            if tok.startswith(("/", "~", "$HOME", "${HOME}")) and "/../" in tok:
                out.append(("command_normalized", os.path.normpath(tok)))
    return out


def blocks(msg):
    """Yield (source, text) for every text-bearing piece of a message."""
    c = msg.get("content")
    role = msg.get("role")
    if isinstance(c, str):
        yield ("prompt" if role == "user" else "text"), c
        return
    for b in c if isinstance(c, list) else []:
        if not isinstance(b, dict):
            continue
        t = b.get("type")
        if t == "text":
            yield ("prompt" if role == "user" else "text"), b.get("text", "")
        elif t == "tool_use":
            yield "tool_use", json.dumps(b.get("input", {}))
        elif t == "tool_result":
            rc = b.get("content")
            if isinstance(rc, str):
                yield "tool_result", rc
            else:
                for x in rc if isinstance(rc, list) else []:
                    if isinstance(x, dict) and x.get("type") == "text":
                        yield "tool_result", x.get("text", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("transcript_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--live-root", default="/homes/rdubash/avrl_live")
    ap.add_argument("--default-run", default="UNLABELED")
    a = ap.parse_args()
    live = re.escape(a.live_root.rstrip("/"))
    live_pat = re.compile(r"(?:" + live + r"|(?:~|\$\{?HOME\}?)/avrl_live)/([A-Za-z0-9_]+)")

    rep = defaultdict(lambda: {"agents": 0, "tool_uses": 0, "hard": [], "soft": [],
                               "hard_by_rule": Counter(), "soft_by_rule": Counter(),
                               "soft_by_rule_nonprompt": Counter(), "tool_results_spill_hard": 0})
    stored = Counter()

    def add(run, kind, rule, agent, label, source, tool, text, m, spill=False):
        r = rep[run]
        r[kind + "_by_rule"][rule] += 1
        if kind == "soft" and source != "prompt":
            r["soft_by_rule_nonprompt"][rule] += 1
        if kind == "hard" and spill:
            r["tool_results_spill_hard"] += 1
        key = (run, kind, rule, source)
        stored[key] += 1
        if stored[key] <= CAP:
            h = {"rule": rule, "agent": agent, "label": label, "source": source, "tool": tool,
                 "match": m.group(0)[:120], "context": snippet(text, m)}
            if kind == "hard":
                h["tool_results_spill"] = spill
            r[kind].append(h)

    for meta_p in sorted(a.transcript_dir.rglob("agent-*.meta.json")):
        jsonl_p = meta_p.with_suffix("").with_suffix(".jsonl")
        try:
            label = json.loads(meta_p.read_text()).get("description", "") or ""
            lines = jsonl_p.read_text().splitlines()
        except (OSError, ValueError, AttributeError) as e:
            print(f"WARNING: skipping {meta_p.name}: {e}", file=sys.stderr)
            continue
        lm = LABEL_RUN.match(label)
        run = lm.group(1) if lm else a.default_run
        agent = meta_p.name[: -len(".meta.json")]
        rep[run]["agents"] += 1
        seen_tool = set()  # streamed chunks can repeat a tool_use block
        for line in lines:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = d.get("message")
            if not isinstance(msg, dict):
                continue
            c = msg.get("content")
            for b in (c if isinstance(c, list) else []):
                if not isinstance(b, dict) or b.get("type") != "tool_use" or (b.get("id") and b["id"] in seen_tool):
                    continue
                seen_tool.add(b.get("id"))
                rep[run]["tool_uses"] += 1
                raw_cmd = (b.get("input") or {}).get("command") or ""
                for field, s in tool_strings(b):
                    for rule, pat in HARD_STATIC:
                        for m in pat.finditer(s):
                            if field == "command_normalized" and m.group(0) in raw_cmd:
                                continue  # already counted on the raw command
                            add(run, "hard", rule, agent, label, "tool_use", b.get("name"), s, m,
                                spill=rule == "claude_dir" and bool(SPILL.match(s, m.end() - len("/.claude"))))
                    for m in live_pat.finditer(s):
                        if m.group(1) != run and not (field == "command_normalized" and m.group(0) in raw_cmd):
                            add(run, "hard", "other_run_base", agent, label, "tool_use", b.get("name"), s, m)
                    if field == "command" and BARE_AA.search(s):
                        m = REL_TOKEN.search(s)
                        if m:
                            add(run, "soft", "active_audit_relative", agent, label, "tool_use", b.get("name"), s, m)
            for source, text in blocks(msg):
                for rule, pat in SOFT_STRINGS:
                    for m in pat.finditer(text):
                        add(run, "soft", rule, agent, label, source, None, text, m)

    out = {"transcript_dir": str(a.transcript_dir), "live_root": a.live_root, "runs": {}}
    for run in sorted(rep):
        r = rep[run]
        nh = sum(r["hard_by_rule"].values())
        out["runs"][run] = {
            "status": "CONTAMINATED" if nh else "CLEAN",
            "agents": r["agents"],
            "tool_uses": r["tool_uses"],
            "hard_count": nh,
            "hard_count_excluding_tool_results_spill": nh - r["tool_results_spill_hard"],
            "soft_count": sum(r["soft_by_rule"].values()),
            "hard_by_rule": dict(r["hard_by_rule"]),
            "soft_by_rule": dict(r["soft_by_rule"]),
            "soft_by_rule_excluding_prompt": dict(r["soft_by_rule_nonprompt"]),
            "hard": r["hard"],
            "soft": r["soft"],
        }
        print(f"{run}: {out['runs'][run]['status']} agents={r['agents']} tool_uses={r['tool_uses']} "
              f"hard={nh} (spill={r['tool_results_spill_hard']}) {dict(r['hard_by_rule'])} "
              f"soft={out['runs'][run]['soft_count']} {dict(r['soft_by_rule'])}")
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
