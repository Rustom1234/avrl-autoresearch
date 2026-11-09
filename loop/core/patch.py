"""SEARCH/REPLACE patching for train.py (aider style).

Where it sits. The model that proposes an edit never writes files. It replies with text, and this module turns that
text into a new train.py (or a precise failure report) before the linter and the GPU see anything.
Stdlib only, no I/O, no model calls. Entry points, from low level to high:
  parse_blocks        reply text -> [(search, replace), ...]
  apply_blocks        apply parsed blocks to a source string
  extract_full_file   the alternative reply form: one fenced block holding the whole file
  apply_text          pick the reply form and apply it
  apply_with_retries  apply_text plus a syntax check and a hint for the retry prompt (what arms.py calls)

The models emit edits as

    <<<<<<< SEARCH
    exact existing lines
    =======
    replacement lines
    >>>>>>> REPLACE

Matching is whole-line, never mid-line, never fuzzy:
  1. exact line-for-line match;
  2. only if that finds nothing: match ignoring trailing whitespace per line.
Each SEARCH must match exactly once at the moment it is applied. Blocks are applied
in order, each against the text produced by the previous blocks. On any failure the
original source is returned unchanged together with a report naming the failed block.

A full-file reply is accepted only when the text has no SEARCH markers and holds
exactly one fenced python block of more than 400 lines (report mode "full_file").

Why so strict. A fuzzy or mid-line match can land an edit somewhere the model did not intend and
still produce a file that trains, which would silently change what the experiment tested. Failing
loudly costs only a free retry; the report says which block failed and why.
"""
import ast
import difflib
import re

# A fenced block counts as a whole-file reply only above this length. The base train.py is longer than
# this, so a short illustrative snippet in a reply can never replace the file.
FULL_FILE_MIN_LINES = 400
HINT_LINE_CHARS = 120   # source lines quoted in a retry hint are cut to this width
FORMAT_REMINDER = ("Reply with blocks of <<<<<<< SEARCH / ======= / >>>>>>> REPLACE, each closed, "
                   "or with one complete fenced python block holding the whole file.")

# Marker lines. Five to nine marker characters are accepted (the canonical form has seven) to tolerate a
# miscounted marker; the marker must be alone on its line.
_SEARCH = re.compile(r"^\s*<{5,9}\s*SEARCH\s*$")
_DIVIDER = re.compile(r"^\s*={5,9}\s*$")
_REPLACE = re.compile(r"^\s*>{5,9}\s*REPLACE\s*$")
_FENCE = re.compile(r"^\s*```\s*([A-Za-z0-9_+-]*)\s*$")


def _split(text):
    """Lines of a model reply, with Windows line endings normalised."""
    return text.replace("\r\n", "\n").split("\n")


def parse_blocks(text):
    """Return [(search, replace), ...] in order. Each string holds its lines, every line
    terminated by "\\n" (so "" means zero lines and "\\n" means one blank line).

    Prose and code fences outside blocks are ignored. Raises ValueError on a malformed
    block (unterminated, missing divider, nested SEARCH marker)."""
    # Three-state line scanner: "out" (between blocks), "search" (after <<<<<<< SEARCH), "replace" (after =======).
    blocks, state, search, repl = [], "out", [], []
    for line in _split(text):
        if state == "out":
            if _SEARCH.match(line):
                state, search, repl = "search", [], []
        elif state == "search":
            if _DIVIDER.match(line):
                state = "replace"
            elif _SEARCH.match(line) or _REPLACE.match(line):
                raise ValueError("block %d: missing ======= divider" % (len(blocks) + 1))
            else:
                search.append(line)
        else:
            if _REPLACE.match(line):
                blocks.append(("".join(x + "\n" for x in search), "".join(x + "\n" for x in repl)))
                state = "out"
            elif _SEARCH.match(line):
                raise ValueError("block %d: missing >>>>>>> REPLACE terminator" % (len(blocks) + 1))
            else:
                repl.append(line)
    if state != "out":
        raise ValueError("block %d: unterminated (no >>>>>>> REPLACE)" % (len(blocks) + 1))
    return blocks


def _lines(s):
    """Inverse of the parse_blocks encoding: "" -> [], "a\\nb\\n" -> ["a", "b"], "\\n" -> [""]."""
    if s == "":
        return []
    parts = s.split("\n")
    if parts[-1] == "":
        parts.pop()
    return parts


def _find(norm_lines, norm_target):
    """Every index i where the line list norm_target occurs in norm_lines starting at line i."""
    n = len(norm_target)
    return [i for i in range(len(norm_lines) - n + 1) if norm_lines[i:i + n] == norm_target]


def _report(mode, n):
    """A fresh success report (keys documented in apply_blocks). mode: search_replace | full_file | none."""
    return {"mode": mode, "ok": True, "n_blocks": n, "failed_block": None,
            "reason": "", "message": "", "fallback_blocks": []}


def _fail(report, k, reason):
    """Mark report as failed at block k (0-based); the message counts blocks from 1 for readers."""
    report.update(ok=False, failed_block=k, reason=reason,
                  message="SEARCH block %d of %d: %s" % (k + 1, report["n_blocks"], reason))
    return report


def apply_blocks(source, blocks):
    """Apply blocks to source. Returns (new_source, report).

    report keys: mode, ok, n_blocks, failed_block (0-based index or None), reason
    ("no match", "multiple matches (k)", "empty SEARCH", ...), message (1-based, for
    humans and for feeding back to a model), fallback_blocks (0-based indices that
    needed the trailing-whitespace fallback). On failure the original source is returned."""
    report = _report("search_replace", len(blocks))
    if not blocks:
        report.update(ok=False, reason="no blocks", message="no SEARCH/REPLACE blocks")
        return source, report
    # Work on a list of lines and remember whether the file ended with a newline, so it is restored as it was.
    trailing_nl = source.endswith("\n")
    lines = source.split("\n")
    if trailing_nl:
        lines.pop()
    for k, (search, replace) in enumerate(blocks):
        target = _lines(search)
        if not target:
            # An empty SEARCH would match everywhere; there is no "insert at top" form.
            return source, _fail(report, k, "empty SEARCH")
        hits = _find(lines, target)
        if not hits:
            # Only fallback: models often drop or add trailing spaces. Leading whitespace is never relaxed
            # because indentation is meaning in Python.
            hits = _find([x.rstrip() for x in lines], [x.rstrip() for x in target])
            if hits:
                report["fallback_blocks"].append(k)
        if not hits:
            return source, _fail(report, k, "no match")
        if len(hits) > 1:
            return source, _fail(report, k, "multiple matches (%d)" % len(hits))
        i = hits[0]
        lines[i:i + len(target)] = _lines(replace)
    return "\n".join(lines) + ("\n" if trailing_nl else ""), report


def extract_full_file(text):
    """Return the file text if `text` holds exactly one python fenced block (lang python,
    py, python3 or untagged) of more than FULL_FILE_MIN_LINES lines, else None. An
    unterminated fence (truncated reply) never counts."""
    # cur is (language tag, lines) while inside a fence, None outside. Only a bare ``` closes a fence, so a
    # reply cut off mid-file leaves cur open and contributes no block.
    blocks, cur = [], None
    for line in _split(text):
        m = _FENCE.match(line)
        if cur is None:
            if m:
                cur = (m.group(1).lower(), [])
        elif m and m.group(1) == "":
            blocks.append(cur)
            cur = None
        else:
            cur[1].append(line)
    py = [b for b in blocks if b[0] in ("", "python", "py", "python3")]
    if len(py) != 1 or len(py[0][1]) <= FULL_FILE_MIN_LINES:
        return None
    return "\n".join(py[0][1]) + "\n"


def apply_text(source, text):
    """Entry point for a raw model reply: SEARCH/REPLACE blocks if any marker is present,
    else a full-file block if one qualifies. Returns (new_source, report)."""
    if any(_SEARCH.match(x) for x in _split(text)):
        try:
            blocks = parse_blocks(text)
        except ValueError as e:
            report = _report("search_replace", 0)
            report.update(ok=False, reason=str(e), message=str(e))
            return source, report
        return apply_blocks(source, blocks)
    full = extract_full_file(text)
    if full is not None:
        return full, _report("full_file", 0)
    report = _report("none", 0)
    msg = "no SEARCH/REPLACE blocks and no single fenced python block over %d lines" % FULL_FILE_MIN_LINES
    report.update(ok=False, reason=msg, message=msg)
    return source, report


def _short(line):
    """line cut to HINT_LINE_CHARS so a hint stays short inside the retry prompt."""
    return line if len(line) <= HINT_LINE_CHARS else line[:HINT_LINE_CHARS - 3] + "..."


def _block_hint(source, blocks, k, reason):
    """Retry hint for a SEARCH block that did not match exactly once. `source` is the text the
    block was applied to (earlier blocks already applied)."""
    target = _lines(blocks[k][0])
    head = target[:2]
    text = "SEARCH block %d %s. It starts with: %s." % (
        k + 1, "matched nothing" if reason == "no match" else "is ambiguous (%s)" % reason,
        " / ".join(repr(_short(x)) for x in head))
    src_lines = source.split("\n")
    after = " (numbered after the earlier blocks were applied)" if k else ""
    if reason == "no match":
        # Point the model at the closest real line (difflib similarity of at least 0.6 to the first
        # non-blank SEARCH line), since the usual cause is a line quoted from memory with a small change.
        first = next((x.strip() for x in head if x.strip()), "")
        stripped = [x.strip() for x in src_lines]
        near = difflib.get_close_matches(first, stripped, n=1, cutoff=0.6) if first else []
        if near:
            i = stripped.index(near[0])
            text += " Nearest source line (line %d%s): %r." % (i + 1, after, _short(src_lines[i]))
        text += " Copy the SEARCH lines exactly from the current source, including indentation."
    else:
        hits = _find(src_lines, target) or _find([x.rstrip() for x in src_lines], [x.rstrip() for x in target])
        text += " It matches at lines %s%s. Add neighbouring lines so it matches exactly once." % (
            ", ".join(str(i + 1) for i in hits[:5]), after)
    return text


def apply_with_retries(source, reply):
    """apply_text plus a syntax check, with the failure classified for a retry prompt. No model calls.

    Returns (new_source, report). report is apply_text's report plus
      failure_class: none | parse | no_match | multi_match | syntax
      retry_hint: short text a model can act on ("" when failure_class is none)
    parse covers an unusable reply (no blocks and no full file, malformed or empty-SEARCH block),
    syntax means the edit applied but the result does not parse (full-file replies are checked too).
    On any failure the original source is returned."""
    new, rep = apply_text(source, reply)
    rep = dict(rep, failure_class="none", retry_hint="")
    if rep["ok"]:
        try:
            ast.parse(new)
            return new, rep
        except (SyntaxError, ValueError, RecursionError, MemoryError) as e:
            line, rows = getattr(e, "lineno", None) or 0, new.split("\n")
            bad = rows[line - 1].strip() if 0 < line <= len(rows) else ""
            rep.update(ok=False, failure_class="syntax", reason="syntax error", message="edit does not parse: %s" % e,
                       retry_hint="The edit applied but the result does not parse: %s (line %d: %r). "
                                  "Fix that line and resend the edit." % (getattr(e, "msg", e), line, _short(bad)))
            return source, rep
    reason, k = rep["reason"], rep["failed_block"]
    if k is not None and (reason == "no match" or reason.startswith("multiple matches")):
        # Rebuild the text block k was matched against (blocks 0..k-1 applied) so that the line numbers
        # in the hint refer to what the model's block actually saw.
        blocks = parse_blocks(reply)
        before = apply_blocks(source, blocks[:k])[0] if k else source
        rep.update(failure_class="no_match" if reason == "no match" else "multi_match",
                   retry_hint=_block_hint(before, blocks, k, reason))
    else:
        rep.update(failure_class="parse", retry_hint="%s. %s" % (rep["message"].rstrip("."), FORMAT_REMINDER))
    return source, rep
