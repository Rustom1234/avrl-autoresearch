"""Program card, role prompts and JSON schemas (DESIGN.md section 10).

Where it sits. Every model call of the loop is built here and sent by arms.py: the system prompt is
program_card(n_slots), the message body comes from one of the role builders below, and the reply must
match SCHEMAS[role]. Nothing in this file calls a model or reads a file.

Roles (which model fills a role is set by the arm's configuration, not here):
  base_step   arms A and B: propose and implement one change in a single call.
  planner     arms C and D: propose a backlog of ideas (planner_widen when the search has stalled).
  voter       a cheap yes/no vote on one idea; the votes seed the scheduler's prior for that idea.
  coder       arms C and D: implement one backlog idea as an edit.
  repairer    fix linter violations or a crash with the smallest edit.
  analyst     write the short memo that later planner and coder prompts carry.
  critic      review a diff and the harness numbers; may veto.

One program card for every arm. Each role builder returns the message body (user role) as one string. The card is
the system prompt of every call so it is byte-identical across arms and roles. train.py is included in full only for
base_step, coder and repairer. Prompts state facts and the output format, without exhortation.

Terms: val_bpb = validation bits per byte, lower is better. slot = one GPU training run. arm = one loop
configuration (A to D). incumbent = the best train.py accepted so far.
"""
import re

# Idea families. The scheduler keeps accept/reject counts per family, and the planner must label each idea.
FAMILIES = ("optimizer", "schedule", "architecture", "batch", "regularization", "data", "other")

# The one worked example of the edit format (parsed by patch.py). It is a real line of the base train.py,
# so the example itself applies cleanly.
EXAMPLE = """<<<<<<< SEARCH
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
=======
WEIGHT_DECAY = 0.1      # cautious weight decay for Muon
>>>>>>> REPLACE"""

# The system prompt. It states the harness rules up front so that no arm loses slots to rules it could not
# know; {n_slots} and {example} are filled by program_card().
CARD = """# autoresearch program card

Goal: lower val_bpb (validation bits per byte, lower is better) of a small GPT trained by train.py.

Setting. One NVIDIA A100 40 GB per run. Training runs for a fixed 300 s of wall-clock time (startup and compile are
not counted). The baseline uses DEVICE_BATCH_SIZE 64 (128 overflows 40 GB). The metric, data and tokenizer are
unchanged from the original task.

What you may change: train.py only (architecture, optimizer, hyperparameters, schedules, batch size, model size, the
training loop). Everything in it is fair game except the facts below. Install nothing.

Harness facts (enforced by the harness, stated so that no slot is wasted):
- The two seed lines `torch.manual_seed(42)` and `torch.cuda.manual_seed(42)` are owned by the harness. Editing them
  invalidates the run.
- prepare.py is read only and is not shown. train.py imports a fixed set of names from it on one line; keep that line
  as it is and do not reference prepare.py internals (paths, files, shards, splits, helpers).
- The official val_bpb is recomputed by the harness from your trained model. The number your script prints is not used.
  Do not touch the evaluation call or the final summary printing.
- A run whose startup plus training takes over 450 s is rejected, as is any run where the harness finds the evaluation
  path altered.
- VRAM is a soft constraint. Simpler is better: a tiny gain that adds ugly code is not worth it, and equal results
  from less code are a win.

Budget. You have {n_slots} slots in total. One slot is one GPU run of about 7.5 minutes. A crash, a replicate run
and an audit run each use a slot. Rewriting an edit that failed to apply is free (at most 2 per slot).

Output format. Give edits to train.py as SEARCH/REPLACE blocks, aider style. Each SEARCH must match the current
file exactly once, line for line, including indentation. Blocks are applied in order. Example:

{example}

Reply as JSON with the fields requested in the task. Put the blocks, as text, in the field `edits_text`."""

# ---- JSON schemas of the replies, one per role (SCHEMAS maps role name to schema) ----
# edits_text carries the SEARCH/REPLACE blocks as plain text; patch.py parses it, the schema does not.
EDIT_SCHEMA = {
    "type": "object",
    "properties": {"description": {"type": "string"}, "edits_text": {"type": "string"},
                   "abort": {"type": "boolean"}, "abort_reason": {"type": "string"}},
    "required": ["description", "edits_text"]}
REPAIR_SCHEMA = {
    "type": "object",
    "properties": {"description": {"type": "string"}, "edits_text": {"type": "string"}},
    "required": ["edits_text"]}
BACKLOG_SCHEMA = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": {
        "type": "object",
        "properties": {"title": {"type": "string"}, "spec": {"type": "string"},
                       "family": {"type": "string", "enum": list(FAMILIES)}},
        "required": ["title", "spec", "family"]}}},
    "required": ["items"]}
VOTE_SCHEMA = {
    "type": "object",
    "properties": {"vote": {"type": "string", "enum": ["yes", "no"]}, "reason": {"type": "string"}},
    "required": ["vote"]}
MEMO_SCHEMA = {"type": "object", "properties": {"memo": {"type": "string"}}, "required": ["memo"]}
CRITIC_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}, "issues": {"type": "array", "items": {"type": "string"}},
                   "veto": {"type": "boolean"}, "recomputed": {"type": "object"}},
    "required": ["ok", "issues", "veto"]}

SCHEMAS = {"base_step": EDIT_SCHEMA, "coder": EDIT_SCHEMA, "repairer": REPAIR_SCHEMA, "planner": BACKLOG_SCHEMA,
           "planner_backlog": BACKLOG_SCHEMA, "planner_refresh": BACKLOG_SCHEMA, "planner_widen": BACKLOG_SCHEMA,
           "voter": VOTE_SCHEMA, "analyst": MEMO_SCHEMA, "critic": CRITIC_SCHEMA}   # "planner" kept for calibration.py


# ---- prompt builders ----
def program_card(n_slots):
    """The system prompt of every call. n_slots is the loop's total GPU-run budget."""
    return CARD.format(n_slots=n_slots, example=EXAMPLE)


def _file(train_py):
    """The whole current train.py as a fenced block (only for roles that must edit it)."""
    return "## Current train.py\n```python\n%s\n```" % train_py.rstrip("\n")


def retry_suffix(prev, hint):
    """Appended to a prompt when the previous reply could not be used (hint comes from the patch applier)."""
    if not hint:
        return ""
    return ("\n\n## Your previous reply could not be used\n%s\nYour previous edits_text was:\n%s\n"
            "Send corrected edits_text." % (hint, (prev or "")[:3000]))


def hyper_block(train_py):
    """The module-level hyperparameter assignments of train.py (for roles that do not get the whole file).

    Collects the UPPERCASE assignments between the "# Hyperparameters" comment and the first seed line,
    which is where the base file keeps them. The planner gets this instead of the full file to keep its
    prompt small."""
    lines, on = [], False
    for ln in train_py.splitlines():
        if ln.startswith("# Hyperparameters"):
            on = True
        elif on and ln.startswith("torch.manual_seed"):
            break
        elif on and re.match(r"[A-Z][A-Z0-9_]* = ", ln):
            lines.append(ln)
    return "\n".join(lines)


def _abort_note(allow_abort):
    """Sentence offering the "abort" reply (decline to spend a slot); empty when the arm does not allow it."""
    return (' If you see no change worth a slot, set "abort": true and give "abort_reason"; edits_text may then be '
            'empty.' if allow_abort else "")


def base_step(train_py, table, last_tail, allow_abort):
    """Arms A and B: one call that both picks an idea and writes the edit. table is the ledger's results
    table, last_tail the sanitized view of the previous run's log (sanitize_log.py)."""
    return "\n\n".join([
        _file(train_py), "## Results so far\n%s" % table, "## Last run\n%s" % (last_tail or "none yet"),
        '## Task\nPropose one change to train.py that you expect to lower val_bpb. Reply as JSON: {"description": '
        'one sentence, "edits_text": SEARCH/REPLACE blocks, "abort": false, "abort_reason": ""}.'
        + _abort_note(allow_abort)])


def coder(train_py, item, table, memo, family_stats, allow_abort):
    """Arms C and D: implement one backlog item (dict with title, family, spec) chosen by the scheduler.
    memo is the analyst's latest notes; family_stats maps family to its Beta counts {alpha, beta}."""
    return "\n\n".join([
        _file(train_py), "## Results so far\n%s" % table,
        "## Notes from earlier experiments\n%s\nFamily outcomes (alpha = 1 + accepted, beta = 1 + not accepted): %s"
        % (memo or "none yet", _fam(family_stats)),
        "## Task\nImplement this idea as a minimal edit to train.py.\nTitle: %s\nFamily: %s\nSpec: %s\n"
        'Reply as JSON: {"description": one sentence, "edits_text": SEARCH/REPLACE blocks, "abort": false, '
        '"abort_reason": ""}.' % (item["title"], item["family"], item["spec"]) + _abort_note(allow_abort)
    ])


def repairer_lint(train_py, violations):
    """Repair prompt after a linter failure. train_py is the rejected candidate, violations the linter's
    hard list of {rule, line, message}."""
    vio = "\n".join("- %s (line %s): %s" % (v.get("rule"), v.get("line"), v.get("message")) for v in violations)
    return "\n\n".join([
        _file(train_py),
        "## Static check failed\nThe file above is your candidate. The checker reports these hard violations:\n%s"
        % vio,
        '## Task\nRemove the violations with the smallest possible edits, keeping the intent of the change. Reply as '
        'JSON: {"description": one sentence, "edits_text": SEARCH/REPLACE blocks against the file shown}.'])


def repairer_crash(train_py, tail):
    """Repair prompt after a crashed run. tail is the last lines of the run log (traceback included)."""
    return "\n\n".join([
        _file(train_py),
        "## The run crashed\nLast lines of the run log:\n```\n%s\n```" % tail,
        '## Task\nFix the cause with the smallest possible edits. If the idea cannot work, change it so that it does. '
        'Reply as JSON: {"description": one sentence, "edits_text": SEARCH/REPLACE blocks against the file shown}.'])


def _fam(family_stats):
    """Family outcome counts on one line, sorted by family so the text is deterministic."""
    return "; ".join("%s a=%d b=%d" % (k, v["alpha"], v["beta"]) for k, v in sorted(family_stats.items())) or "none"


def _backlog_titles(items):
    """One line per backlog item (id, title, family, status), shown so the planner does not repeat ideas."""
    return "\n".join("- [%s] %s (%s, %s)" % (i["id"], i["title"], i["family"], i["status"]) for i in items) or "none"


def planner(train_py, n_items, table, memo, family_stats, existing, widen=False):
    """Ask for n_items new backlog ideas. The planner sees only the hyperparameter block, never the full
    file, and each idea must stand alone because the coder implements it without seeing the others.
    widen=True is the call made after a run of rejections: it asks for untried families and bolder ideas."""
    kind = ("The search has stalled. Propose %d ideas from families that have not been tried recently, including at "
            "least a few bolder ones." if widen else "Propose %d distinct ideas.") % n_items
    return "\n\n".join([
        "## Hyperparameters of the current train.py\n%s" % hyper_block(train_py),
        "## Results so far\n%s" % table,
        "## Notes from earlier experiments\n%s\nFamily outcomes: %s" % (memo or "none yet", _fam(family_stats)),
        "## Ideas already in the backlog (do not repeat)\n%s" % _backlog_titles(existing),
        "## Task\n%s Each idea is one self-contained change to train.py that a coder can implement from one paragraph, "
        "without seeing the other ideas. Give each a short title, a spec paragraph, and a family from: %s. "
        'Reply as JSON: {"items": [{"title", "spec", "family"}]}.' % (kind, ", ".join(FAMILIES))])


def voter(item):
    """Yes/no plausibility vote on one idea. Deliberately context-free (no table, no file): several
    independent votes per idea are taken, and their agreement is the signal."""
    return ("An idea for lowering val_bpb of a small GPT trained for 300 s on one A100:\nTitle: %s\nSpec: %s\n"
            'Will it plausibly lower val_bpb under that budget? Reply as JSON: {"vote": "yes" or "no", "reason": one '
            "sentence}." % (item["title"], item["spec"]))


def analyst(table, memo, family_stats):
    """Ask for the next memo (at most 150 words). The memo may only reuse numbers from the table, so no
    figure enters later prompts that the harness did not produce."""
    return "\n\n".join([
        "## Results so far\n%s" % table,
        "## Previous notes\n%s" % (memo or "none"), "## Family outcomes\n%s" % _fam(family_stats),
        '## Task\nWrite notes for the next planning round in at most 150 words: which kinds of changes helped, which '
        'did not, and what to try next. Use only numbers that appear above. Reply as JSON: {"memo": text}.'])


def critic(diff, numbers, reason):
    """Review prompt. diff is candidate against incumbent, numbers the harness values as text, reason why
    this record was sent for review (for example an audit or a borderline accept). A veto is reserved for
    metric tampering or a result-voiding bug, so the critic cannot reject on taste."""
    return "\n\n".join([
        "## Why you are asked\n%s" % reason,
        "## Diff of the candidate against the incumbent\n```diff\n%s\n```" % diff,
        "## Numbers computed by the harness code\n%s" % numbers,
        "## Task\nReview the diff for (1) any attempt to change how val_bpb is measured, the data, the seed lines or "
        "the evaluation call, (2) bugs that would make the result meaningless, (3) claims that do not match the "
        "numbers above. Recompute z from the numbers if they allow it. Set veto true only for (1) or (2). Reply as "
        'JSON: {"ok": bool, "issues": [strings], "veto": bool, "recomputed": {"z": number or null}}.'])
