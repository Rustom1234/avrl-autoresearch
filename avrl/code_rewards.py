"""Code-generation reward machinery for E10 (MBPP/HumanEval) phase-1 GRPO. See p1_code_grpo.py
(CLI + GRPOTrainer wiring) and avrl/exp/E08_mbpp.py / exp/_e08_verify_worker.py, which this file
ports its prompt-extraction regexes, assert-parsing, and sandboxed-execution protocol from.
E08_mbpp.py cannot be imported directly (it parses sys.argv and runs top-level code at import
time), so the small pieces of it this file needs are ported here with the same logic, not
imported; the actual execution sandbox is avrl/exp/_e10_verify_worker.py, a near-verbatim copy of
_e08_verify_worker.py (see that file's docstring for why it is a copy, not a shared import, and
for the one addition -- self-imposed resource limits -- beyond E08's original).

This file is pure stdlib + numpy: no torch/transformers/trl import anywhere, so py_compile and
every function here are checkable, and testable, on a machine with none of those installed
(mirrors judge_rewards.py's and p1_grpo.py's module-docstring convention).

Fairness note (ties to allocators.py's invariants -- see its module docstring): nothing in this
file ever mixes a gold-derived signal (the real-asserts `correct` flag) into a feat_vec; that
mixing, and the gold-vs-proxy reward split, happens in p1_code_grpo.py's CodeRewardState, exactly
mirroring p1_grpo.py's RewardState. This file only provides the pure prompt/extraction/sandbox
primitives both CodeRewardState (training) and the eval callbacks (periodic + --eval_only) share.
"""
import ast
import json
import os
import re
import signal
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.abspath(__file__))  # avrl/
WORKER = os.path.join(ROOT, "exp", "_e10_verify_worker.py")

INSTRUCTION = (
    "Write a single Python function that solves the task below. Respond with exactly one "
    "python code block containing the function definition and any needed imports. No example "
    "usage, no explanation, no text outside the code block."
)
HUMANEVAL_SETUP = "from typing import *\n"  # permissive import safety net (standard HumanEval-eval
                                             # convention: the prompt itself may name typing types
                                             # the model forgets to re-import); grading substance
                                             # (correctness of the model's own code) is unaffected.

CODE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)  # a fully closed fence anywhere
CLOSE_RE = re.compile(r"```")


def build_instruction_content(problem_text, first_assert=None):
    """Instruction + problem (+ first assert, if given), no fence/continuation suffix. Used as a
    chat 'user' message for a model WITH a chat template (the Instruct reference bar); see
    build_prompt() for the plain-text, fence-opened version used for everything else (OLMo-2-
    0425-1B base has no chat template)."""
    parts = [INSTRUCTION, "", "Problem:", problem_text.strip()]
    if first_assert:
        parts += ["", "Your code should satisfy this test:", first_assert.strip()]
    return "\n".join(parts)


def build_prompt(problem_text, first_assert=None):
    """Plain-text prompt for OLMo-2-0425-1B base (no chat template, so there is no
    apply_chat_template to lean on): instruction, problem, first assert (MBPP's problem text
    alone does not name the function the asserts expect; the first test_list entry does -- same
    device E08_mbpp.py's SYS/user message uses), a bare 'Answer:' line, then an OPENED ```python
    fence with no closing fence. The model's completion is expected to continue directly inside
    that fence (a base model has no instruction-following prior to lean on either, so the open
    fence is doing the work a chat template's generation-prompt would). Used for every training
    prompt and for evaluating the base/trained policy; see extract_code() for the matching
    extraction. Design choice, stated per Karpathy rule 1: the SAME instruction/problem/assert
    content (build_instruction_content) is reused verbatim for the chat-template path so the two
    prompt styles differ only in the chat wrapper / fence suffix, not in task content."""
    return build_instruction_content(problem_text, first_assert) + "\n\nAnswer:\n```python\n"


def extract_code(text):
    """Extract a candidate program from a completion. Two prompt styles feed this function, and
    they are NOT interchangeable:
      - build_prompt() (base-model / training path) puts the OPENING ```python fence in the
        PROMPT, not the completion. The completion is continuing an already-open fence, so its
        FIRST bare ``` is always the real closing delimiter. A later, fully-closed fence pair
        (e.g. a base model rambling on with 'Example usage:\\n```...```' after correctly closing
        its own code) is a doc-style continuation, not the real program, and must not win.
      - The chat-templated Instruct path generates its OWN opening fence inside the completion
        (no open fence in the prompt to continue), so there the fully-closed-fence-anywhere
        extraction is correct and expected.
    Dispatch on which case applies by checking whether the completion itself opens with a fence:
      1. Completion starts with ``` (Instruct self-fenced response): use CODE_RE to pull the
         first fully-closed ```...``` block's contents.
      2. Otherwise (continuing the prompt's already-open fence): the FIRST bare ``` closes it;
         everything before it is the code. No ``` at all: take the whole completion (open-ended,
         same fallback E08_mbpp.py's CODE_RE_OPEN uses for a missing closing fence).
    Returns None (not an empty string) if the result is blank, so callers can short-circuit
    exactly like E08_mbpp.py's `code is None` path (NO_CODE)."""
    if text.lstrip().startswith("```"):
        m = CODE_RE.search(text)
        code = m.group(1) if m else text
    else:
        m = CLOSE_RE.search(text)
        code = text[:m.start()] if m else text
    code = code.strip()
    return code or None


def parse_call(assert_src):
    """Left side of a simple '== ' comparison in a test assert, as source text -- ported
    verbatim from E08_mbpp.py's parse_call(). None if the assert does not have that shape (rare
    in MBPP; such tests still run for the gold check, just not the signature)."""
    try:
        node = ast.parse(assert_src.strip()).body[0]
        if (isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare)
                and len(node.test.ops) == 1 and isinstance(node.test.ops[0], ast.Eq)):
            return ast.unparse(node.test.left)
    except Exception:
        pass
    return None


def build_mbpp_task(code, test_list, test_setup_code):
    """(code, setup, tests, calls) for one MBPP row + one candidate program, ready for
    verify_program()/run_batch(). `calls` are the bare call expressions parsed from each test's
    '==' left side -- the cheap signature's probe inputs; see _e10_verify_worker.py's docstring
    for why the signature is computed before the real asserts ever run."""
    tests = list(test_list)
    calls = [parse_call(t) for t in tests]
    return (code, test_setup_code or "", tests, calls)


def build_humaneval_task(code, test_src, entry_point):
    """(code, setup, tests, calls) for one HumanEval row + one candidate program. HumanEval's
    `test` field is a `check(candidate)` harness, not simple per-assert '==' comparisons, so
    there is no per-input signature here (HumanEval is eval-only in this experiment -- see
    PROJECT.md/the task spec -- never used for the cheap-proxy training reward); `calls=[None]`
    means verify_program() reports sig=["UNPARSED"], which callers evaluating HumanEval simply
    ignore and read only `correct`. The single `tests` entry defines `check`/METADATA via the
    harness text and then calls check(<entry_point>) against the candidate's own namespace --
    one exec(t, ns) statement is enough for _e10_verify_worker.py's existing per-test loop."""
    tests = [test_src + f"\ncheck({entry_point})\n"]
    return (code, HUMANEVAL_SETUP, tests, [None])


def verify_program(code, setup, tests, calls, timeout=5.0):
    """One sandboxed execution of a candidate program, via avrl/exp/_e10_verify_worker.py.
    Ported from E08_mbpp.py's verify() (same NO_CODE short-circuit, same worker I/O protocol,
    same TIMEOUT/PARENT_ERROR fallbacks) with one change beyond E08_mbpp.py: the worker is
    launched with start_new_session=True (its own process group) and, on timeout, the WHOLE
    group is SIGKILLed via os.killpg -- not just the direct child subprocess.run would kill --
    so a forkbomb's grandchildren die too. E08_mbpp.py graded already-generated analysis
    rollouts (effectively trusted, exploratory data); these are live RL rollouts from a policy
    actively being optimized against the reward this function returns, so a reward-hacking
    policy has more incentive to probe the sandbox. The worker's own resource.setrlimit calls
    (RLIMIT_AS, RLIMIT_CPU; see its docstring for why RLIMIT_NPROC is deliberately NOT set) are
    the other half of that defense-in-depth and need no cooperation from this function."""
    n = len(tests)
    if not code:
        return dict(correct=False, sig=tuple(["NO_CODE"] * n), syntax_error=False, exec_exception=False)
    task = json.dumps(dict(setup=setup, code=code, tests=tests, calls=calls))
    try:
        proc = subprocess.Popen(
            [sys.executable, WORKER], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, start_new_session=True,
        )
        try:
            stdout, _ = proc.communicate(input=task, timeout=timeout)
            res = json.loads(stdout.strip().splitlines()[-1])
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.communicate()  # reap, discard
            res = dict(correct=False, sig=["TIMEOUT"] * n, syntax_error=False, exec_exception=True)
    except Exception:
        res = dict(correct=False, sig=["PARENT_ERROR"] * n, syntax_error=False, exec_exception=True)
    return dict(correct=bool(res["correct"]), sig=tuple(res["sig"]),
                syntax_error=bool(res["syntax_error"]), exec_exception=bool(res["exec_exception"]))


def run_batch(tasks, timeout=5.0, max_workers=None):
    """Run every (code, setup, tests, calls) tuple in `tasks` through verify_program(), in
    parallel, via a thread pool. subprocess.Popen/communicate releases the GIL while the child
    runs, so threads -- not processes -- are enough to get one OS process (and real multi-core
    parallelism) per program; this is E08_mbpp.py's own pattern
    (`ThreadPoolExecutor(max_workers=a.workers)` around its per-rollout verify() call). Deliberately
    NOT a preexec_fn-based approach to resource limiting (the subprocess docs warn preexec_fn is
    unsafe combined with a multithreaded caller, since fork() only duplicates the calling thread
    and another thread could be mid-malloc at that instant); see _e10_verify_worker.py, which
    applies its limits to itself, after exec, in a fresh single-threaded process instead.

    Callers should flatten a whole step's n_groups*G programs into one `tasks` list (not call
    this once per group) so all of them run concurrently -- e.g. 64 at prompts_per_step=8, G=8 --
    not G at a time. Returns a list of verify_program() result dicts, same order as `tasks`."""
    if not tasks:
        return []
    max_workers = max_workers or min(len(tasks), os.cpu_count() or 8)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(lambda t: verify_program(*t, timeout=timeout), tasks))


def group_signature_stats(results, G):
    """Majority-signature vote over one group's G verify_program() results -- ported from
    E08_mbpp.py's per-pid loop. Returns (majority_sig, agreement, n_distinct, majority_correct):
    majority_correct is the REAL-asserts `correct` flag of the (a) majority-signature member(s),
    i.e. gold, used by callers for reward (verified groups) or logging only (unverified groups --
    never fed to an allocator as a feature); see CodeRewardState in p1_code_grpo.py."""
    sigs = [r["sig"] for r in results]
    counts = Counter(sigs)
    majority_sig, maj_count = counts.most_common(1)[0]
    agreement = maj_count / G
    n_distinct = len(counts)
    maj_members = [r for r in results if r["sig"] == majority_sig]
    majority_correct = int(round(sum(r["correct"] for r in maj_members) / len(maj_members)))
    return majority_sig, agreement, n_distinct, majority_correct


def visible_signature_stats(results, visible_pass):
    """Confidence/n_distinct for the visible-test cheap proxy (p1_code_grpo.py's --proxy
    visible, E10b): the same output-signature majority vote group_signature_stats() does, but
    restricted to the completions that pass the single visible test (test_list[0]). Only `sig`
    (gold-free, same field the default --proxy signature vote reads) and the caller's own
    visible_pass flags are read here -- no `correct` (gold) -- so this is safe to feed into a
    feat_vec exactly like group_signature_stats()'s output. If no completion passes the visible
    test, no completion the cheap grader could wrongly reward exists, so no false positive is
    possible: confidence is defined as 1.0 (vacuously trustworthy) and n_distinct as 0."""
    sigs = [r["sig"] for r, v in zip(results, visible_pass) if v]
    if not sigs:
        return 1.0, 0
    counts = Counter(sigs)
    _, maj_count = counts.most_common(1)[0]
    return maj_count / len(sigs), len(counts)
