"""Child process for p1_code_grpo.py / code_rewards.py's sandboxed program execution (E10, MBPP +
HumanEval). Protocol, control flow, and the signature-before-gold ordering are ported verbatim
from exp/_e08_verify_worker.py (read that file's docstring first for why the signature is computed
BEFORE the real asserts ever run -- it keeps the cheap proxy honestly blind to gold). This is a
separate copy, not an import of _e08_verify_worker.py, so E08's own artifact is untouched; the only
change versus E08's worker is the resource.setrlimit block below, applied to this process itself
before stdin is even read.

Why self-imposed limits, and why here rather than in the parent (code_rewards.verify_program): the
parent launches this script via subprocess.Popen from a thread pool (avrl/code_rewards.run_batch).
A preexec_fn passed to Popen runs in the PARENT's address space between fork() and exec(), and the
Python docs warn this is unsafe to combine with a multithreaded caller (fork() only duplicates the
calling thread; another thread could be mid-malloc, holding a lock this process never releases).
Setting limits here instead -- after exec(), in a brand-new, single-threaded process -- has none of
that hazard and is exactly as effective. E08_mbpp.py's worker graded already-generated analysis
rollouts (effectively trusted, exploratory data); this worker grades live RL rollouts from a policy
being actively optimized against the reward these asserts produce, so a reward-hacking policy has a
real incentive to probe the sandbox (unbounded allocation, a runaway loop) to stall or crash the
verifier -- hence the added limits.

RLIMIT_NPROC is deliberately NOT set: it caps the number of processes for the REAL UID, not this
process's subtree, and this worker shares its uid (rdubash) with the user's other live sessions on
this shared, multi-session machine. Capping it from inside a short-lived sandboxed worker risks
starving unrelated shells/sessions/daemons under the same uid of the ability to fork, with no
reliable way to pick a safe number (many parallel workers race to set the SAME uid-wide limit
concurrently). RLIMIT_AS and RLIMIT_CPU are per-process (safe) and, together with the PARENT's
process-group SIGKILL on a 5s wall-clock timeout (code_rewards.verify_program), are the sandbox: a
forkbomb's grandchildren die with the group, not just this direct child.

Reads a JSON task from stdin: {"setup": str, "code": str, "tests": [str, ...], "calls": [str|null, ...]}.
Prints exactly one JSON line to stdout: {"correct": bool, "sig": [str, ...], "syntax_error": bool,
"exec_exception": bool}. Network is disabled.
"""
import resource

_MEM_BYTES = 2 * 1024 ** 3  # 2 GiB address space -- generous for stdlib-only MBPP/HumanEval solutions
_CPU_SECONDS = 8            # backstop behind the parent's 5s wall-clock subprocess timeout

for _res, _val in ((resource.RLIMIT_AS, _MEM_BYTES), (resource.RLIMIT_CPU, _CPU_SECONDS)):
    try:
        resource.setrlimit(_res, (_val, _val))
    except (ValueError, OSError):
        pass  # best-effort: some limits may already be capped lower by the OS/container

import contextlib
import io
import json
import socket
import sys


def _no_net(*a, **k):
    raise OSError("network disabled for E10 verification")


socket.socket = _no_net
socket.create_connection = _no_net

task = json.loads(sys.stdin.read())
setup, code, tests, calls = task["setup"], task["code"], task["tests"], task["calls"]
n = len(tests)
out = {"correct": False, "sig": ["UNDEFINED"] * n, "syntax_error": False, "exec_exception": False}
buf = io.StringIO()
try:
    with contextlib.redirect_stdout(buf):
        ns = {}
        try:
            compiled = compile(setup + "\n" + code, "<prog>", "exec")
        except SyntaxError:
            out["syntax_error"] = True
            out["sig"] = ["SYNTAX_ERROR"] * n
        else:
            try:
                exec(compiled, ns)
            except Exception as e:
                out["exec_exception"] = True
                out["sig"] = [f"DEFINE_ERROR:{type(e).__name__}"] * n
            else:
                sig = []
                for c in calls:
                    if c is None:
                        sig.append("UNPARSED")
                        continue
                    try:
                        sig.append(repr(eval(c, ns)))
                    except Exception as e:
                        sig.append(f"EXC:{type(e).__name__}")
                        out["exec_exception"] = True
                out["sig"] = sig
                correct = True
                for t in tests:
                    try:
                        exec(t, ns)
                    except Exception:
                        correct = False
                        break
                out["correct"] = correct
except Exception as e:
    out["exec_exception"] = True
    out["sig"] = [f"HARNESS_ERROR:{type(e).__name__}"] * n
sys.stdout.write(json.dumps(out) + "\n")
