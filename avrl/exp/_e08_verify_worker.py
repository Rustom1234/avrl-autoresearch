"""Child process for E08_mbpp.py's subprocess-isolated program execution. One call verifies one rollout.

Reads a JSON task from stdin: {"setup": str, "code": str, "tests": [str, ...], "calls": [str|null, ...]}.
"calls" are bare call expressions parsed from the left side of each test's "==" (e.g. "f(1, 2)" from
"assert f(1, 2) == 3"); null if a test could not be parsed into that shape.

Order of operations matters for the project's claim: the signature (bare calls) is computed FIRST, with
no access to expected values, before the real asserts are ever executed. This keeps the cheap proxy
honestly blind to gold.

Prints exactly one JSON line to stdout: {"correct": bool, "sig": [str, ...], "syntax_error": bool,
"exec_exception": bool}. The candidate program's own prints are captured and discarded so they cannot
corrupt this output. Network is disabled (socket.socket raises) since generated code should not need it
and the parent runs this with a 5 second subprocess timeout and no other sandboxing.
"""
import contextlib, io, json, socket, sys


def _no_net(*a, **k):
    raise OSError("network disabled for E08 verification")


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
