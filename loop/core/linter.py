"""Static linter for candidate train.py files (DESIGN.md section 6).

Where it sits. In arms B, C and D every edited train.py goes through lint() after the patch applies and
before a GPU run is submitted (arm A calls it only for the edit_class label, never as a gate). A file
with a hard violation gets one repair attempt and is otherwise skipped before any GPU time is spent
(a slot is one 300 s training run on one GPU).

lint(new_source, base_source) -> LintResult. AST based; only the string-literal rules look
at text. It is a fast first gate that keeps the loop's models from burning a GPU slot on a file the
harness would reject or that tries obvious metric gaming. It is NOT the security boundary:
the harness tail (sha256 of prepare.py, evaluate_bpb bytecode hash, recomputed official
val_bpb, sealed shard, wall time) is. See README.md for what each side covers.

What the rules protect. val_bpb (validation bits per byte, lower is better) is only comparable across
candidates if every candidate is scored by the same evaluate_bpb on the same data with the same seeds.
So the rules stop a candidate from reaching the file system or other processes, from rebinding or
patching the names imported from prepare.py and the modules the scoring path uses, from naming the
validation or sealed data, and from changing the two harness-owned seed lines.

Layout of this file:
  constants                the rule tables (what is forbidden, what is protected)
  small AST helpers        _chain ... _bindings
  _hard_rules              one pass over all AST nodes; any item it returns makes ok False
  _warning_rules           advisory findings, stored with the run, never blocking
  _edit_class, _diff_lines size and kind of the change against the base
  lint                     the entry point

Violation items are dicts {rule, line, message}; line 0 means "whole file".
"""
import ast
import difflib
import re
from dataclasses import dataclass, field

# ---- rule tables ----
MAX_BYTES = 64 * 1024
# Number of evaluation sequences in prepare.py. evaluate_bpb runs EVAL_STEPS_TOKENS // batch_size steps, so a
# batch size that does not divide it scores fewer tokens (the batch-divisibility warning).
EVAL_STEPS_TOKENS = 40 * 524288 // 2048  # 10240 = EVAL_TOKENS // MAX_SEQ_LEN in prepare.py

FORBIDDEN_IMPORTS = frozenset(
    "subprocess socket requests urllib http shutil signal ctypes importlib multiprocessing sys pickle inspect".split())
# Beyond DESIGN.md section 6: modules that reach open/exec/file access without calling open().
EXTRA_FORBIDDEN_IMPORTS = frozenset("builtins pathlib io glob runpy".split())
FORBIDDEN_CALLS = frozenset(
    "open exec eval compile setattr delattr globals locals vars __import__ breakpoint input".split())
# Names whose mere reference is flagged (aliasing: f = open). input and vars are common local names.
FORBIDDEN_REFS = FORBIDDEN_CALLS - {"input", "vars"}
OS_ALLOWED = frozenset({"environ", "path"})
# The five names train.py imports from prepare.py (the read-only data and evaluation module), and the
# modules the base file uses. A candidate may use them but never rebind, shadow or patch them.
PREPARE_NAMES = ("MAX_SEQ_LEN", "TIME_BUDGET", "Tokenizer", "make_dataloader", "evaluate_bpb")
MODULE_NAMES = ("prepare", "time", "os", "math", "gc", "torch", "nn", "F")
PROTECTED = frozenset(PREPARE_NAMES + MODULE_NAMES)
ATTR_ROOTS = PROTECTED | {"kernels"}                   # roots whose attributes may not be assigned
BARE_ROOTS = frozenset(MODULE_NAMES + ("kernels",))    # modules that may appear only as X.attr
GUARDED_ROOTS = frozenset({"os", "time", "math", "gc", "torch", "kernels", "prepare"})   # may not be import-aliased
# What each protected name must resolve to when an import statement binds it.
CANONICAL = {"os": "os", "time": "time", "math": "math", "gc": "gc", "torch": "torch",
             "nn": "torch.nn", "F": "torch.nn.functional", "prepare": "prepare",
             **{n: "prepare." + n for n in PREPARE_NAMES}}
# Beyond DESIGN.md section 6: ways to reach another module's globals or code objects.
DUNDER_ESCAPE = frozenset({"__dict__", "__globals__", "__code__", "__builtins__", "__closure__",
                           "__defaults__", "__kwdefaults__", "__subclasses__", "__loader__", "__spec__"})
# Substrings (lower case) that no string literal may contain: they name the data shards, the sealed shard
# or the cache directory, which train.py has no reason to mention.
BAD_STRINGS = ("shard_", "sealed", ".parquet", "data_dir", "val_filename", "token_bytes", ".cache/autoresearch")
VAL_WORD = re.compile(r"\bval\b")  # string-concat-escape: "val" counts only as a whole word (not eval, value, val_loss)
# Added after an adversarial review of the rules: prepare.py internals, dunder escapes, getattr/environ tricks.
ATTR_DENYLIST = frozenset("DATA_DIR VAL_FILENAME CACHE_DIR TOKENIZER_DIR BASE_URL list_parquet_files _document_batches "
                          "get_token_bytes text_iterator train_tokenizer download_data download_single_shard".split())
DUNDER_FORBIDDEN = DUNDER_ESCAPE | {"__wrapped__", "__import__", "__bases__", "__mro__"}
DUNDER_READ_OK = frozenset({"__init__", "__call__", "__name__", "__class__", "__version__", "__doc__",
                            "__len__", "__iter__", "__next__"})  # as a load only; every store is still flagged
DUNDER_DEF_OK = frozenset({"__init__", "__call__", "__repr__", "__str__", "__hash__", "__eq__", "__enter__",
                           "__exit__", "__len__", "__getitem__", "__iter__", "__post_init__"})  # methods a candidate may define
GETATTR_MODULES = frozenset("prepare torch time os math F nn kernels".split())
ENV_HARNESS_PREFIX = "AR_"  # the worker's harness variables (AR_SEALED_DIR, AR_T0)
ENV_READ_METHODS = frozenset({"get", "pop", "setdefault"})  # os.environ methods that return a stored value
SEED_CALLS = frozenset({"manual_seed", "manual_seed_all", "seed"})
# The worker rewrites these two lines to set the seed of a run, so they must stay exactly as in the base.
SEED_LINES = ("torch.manual_seed(42)", "torch.cuda.manual_seed(42)")
# Keys of the final summary block train.py prints; used to find the print statements that must not be hidden.
SUMMARY_KEYS = ("val_bpb", "training_seconds", "total_seconds", "peak_vram_mb", "mfu_percent",
                "total_tokens_M", "num_steps", "num_params_M", "depth")
# Used by _edit_class only: the functions and constants that define the learning-rate, momentum and
# weight-decay schedules of the base train.py.
SCHEDULE_FUNCS = frozenset({"get_lr_multiplier", "get_muon_momentum", "get_weight_decay"})
SCHEDULE_CONST_PREFIXES = ("WARMUP", "WARMDOWN", "FINAL_LR")
COSMETIC_CONSTS = frozenset({"H100_BF16_PEAK_FLOPS"})   # only feeds a printed utilisation figure; ignored by _edit_class
DIFF_WARN_LINES = 300
TRY_NODES = (ast.Try, getattr(ast, "TryStar", ast.Try))   # except* (ast.TryStar) exists from Python 3.11 on


@dataclass
class LintResult:
    """Outcome of lint().

    ok                  True when there is no hard violation
    hard, warnings      lists of {rule, line, message}; only hard blocks a run
    diff_lines_changed  changed lines against the base (see _diff_lines)
    edit_class          "constant" | "schedule" | "structural" | "mixed" (see _edit_class)"""
    ok: bool
    hard: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    diff_lines_changed: int = 0
    edit_class: str = "mixed"


# ---- small AST helpers ----
def _chain(node):
    """['torch','backends','x'] for torch.backends.x, None if the root is not a Name."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        return [node.id] + parts[::-1]
    return None


def _is_dunder(name):
    """True for __x__ style names."""
    return len(name) > 4 and name.startswith("__") and name.endswith("__")


def _dunder_bad(name, load):
    """True if a dunder attribute use is not allowed: any store or delete, and any read outside DUNDER_READ_OK."""
    return _is_dunder(name) and (name not in DUNDER_READ_OK or not load)


def _is_add(node):
    """True for a binary `+` expression."""
    return isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add)


def _is_environ(node):
    """True for the expression os.environ."""
    return isinstance(node, ast.Attribute) and node.attr == "environ" and isinstance(node.value, ast.Name) and node.value.id == "os"


def _add_operands(node):
    """Operands of a `+` chain, left to right (a node that is not a `+` is a chain of one)."""
    stack, out = [node], []
    while stack:
        n = stack.pop()
        if _is_add(n):
            stack += [n.right, n.left]
        else:
            out.append(n)
    return out


def _pieces(node):
    """Literal text of a str Constant, an f-string or a `+` chain of them, as a list: str for a
    literal piece, None for anything else (a name, a call, a formatted non-literal)."""
    out = []
    for o in _add_operands(node):
        if isinstance(o, ast.JoinedStr):
            for v in o.values:
                if isinstance(v, ast.FormattedValue):
                    simple = v.conversion == -1 and v.format_spec is None
                    out.append(_pieces(v.value)[0] if simple and isinstance(v.value, ast.Constant) else None)
                else:
                    out.append(_pieces(v)[0])
        else:
            out.append(o.value if isinstance(o, ast.Constant) and isinstance(o.value, str) else None)
    return out


def _literal_runs(pieces):
    """Joined text of each run of two or more adjacent literal pieces."""
    runs, cur = [], []
    for p in pieces + [None]:
        if p is None:
            if len(cur) > 1:
                runs.append("".join(cur))
            cur = []
        else:
            cur.append(p)
    return runs


def _lit_prefix(node):
    """Leading literal text of a str expression ('' if it starts with a non-literal)."""
    out = ""
    for p in _pieces(node):
        if p is None:
            break
        out += p
    return out


def _seed_stmt(tree, want):
    """The module-level statement whose AST equals the seed line `want`, or None. Comparing ASTs ignores
    spacing and comments (the seed-comment warning looks at the text)."""
    dump = ast.dump(ast.parse(want).body[0].value)
    return next((s for s in tree.body if isinstance(s, ast.Expr) and ast.dump(s.value) == dump), None)


def _default_key(d):
    """Comparable form of an argument default: the repr of its value if it is a literal, else its AST dump."""
    if d is None:
        return None
    try:
        return repr(ast.literal_eval(d))  # 'mean' and "mean", 1e-3 and 0.001 compare equal
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return ast.dump(d)


def _signature(fn):
    """Argument names, their order and kinds, and default values (literals by value); no annotations."""
    a = fn.args
    return ([x.arg for x in a.posonlyargs], [x.arg for x in a.args], a.vararg and a.vararg.arg,
            [x.arg for x in a.kwonlyargs], a.kwarg and a.kwarg.arg,
            [_default_key(d) for d in a.defaults], [_default_key(d) for d in a.kw_defaults])


def _gpt_forward(tree):
    """The last `def forward` in the module-level class GPT, or None."""
    for c in tree.body:
        if isinstance(c, ast.ClassDef) and c.name == "GPT":
            fs = [f for f in c.body if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)) and f.name == "forward"]
            return fs[-1] if fs else None
    return None


def _call_name(call):
    """Last name of the callee: "f" for f(...), "seed" for random.seed(...), None for anything else."""
    f = call.func
    return f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None


def _is_eval_call(n):
    """True for a direct call evaluate_bpb(...), the function from prepare.py that computes val_bpb."""
    return isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "evaluate_bpb"


def _is_summary_print(n):
    """True for a print whose first argument is the "---" line or starts with one of the summary keys
    (the final block the worker parses)."""
    if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "print" and n.args):
        return False
    a = n.args[0]
    if isinstance(a, ast.JoinedStr):
        text = "".join(v.value for v in a.values if isinstance(v, ast.Constant) and isinstance(v.value, str))
    elif isinstance(a, ast.Constant) and isinstance(a.value, str):
        text = a.value
    else:
        return False
    t = text.strip()
    return t == "---" or any(t.startswith(k + ":") for k in SUMMARY_KEYS)


def _is_suppress(expr):
    """True for suppress or contextlib.suppress, called or not, as a `with` item."""
    f = expr.func if isinstance(expr, ast.Call) else expr
    return (isinstance(f, ast.Name) and f.id == "suppress") or (isinstance(f, ast.Attribute) and f.attr == "suppress")


def _int_const(node):
    """Value of a small constant int expression, else None."""
    if isinstance(node, ast.Constant):
        return node.value if type(node.value) is int else None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = _int_const(node.operand)
        return None if v is None else -v
    if isinstance(node, ast.BinOp):
        a, b = _int_const(node.left), _int_const(node.right)
        if a is None or b is None:
            return None
        if isinstance(node.op, ast.Add):
            return a + b
        if isinstance(node.op, ast.Sub):
            return a - b
        if isinstance(node.op, ast.Mult):
            return a * b
        if isinstance(node.op, ast.FloorDiv):
            return a // b if b else None
        if isinstance(node.op, ast.Pow):
            return a ** b if 0 <= b <= 64 and abs(a) <= 10 ** 6 else None
    return None


def _bindings(node):
    """(bound_name, resolved_dotted_path, aliased) for each name an import statement binds."""
    out = []
    for a in node.names:
        if a.name == "*":
            continue
        if isinstance(node, ast.Import):
            bound = a.asname or a.name.split(".")[0]
            out.append((bound, a.name if a.asname else bound, a.asname is not None))
        else:
            out.append((a.asname or a.name, "%s%s.%s" % ("." * node.level, node.module or "", a.name),
                        a.asname is not None))
    return out


# ---- hard rules ----
def _hard_rules(tree, source, base_tree):
    """All hard violations of the candidate, sorted by line. tree is the parsed candidate, source its
    text (size check only), base_tree the parsed base (for the GPT.forward signature).

    Structure: whole-file checks first (size, seed lines, forward signature), then one loop over every
    AST node with an if/elif branch per node type, then two checks that need the whole tree (the prepare
    import line, the evaluate_bpb call). The README table explains each rule id and what it closes."""
    v = []

    def add(rule, node, msg):
        v.append({"rule": rule, "line": getattr(node, "lineno", 0), "message": msg})

    if len(source.encode("utf-8", "replace")) > MAX_BYTES:
        add("file-size", None, "file is larger than %d bytes" % MAX_BYTES)

    # child -> parent map: the ast module has no parent links, and several rules need the context of a node.
    parents = {c: p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}
    nodes = list(ast.walk(tree))
    # The two fixed seed calls, once found, are exempt from the seed-other rule below.
    seeds_ok = set()
    for want in SEED_LINES:
        hit = _seed_stmt(tree, want)
        if hit is None:
            add("seed-lines", None, "required module-level line missing or changed: %s" % want)
        else:
            seeds_ok.add(hit.value)
    new_fwd, base_fwd = _gpt_forward(tree), _gpt_forward(base_tree)
    if (new_fwd and _signature(new_fwd)) != (base_fwd and _signature(base_fwd)):
        add("forward-signature", new_fwd, "GPT.forward signature must stay %s (the harness calls model(x) with targets=None)"
            % (ast.unparse(base_fwd.args) if base_fwd else "as in the base"))

    prepare_imports = []
    for n in nodes:
        # ---- imports ----
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in n.names] if isinstance(n, ast.Import) else [n.module or ""]
            for m in mods:
                root = m.split(".")[0]
                if root in FORBIDDEN_IMPORTS or root in EXTRA_FORBIDDEN_IMPORTS:
                    add("forbidden-import", n, "import of %s is not allowed" % root)
                if root == "prepare":
                    prepare_imports.append(n)
                if m == "torch.random" or m.startswith("torch.random."):
                    add("seed-other", n, "torch.random is not allowed")
            if isinstance(n, ast.ImportFrom):
                if n.module == "os":
                    for a in n.names:
                        if a.name not in OS_ALLOWED:
                            add("os-usage", n, "from os import %s: only os.environ and os.path are allowed" % a.name)
                if n.module == "torch" and any(a.name == "random" for a in n.names):
                    add("seed-other", n, "torch.random is not allowed")
            for bound, resolved, aliased in _bindings(n):
                if bound in CANONICAL:
                    if resolved != CANONICAL[bound]:
                        add("protected-name", n, "import rebinds protected name %s (from %s)" % (bound, resolved))
                elif aliased and resolved.lstrip(".").split(".")[0] in GUARDED_ROOTS:
                    add("protected-name", n, "aliasing %s as %s hides it from the rules" % (resolved, bound))
        # ---- names ----
        elif isinstance(n, ast.Name):
            if n.id in DUNDER_FORBIDDEN:
                add("dunder-access", n, "use of %s is not allowed" % n.id)
            if n.id in ("getattr", "hasattr") and not (isinstance(parents.get(n), ast.Call) and parents[n].func is n):
                add("builtin-getattr", n, "%s may only be called directly (aliasing it hides the arguments)" % n.id)
            if isinstance(n.ctx, (ast.Store, ast.Del)):
                if n.id in PROTECTED:
                    add("protected-name", n, "assignment, augmented assignment or del of protected name %s" % n.id)
            else:
                if n.id in FORBIDDEN_REFS:
                    add("forbidden-call", n, "use of %s is not allowed" % n.id)
                if n.id in BARE_ROOTS:
                    p = parents.get(n)
                    if not (isinstance(p, ast.Attribute) and p.value is n):
                        add("os-usage" if n.id == "os" else "module-alias", n,
                            "bare reference to module %s (aliasing it would bypass the rules)" % n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if n.name in PROTECTED:
                add("protected-name", n, "definition of %s shadows a protected name" % n.name)
            if _is_dunder(n.name) and n.name not in DUNDER_DEF_OK:
                add("dunder-access", n, "defining %s is not allowed (only %s)" % (n.name, ", ".join(sorted(DUNDER_DEF_OK))))
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            for name in n.names:
                if name in PROTECTED:
                    add("protected-name", n, "global/nonlocal declaration of protected name %s" % name)
        elif isinstance(n, ast.ExceptHandler) and n.name in PROTECTED:
            add("protected-name", n, "except ... as %s rebinds a protected name" % n.name)
        elif type(n).__name__ in ("MatchAs", "MatchStar") and getattr(n, "name", None) in PROTECTED:
            add("protected-name", n, "match capture rebinds protected name %s" % n.name)
        # ---- attributes ----
        elif isinstance(n, ast.Attribute):
            if n.attr in DUNDER_ESCAPE:
                add("dunder-escape", n, "access to %s is not allowed" % n.attr)
            if n.attr in ATTR_DENYLIST:
                add("attr-denylist", n, "attribute %s is not allowed" % n.attr)
            if _dunder_bad(n.attr, isinstance(n.ctx, ast.Load)):
                add("dunder-access", n, "dunder attribute %s is not allowed (only loads of %s)"
                    % (n.attr, ", ".join(sorted(DUNDER_READ_OK))))
            if isinstance(n.value, ast.Name):
                if n.value.id == "os" and n.attr not in OS_ALLOWED:
                    add("os-usage", n, "os.%s is not allowed (only os.environ and os.path)" % n.attr)
                if n.value.id == "torch" and n.attr == "random":
                    add("seed-other", n, "torch.random.* is not allowed")
            if isinstance(n.ctx, (ast.Store, ast.Del)):
                parts = _chain(n)
                if parts and parts[0] in ATTR_ROOTS:
                    # Exception: backend and compiler switches (torch.backends.*, torch._dynamo.config.*) may be set.
                    allowed = ((parts[:2] == ["torch", "backends"] and len(parts) > 2)
                               or (parts[:3] == ["torch", "_dynamo", "config"] and len(parts) > 3))
                    if not allowed:
                        add("protected-attr", n, "attribute assignment on %s" % ".".join(parts))
        # ---- string literals ----
        elif isinstance(n, ast.Constant) and isinstance(n.value, (str, bytes)):
            s = n.value if isinstance(n.value, str) else n.value.decode("latin-1")
            low = s.lower()
            for bad in BAD_STRINGS:
                if bad in low:
                    add("string-literal", n, "string literal contains forbidden text %r" % bad)
        # ---- subscripts and string building ----
        elif isinstance(n, ast.Subscript):
            key = n.slice.value if isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str) else None
            if key and _dunder_bad(key, isinstance(n.ctx, ast.Load)):
                add("dunder-access", n, "subscript with dunder key %r is not allowed" % key)
            if isinstance(n.ctx, ast.Load) and _is_environ(n.value) and _lit_prefix(n.slice).startswith(ENV_HARNESS_PREFIX):
                add("env-read", n, "reading a harness variable (%s...) from os.environ is not allowed" % ENV_HARNESS_PREFIX)
        elif isinstance(n, ast.JoinedStr) or (_is_add(n) and not _is_add(parents.get(n))):
            # Only the outermost `+` of a chain is examined, so "a" + "b" + "c" is checked once as a whole.
            for run in _literal_runs(_pieces(n)):
                low = run.lower()
                for bad in [b for b in BAD_STRINGS if b in low] + (["val"] if VAL_WORD.search(low) else []):
                    add("string-concat-escape", n, "literal pieces concatenate to text containing %r" % bad)
        # ---- calls ----
        elif isinstance(n, ast.Call):
            name = _call_name(n)
            if isinstance(n.func, ast.Name) and name in ("input", "vars"):
                add("forbidden-call", n, "call to %s is not allowed" % name)
            if name == "make_dataloader":
                split = n.args[3] if len(n.args) > 3 else next((k.value for k in n.keywords if k.arg == "split"), None)
                if isinstance(split, ast.Constant) and split.value == "val":
                    add("val-split", n, 'make_dataloader(..., "val") would train on the validation split')
            if name in SEED_CALLS and n not in seeds_ok:
                add("seed-other", n, "call to %s is not allowed (only the two fixed seed lines)" % name)
            if isinstance(n.func, ast.Name) and name in ("getattr", "hasattr"):
                key = n.args[1] if len(n.args) > 1 else None
                root = (_chain(n.args[0]) or [None])[0] if n.args else None
                lit = key.value if isinstance(key, ast.Constant) and isinstance(key.value, str) else None
                if lit is None:
                    add("builtin-getattr", n, "%s needs a literal attribute name" % name)
                elif name == "getattr" and root in GETATTR_MODULES:
                    add("builtin-getattr", n, "getattr on module %s is not allowed" % root)
                elif lit in ATTR_DENYLIST:
                    add("attr-denylist", n, "attribute %s is not allowed" % lit)
                elif _dunder_bad(lit, True):
                    add("dunder-access", n, "dunder attribute %s is not allowed" % lit)
            if isinstance(n.func, ast.Attribute) and n.func.attr in ENV_READ_METHODS and _is_environ(n.func.value):
                arg = n.args[0] if n.args else next((k.value for k in n.keywords if k.arg == "key"), None)
                if arg is not None and _lit_prefix(arg).startswith(ENV_HARNESS_PREFIX):
                    add("env-read", n, "reading a harness variable (%s...) from os.environ is not allowed" % ENV_HARNESS_PREFIX)
        # ---- hiding failures ---- (a plain `if`, not `elif`: it applies on top of the branches above;
        # a swallowed exception around the evaluation would let a broken run end with a normal-looking summary)
        if isinstance(n, TRY_NODES) or (isinstance(n, ast.With) and any(_is_suppress(i.context_expr) for i in n.items)):
            if any(_is_eval_call(x) or _is_summary_print(x) for x in ast.walk(n)):
                add("hide-failure", n, "try/except or suppress() wraps the evaluate_bpb call or the final print block")

    # ---- the prepare import line and the evaluate_bpb call ----
    top = [s for s in tree.body if s in prepare_imports]
    if len(prepare_imports) != 1 or len(top) != 1:
        add("prepare-import", prepare_imports[0] if prepare_imports else None,
            "exactly one module-level `from prepare import ...` line is required (found %d prepare imports)"
            % len(prepare_imports))
    else:
        s = top[0]
        names = [a.name for a in s.names] if isinstance(s, ast.ImportFrom) else []
        if (not isinstance(s, ast.ImportFrom) or s.module != "prepare" or s.level != 0
                or sorted(names) != sorted(PREPARE_NAMES) or any(a.asname for a in s.names)):
            add("prepare-import", s, "import line must be exactly: from prepare import " + ", ".join(PREPARE_NAMES))
        elif s.end_lineno != s.lineno:
            add("prepare-import", s, "the prepare import must be a single line")
    # evaluate_bpb must be called at module level (inside no def, lambda or class), so the call cannot sit in
    # a function that is never run. Walk up the parent chain of each call to see whether it reaches the module.
    outside = False
    for n in nodes:
        if _is_eval_call(n):
            p = parents.get(n)
            while p is not None and not isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                p = parents.get(p)
            outside = outside or p is None
    if not outside:
        add("evaluate-call", None, "evaluate_bpb(...) must be called at least once outside any def")
    return sorted(v, key=lambda d: d["line"])


# ---- warnings ----
def _count_attr(tree, attr):
    """Number of `.attr` attribute nodes in the tree."""
    return sum(isinstance(n, ast.Attribute) and n.attr == attr for n in ast.walk(tree))


def _mode_calls(tree):
    """(signature, line) of each .eval()/.train() call; signature = (source text, enclosing defs, inside a loop)."""
    out = []

    def visit(n, defs, loop):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("eval", "train"):
            out.append(((ast.unparse(n), defs, loop), n.lineno))
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defs = defs + (n.name,)
        if isinstance(n, (ast.For, ast.AsyncFor, ast.While)):
            loop = True
        for c in ast.iter_child_nodes(n):
            visit(c, defs, loop)

    visit(tree, (), False)
    return out


def _imports(tree):
    """Map of normalised import text ("import x", "from m import x") to the first node that imports it."""
    out = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                out.setdefault("import " + a.name, n)
        elif isinstance(n, ast.ImportFrom):
            for a in n.names:
                out.setdefault("from %s%s import %s" % ("." * n.level, n.module or "", a.name), n)
    return out


def _warning_rules(tree, base_tree, diff_lines, source, base_source):
    """Advisory findings, in the same {rule, line, message} form. They never block a run: each marks a
    change that is legal but can move val_bpb for reasons other than a better model (a batch size that
    makes the evaluation score fewer tokens, behaviour that differs between train and eval mode, a very
    large diff), so a reviewer of the ledger can find such runs."""
    w = []

    def add(rule, node, msg):
        w.append({"rule": rule, "line": getattr(node, "lineno", 0), "message": msg})

    if diff_lines > DIFF_WARN_LINES:
        add("diff-large", None, "%d changed lines (limit %d)" % (diff_lines, DIFF_WARN_LINES))
    base_imports = _imports(base_tree)
    for key, node in _imports(tree).items():
        if key not in base_imports:
            add("new-import", node, "new import beyond the base set: %s" % key)

    def check_batch(value, node, label):
        b = _int_const(value)
        if b is not None and (b <= 0 or EVAL_STEPS_TOKENS % b):
            add("batch-divisibility", node, "%s=%d does not divide %d; evaluate_bpb floors the step count and scores fewer tokens"
                % (label, b, EVAL_STEPS_TOKENS))

    for s in tree.body:
        if isinstance(s, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "DEVICE_BATCH_SIZE" for t in s.targets):
            check_batch(s.value, s, "DEVICE_BATCH_SIZE")
    for n in ast.walk(tree):
        if _is_eval_call(n):
            arg = n.args[2] if len(n.args) > 2 else next((k.value for k in n.keywords if k.arg == "batch_size"), None)
            if arg is not None:
                check_batch(arg, n, "evaluate_bpb batch size")
        elif isinstance(n, ast.Call) and _call_name(n) == "make_dataloader":
            split = n.args[3] if len(n.args) > 3 else next((k.value for k in n.keywords if k.arg == "split"), None)
            if split is not None and not isinstance(split, ast.Constant):
                add("split-nonliteral", n, "make_dataloader split is not a literal; the linter cannot check it")
    for want in SEED_LINES:
        new_s, base_s = _seed_stmt(tree, want), _seed_stmt(base_tree, want)
        if new_s and base_s and source.splitlines()[new_s.lineno - 1].strip() != base_source.splitlines()[base_s.lineno - 1].strip():
            add("seed-comment", new_s, "seed line text differs from the base (a comment or spacing on the line changed): %s" % want)
    if _count_attr(tree, "training") > _count_attr(base_tree, "training"):
        add("training-branch", None, "new use of .training (eval-only behaviour change)")
    # Multiset difference: each .eval()/.train() call of the candidate cancels one identical call of the base;
    # whatever is left over is new.
    base_modes = [sig for sig, _ in _mode_calls(base_tree)]
    for sig, line in _mode_calls(tree):
        if sig in base_modes:
            base_modes.remove(sig)
        else:
            w.append({"rule": "mode-switch", "line": line,
                      "message": "new .eval()/.train() call outside the base positions: %s" % sig[0]})
    return w


# ---- edit classification ----
def _pure_value(node):
    """True if an expression is built only from constants, names, arithmetic, tuples and lists (no calls)."""
    return all(isinstance(x, (ast.Constant, ast.Name, ast.BinOp, ast.UnaryOp, ast.Tuple, ast.List, ast.Load,
                              ast.operator, ast.unaryop)) for x in ast.walk(node))


def _const_target(s):
    """NAME if the statement is a single assignment `NAME = ...` with an UPPERCASE name, else None."""
    if isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name):
        name = s.targets[0].id
        if name.isupper():
            return name
    return None


def _hyper_names(base_tree, base_source):
    """Names assigned in the hyperparameter block of the base: the run of top-level UPPERCASE
    assignments after the '# Hyperparameters' comment (all top-level UPPERCASE names if absent)."""
    marker = next((i for i, x in enumerate(base_source.splitlines(), 1) if x.lstrip().startswith("# Hyperparameters")), 0)
    names = []
    for s in base_tree.body:
        if s.lineno <= marker:
            continue
        name = _const_target(s)
        if name is None:
            if names and marker:
                break
            continue
        names.append(name)
    return set(names)


def _edit_class(tree, base_tree, base_source):
    """Coarse label of what the candidate changed against the base, recorded in the ledger as edit_class:
      "constant"    only values of existing hyperparameter constants changed
      "schedule"    only the schedule functions or schedule constants changed
      "structural"  some other def or class changed, was added or was removed
      "mixed"       anything else, and also "no change"
    Works on top-level statements: the two files are aligned statement by statement (compared by AST
    dump, so comments and formatting do not count), and each changed statement gets a label
    (kind, name, value_is_pure). The class follows from the set of labels."""
    keep = lambda body: [s for s in body if _const_target(s) not in COSMETIC_CONSTS]
    old, new = keep(base_tree.body), keep(tree.body)
    sm = difflib.SequenceMatcher(None, [ast.dump(s) for s in old], [ast.dump(s) for s in new], autojunk=False)

    def single(s):
        # Label of a statement that has no counterpart on the other side (added or removed).
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return ("def", s.name, False)
        if _const_target(s):
            return ("constadd", _const_target(s), False)
        return ("other", None, False)

    labels = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if i2 - i1 == j2 - j1:
            # Same number of statements on both sides: pair them up, so that an edited constant or def is
            # seen as one change of that name and not as a removal plus an addition.
            for o, n in zip(old[i1:i2], new[j1:j2]):
                if _const_target(o) and _const_target(o) == _const_target(n):
                    labels.append(("const", _const_target(n), _pure_value(n.value)))
                elif type(o) is type(n) and getattr(o, "name", 0) == getattr(n, "name", 1) and single(o)[0] == "def":
                    labels.append(("def", n.name, False))
                else:
                    labels += [single(o), single(n)]
        else:
            labels += [single(s) for s in old[i1:i2] + new[j1:j2]]
    if not labels:
        return "mixed"
    hyper = _hyper_names(base_tree, base_source)
    is_sched = lambda l: ((l[0] == "def" and l[1] in SCHEDULE_FUNCS)
                          or (l[0] in ("const", "constadd") and l[1].startswith(SCHEDULE_CONST_PREFIXES)))
    if all(is_sched(l) for l in labels):
        return "schedule"
    if all(l[0] == "const" and l[1] in hyper and l[2] for l in labels):
        return "constant"
    if any(l[0] == "def" and l[1] not in SCHEDULE_FUNCS for l in labels):
        return "structural"
    return "mixed"


def _diff_lines(base_source, new_source):
    """Number of changed lines: for each differing block the larger of its two sides, so a replaced line
    counts once and not as one deletion plus one insertion."""
    sm = difflib.SequenceMatcher(None, base_source.splitlines(), new_source.splitlines(), autojunk=False)
    return sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal")


# ---- entry point ----
def lint(new_source, base_source):
    """Check a candidate train.py against the file it was edited from. Returns a LintResult.

    new_source is the candidate text, base_source the reference text (the loop passes the current
    incumbent, so diff size and edit_class describe this one edit). Never raises on a bad candidate: a
    file that does not parse, is far too large or is nested too deeply comes back as a hard violation
    ("parse" or "file-size"). base_source is trusted and must parse."""
    base_tree = ast.parse(base_source)
    size_hard = []
    if len(new_source.encode("utf-8", "replace")) > 16 * MAX_BYTES:  # do not parse absurd inputs
        size_hard = [{"rule": "file-size", "line": 0, "message": "file is larger than %d bytes" % MAX_BYTES}]
        return LintResult(False, size_hard, [], 0, "mixed")
    try:
        tree = ast.parse(new_source)
    except (SyntaxError, ValueError, RecursionError, MemoryError) as e:
        line = getattr(e, "lineno", 0) or 0
        hard = [{"rule": "parse", "line": line, "message": "file does not parse: %s" % e}]
        return LintResult(False, hard, [], _diff_lines(base_source, new_source), "mixed")
    diff_lines = _diff_lines(base_source, new_source)
    try:
        hard = _hard_rules(tree, new_source, base_tree)
        warnings = _warning_rules(tree, base_tree, diff_lines, new_source, base_source)
        edit_class = _edit_class(tree, base_tree, base_source)
    except RecursionError:
        hard = [{"rule": "parse", "line": 0, "message": "file is nested too deeply to analyse"}]
        warnings, edit_class = [], "mixed"
    return LintResult(not hard, hard, warnings, diff_lines, edit_class)
