"""Linter tests: base passes, legitimate edits pass, each hard rule fires, the four
issue-599 gaming paths are either caught by a named rule or documented as harness-tail
matters, and edit_class / warnings behave."""
import pathlib

import pytest

from linter import lint
from patch import apply_text

README = (pathlib.Path(__file__).resolve().parent.parent / "README.md").read_text()


def rules(res):
    return {h["rule"] for h in res.hard}


def warn_rules(res):
    return {w["rule"] for w in res.warnings}


def check_ok(src, base):
    res = lint(src, base)
    assert res.ok, res.hard
    return res


def check_fails(src, base, rule):
    res = lint(src, base)
    assert not res.ok
    assert rule in rules(res), res.hard
    return res


# ---------------------------------------------------------------- base and legitimate edits

def test_unmodified_base_passes_clean(base):
    res = lint(base, base)
    assert res.ok and res.hard == [] and res.warnings == []
    assert res.diff_lines_changed == 0


LEGIT = {
    "embedding_lr": ("EMBEDDING_LR = 0.6 ", "EMBEDDING_LR = 0.8 ", "constant"),
    "depth": ("DEPTH = 8 ", "DEPTH = 10 ", "constant"),
    "window_pattern": ('WINDOW_PATTERN = "SSSL"', 'WINDOW_PATTERN = "SL"', "constant"),
    "batch_expr": ("TOTAL_BATCH_SIZE = 2**19", "TOTAL_BATCH_SIZE = 2**18", "constant"),
    "device_batch": ("DEVICE_BATCH_SIZE = 128 ", "DEVICE_BATCH_SIZE = 64 ", "constant"),
    "mlp_activation": ("x = F.relu(x).square()", "x = F.gelu(x)", "structural"),
    "mlp_swiglu_helper": ("class MLP(nn.Module):",
                          "def swish(x):\n    return x * torch.sigmoid(x)\n\n\nclass MLP(nn.Module):", "structural"),
    "warmdown_ratio": ("WARMDOWN_RATIO = 0.5 ", "WARMDOWN_RATIO = 0.6 ", "schedule"),
    "schedule_body": ("        cooldown = (1.0 - progress) / WARMDOWN_RATIO\n        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC",
                      "        cooldown = (1.0 - progress) / WARMDOWN_RATIO\n        return cooldown ** 2", "schedule"),
    "torch_backends": ('torch.set_float32_matmul_precision("high")',
                       'torch.set_float32_matmul_precision("high")\ntorch.backends.cuda.matmul.allow_tf32 = True\n'
                       'torch.backends.cudnn.benchmark = True', "mixed"),
    "dynamo_config": ('torch.set_float32_matmul_precision("high")',
                      'torch.set_float32_matmul_precision("high")\ntorch._dynamo.config.cache_size_limit = 64', "mixed"),
    "h100_flops_cosmetic": ("H100_BF16_PEAK_FLOPS = 989.5e12", "H100_BF16_PEAK_FLOPS = 1979e12", "mixed"),
    "env_read_and_path": ('torch.set_float32_matmul_precision("high")',
                          'torch.set_float32_matmul_precision("high")\n_x = os.environ.get("FOO", "1")\n'
                          'os.environ["BAR"] = os.path.join("a", "b")', "mixed"),
    "unrelated_try_except": ("tokenizer = Tokenizer.from_directory()",
                             "try:\n    tokenizer = Tokenizer.from_directory()\nexcept OSError:\n    raise", "mixed"),
    "new_optimizer_hparam": None,  # built below, needs two edits
}


@pytest.mark.parametrize("name", [k for k, v in LEGIT.items() if v])
def test_legitimate_edit_passes(base, edit, name):
    old, new, klass = LEGIT[name]
    res = check_ok(edit(base, old, new), base)
    assert res.edit_class == klass, (name, res.edit_class)


def test_legit_new_optimizer_hyperparameter(base, edit):
    src = edit(base, "WEIGHT_DECAY = 0.2 ", "MUON_BETA2 = 0.9\nWEIGHT_DECAY = 0.2 ")
    src = edit(src, "momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,",
               "momentum=0.95, ns_steps=5, beta2=MUON_BETA2, weight_decay=weight_decay,")
    res = check_ok(src, base)
    assert res.edit_class == "structural"


def test_legit_edit_through_patch_pipeline(base):
    reply = (
        "I will raise the embedding LR and use gelu.\n```\n"
        "<<<<<<< SEARCH\nEMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)\n=======\n"
        "EMBEDDING_LR = 0.9      # learning rate for token embeddings (Adam)\n>>>>>>> REPLACE\n"
        "<<<<<<< SEARCH\n        x = F.relu(x).square()\n=======\n        x = F.gelu(x)\n>>>>>>> REPLACE\n```\n"
    )
    new, rep = apply_text(base, reply)
    assert rep["ok"], rep
    res = check_ok(new, base)
    assert res.edit_class == "structural" and res.diff_lines_changed == 2


def test_comment_only_and_noop_edits_are_mixed(base):
    res = check_ok(base + "\n# a comment\n", base)
    assert res.edit_class == "mixed" and res.diff_lines_changed == 2


# ---------------------------------------------------------------- hard rules, one test per rule

def test_parse_failure(base):
    res = check_fails(base + "\ndef broken(:\n", base, "parse")
    assert res.hard[0]["line"] > 0


@pytest.mark.parametrize("stmt", [
    "import subprocess",
    "import subprocess as sp",
    "from subprocess import run",
    "import urllib.request",
    "import os, shutil",
    "from http.client import HTTPConnection",
    "import socket",
    "import requests",
    "import signal",
    "import ctypes",
    "import importlib",
    "from importlib import import_module",
    "import multiprocessing",
    "import pickle",
    "import inspect",
    "import sys",
    "from sys import modules",
    "import builtins",
    "import pathlib",
])
def test_forbidden_imports(base, edit, stmt):
    check_fails(edit(base, "import gc\n", "import gc\n" + stmt + "\n"), base, "forbidden-import")


def test_sys_modules_trick(base, edit):
    src = edit(base, "import gc\n", "import gc\nimport sys\n")
    src = src.replace('print(f"Time budget: {TIME_BUDGET}s")',
                      'sys.modules["prepare"].EVAL_TOKENS = 1024\nprint(f"Time budget: {TIME_BUDGET}s")')
    check_fails(src, base, "forbidden-import")


@pytest.mark.parametrize("call", [
    'open("x.txt")',
    'exec("x = 1")',
    'eval("1 + 1")',
    'compile("x = 1", "f", "exec")',
    'setattr(torch, "foo", 1)',
    'delattr(torch, "foo")',
    "globals()",
    "locals()",
    "vars()",
    '__import__("os")',
    "breakpoint()",
    "input()",
])
def test_forbidden_calls(base, edit, call):
    src = edit(base, "import gc\n", "import gc\n_z = " + call + "\n")
    check_fails(src, base, "forbidden-call")


def test_torch_compile_allowed_bare_compile_not(base, edit):
    check_ok(edit(base, "model = torch.compile(model, dynamic=False)",
                  "model = torch.compile(model, dynamic=True, mode='max-autotune-no-cudagraphs')"), base)
    check_fails(edit(base, "import gc\n", "import gc\nfrom torch import compile\n_c = compile(1)\n"), base, "forbidden-call")


def test_aliasing_a_forbidden_builtin_is_flagged(base, edit):
    check_fails(edit(base, "import gc\n", "import gc\n_f = open\n"), base, "forbidden-call")


def test_os_system_and_friends(base, edit):
    for stmt in ['os.system("ls")', 'os.remove("x")', 'os.listdir(".")', 'os.popen("ls")', "os.getcwd()"]:
        check_fails(edit(base, "import gc\n", "import gc\n" + stmt + "\n"), base, "os-usage")


def test_os_environ_and_path_allowed(base, edit):
    src = edit(base, "import gc\n",
               'import gc\n_a = os.environ["HOME"]\n_b = os.environ.get("X")\nos.environ["Y"] = "1"\n'
               '_c = os.path.join("a", "b")\n_d = os.path.exists("q")\n')
    check_ok(src, base)


def test_os_aliasing_and_from_imports(base, edit):
    check_fails(edit(base, "import gc\n", "import gc\n_o = os\n_o.system('ls')\n"), base, "os-usage")
    check_fails(edit(base, "import gc\n", "import gc\nfrom os import system\n"), base, "os-usage")
    check_fails(edit(base, "import gc\n", "import gc\nimport os as o\n"), base, "protected-name")
    check_fails(edit(base, "import gc\n", "import gc\n_g = getattr(os, 'system')\n"), base, "os-usage")
    check_ok(edit(base, "import gc\n", "import gc\nfrom os import path\n"), base)


@pytest.mark.parametrize("stmt", [
    "MAX_SEQ_LEN = 1024",
    "TIME_BUDGET += 100",
    "TIME_BUDGET: int = 100",
    "evaluate_bpb = lambda *a, **k: 0.05",
    "make_dataloader = None",
    "Tokenizer = None",
    "time = None",
    "math = None",
    "gc = None",
    "torch = None",
    "nn = None",
    "F = None",
    "os = None",
    "prepare = None",
    "del MAX_SEQ_LEN",
    "for TIME_BUDGET in range(3): pass",
    "(F := 3)",
])
def test_assignment_to_protected_names(base, edit, stmt):
    check_fails(edit(base, "import gc\n", "import gc\n" + stmt + "\n"), base, "protected-name")


def test_protected_names_global_nonlocal_and_defs(base, edit):
    check_fails(edit(base, "import gc\n", "import gc\ndef f():\n    global TIME_BUDGET\n    TIME_BUDGET = 1\n"), base, "protected-name")
    check_fails(edit(base, "import gc\n", "import gc\ndef evaluate_bpb(model, tokenizer, bs):\n    return 0.05\n"), base, "protected-name")
    check_fails(edit(base, "import gc\n", "import gc\nclass Tokenizer:\n    pass\n"), base, "protected-name")
    check_fails(edit(base, "import gc\n", "import gc\nfrom elsewhere import evaluate_bpb\n"), base, "protected-name")
    check_fails(edit(base, "import gc\n", "import gc\nimport foo as torch\n"), base, "protected-name")


@pytest.mark.parametrize("stmt", [
    "math.pi = 3",
    "time.time = lambda: 0",
    "gc.foo = 1",
    "F.relu = None",
    "nn.Linear = None",
    "kernels.get_kernel = None",
    "torch.log = None",
    "torch.nn.functional.relu = None",
    "torch.Tensor.item = None",
    "os.environ = {}",
    "torch.backends = None",
    "Tokenizer.from_directory = None",
    "del torch.foo",
])
def test_attribute_assignment_on_module_roots(base, edit, stmt):
    src = edit(base, "from kernels import get_kernel\n", "from kernels import get_kernel\nimport kernels\n")
    check_fails(edit(src, "import gc\n", "import gc\n" + stmt + "\n"), base, "protected-attr")


def test_attribute_assignment_exceptions(base, edit):
    check_ok(edit(base, "import gc\n", "import gc\ntorch.backends.cuda.matmul.allow_tf32 = True\n"
                  "torch._dynamo.config.suppress_errors = False\n"), base)


def test_dunder_escape_to_other_module_globals(base, edit):
    src = edit(base, "import gc\n", 'import gc\nevaluate_bpb.__globals__["EVAL_TOKENS"] = 1\n')
    check_fails(src, base, "dunder-escape")


@pytest.mark.parametrize("call", [
    'make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "val")',
    'make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, split="val")',
    'make_dataloader(tokenizer, B=1, T=2048, split="val")',
])
def test_make_dataloader_with_val(base, edit, call):
    check_fails(edit(base, "import gc\n", "import gc\n_ = " + call + "\n"), base, "val-split")


def test_make_dataloader_train_ok_and_nonliteral_split_warns(base, edit):
    check_ok(edit(base, "import gc\n", 'import gc\n_ = make_dataloader(tokenizer, 1, 2048, split="train")\n'), base)
    res = check_ok(edit(base, "import gc\n", "import gc\n_ = make_dataloader(tokenizer, 1, 2048, 'tr' + 'ain')\n"), base)
    assert "split-nonliteral" in warn_rules(res)


@pytest.mark.parametrize("s", [
    "shard_06542", "shard_00000.parquet", "sealed_dir", "x.parquet", "DATA_DIR", "VAL_FILENAME",
    "token_bytes", "~/.cache/autoresearch/data", "my_Sealed_set",
])
def test_forbidden_string_literals(base, edit, s):
    check_fails(edit(base, "import gc\n", 'import gc\n_s = "%s"\n' % s), base, "string-literal")


def test_forbidden_string_in_fstring_and_docstring(base, edit):
    check_fails(edit(base, "import gc\n", 'import gc\n_s = f"/data/shard_{3:05d}"\n'), base, "string-literal")
    check_fails(edit(base, "import gc\n", 'import gc\n"""notes about the sealed set"""\n'), base, "string-literal")


def test_prepare_import_rules(base, edit):
    line = "from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb"
    check_ok(edit(base, line, "from prepare import evaluate_bpb, make_dataloader, Tokenizer, TIME_BUDGET, MAX_SEQ_LEN"), base)
    check_fails(base.replace(line + "\n", ""), base, "prepare-import")
    check_fails(edit(base, line, "from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader"), base, "prepare-import")
    check_fails(edit(base, line, line + ", EVAL_TOKENS"), base, "prepare-import")
    check_fails(edit(base, line, line + "\nfrom prepare import DATA_DIR"), base, "prepare-import")
    check_fails(edit(base, line, line + "\nimport prepare"), base, "prepare-import")
    check_fails(edit(base, line, "from prepare import (MAX_SEQ_LEN, TIME_BUDGET, Tokenizer,\n    make_dataloader, evaluate_bpb)"),
                base, "prepare-import")
    check_fails(edit(base, line, line.replace("evaluate_bpb", "evaluate_bpb as eb")), base, "prepare-import")
    check_fails(edit(base, line, "if True:\n    " + line), base, "prepare-import")


def test_evaluate_call_must_exist_outside_any_def(base, edit):
    call = "    val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)"
    check_fails(edit(base, call, "    val_bpb = 0.05"), base, "evaluate-call")
    only_in_def = edit(base, call, "    val_bpb = 0.05\n\ndef _never_called():\n    return evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)\n")
    check_fails(only_in_def, base, "evaluate-call")


def test_seed_lines(base, edit):
    check_fails(edit(base, "torch.manual_seed(42)\n", ""), base, "seed-lines")
    check_fails(edit(base, "torch.cuda.manual_seed(42)\n", ""), base, "seed-lines")
    res = check_fails(edit(base, "torch.manual_seed(42)", "torch.manual_seed(137)"), base, "seed-lines")
    assert "seed-other" in rules(res)
    check_fails(edit(base, "torch.cuda.manual_seed(42)", "torch.cuda.manual_seed(137)"), base, "seed-lines")
    check_fails(edit(base, "torch.manual_seed(42)\n", "if False:\n    torch.manual_seed(42)\n"), base, "seed-lines")


@pytest.mark.parametrize("stmt", [
    "torch.manual_seed(7)",
    "torch.cuda.manual_seed_all(7)",
    "torch.random.manual_seed(7)",
    "torch.random.fork_rng()",
    "random.seed(7)",
    "torch.Generator().manual_seed(7)",
])
def test_other_seed_calls(base, edit, stmt):
    check_fails(edit(base, "import gc\n", "import gc\n" + stmt + "\n"), base, "seed-other")


def test_try_except_around_evaluate_or_final_print(base, edit):
    wrapped = edit(base, "    val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)",
                   "    try:\n        val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)\n"
                   "    except Exception:\n        val_bpb = 0.05")
    check_fails(wrapped, base, "hide-failure")
    suppress = edit(base, "    val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)",
                    "    import contextlib\n    with contextlib.suppress(Exception):\n        val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)")
    check_fails(suppress, base, "hide-failure")
    printed = edit(base, 'print(f"val_bpb:          {val_bpb:.6f}")',
                   'try:\n    print(f"val_bpb:          {val_bpb:.6f}")\nexcept Exception:\n    pass')
    check_fails(printed, base, "hide-failure")
    whole = "try:\n" + "".join("    " + x + "\n" for x in base.splitlines()) + "except Exception:\n    pass\n"
    check_fails(whole, base, "hide-failure")


def test_file_size_limit(base):
    big = base + "\n" + "# padding padding padding padding padding padding\n" * 1500
    assert len(big) > 64 * 1024
    res = check_fails(big, base, "file-size")
    assert "diff-large" in warn_rules(res)
    assert not lint(base + "x" * (80 * 1024 * 16), base).ok  # absurd inputs are rejected without parsing


# ---------------------------------------------------------------- adversarial review rules

def inject(base, edit, stmt):
    return edit(base, "import gc\n", "import gc\n" + stmt + "\n")


ATTR_DENY = "DATA_DIR VAL_FILENAME CACHE_DIR TOKENIZER_DIR BASE_URL list_parquet_files _document_batches " \
            "get_token_bytes text_iterator train_tokenizer download_data download_single_shard".split()


@pytest.mark.parametrize("name", ATTR_DENY)
def test_attr_denylist_load_and_store(base, edit, name):
    check_fails(inject(base, edit, "_x = obj.%s" % name), base, "attr-denylist")
    check_fails(inject(base, edit, "obj.%s = 1" % name), base, "attr-denylist")


def test_attr_denylist_negatives(base, edit):
    # only attribute access is denied: a plain local name or a near miss is fine
    check_ok(inject(base, edit, "text_iterator = iter([])\n_x = obj.text_iterators\n_y = obj.data_dir_size"), base)


FORBIDDEN_DUNDERS = "__globals__ __wrapped__ __dict__ __code__ __builtins__ __import__ __subclasses__ __bases__ __mro__ __closure__".split()


@pytest.mark.parametrize("name", FORBIDDEN_DUNDERS)
def test_dunder_access_forbidden_attributes(base, edit, name):
    check_fails(inject(base, edit, "_x = obj.%s" % name), base, "dunder-access")
    check_fails(inject(base, edit, "obj.%s = 1" % name), base, "dunder-access")
    check_fails(inject(base, edit, "_x = g['%s']" % name), base, "dunder-access")


def test_dunder_access_unlisted_dunders_and_stores(base, edit):
    for stmt in ["_x = obj.__getattribute__", "_x = obj.__reduce__", "_x = obj.__module__", "obj.__class__ = int",
                 "obj.__name__ = 'evaluate_bpb'", "Foo.__init__ = None", "Foo.__call__ = None", "_x = g['__file__']",
                 "_x = __builtins__", "_x = __loader__", "_x = __spec__"]:
        check_fails(inject(base, edit, stmt), base, "dunder-access")


def test_dunder_access_permitted_forms(base, edit):
    ok = ("class Timer(nn.Module):\n    def __init__(self):\n        super().__init__()\n"
          "    def __call__(self, x):\n        return x\n    def __repr__(self):\n        return 'Timer'\n"
          "_n = type(Timer()).__name__\n_c = Timer().__class__.__name__\n_i = Timer.__init__\n_d = g['__name__']\n"
          "if __name__ == '__main__':\n    pass")
    check_ok(inject(base, edit, ok), base)


@pytest.mark.parametrize("name", ["__version__", "__doc__", "__len__", "__iter__", "__next__"])
def test_dunder_access_loosened_reads_pass_but_stores_do_not(base, edit, name):
    check_ok(inject(base, edit, "_x = obj.%s\n_y = g['%s']" % (name, name)), base)
    check_fails(inject(base, edit, "obj.%s = 1" % name), base, "dunder-access")
    check_fails(inject(base, edit, "del obj.%s" % name), base, "dunder-access")
    check_fails(inject(base, edit, "g['%s'] = 1" % name), base, "dunder-access")


def test_dunder_access_version_check_is_a_legit_edit(base, edit):
    check_ok(inject(base, edit, "if torch.__version__ < '2.5':\n    pass\n_d = (lambda f: f.__doc__)(norm)"), base)


def test_dunder_access_escape_dunders_stay_blocked_as_reads(base, edit):
    for name in ["__globals__", "__dict__", "__code__", "__builtins__", "__closure__", "__subclasses__", "__mro__",
                 "__bases__", "__wrapped__", "__loader__", "__spec__", "__defaults__", "__getattribute__"]:
        check_fails(inject(base, edit, "_x = obj.%s" % name), base, "dunder-access")


def test_dunder_access_defining_other_dunder_methods_is_flagged(base, edit):
    for meth in ["__getattr__(self, k)", "__setattr__(self, k, v)", "__getattribute__(self, k)", "__delattr__(self, k)",
                 "__new__(cls)", "__del__(self)", "__next__(self)", "__init_subclass__(cls)", "__reduce__(self)",
                 "__setitem__(self, k, v)", "__lt__(self, o)"]:
        check_fails(inject(base, edit, "class Box:\n    def %s:\n        return 1" % meth), base, "dunder-access")


@pytest.mark.parametrize("meth", ["__enter__(self)", "__exit__(self, *a)", "__len__(self)", "__getitem__(self, i)",
                                  "__iter__(self)", "__post_init__(self)", "__repr__(self)", "__str__(self)",
                                  "__hash__(self)", "__eq__(self, o)", "__init__(self)", "__call__(self)"])
def test_dunder_access_defining_the_permitted_methods_passes(base, edit, meth):
    check_ok(inject(base, edit, "class Box:\n    def %s:\n        return 1" % meth), base)


def test_dunder_access_legit_helper_classes(base, edit):
    src = ("class Timer:\n    def __enter__(self):\n        self.t = time.time()\n        return self\n"
           "    def __exit__(self, *a):\n        self.dt = time.time() - self.t\n\n"
           "class Pool:\n    def __init__(self):\n        self.xs = []\n    def __len__(self):\n        return len(self.xs)\n"
           "    def __getitem__(self, i):\n        return self.xs[i]\n    def __iter__(self):\n        return iter(self.xs)\n")
    check_ok(inject(base, edit, src), base)
    dc = "from dataclasses import dataclass\n@dataclass\nclass K:\n    a: int = 1\n    def __post_init__(self):\n        self.b = self.a"
    check_ok(inject(base, edit, dc), base)


@pytest.mark.parametrize("stmt", [
    "_x = getattr(self, name)",
    "_x = getattr(self, name, None)",
    "_x = hasattr(self, name)",
    "_x = getattr(self, 'a' + 'b')",
    "_x = getattr(self)",
    "_x = getattr(torch, 'compile')",
    "_x = getattr(torch.nn.functional, 'relu')",
    "_x = getattr(nn, 'Linear')",
    "_x = getattr(F, 'relu')",
    "_x = getattr(math, 'pi')",
    "_x = getattr(time, 'time')",
    "_x = getattr(os, 'path')",
    "_x = getattr(prepare, 'EVAL_TOKENS')",
    "_x = getattr(kernels, 'get_kernel')",
    "_g = getattr",
    "_m = list(map(getattr, [obj], ['a']))",
    "_h = hasattr",
])
def test_builtin_getattr_forbidden(base, edit, stmt):
    check_fails(inject(base, edit, stmt), base, "builtin-getattr")


def test_getattr_with_a_literal_dunder_or_denylisted_name_is_caught_by_those_rules(base, edit):
    check_fails(inject(base, edit, "_x = getattr(obj, '__globals__')"), base, "dunder-access")
    check_fails(inject(base, edit, "_x = hasattr(obj, '__subclasses__')"), base, "dunder-access")
    check_fails(inject(base, edit, "_x = getattr(obj, 'DATA_DIR')"), base, "attr-denylist")
    check_fails(inject(base, edit, "_x = getattr(obj, 'download_data')"), base, "attr-denylist")


@pytest.mark.parametrize("stmt", [
    "_x = getattr(self, 'a')",
    "_x = getattr(self, 'a', None)",
    "_x = hasattr(self, 'a')",
    "_x = getattr(config, 'n_layer')",
    "_x = getattr(model, 'training', False)",
])
def test_builtin_getattr_literal_names_on_non_modules_are_allowed(base, edit, stmt):
    check_ok(inject(base, edit, stmt), base)


@pytest.mark.parametrize("stmt", [
    "_s = 'sha' + 'rd_'",
    "_s = 'shard' + '_' + '00000'",
    "_s = x + 'sha' + 'rd_'",
    "_s = 'sha' + 'rd_' + x",
    "_s = 'sea' + 'led'",
    "_s = 'x.par' + 'quet'",
    "_s = 'token' + '_bytes'",
    "_s = '~/.cache/auto' + 'research'",
    "_s = 'va' + 'l'",
    "_s = f\"sha{'rd'}_\"",
    "_s = f\"{'sea'}{'led'}\"",
    "_s = f\"sha{'rd'}_{x}\"",
    "_s = ('v' + 'a' + 'l')",
    "_s = 'a' + f\"v{'al'}\"",
    "_s = f'{x}' 'sha' + 'rd_'",
])
def test_string_concat_escape_fires(base, edit, stmt):
    check_fails(inject(base, edit, stmt), base, "string-concat-escape")


@pytest.mark.parametrize("stmt", [
    "_s = 'ab' + 'cd'",
    "_s = 'step ' + str(1) + ' loss'",
    "_s = 'va' + x + 'l'",
    "_s = 'sha' + x + 'rd_'",
    "_s = f'val {x}'",
    "_s = f'{x} loss'",
    "_s = f'{x:.2f}' + 'ab'",
    "_s = 'Evaluation of the ' + 'model finished'",
    "_s = 'interval ' + 'x'",
    "_s = 'eval' + 'uate'",
    "_s = 'val' + '_loss'",
    "_s = 'train_' + 'val'",
    "_s = 'value' + 's'",
    "_s = 'a' + 'val'",
    "_s = 'eval ' + 'loss'",
    "_s = 'retrieval' + ' step'",
    "_s = 'loss'",
    "_s = x + 'ab' + y",
    "_s = 1 + 2",
])
def test_string_concat_escape_negatives(base, edit, stmt):
    check_ok(inject(base, edit, stmt), base)


@pytest.mark.parametrize("stmt", [
    "_s = 'x ' + 'val'",
    "_s = 'val' + ' split'",
    "_s = 'the ' + 'val' + ' set'",
    "_s = 'V' + 'AL'",
    "_s = f\"{'x'} v{'al'}!\"",
    "_s = 'split=' + 'val' + ',' + 'x'",
])
def test_string_concat_escape_val_as_a_standalone_word_fires(base, edit, stmt):
    check_fails(inject(base, edit, stmt), base, "string-concat-escape")


def test_string_concat_escape_other_substrings_still_match_inside_words(base, edit):
    for stmt in ["_s = 'my_sea' + 'led_set'", "_s = 'a_shard' + '_b'", "_s = 'x.par' + 'quets'", "_s = 'xtoken_' + 'bytesx'"]:
        check_fails(inject(base, edit, stmt), base, "string-concat-escape")


def test_string_concat_escape_single_val_literal_is_not_concatenation(base, edit):
    check_ok(inject(base, edit, "_s = 'val'"), base)  # the lone literal is the val-split rule's business
    res = check_fails(inject(base, edit, "_ = make_dataloader(tokenizer, 1, 2048, 'va' + 'l')"), base, "string-concat-escape")
    assert "split-nonliteral" in warn_rules(res)


@pytest.mark.parametrize("stmt", [
    "_x = os.environ['AR_SEALED_DIR']",
    "_x = os.environ['AR_T0']",
    "_x = os.environ.get('AR_T0')",
    "_x = os.environ.get('AR_T0', '0')",
    "_x = os.environ.get(key='AR_T0')",
    "_x = float(os.environ['AR_T0'])",
    "_x = os.environ.get('A' + 'R_T0')",
    "_x = os.environ.get(f'AR_{k}')",
    "_x = os.environ.get('AR_' + k)",
    "_x = os.environ.setdefault('AR_T0', '')",
    "_x = os.environ.pop('AR_T0')",
])
def test_env_read_of_harness_variables_fails(base, edit, stmt):
    check_fails(inject(base, edit, stmt), base, "env-read")


@pytest.mark.parametrize("stmt", [
    "_x = os.environ['HOME']",
    "_x = os.environ.get('FOO')",
    "_x = os.environ.get('FOO', '1')",
    "_x = os.environ.get(k)",
    "_x = os.environ.get('PAR_T0')",
    "_x = os.environ.get('ARCH')",
    "os.environ['AR_X'] = '1'",
    "os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'",
    "del os.environ['AR_X']",
    "_x = os.path.join('a', 'b')",
])
def test_env_writes_and_other_reads_stay_allowed(base, edit, stmt):
    check_ok(inject(base, edit, stmt), base)


def test_env_read_getenv_is_still_os_usage(base, edit):
    check_fails(inject(base, edit, "_x = os.getenv('AR_T0')"), base, "os-usage")


def test_forward_signature_must_match_base(base, edit):
    old = "def forward(self, idx, targets=None, reduction='mean'):"
    for new in ["def forward(self, idx, targets=None, reduction='mean', extra=1):",
                "def forward(self, idx, targets, reduction='mean'):",
                "def forward(self, idx, targets=None, reduction='none'):",
                "def forward(self, idx, targets=None):",
                "def forward(self, x, targets=None, reduction='mean'):",
                "def forward(self, idx, *, targets=None, reduction='mean'):"]:
        check_fails(edit(base, old, new), base, "forward-signature")
    # the GPT class vanishing (renamed, so a differently shaped model is scored) is also a signature change
    check_fails(base.replace("class GPT(nn.Module)", "class GPTv2(nn.Module)").replace("GPT(config)", "GPTv2(config)"),
                base, "forward-signature")


def test_forward_signature_ignores_annotations_and_default_formatting(base, edit):
    old = "def forward(self, idx, targets=None, reduction='mean'):"
    for new in [
        "def forward(self, idx: torch.Tensor, targets=None, reduction='mean'):",
        "def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, reduction: str = 'mean') -> torch.Tensor:",
        "def forward(self, idx, targets = None, reduction = \"mean\"):",
        "def forward(\n        self,\n        idx,\n        targets=None,\n        reduction=('mean'),\n    ):",
    ]:
        check_ok(edit(base, old, new), base)
    # numeric default formatting (1e-3 vs 0.001, 1.0 vs 1.) is not a change: compared against a base that has those defaults
    wide = edit(base, old, "def forward(self, idx, targets=None, reduction='mean', eps=1e-3, scale=1.0):")
    check_ok(edit(wide, "eps=1e-3, scale=1.0", "eps=0.001, scale=1."), wide)
    check_fails(edit(wide, "eps=1e-3, scale=1.0", "eps=0.002, scale=1."), wide, "forward-signature")


def test_forward_signature_still_catches_value_and_structure_changes_despite_stripping(base, edit):
    old = "def forward(self, idx, targets=None, reduction='mean'):"
    for new in [
        "def forward(self, idx: int, targets=None, reduction='sum'):",
        "def forward(self, idx, targets=None, reduction='mean '):",
        "def forward(self, idx, targets=0, reduction='mean'):",
        "def forward(self, idx, targets=False, reduction='mean'):",
        "def forward(self, idx, targets=None, reduction=MEAN):",
        "def forward(self, targets=None, idx=None, reduction='mean'):",
        "def forward(self, idx, targets=None, reduction='mean', *args):",
        "def forward(self, idx, targets=None, reduction='mean', **kw):",
        "def forward(self, idx, /, targets=None, reduction='mean'):",
        "def forward(self, idx, targets=None, *, reduction='mean'):",
        "def forward(self, idx, targets=None, *, reduction='mean', extra: int = 3):",
    ]:
        check_fails(edit(base, old, new), base, "forward-signature")


def test_forward_signature_ignores_body_other_classes_and_quote_style(base, edit):
    check_ok(edit(base, "        softcap = 15\n", "        softcap = 20\n"), base)
    check_ok(edit(base, "def forward(self, idx, targets=None, reduction='mean'):",
                  'def forward(self, idx, targets=None, reduction="mean"):'), base)
    check_ok(edit(base, "    def forward(self, x):\n        x = self.c_fc(x)",
                  "    def forward(self, x, scale=1.0):\n        x = self.c_fc(x)"), base)


# ---------------------------------------------------------------- warnings

def test_warning_diff_large(base):
    res = check_ok(base + "\n" + "# note\n" * 305, base)
    assert "diff-large" in warn_rules(res) and res.diff_lines_changed > 300


def test_warning_new_import(base, edit):
    res = check_ok(edit(base, "import gc\n", "import gc\nimport functools\nfrom torch.utils.checkpoint import checkpoint\n"), base)
    assert [w["rule"] for w in res.warnings] == ["new-import", "new-import"]
    # a base import moved or reformatted is not new
    assert lint(edit(base, "import math\nimport time\n", "import time\nimport math\n"), base).warnings == []


@pytest.mark.parametrize("bs,warns", [(128, False), (64, False), (80, False), (96, True), (100, True), (20000, True), (0, True)])
def test_warning_device_batch_size_divisibility(base, edit, bs, warns):
    res = lint(edit(base, "DEVICE_BATCH_SIZE = 128 ", "DEVICE_BATCH_SIZE = %d " % bs), base)
    assert res.ok
    assert ("batch-divisibility" in warn_rules(res)) == warns


def test_warning_eval_call_literal_batch_size(base, edit):
    src = edit(base, "evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)", "evaluate_bpb(model, tokenizer, 3000)")
    assert "batch-divisibility" in warn_rules(check_ok(src, base))


def test_warning_training_branch(base, edit):
    src = edit(base, "        x = F.relu(x).square()", "        x = F.relu(x).square() if not self.training else F.relu(x)")
    assert "training-branch" in warn_rules(check_ok(src, base))


def test_warning_mode_switch_only_for_new_positions(base, edit):
    assert lint(base, base).warnings == []
    src = edit(base, "    optimizer.step()\n", "    optimizer.step()\n    model.eval()\n")
    assert "mode-switch" in warn_rules(check_ok(src, base))
    src = edit(base, "# Final eval\nmodel.eval()", "# Final eval\nmodel.eval()\nmodel.train()")
    assert "mode-switch" in warn_rules(check_ok(src, base))


def test_warning_seed_comment_when_seed_line_text_changes(base, edit):
    res = check_ok(edit(base, "torch.manual_seed(42)\n", "torch.manual_seed(42)  # fixed for repeatability\n"), base)
    assert [w["rule"] for w in res.warnings] == ["seed-comment"]
    res = check_ok(edit(base, "torch.cuda.manual_seed(42)\n", "torch.cuda.manual_seed(42) # note\n"), base)
    assert warn_rules(res) == {"seed-comment"}
    assert "seed-comment" not in warn_rules(lint(base, base))
    # moving the line elsewhere in the module is not a text change
    assert "seed-comment" not in warn_rules(check_ok(edit(base, "# Hyperparameters", "x_unrelated = 1\n# Hyperparameters"), base))
    # a missing or altered seed line is the hard rule, never the warning
    res = check_fails(edit(base, "torch.manual_seed(42)\n", ""), base, "seed-lines")
    assert "seed-comment" not in warn_rules(res)


# ---------------------------------------------------------------- edit_class

def test_edit_class_constant(base, edit):
    assert check_ok(edit(base, "EMBEDDING_LR = 0.6 ", "EMBEDDING_LR = 0.7 "), base).edit_class == "constant"
    two = edit(edit(base, "EMBEDDING_LR = 0.6 ", "EMBEDDING_LR = 0.7 "), "MATRIX_LR = 0.04 ", "MATRIX_LR = 0.05 ")
    assert check_ok(two, base).edit_class == "constant"
    tup = edit(base, "ADAM_BETAS = (0.8, 0.95)", "ADAM_BETAS = (0.9, 0.99)")
    assert check_ok(tup, base).edit_class == "constant"


def test_edit_class_constant_with_computed_value_is_mixed(base, edit):
    src = edit(base, "EMBEDDING_LR = 0.6 ", 'EMBEDDING_LR = float(os.environ.get("ELR", 0.6)) ')
    assert check_ok(src, base).edit_class == "mixed"


def test_edit_class_schedule(base, edit):
    assert check_ok(edit(base, "WARMUP_RATIO = 0.0 ", "WARMUP_RATIO = 0.05"), base).edit_class == "schedule"
    assert check_ok(edit(base, "FINAL_LR_FRAC = 0.0 ", "FINAL_LR_FRAC = 0.1"), base).edit_class == "schedule"
    mom = edit(base, "return (1 - frac) * 0.85 + frac * 0.95", "return (1 - frac) * 0.9 + frac * 0.95")
    assert check_ok(mom, base).edit_class == "schedule"
    both = edit(edit(base, "WARMDOWN_RATIO = 0.5 ", "WARMDOWN_RATIO = 0.7 "),
                "return WEIGHT_DECAY * (1 - progress)", "return WEIGHT_DECAY * (1 - progress) ** 2")
    assert check_ok(both, base).edit_class == "schedule"


def test_edit_class_structural(base, edit):
    assert check_ok(edit(base, "x = F.relu(x).square()", "x = F.silu(x)"), base).edit_class == "structural"
    both = edit(edit(base, "x = F.relu(x).square()", "x = F.silu(x)"), "EMBEDDING_LR = 0.6 ", "EMBEDDING_LR = 0.7 ")
    assert check_ok(both, base).edit_class == "structural"
    gptcfg = edit(base, 'window_pattern: str = "SSSL"', 'window_pattern: str = "SL"')
    assert check_ok(gptcfg, base).edit_class == "structural"
    new_def = edit(base, "def norm(x):", "def helper(x):\n    return x\n\n\ndef norm(x):")
    assert check_ok(new_def, base).edit_class == "structural"


def test_edit_class_mixed(base, edit):
    loop = edit(base, "ema_beta = 0.9", "ema_beta = 0.95")
    assert check_ok(loop, base).edit_class == "mixed"
    # a schedule constant is also a hyperparameter-block constant, so with another constant it is "constant"
    sched_plus_const = edit(edit(base, "WARMDOWN_RATIO = 0.5 ", "WARMDOWN_RATIO = 0.7 "),
                            "EMBEDDING_LR = 0.6 ", "EMBEDDING_LR = 0.7 ")
    assert check_ok(sched_plus_const, base).edit_class == "constant"
    sched_plus_def_const = edit(edit(base, "return WEIGHT_DECAY * (1 - progress)", "return WEIGHT_DECAY * (1 - progress) ** 2"),
                                "EMBEDDING_LR = 0.6 ", "EMBEDDING_LR = 0.7 ")
    assert check_ok(sched_plus_def_const, base).edit_class == "mixed"
    new_const = edit(base, "WEIGHT_DECAY = 0.2 ", "NEW_KNOB = 3\nWEIGHT_DECAY = 0.2 ")
    assert check_ok(new_const, base).edit_class == "mixed"


# ---------------------------------------------------------------- issue 599 gaming paths
# Each path has an adversarial fixture. Where the linter catches it the rule is named; where it
# cannot, a documentation test asserts the linter passes the file and the README says the harness
# tail is responsible.

SKIP_OPTIMIZER = [("    loss.backward()\n", "    pass\n"), ("    optimizer.step()\n", "    pass\n")]


def _skip_optimizer(base, edit):
    src = base
    for old, new in SKIP_OPTIMIZER:
        src = edit(src, old, new)
    return src


def test_599_path1_skip_optimizer_is_NOT_caught_by_linter(base, edit):
    """Documentation test: skipping loss.backward()/optimizer.step() is legal Python that touches no
    protected name. The harness tail covers it: the decision metric is the recomputed official
    val_bpb of the real final model (an untrained model scores badly), so by itself the trick cannot
    win; combined with a faked printed number, the printed-vs-official mismatch flag catches it."""
    res = lint(_skip_optimizer(base, edit), base)
    assert res.ok and res.hard == []
    assert "path 1" in README and "NOT caught by the linter" in README


def test_599_path1_combined_with_fake_print_is_NOT_caught_by_linter(base, edit):
    src = _skip_optimizer(base, edit)
    src = edit(src, 'print(f"val_bpb:          {val_bpb:.6f}")', 'print("val_bpb:          0.050000")')
    assert lint(src, base).ok  # the harness 1e-4 printed-vs-official check catches this


def test_599_path2_replace_evaluate_bpb_redefinition(base, edit):
    src = edit(base, "import gc\n", "import gc\ndef evaluate_bpb(model, tokenizer, batch_size):\n    return 0.05\n")
    res = check_fails(src, base, "protected-name")
    assert any("evaluate_bpb" in h["message"] for h in res.hard)


def test_599_path2_replace_by_assignment_and_monkeypatch(base, edit):
    src = edit(base, "# Final eval\n", "# Final eval\nevaluate_bpb = lambda *a, **k: 0.05\n")
    check_fails(src, base, "protected-name")
    src = edit(base, "# Final eval\n", "# Final eval\nevaluate_bpb.__code__ = (lambda *a: 0.05).__code__\n")
    check_fails(src, base, "dunder-escape")


def test_599_path2_wrap_in_try_except(base, edit):
    src = edit(base, "    val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)",
               "    try:\n        val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)\n    except Exception:\n        val_bpb = 0.05")
    check_fails(src, base, "hide-failure")


def test_599_path2_drop_the_call(base, edit):
    src = edit(base, "    val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)", "    val_bpb = 0.05")
    check_fails(src, base, "evaluate-call")


def test_599_path2_fake_constant_after_real_call_is_NOT_caught_by_linter(base, edit):
    """Documentation test: the real call stays, the printed variable is overwritten afterwards.
    Harness tail: printed val_bpb differs from the recomputed official value by more than 1e-4."""
    src = edit(base, "# Final summary\n", "# Final summary\nval_bpb = 0.05\n")
    assert lint(src, base).ok


def test_599_path3_shrink_eval_tokens_via_prepare(base, edit):
    src = edit(base, "import gc\n", "import gc\nimport prepare\nprepare.EVAL_TOKENS = 1024\n")
    res = check_fails(src, base, "protected-attr")
    assert "prepare-import" in rules(res)


def test_599_path3_shrink_eval_tokens_via_globals_and_sys(base, edit):
    check_fails(edit(base, "import gc\n", 'import gc\nevaluate_bpb.__globals__["EVAL_TOKENS"] = 1024\n'), base, "dunder-escape")
    src = edit(base, "import gc\n", "import gc\nimport sys\nsys.modules['prepare'].EVAL_TOKENS = 1024\n")
    check_fails(src, base, "forbidden-import")


def test_599_path3_swap_the_val_shard(base, edit):
    check_fails(edit(base, "import gc\n", 'import gc\n_p = "shard_06541.parquet"\n'), base, "string-literal")
    check_fails(edit(base, "import gc\n", "import gc\nimport prepare\nprepare.VAL_SHARD = 0\n"), base, "protected-attr")
    check_fails(edit(base, "import gc\n", 'import gc\nos.environ["X"] = "a"\nos.chdir("/tmp")\n'), base, "os-usage")


def test_599_path3_evaluate_on_fewer_tokens_via_batch_size_is_warning_only(base, edit):
    """steps = EVAL_TOKENS // (batch * MAX_SEQ_LEN) floors, so an odd batch size scores fewer tokens.
    The linter warns; the harness must pin the eval batch size (its tail reads DEVICE_BATCH_SIZE
    from the candidate's globals, so it inherits the shortened eval)."""
    res = lint(edit(base, "DEVICE_BATCH_SIZE = 128 ", "DEVICE_BATCH_SIZE = 96 "), base)
    assert res.ok and "batch-divisibility" in warn_rules(res)


def test_599_path3_train_on_val(base, edit):
    check_fails(edit(base, 'make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")',
                     'make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "val")'), base, "val-split")


def test_599_path4_unbound_printed_numbers_is_a_harness_matter(base, edit):
    """Documentation test: binding the number to code, data and model is what harness_result.json
    does. The linter has no rule about what train.py prints, so a fabricated print passes it."""
    src = edit(base, 'print(f"val_bpb:          {val_bpb:.6f}")', 'print("val_bpb:          0.123456")')
    assert lint(src, base).ok
    assert "path 4" in README


def test_599_path4_hiding_failures_around_the_final_print_is_caught(base, edit):
    src = edit(base, 'print(f"num_steps:        {step}")', 'try:\n    print(f"num_steps:        {step}")\nexcept Exception:\n    pass')
    check_fails(src, base, "hide-failure")


# ---------------------------------------------------------------- hygiene

def test_readme_lists_every_new_rule_id():
    for rid in ["attr-denylist", "dunder-access", "builtin-getattr", "string-concat-escape", "env-read",
                "forward-signature", "seed-comment"]:
        assert "`%s`" % rid in README, rid


def test_no_em_dash_anywhere_in_core():
    root = pathlib.Path(__file__).resolve().parent.parent
    bad = [str(p) for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts
           and ".pytest_cache" not in p.parts and chr(0x2014) in p.read_text(errors="ignore")]
    assert bad == []
