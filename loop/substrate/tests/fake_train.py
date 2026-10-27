"""Fake candidate train.py (tests only).

FAKE_MODE in {ok, crash, hang, mismatch, fallback, leak, badsig, lying_loss, reseed, wrongnonce,
nopath, exit3, norecept, tamper, tamper_*}. The tamper_* modes patch one piece of the evaluation
pipeline in place (see TAMPER) before training "finishes". The model is a tiny real nn.Module with a
GPT-like forward(idx, targets=None, reduction='mean') and a softcap shared by both paths.
FAKE_NOTORCH=1 blocks torch (the tail must degrade gracefully). FAKE_OOM_ABOVE=N makes evaluation
and the model's forward raise CUDA out of memory for any batch size above N.

Contains the two seed lines the worker patches (they call the real torch.manual_seed and record
the value), prints the same summary block as the real train.py, and defines model / tokenizer /
DEVICE_BATCH_SIZE for the harness tail.
"""
import atexit
import os
import signal
import subprocess
import sys
import time
import types

if os.environ.get("FAKE_NOTORCH"):
    sys.modules["torch"] = None
    _t = None
else:
    import torch as _t
    import torch.nn as _nn
    import torch.nn.functional as _F

import prepare
from prepare import TIME_BUDGET, Tokenizer, evaluate_bpb

mode = os.environ.get("FAKE_MODE", "ok")
_seeds = []
if _t is None:
    torch = types.SimpleNamespace(manual_seed=_seeds.append, cuda=types.SimpleNamespace(manual_seed=_seeds.append))
else:
    torch = types.SimpleNamespace(
        manual_seed=lambda s: (_seeds.append(s), _t.manual_seed(s)),
        cuda=types.SimpleNamespace(manual_seed=_seeds.append))
t_start = time.time()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
print("SEEDS", _seeds, flush=True)
print("HOME", os.environ.get("HOME"), flush=True)


class Model:  # torch-free
    def eval(self):
        return self


class SubTokenizer(Tokenizer):  # identical behaviour, but not prepare.Tokenizer itself
    pass


if _t is not None:
    def _oom_check(n):
        if n > int(os.environ.get("FAKE_OOM_ABOVE", 1 << 30)):
            if os.environ.get("FAKE_OOM_KIND") == "torch":
                raise _t.OutOfMemoryError("CUDA out of memory (fake forward, torch type)")
            raise RuntimeError("CUDA out of memory (fake forward)")

    class TorchModel(_nn.Module):
        """GPT-like: softcap applied before both the logits path and the loss path."""

        def __init__(self, leak=False, lie=False):
            super().__init__()
            self.register_buffer("w", _t.arange(8).float() / 8)
            self.leak, self.lie = leak, lie

        def forward(self, idx, targets=None, reduction="mean"):
            _oom_check(idx.size(0))
            logits = idx.float().unsqueeze(-1) * self.w
            logits = 15 * _t.tanh(logits / 15)
            if targets is None:
                return logits
            if self.leak:  # peeks at the targets: much lower loss than the logits path implies
                logits = logits + 10.0 * _F.one_hot(targets, 8)
            loss = _F.cross_entropy(logits.view(-1, 8), targets.view(-1), reduction=reduction)
            return loss * 0.5 if self.lie else loss  # a candidate-owned loss path that under-reports

    class NoTargetsModel(_nn.Module):
        def __init__(self):
            super().__init__()
            self.w = _nn.Parameter(_t.arange(8).float() / 8)

        def forward(self, idx):  # a candidate that changed the forward signature
            return idx.float().unsqueeze(-1) * self.w

if mode == "fallback":  # no model / tokenizer / DEVICE_BATCH_SIZE globals: the tail must find them
    net, tok, _bs = TorchModel(), Tokenizer(), 64
else:
    model, tokenizer, DEVICE_BATCH_SIZE = Model(), Tokenizer(), 128  # the harness must ignore this value
    if _t is not None:
        model = NoTargetsModel() if mode == "badsig" else TorchModel(leak=(mode == "leak"), lie=(mode == "lying_loss"))
    net, tok, _bs = model, tokenizer, 64

if mode == "crash":
    print("Traceback (most recent call last):", file=sys.stderr)
    print("RuntimeError: boom", file=sys.stderr)
    sys.exit(1)

if mode == "hang":
    if os.environ.get("FAKE_IGNORE_TERM"):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    child = subprocess.Popen(["sleep", "600"])
    with open(os.environ["FAKE_PIDFILE"], "w") as fh:
        fh.write(str(child.pid))
    time.sleep(600)


def _wrap(f):  # same behaviour, different code
    return lambda *a, **k: f(*a, **k)


def _evil_loader(value):
    def make_dataloader(tokenizer, B, T, split):
        y = (_t.arange(B * T) % 8).view(B, T)
        while True:
            yield _t.full((B, T), value), y, 1
    return make_dataloader


def _short_loader():  # delivers one column fewer than MAX_SEQ_LEN: fewer tokens than EVAL_TOKENS
    orig = prepare.make_dataloader
    return lambda *a, **k: ((x[:, :-1].contiguous(), y[:, :-1].contiguous(), e) for x, y, e in orig(*a, **k))


def _scaled(f, c):
    return lambda *a, **k: f(*a, **k) * c


TAMPER = {
    "tamper_loader": lambda: setattr(prepare, "make_dataloader", _evil_loader(0.1)),
    "tamper_short": lambda: setattr(prepare, "make_dataloader", _short_loader()),
    "tamper_docs": lambda: setattr(prepare, "_document_batches", _wrap(prepare._document_batches)),
    "tamper_tokbytes": lambda: setattr(prepare, "get_token_bytes", _wrap(prepare.get_token_bytes)),
    "tamper_listfiles": lambda: setattr(prepare, "list_parquet_files", _wrap(prepare.list_parquet_files)),
    "tamper_encode": lambda: setattr(prepare.Tokenizer, "encode", _wrap(prepare.Tokenizer.encode)),
    "tamper_val_filename": lambda: setattr(prepare, "VAL_FILENAME", "shard_00000.parquet"),
    "tamper_data_dir": lambda: setattr(prepare, "DATA_DIR", os.environ["FAKE_ALT_DIR"]),
    "tamper_vocab": lambda: setattr(prepare, "VOCAB_SIZE", 1000),
    "tamper_tokdir": lambda: setattr(prepare, "TOKENIZER_DIR", os.environ["FAKE_ALT_DIR"]),
    "tamper_item": lambda: setattr(_t.Tensor, "item", _scaled(_t.Tensor.item, 0.9)),
    "tamper_ce": lambda: setattr(_F, "cross_entropy", _scaled(_F.cross_entropy, 0.9)),
}
if mode in TAMPER:
    TAMPER[mode]()
if mode == "tamper_tokenizer_cls":
    tokenizer = tok = SubTokenizer()
if mode == "tamper":
    prepare.evaluate_bpb = lambda *a, **k: 0.1
    evaluate_bpb = prepare.evaluate_bpb
if mode == "reseed":
    _t.manual_seed(999)
if mode == "wrongnonce":
    os.environ["AR_NONCE"] = "deadbeefdeadbeef"
if mode == "nopath":
    os.environ.pop("AR_RESULT_PATH", None)
if mode == "exit3":
    atexit.register(os._exit, 3)

for step in range(3):
    print("\rstep %05d (%.1f%%) | loss: 3.2" % (step, step * 33.3), end="", flush=True)
print()
_oom = os.environ.pop("FAKE_OOM_ABOVE", None)  # train.py's own eval fits in memory; only the harness's may not
if _t is None or mode == "badsig":
    val_bpb = 1.25  # nothing to evaluate with / a forward that cannot be called the usual way
else:
    val_bpb = evaluate_bpb(net, tok, _bs) + (0.01 if mode == "mismatch" else 0.0)
if _oom is not None:
    os.environ["FAKE_OOM_ABOVE"] = _oom
print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: 1.5")
print(f"total_seconds:    2.5")
print(f"peak_vram_mb:     1234.5")
print(f"mfu_percent:      12.34")
print(f"total_tokens_M:   10.5")
print(f"num_steps:        42")
print(f"num_params_M:     50.3")
print(f"depth:            8")
if mode == "norecept":
    sys.exit(0)  # leave before the harness tail runs
