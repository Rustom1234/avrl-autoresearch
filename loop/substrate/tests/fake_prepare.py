"""Stand-in for autoresearch prepare.py (tests only; installed as BASE/prepare.py).

Same shape as the real module: constants, DATA_DIR / TOKENIZER_DIR under expanduser("~") (so the
worker's HOME override is exercised), module-level DATA_DIR / VAL_FILENAME read at call time by
_document_batches, an infinite make_dataloader (with a nested function and a lambda) yielding
(x, y, epoch) with B x MAX_SEQ_LEN tensors, get_token_bytes, Tokenizer.encode, and an
evaluate_bpb that is a copy of the real algorithm (decorated with torch.no_grad).

Data model: a "shard" is a text file holding one number v. x is filled with v; y cycles 0..7;
token_bytes.pt is [0..7] (token 0 is special, 0 bytes). The fake models turn v into logits.
"""
import math
import os

try:
    import torch
except ImportError:  # FAKE_NOTORCH=1 blocks torch
    torch = None

MAX_SEQ_LEN = 2048
TIME_BUDGET = 300
EVAL_TOKENS = 40 * 524288
VOCAB_SIZE = 8192
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")
VAL_FILENAME = "shard_06542.parquet"


def _no_grad(f):
    return torch.no_grad()(f) if torch is not None else f


class Tokenizer:
    def get_bos_token_id(self):
        return 0

    def encode(self, text, prepend=None):
        return [len(text)]


def get_token_bytes(device="cpu"):
    with open(os.path.join(TOKENIZER_DIR, "token_bytes.pt"), "rb") as f:
        return torch.load(f, map_location=device)


def list_parquet_files():
    return sorted(os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".parquet"))


def _document_batches(split):
    paths = list_parquet_files()
    assert len(paths) > 0, "No parquet files found."
    val_path = os.path.join(DATA_DIR, VAL_FILENAME)
    paths = [p for p in paths if p != val_path] if split == "train" else [val_path]
    while True:
        for p in paths:
            with open(p) as fh:
                yield fh.read()


def make_dataloader(tokenizer, B, T, split):
    def refill():
        return _document_batches(split)

    y = (torch.arange(B * T) % 8).view(B, T)
    docs = refill()
    while True:
        text = next(docs)
        tokenizer.encode(text, prepend=min([0, 1], key=lambda i: i))
        yield torch.full((B, T), float(text)), y, 1


def _maybe_oom(batch_size):
    if batch_size > int(os.environ.get("FAKE_OOM_ABOVE", 1 << 30)):
        if os.environ.get("FAKE_OOM_KIND") == "torch":
            raise torch.OutOfMemoryError("CUDA out of memory (fake, torch type)")
        raise RuntimeError("CUDA out of memory. Tried to allocate 4.00 GiB (fake)")


@_no_grad
def evaluate_bpb(model, tokenizer, batch_size):
    token_bytes = get_token_bytes(device="cpu")
    _maybe_oom(batch_size)
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    steps = EVAL_TOKENS // (batch_size * MAX_SEQ_LEN)
    total_nats = 0.0
    total_bytes = 0
    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction="none").view(-1)
        y_flat = y.view(-1)
        nbytes = token_bytes[y_flat]
        mask = nbytes > 0
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()
    return total_nats / (math.log(2) * total_bytes)
