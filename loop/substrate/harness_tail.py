"""Harness tail: Python source appended to every candidate train.py copy.

The tail checks the evaluation pipeline for tampering, scores the trained model itself from raw
logits (official shard, and the sealed shard when AR_SEALED_DIR is set), re-runs the pinned
evaluate_bpb as a secondary number, and writes the receipt (AR_RESULT_PATH, else harness_result.json
next to train.py).

CLI: `python harness_tail.py PREPARE_PY` prints, for this interpreter, a pinned_hashes.json
fragment (prepare.py sha256 plus the code hashes below) to paste into BASE/pinned_hashes.json.
"""
import hashlib
import json
import sys

# Code hash recipe: sha256 over co_code + repr(co_consts) + repr(co_names), with nested code objects
# in co_consts replaced by their own hash (their repr has an address and a file path). Hashes are
# keyed by "major.minor" because bytecode differs per Python version. Computed from
# karpathy/autoresearch @ 228791f prepare.py (file sha256 4f2ba9cb...).
# evaluate_bpb is hashed UNWRAPPED (it is decorated with torch.no_grad).
PINNED_EVAL_CODE_SHA256 = {
    "3.10": "12a940887adffbb9e051687211c8d97b3b1e39291498bca1ca774e103fa4485f",
    "3.12": "fe3c17a59289e56c399ad424b2e6c32d8a1349fbaa2ba3399995625e9b17638d",
    "3.13": "f801ae385ac64086a0e35e848ce150d7b94ce3f13263c163d8fd825131d7c5ff",
    "3.14": "af319d1593987c82898fde480e60b8221964518c4b4048ca2ed3eb43cdfafafb",
}
# The rest of the evaluation pipeline: version -> function name -> hash.
PIPELINE_FUNCS = ("make_dataloader", "_document_batches", "get_token_bytes", "list_parquet_files",
                  "Tokenizer.encode")
PINNED_PIPELINE_CODE_SHA256 = {
    "3.10": {
        "make_dataloader": "b6817facede01f315d87238174e9e3345a84db349faa83cef58d07d5603a76bd",
        "_document_batches": "997511e3a407a2234aee337dbcb37bfbee53fafb1602c4def11cf4d77a221c28",
        "get_token_bytes": "8c0146f8f856d86deae46863ed7c4f9364bfa245b83afadc1e7bb9a409b38785",
        "list_parquet_files": "7a52eaeb5241fc60ceac9b5c4176f235823c89d6659148ae40251686b706411a",
        "Tokenizer.encode": "ca563da9683ab84317e48dae89d77fb18406034d203026d1373e9097b357f56e",
    },
    "3.12": {
        "make_dataloader": "533ea2475cbdd575c4a52c1881104250a981eb586014fb1dae6aab980c6acf15",
        "_document_batches": "1d3a459dfa3b37112d320b039d9938c5b3c95f777594bbd1ad819356999ef612",
        "get_token_bytes": "2dc74800961445a9fd4c6aeeff9d260cc6b6147012c6988bd4969bb336535318",
        "list_parquet_files": "149059c1f1905e7f1d1b165a9e52360ce7242c1e612fcf26d990dbb7f1ac6176",
        "Tokenizer.encode": "034827e861093892878b044cf0ecd9f5bd7f556317f8815bba6319c3993f1f0a",
    },
    "3.13": {
        "make_dataloader": "775e9e88a44dc080fbae3e9947a23d22d1606188c943b47dbd5a99ae6d179799",
        "_document_batches": "bd76d53d4e0902a2dbbbd3345f303dd18ccbc4244b06b6fc5e23dfb75e74ea63",
        "get_token_bytes": "0e014cb8d879ee7d0e70756060e4aa097c4c3ea300491260e8657678fec2ab33",
        "list_parquet_files": "f20c04a5a4189541f095d25c1256cbc4d61e8e43e9d36712d7f94a06b584abb2",
        "Tokenizer.encode": "5ea73c99e341bf6d0e458da743948da78304170baefd21e845a5bc6bb8e0bf19",
    },
    "3.14": {
        "make_dataloader": "13283a4f332d61644173099fc9564f65441a04b7606d034abccde6cc15be3cfd",
        "_document_batches": "0e3354547b30f5e198bcc5fae051f2d8693547a603cccf93228e126482153ba5",
        "get_token_bytes": "6822b9e39b590b99a7bdd056e4561680f031a46ecd3de7ce7f5d834022fdaf86",
        "list_parquet_files": "82c3b81a093e579ecd0864848bd347045c829bd2d6bb44cd776c5c2504b375f1",
        "Tokenizer.encode": "874921063b0ae653b5cb7040e6200eaa05ab34af5ce91bcfc1e3f9604771ee19",
    },
}


def eval_code_hash(code):
    consts = tuple(eval_code_hash(k) if hasattr(k, "co_code") else k for k in code.co_consts)
    return hashlib.sha256(
        code.co_code + repr(consts).encode() + repr(code.co_names).encode()
    ).hexdigest()


def prepare_code_hashes(prepare_src):
    """name -> hash for evaluate_bpb and PIPELINE_FUNCS, as this interpreter compiles prepare.py source."""
    wanted = ("evaluate_bpb",) + tuple(n for n in PIPELINE_FUNCS if "." not in n)
    found = {}
    for c in compile(prepare_src, "prepare.py", "exec").co_consts:
        name = getattr(c, "co_name", None)
        if name in wanted and hasattr(c, "co_code"):
            found[name] = eval_code_hash(c)
        elif name == "Tokenizer" and hasattr(c, "co_code"):  # class body: methods are its code constants
            for k in c.co_consts:
                if getattr(k, "co_name", None) == "encode":
                    found["Tokenizer.encode"] = eval_code_hash(k)
    missing = [n for n in ("evaluate_bpb",) + PIPELINE_FUNCS if n not in found]
    if missing:
        raise ValueError("not found in prepare source: %s" % missing)
    return found


def prepare_eval_code_hash(prepare_src):
    return prepare_code_hashes(prepare_src)["evaluate_bpb"]


_TAIL_TEMPLATE = r'''

# ---- autoresearch harness tail (appended by worker.py, not part of the candidate) ----
def _ar_harness(_g):
    import time
    tail_start = time.time()  # external time check: the worker compares this to its own launch time
    import importlib, json, os, sys, hashlib, contextlib, gc, io
    try:
        import torch
    except Exception:
        torch = None
    LN2 = 0.6931471805599453  # literal: do not trust math.log in a process the candidate has run in
    out = {"official_val_bpb": None, "sealed_val_bpb": None, "harness_eval_seconds": None,
           "official_via_prepare": None, "official_via_prepare_diff": None,
           "official_tokens_scored": None, "official_bytes_scored": None,
           "sealed_tokens_scored": None, "sealed_bytes_scored": None, "score_shape_bad": None,
           "canary_ok": None, "canary_value": None, "canary_expected": None,
           "prepare_code_hash_ok": None, "pipeline_code_hash_ok": None, "constants_ok": None,
           "constants_bad": None, "pipeline_bad": None, "eval_code_sha256": None,
           "pipeline_code_sha256": None, "val_shard_path": None, "val_shard_sha256": None,
           "data_files_sha256": None, "data_files_ok": None,
           "eval_batch_size_used": None, "sealed_eval_batch_size_used": None,
           "target_leak_flag": None, "target_leak_loss_a": None, "target_leak_loss_b": None,
           "target_leak_error": None, "initial_seed": None, "seed_ok": None,
           "tail_start_epoch": tail_start, "nonce": os.environ.get("AR_NONCE"),
           "peak_vram_mb": None, "wall_from_AR_T0": None, "error": None}
    errs = []
    t_eval = [0.0]
    pyver = "%d.%d" % sys.version_info[:2]
    try:
        tdir = os.path.realpath(os.path.dirname(torch.__file__))
    except Exception:
        tdir = None

    def _h(c):  # same recipe as harness_tail.eval_code_hash
        consts = tuple(_h(k) if hasattr(k, "co_code") else k for k in c.co_consts)
        return hashlib.sha256(c.co_code + repr(consts).encode() + repr(c.co_names).encode()).hexdigest()

    def _is_oom(e):
        t = getattr(torch, "OutOfMemoryError", None)
        return (t is not None and isinstance(e, t)) or "out of memory" in str(e).lower()

    def _retrying(fn, start=64):
        """fn(bs) -> value; returns (value, bs). Fixed batch 64; on CUDA OOM empty the cache, retry 32, then 16."""
        sizes = [b for b in (64, 32, 16) if b <= start]
        for i, bs in enumerate(sizes):
            try:
                return fn(bs), bs
            except Exception as e:
                if not _is_oom(e) or i == len(sizes) - 1:
                    raise
            gc.collect()  # outside the except block, so the failed attempt's tensors are released
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def _ctx():
        st = contextlib.ExitStack()
        if torch is not None:
            st.enter_context(torch.no_grad())
            st.enter_context(torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16))
        return st

    def _inspect(f):  # (hash of the innermost function, every wrapper is torch's)
        ok = True
        for _ in range(8):
            if not hasattr(f, "__wrapped__"):
                break
            ok = ok and tdir is not None and os.path.realpath(f.__code__.co_filename).startswith(tdir + os.sep)
            f = f.__wrapped__
        return _h(f.__code__), ok and not hasattr(f, "__wrapped__")

    def _sha_file(path):
        sh = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                sh.update(chunk)
        return sh.hexdigest()

    try:
        _p = importlib.import_module("prepare")
        # All integrity checks run here, before the official eval and before any sealed redirect.
        g = lambda n: getattr(_p, n, None)
        out["constants_bad"] = [n for n, ok in (
            ("TIME_BUDGET", g("TIME_BUDGET") == 300),
            ("EVAL_TOKENS", g("EVAL_TOKENS") == 40 * 524288),
            ("MAX_SEQ_LEN", g("MAX_SEQ_LEN") == 2048),
            ("VOCAB_SIZE", g("VOCAB_SIZE") == 8192),
            ("VAL_FILENAME", g("VAL_FILENAME") == "shard_06542.parquet"),
            ("DATA_DIR", str(g("DATA_DIR")).endswith(".cache/autoresearch/data")),
            ("TOKENIZER_DIR", str(g("TOKENIZER_DIR")).endswith(".cache/autoresearch/tokenizer")),
        ) if not ok]
        out["constants_ok"] = not out["constants_bad"]

        try:
            h, ok = _inspect(_p.evaluate_bpb)
            out["eval_code_sha256"] = h
            pin = __PINNED__.get(pyver)
            out["prepare_code_hash_ok"] = None if pin is None else bool(ok and h == pin)
        except Exception:
            out["prepare_code_hash_ok"] = False

        model = _g.get("model")
        tokenizer = _g.get("tokenizer")
        if model is None and torch is not None:
            model = next((v for v in _g.values() if isinstance(v, torch.nn.Module)), None)
        if tokenizer is None:
            tokenizer = next((v for v in _g.values()
                              if not isinstance(v, type) and hasattr(v, "encode")
                              and hasattr(v, "get_bos_token_id")), None)

        pins = __PIPELINE__.get(pyver)
        out["pipeline_code_sha256"], bad = {}, []
        for name in __PIPELINE_NAMES__:
            try:
                obj = _p
                for part in name.split("."):
                    obj = getattr(obj, part)
                out["pipeline_code_sha256"][name], ok = _inspect(obj)
                if not ok or (pins is not None and out["pipeline_code_sha256"][name] != pins.get(name)):
                    bad.append(name)
            except Exception:
                bad.append(name)
        if tokenizer is None or type(tokenizer) is not getattr(_p, "Tokenizer", None):
            bad.append("Tokenizer.type")  # the Tokenizer.encode hash only means something for a real Tokenizer
        out["pipeline_bad"] = bad
        out["pipeline_code_hash_ok"] = (not bad) if pins is not None else (False if bad else None)

        # Seed binding: the worker passes the job seed in AR_SEED; the default generator must carry it.
        try:
            out["initial_seed"] = int(torch.initial_seed())
            if os.environ.get("AR_SEED"):
                out["seed_ok"] = out["initial_seed"] == int(os.environ["AR_SEED"])
        except Exception:
            pass

        # Data files: record (and compare with the pins, when there are any) the sha256 of every file
        # the score depends on. token_bytes.pt is read once, hashed, and loaded from those same bytes.
        data_pins = __DATA_FILES__
        files, tb_cpu = {}, None
        try:
            vp = os.path.join(_p.DATA_DIR, _p.VAL_FILENAME)
            out["val_shard_path"] = vp
            out["val_shard_sha256"] = files[os.path.basename(vp)] = _sha_file(vp)
        except Exception as e:
            errs.append("val shard hash: %s: %s" % (type(e).__name__, str(e)[:300]))
        try:
            with open(os.path.join(_p.TOKENIZER_DIR, "token_bytes.pt"), "rb") as fh:
                raw = fh.read()
            files["token_bytes.pt"] = hashlib.sha256(raw).hexdigest()
            tb_cpu = torch.load(io.BytesIO(raw), map_location="cpu")
        except Exception as e:
            errs.append("token_bytes: %s: %s" % (type(e).__name__, str(e)[:300]))
        try:
            files["tokenizer.pkl"] = _sha_file(os.path.join(_p.TOKENIZER_DIR, "tokenizer.pkl"))
        except Exception as e:
            errs.append("tokenizer.pkl hash: %s: %s" % (type(e).__name__, str(e)[:300]))
        out["data_files_sha256"] = files
        out["data_files_ok"] = all(files.get(k) == v for k, v in data_pins.items()) if data_pins else None

        if model is None or tokenizer is None:
            raise RuntimeError("model or tokenizer not found in train.py globals")
        model.eval()

        def _bpb(bs):  # secondary: the pinned prepare.evaluate_bpb (its loss path belongs to the candidate)
            t = time.time()
            try:
                with _ctx():
                    return float(_p.evaluate_bpb(model, tokenizer, bs))
            finally:
                t_eval[0] += time.time() - t

        def _score(mdl, bs, steps):
            """Harness-owned bpb. Raw logits from mdl(x.clone()) with NO targets, nats from our own
            F.cross_entropy, bytes from our own token_bytes, Python accumulation, literal LN2."""
            F = torch.nn.functional
            loader = _p.make_dataloader(tokenizer, bs, _p.MAX_SEQ_LEN, "val")
            nats_sum, bytes_sum, tokens, masked, tb = 0.0, 0, 0, 0, tb_cpu
            t = time.time()
            try:
                with _ctx():
                    for _ in range(steps):
                        x, y = next(loader)[:2]
                        logits = mdl(x.clone())
                        yf = y.view(-1)
                        nats = F.cross_entropy(logits.view(-1, logits.size(-1)), yf, reduction="none")
                        if tb.device != yf.device:
                            tb = tb.to(yf.device)
                        nbytes = tb[yf]
                        mask = nbytes > 0
                        nats_sum += (nats.double() * mask).sum().item()
                        # counts via int(), not Tensor.item: a patched item() can then only move the numerator,
                        # which the canary catches (a uniform scale of both would cancel in the ratio)
                        bytes_sum += int(nbytes.sum())
                        masked += int(mask.sum())
                        tokens += yf.numel()
            finally:
                t_eval[0] += time.time() - t
            return {"bpb": nats_sum / (LN2 * bytes_sum), "bytes": bytes_sum, "tokens": tokens, "masked": masked}

        class _Zeros:  # harness-owned stand-in model: uniform over 8192 tokens
            def __call__(self, x):
                return torch.zeros(x.size(0), x.size(1), 8192, device=x.device)

        def _canary():
            # uniform logits cost exactly log2(8192) = 13 bits per token, so bpb = 13 * masked / bytes.
            # Small fixed batch: the (B, T, 8192) logits of 64 rows would be 4 GB for no benefit here.
            r = _score(_Zeros(), 4, 2)
            out["canary_value"], out["canary_expected"] = r["bpb"], 13.0 * r["masked"] / r["bytes"]
            return abs(out["canary_value"] - out["canary_expected"]) < 1e-6

        def _full(bs):
            steps = _p.EVAL_TOKENS // (bs * _p.MAX_SEQ_LEN)
            if steps * bs * _p.MAX_SEQ_LEN != _p.EVAL_TOKENS:
                out["score_shape_bad"] = True
                raise ValueError("steps * B * T != EVAL_TOKENS")
            return _score(model, bs, steps)

        def _own(split, bs_start):
            r, bs = _retrying(_full, start=bs_start)
            bad_shape = r["tokens"] != 40 * 524288
            out["score_shape_bad"] = bool(out["score_shape_bad"] or bad_shape)
            out[split + "_tokens_scored"], out[split + "_bytes_scored"] = r["tokens"], r["bytes"]
            return (None if bad_shape else r["bpb"]), bs

        def _leak(bs):
            # base GPT.forward softcaps the logits BEFORE both paths, so model(x) returns exactly the
            # logits the loss is computed from: for an unmodified model loss_a == loss_b.
            if torch is None:
                raise RuntimeError("torch unavailable")
            x, y = next(_p.make_dataloader(tokenizer, bs, _p.MAX_SEQ_LEN, "val"))[:2]
            with _ctx():
                a = model(x, y, reduction="none").float().mean().item()
                logits = model(x)
                b = torch.nn.functional.cross_entropy(
                    logits.float().view(-1, logits.size(-1)), y.view(-1)).item()
            del logits
            return a, b

        try:  # a model whose forward peeks at the targets scores better with them than without
            (a, b), _ = _retrying(_leak)
            out["target_leak_loss_a"], out["target_leak_loss_b"] = a, b
            out["target_leak_flag"] = bool(abs(a - b) > 0.01)
        except Exception as e:
            out["target_leak_error"] = "%s: %s" % (type(e).__name__, str(e)[:300])

        try:
            if torch is None:
                raise RuntimeError("torch unavailable")
            if tb_cpu is None:
                raise RuntimeError("token_bytes.pt could not be loaded")
            out["canary_ok"] = _canary()
            out["official_val_bpb"], out["eval_batch_size_used"] = _own("official", 64)
            try:
                v, _ = _retrying(_bpb, start=out["eval_batch_size_used"])
                out["official_via_prepare"] = v
                if out["official_val_bpb"] is not None:
                    out["official_via_prepare_diff"] = out["official_val_bpb"] - v
            except Exception as e:
                errs.append("via prepare: %s: %s" % (type(e).__name__, str(e)[:300]))
        except Exception as e:
            import traceback; traceback.print_exc()
            errs.append("official: %s: %s" % (type(e).__name__, str(e)[:300]))

        sd = os.environ.get("AR_SEALED_DIR")
        if sd:
            try:
                names = [n for n in os.listdir(sd) if n.endswith(".parquet")]
                if len(names) != 1:
                    raise RuntimeError("expected exactly one .parquet in %s, found %d" % (sd, len(names)))
                if torch is None or tb_cpu is None:
                    raise RuntimeError("torch or token_bytes unavailable")
                _p.DATA_DIR = sd
                _p.VAL_FILENAME = names[0]
                out["sealed_val_bpb"], out["sealed_eval_batch_size_used"] = _own(
                    "sealed", out["eval_batch_size_used"] or 64)
            except Exception as e:
                import traceback; traceback.print_exc()
                errs.append("sealed: %s: %s" % (type(e).__name__, str(e)[:300]))
    except Exception as e:
        import traceback; traceback.print_exc()
        errs.append("%s: %s" % (type(e).__name__, str(e)[:300]))
    finally:
        try:
            if torch is not None:
                out["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024
        except Exception:
            pass
        if os.environ.get("AR_T0"):
            out["wall_from_AR_T0"] = time.time() - float(os.environ["AR_T0"])
        out["harness_eval_seconds"] = t_eval[0]
        out["error"] = "; ".join(errs) or None
        try:
            path = os.environ.get("AR_RESULT_PATH") or os.path.join(
                os.path.dirname(os.path.abspath(_g.get("__file__") or "train.py")), "harness_result.json")
            with open(path + ".tmp", "w") as fh:
                json.dump(out, fh)
            os.replace(path + ".tmp", path)
        except Exception as e:
            print("harness: could not write the harness result: %r" % (e,))
        print("harness_official_val_bpb: %s" % (out["official_val_bpb"],), flush=True)


_ar_harness(globals())
'''


def build_tail(pinned=None, pipeline=None, data_files=None):
    """Tail source with the pin tables baked in: evaluate_bpb code hashes (version -> sha256), pipeline
    code hashes (version -> name -> sha256) and data file hashes (file name -> sha256, may be empty)."""
    pinned = PINNED_EVAL_CODE_SHA256 if pinned is None else pinned
    pipeline = PINNED_PIPELINE_CODE_SHA256 if pipeline is None else pipeline
    return (_TAIL_TEMPLATE.replace("__PINNED__", repr(dict(pinned)))
            .replace("__PIPELINE_NAMES__", repr(PIPELINE_FUNCS))
            .replace("__PIPELINE__", repr(dict(pipeline)))
            .replace("__DATA_FILES__", repr(dict(data_files or {}))))


HARNESS_TAIL = build_tail()


if __name__ == "__main__":
    with open(sys.argv[1], "rb") as fh:
        data = fh.read()
    key = "%d.%d" % sys.version_info[:2]
    hashes = prepare_code_hashes(data.decode("utf-8"))
    print(json.dumps({
        "prepare_py_sha256": hashlib.sha256(data).hexdigest(),
        "evaluate_bpb_code_sha256": {key: hashes.pop("evaluate_bpb")},
        "pipeline_code_sha256": {key: hashes},
    }, indent=2))
