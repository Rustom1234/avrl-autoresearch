"""CPU smoke test of the structural reference edits (no GPU, no data, no model calls).

For each coder reference in references.json it applies the SEARCH/REPLACE text to the base train.py, keeps the model and
optimizer code of the result (classes, functions, hyperparameters), stubs flash attention with torch SDPA, builds a tiny
GPT on the meta device the way train.py does, and checks:
  G1 every parameter is finite after init_weights (parameters are NaN-filled first, so a missing init shows up), and
     num_scaling_params and estimate_flops run (train.py calls both right after init_weights)
  G2 the training loss is finite and every parameter gets a gradient (except the ones a spec says go unused)
  G3 setup_optimizer puts every parameter in exactly one group and gives Muon only same-shape 2-D tensors
  G4 one optimizer step keeps all parameters finite, and a second forward still works
  G5 the evaluation path is untouched: forward(..., reduction='none') equals plain per-token cross entropy of the
     logits that forward(idx) returns
plus a few per-item probes (rope base, GQA, z-loss, grad clip, LR split, ...).

    .venv/bin/python -B loop/runs/calibration_requal/smoke_references.py [spec_id ...]

Needs torch. The GPU is hidden on purpose.
"""
import ast
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "loop" / "core"))
import calibration as cal  # noqa: E402
import patch  # noqa: E402

BASE = cal.BASE_PATH.read_text()
V, T, B = 256, 32, 2
UNUSED_GRAD = {"rstruct-04": {"x0_lambdas"}}


class FA3:
    """Stand-in for the flash attention interface: causal SDPA, window ignored, GQA by repeating kv heads."""

    @staticmethod
    def flash_attn_func(q, k, v, softmax_scale=None, causal=False, window_size=(-1, -1), **kw):
        h, hk = q.size(2), k.size(2)
        if hk != h:
            k, v = k.repeat_interleave(h // hk, dim=2), v.repeat_interleave(h // hk, dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=softmax_scale).transpose(1, 2)


def _is_compile_deco(d):
    return isinstance(d, ast.Call) and ast.unparse(d.func) == "torch.compile"


def load_namespace(src):
    """Exec the imports, classes, functions and uppercase constants of train.py (no GPU setup, no training loop)."""
    keep = []
    for node in ast.parse(src).body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom) and node.module in ("kernels", "prepare"):
                continue
            keep.append(node)
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            node.decorator_list = [d for d in node.decorator_list if not _is_compile_deco(d)]
            keep.append(node)
        elif isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) for t in node.targets) and any(
                t.id.isupper() or t.id == "polar_express_coeffs" for t in node.targets):
            keep.append(node)
    ns = {"fa3": FA3, "__name__": "smoke"}
    exec(compile(ast.Module(keep, []), "train.py", "exec"), ns)
    ns.update(MAX_SEQ_LEN=T, vocab_size=V, ASPECT_RATIO=16, HEAD_DIM=32)
    return ns


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def build(ns, depth=4):
    config = ns["build_model_config"](depth)
    with torch.device("meta"):
        model = ns["GPT"](config)
    model.to_empty(device="cpu")
    for p in model.parameters():
        p.data.fill_(float("nan"))
    model.init_weights()
    return config, model


def run_checks(spec_id, src):
    torch.manual_seed(0)
    ns = load_namespace(src)
    config, model = build(ns)
    for n, p in model.named_parameters():
        check(torch.isfinite(p.float()).all(), f"G1 {n} not finite after init_weights")
    model.num_scaling_params()
    model.estimate_flops()
    idx, tgt = torch.randint(0, V, (B, T)), torch.randint(0, V, (B, T))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = model(idx, tgt)
    check(torch.isfinite(loss), "G2 loss not finite")
    loss.backward()
    skip = UNUSED_GRAD.get(spec_id, set())
    for n, p in model.named_parameters():
        check(p.grad is not None or n.split(".")[0] in skip, f"G2 no gradient for {n}")
    opt = model.setup_optimizer()
    seen = [id(p) for g in opt.param_groups for p in g["params"]]
    check(len(seen) == len(set(seen)) == len(list(model.parameters())), "G3 a parameter is in two groups or none")
    for g in opt.param_groups:
        if g["kind"] == "muon":
            check(all(p.ndim == 2 and p.shape == g["params"][0].shape for p in g["params"]), "G3 muon group has a non-2D or mixed-shape tensor")
    probes(spec_id, ns, config, model, opt, idx, tgt, loss)
    clip_calls = []
    real_clip = torch.nn.utils.clip_grad_norm_
    torch.nn.utils.clip_grad_norm_ = lambda *a, **k: (clip_calls.append((a, k)), real_clip(*a, **k))[1]
    try:
        opt.step()
    finally:
        torch.nn.utils.clip_grad_norm_ = real_clip
    if spec_id == "rstruct-10":
        check(len(clip_calls) == 1 and (clip_calls[0][0][1:] or [clip_calls[0][1].get("max_norm")])[0] == 1.0,
              "clip_grad_norm_(params, 1.0) not called once in step")
    for n, p in model.named_parameters():
        check(torch.isfinite(p.float()).all(), f"G4 {n} not finite after the optimizer step")
    model.zero_grad(set_to_none=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        check(torch.isfinite(model(idx, tgt)), "G4 second forward not finite")
        logits = model(idx)
        none = model(idx, tgt, reduction="none")
    ce = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1), reduction="none")
    check(none.shape == ce.shape and torch.allclose(none, ce, atol=1e-5), "G5 reduction='none' is not plain cross entropy of the logits")


def probes(spec_id, ns, config, model, opt, idx, tgt, loss):
    groups = opt.param_groups
    if spec_id == "rstruct-06":
        check(config.n_kv_head == max(1, config.n_head // 2) and config.n_kv_head < config.n_head, "GQA: n_kv_head not half of n_head")
    if spec_id == "rstruct-07":
        with torch.autocast("cpu", dtype=torch.bfloat16):
            logits = model(idx)
        ce = F.cross_entropy(logits.view(-1, V), tgt.view(-1))
        check(loss.item() > ce.item() + 1e-7, "z-loss: mean loss is not above plain cross entropy")
    if spec_id == "rstruct-14":
        check(model.x0_lambdas.shape == (config.n_layer, config.n_embd), "x0_lambdas shape")
    if spec_id == "rstruct-16":
        check(torch.all(model.block_gates == 1.0), "block_gates init")
    if spec_id == "rstruct-17":
        check(torch.all(model.logit_bias == 0.0), "logit_bias init")
    if spec_id == "rstruct-18":
        lrs = sorted({g["lr"] for g in groups if g["kind"] == "muon"})
        check(len(lrs) == 2 and abs(lrs[1] / lrs[0] - 1.5) < 1e-9, f"muon lrs {lrs}")
        mlp_ids = {id(p) for b in model.transformer.h for p in b.mlp.parameters()}
        for g in groups:
            if g["kind"] == "muon":
                check({id(p) in mlp_ids for p in g["params"]} in ({True}, {False}), "muon group mixes mlp and attention tensors")
                check((g["lr"] == max(lrs)) == (id(g["params"][0]) in mlp_ids), "the 1.5x lr is not on the mlp groups")
    if spec_id == "rstruct-19":
        gates = {id(b.attn.ve_gate.weight) for b in model.transformer.h if b.attn.ve_gate is not None}
        check(gates and all(g["kind"] == "adamw" for g in groups if gates & {id(p) for p in g["params"]}), "ve_gate not in an adamw group")
    if spec_id == "rstruct-20":
        check(all(torch.all(b.attn.attn_gate.weight == 0) for b in model.transformer.h), "attn_gate init")
    if spec_id == "rstruct-21":
        check(model.skip_weights.shape == (config.n_layer // 2,) and torch.all(model.skip_weights == 0), "skip_weights")
    if spec_id == "rstruct-22":
        check(not hasattr(model, "value_embeds") and model.value_embed.weight.dtype == torch.bfloat16, "shared value_embed missing or not bf16")
        check(model.value_embed.weight.shape == (V, config.n_kv_head * (config.n_embd // config.n_head)), "shared value_embed has the wrong shape")
    if spec_id == "rstruct-23":
        check(torch.all(model.pre_head.weight == 0), "pre_head init")
    if spec_id == "rstruct-24":
        with torch.autocast("cpu", dtype=torch.bfloat16):
            logits = model(idx)
        ce = F.cross_entropy(logits.view(-1, V), tgt.view(-1))
        check(abs(loss.item() - ce.item()) > 1e-6, "mtp: mean loss equals plain cross entropy")
        with torch.autocast("cpu", dtype=torch.bfloat16):
            check(model(idx).shape == (B, T, V), "mtp: targets=None path changed")
    if spec_id == "rstruct-25":
        cos, _ = model._precompute_rotary_embeddings(model.rotary_seq_len, config.n_embd // config.n_head, base=50000)
        check(torch.equal(model.cos, cos), "rotary cos table does not use base 50000 after init_weights")
        check(config.rope_base == 50000, "build_model_config does not pass ROPE_BASE")
    if spec_id == "rstruct-26":
        check(ns["MUON_BETA2"] == 0.9, "MUON_BETA2 is not 0.9")
        opt7 = model.setup_optimizer(muon_beta2=0.7)
        check(all(g["beta2"] == 0.7 for g in opt7.param_groups if g["kind"] == "muon"), "muon_beta2 does not reach the Muon groups")
    if spec_id == "rstruct-27":
        w = model.transformer.h[0].attn.v_conv.weight
        v = torch.randn(B, T, w.size(0))
        out = model.transformer.h[0].attn.v_conv(F.pad(v.transpose(1, 2), (2, 0))).transpose(1, 2)
        check(torch.allclose(out, v, atol=1e-6), "v_conv is not the identity at init")
    if spec_id == "rstruct-28":
        check(all(b.mlp.c_fc.bias is not None and torch.all(b.mlp.c_fc.bias == 0) for b in model.transformer.h), "c_fc bias")
    if spec_id == "rstruct-29":
        check(model.transformer.h[-1].mlp is None and all(b.mlp is not None for b in model.transformer.h[:-1]), "last block mlp")
    if spec_id == "rstruct-30":
        check(torch.all(model.final_gain == 1.0), "final_gain init")


def main(argv):
    refs = json.loads((HERE / "references.json").read_text())["coder"]
    ids = argv or sorted(refs)
    bad = 0
    for i in ids:
        new, rep = patch.apply_text(BASE, refs[i])
        try:
            check(rep["ok"], "reference does not apply")
            run_checks(i, new)
            print("ok   ", i)
        except Exception as e:  # report every failing item, not only the first
            bad += 1
            print("FAIL ", i, type(e).__name__, str(e)[:300])
    print(f"{len(ids) - bad}/{len(ids)} references pass the CPU smoke test")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
