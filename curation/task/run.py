"""One evaluation: curate (CPU, 180 s) -> train 300 fixed steps on exactly T tokens in curator order -> dev and sealed bpb.

    python run.py --curate curate.py --seed 0 --out result.json [--pool v1]
"""
import argparse, json, math, os, subprocess, sys, tempfile, time
import numpy as np
import tiktoken, torch, torch.nn as nn, torch.nn.functional as F
from common import DATA, SEQ, BATCH_TOKENS, STEPS, T

D, NL, H, V, MICRO = 384, 6, 6, 50304, 16
LR, WARM, LR_END = 1e-3, 15, 0.1   # 1e-3 is the best of 3e-4..4e-3 on the raw pool (results/scratch)


class Block(nn.Module):
    def __init__(s):
        super().__init__()
        s.l1, s.l2 = nn.LayerNorm(D), nn.LayerNorm(D)
        s.qkv, s.o = nn.Linear(D, 3 * D, bias=False), nn.Linear(D, D, bias=False)
        s.f1, s.f2 = nn.Linear(D, 4 * D, bias=False), nn.Linear(4 * D, D, bias=False)

    def forward(s, x):
        B, L, _ = x.shape
        q, k, v = s.qkv(s.l1(x)).view(B, L, 3, H, D // H).permute(2, 0, 3, 1, 4)
        x = x + s.o(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, L, D))
        return x + s.f2(F.gelu(s.f1(s.l2(x))))


class GPT(nn.Module):
    def __init__(s):
        super().__init__()
        s.te, s.pe = nn.Embedding(V, D), nn.Embedding(SEQ, D)
        s.blocks = nn.ModuleList([Block() for _ in range(NL)])
        s.lf = nn.LayerNorm(D)
        for n, p in s.named_parameters():
            if p.dim() == 2:
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * NL) if n.endswith(("o.weight", "f2.weight")) else 0.02)

    def forward(s, x):
        h = s.te(x) + s.pe(torch.arange(x.shape[1], device=x.device))
        for b in s.blocks:
            h = b(h)
        return F.linear(s.lf(h), s.te.weight)  # tied output head


def build_stream(idx, tokens, offs):
    seq = np.concatenate([tokens[offs[i]:offs[i + 1]] for i in idx])
    raw_tokens = len(seq)
    while len(seq) < T + 1:                    # short lists are cycled
        seq = np.concatenate([seq, seq])
    return seq[:T + 1], raw_tokens


@torch.no_grad()
def eval_bpb(model, toks, nbytes):
    model.eval()
    n = (len(toks) - 1) // SEQ
    x = torch.from_numpy(toks[:n * SEQ + 1].astype(np.int64)).cuda()
    xs = torch.stack([x[i * SEQ:(i + 1) * SEQ] for i in range(n)]); ys = torch.stack([x[i * SEQ + 1:(i + 1) * SEQ + 1] for i in range(n)])
    tot = 0.0
    for i in range(0, n, 16):
        with torch.autocast("cuda", torch.bfloat16):
            lg = model(xs[i:i + 16])
        tot += F.cross_entropy(lg.float().view(-1, V), ys[i:i + 16].reshape(-1), reduction="sum").item()
    model.train()
    nb = int(nbytes[toks[1:n * SEQ + 1]].sum())
    return tot / math.log(2) / nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curate", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True); ap.add_argument("--pool", default="v1")
    a = ap.parse_args()
    pool = DATA / f"pool_{a.pool}"
    here = os.path.dirname(os.path.abspath(__file__))
    t0 = time.time()
    with tempfile.NamedTemporaryFile(suffix=".json", dir=DATA) as tf:
        env = dict(os.environ, CURATE_SEED=str(a.seed), CUDA_VISIBLE_DEVICES="")
        subprocess.run([sys.executable, f"{here}/curate_runner.py", a.curate, str(pool), tf.name], env=env, timeout=180, check=True)
        idx = json.load(open(tf.name))
    t_curate = time.time() - t0
    tokens, offs = np.load(pool / "pool_tokens.npy"), np.load(pool / "pool_offsets.npy")
    stream, raw_tokens = build_stream(idx, tokens, offs)
    enc = tiktoken.get_encoding("gpt2")
    nbytes = np.zeros(V, dtype=np.int64)
    for t in range(enc.n_vocab - 1):
        nbytes[t] = len(enc.decode_single_token_bytes(t))   # EOT (50256) costs nats but 0 bytes

    t1 = time.time()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    model = GPT().cuda()
    decay = [p for p in model.parameters() if p.dim() == 2]; nodecay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=0.1), dict(params=nodecay, weight_decay=0.0)], lr=LR, betas=(0.9, 0.95), fused=True)
    data = torch.from_numpy(stream.astype(np.int64))
    per_micro = MICRO * SEQ
    losses = []
    for step in range(STEPS):
        lr = LR * (step + 1) / WARM if step < WARM else LR * (LR_END + (1 - LR_END) * 0.5 * (1 + math.cos(math.pi * (step - WARM) / (STEPS - WARM))))
        for g in opt.param_groups: g["lr"] = lr
        base = step * BATCH_TOKENS
        for m in range(BATCH_TOKENS // per_micro):
            o = base + m * per_micro
            xb = data[o:o + per_micro + 1].cuda(non_blocking=True)
            x, y = xb[:-1].view(MICRO, SEQ), xb[1:].view(MICRO, SEQ)
            with torch.autocast("cuda", torch.bfloat16):
                lg = model(x)
            loss = F.cross_entropy(lg.float().view(-1, V), y.reshape(-1)) / (BATCH_TOKENS // per_micro)
            loss.backward()
            losses.append(loss.item() * (BATCH_TOKENS // per_micro)) if m == 0 and step % 50 == 0 else None
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
    res = dict(curate=a.curate, seed=a.seed, pool=a.pool, tokens_used=T, curator_tokens_before_cycle=int(raw_tokens), n_selected=len(idx),
               dev_bpb=eval_bpb(model, np.load(pool / "dev_tokens.npy"), nbytes),
               sealed_bpb=eval_bpb(model, np.load(pool / "sealed_tokens.npy"), nbytes),
               train_loss_every50=losses, curate_seconds=round(t_curate, 1), seconds=round(time.time() - t1, 1),
               host=os.uname().nodename)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "train_loss_every50"}))


if __name__ == "__main__":
    main()
