#!/usr/bin/env python3
"""Step 6: train the cheap planner (LoRA) on the ideas of the strong planner that were kept.

    .venv-rl/bin/python curation/loop/s6_train_planner.py --data curation/runs/S6/data/train.jsonl \
        --out curation/runs/S6/adapter

Each example is one planner call: the system prompt, the prompt the strong planner saw, and its reply as JSON. The loss is
on the reply only. A few loops are held out to report the loss on unseen prompts; nothing is tuned on them.
One GPU, one process, no data-loader workers; run it inside a memory-capped scope on a shared machine.
"""
import argparse, json, math, random, time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoTokenizer

TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


def encode(tok, ex, max_tokens):
    """(input ids, labels): labels are -100 on the prompt. None if the example is longer than max_tokens."""
    msgs = [{"role": "system", "content": ex["system"]}, {"role": "user", "content": ex["user"]}]
    head = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    a = tok(head, add_special_tokens=False)["input_ids"]
    b = tok(ex["assistant"] + "<|im_end|>\n", add_special_tokens=False)["input_ids"]
    if len(a) + len(b) > max_tokens:
        return None
    return torch.tensor([a + b]), torch.tensor([[-100] * len(a) + b])


def reply_loss(model, x, y):
    """Cross-entropy on the reply tokens only. The output layer is applied to the reply positions alone: logits for a
    whole 7,000-token prompt over the vocabulary would take 3 GB of GPU memory."""
    core = model.base_model.model
    n = int((y[0] != -100).sum())                       # the reply is the last n tokens
    h = core.model(input_ids=x.cuda()).last_hidden_state
    logits = core.lm_head(h[0, -n - 1:-1])
    return F.cross_entropy(logits.float(), x[0, -n:].cuda())


@torch.no_grad()
def mean_loss(model, items):
    model.eval()
    out = [reply_loss(model, x, y).item() for x, y in items]
    model.train()
    return sum(out) / max(len(out), 1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--holdout-loops", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20261111)
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    rows = [json.loads(l) for l in open(a.data)]
    held = set(rng.sample(sorted({r["loop"] for r in rows}), a.holdout_loops))
    tok = AutoTokenizer.from_pretrained(a.base)
    enc = [(r, encode(tok, r, a.max_tokens)) for r in rows]
    train = [e for r, e in enc if e is not None and r["loop"] not in held]
    val = [e for r, e in enc if e is not None and r["loop"] in held]
    print("examples %d: train %d, held out %d (loops %s), too long %d" % (
        len(rows), len(train), len(val), " ".join(sorted(held)), sum(e is None for _, e in enc)), flush=True)

    model = AutoModelForImageTextToText.from_pretrained(a.base, dtype=torch.bfloat16, device_map={"": 0})
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, target_modules=TARGETS,
                                             task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    steps = a.epochs * math.ceil(len(train) / a.accum)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 3) * 0.5 * (1 + math.cos(math.pi * s / steps)))
    log = {"args": vars(a), "held_out_loops": sorted(held), "n_train": len(train), "n_val": len(val), "epochs": []}
    log["val_loss_before"] = mean_loss(model, val)
    print("held-out loss before training: %.4f" % log["val_loss_before"], flush=True)
    model.train()
    t0 = time.time()
    for ep in range(a.epochs):
        rng.shuffle(train)
        total = 0.0
        for i, (x, y) in enumerate(train):
            loss = reply_loss(model, x, y)
            (loss / a.accum).backward()
            total += loss.item()
            if (i + 1) % a.accum == 0 or i + 1 == len(train):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
        row = {"epoch": ep + 1, "train_loss": total / len(train), "val_loss": mean_loss(model, val),
               "minutes": (time.time() - t0) / 60, "gpu_gb_peak": torch.cuda.max_memory_allocated() / 2 ** 30}
        log["epochs"].append(row)
        print(json.dumps(row), flush=True)
    out = Path(a.out)
    model.save_pretrained(out)
    (out / "train_log.json").write_text(json.dumps(log, indent=1))
    print("adapter saved to %s" % out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
