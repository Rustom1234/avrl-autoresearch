"""Cheap-proxy / expensive-judge reward machinery for E09 (open-ended, free-text) phase-1 GRPO.
See p1_judge_grpo.py (CLI + GRPOTrainer wiring) and avrl/exp/E09_judge_task.py (the rubric, the
reward-model scoring pattern, and the cheap-feature definitions this file mirrors).

This file imports torch/transformers/sentence_transformers only inside the classes that need
them (RewardModelScorer, HFJudge, MiniLMConsistency), same convention as p1_grpo.py, so
py_compile and all the pure reward math below (medians, feature vectors, judge-score parsing)
are checkable on a machine with none of those packages installed.

Fairness note (ties to allocators.py's invariants -- see its module docstring): nothing in this
file ever receives gold/judge information alongside a feat_vec or cell; cheap_feature_vector()
is built purely from RM scores, lengths, truncation flags, q_len, consistency and prompt PCA
components. Gold enters the system only through gold_rewards()/is_gold_positive(), which
p1_judge_grpo.py's JudgeRewardState calls strictly on the side, never mixing the result into a
feat_vec (mirrors p1_grpo.py's RewardState stripping gold before calling allocator.select()).
"""
import re

import numpy as np

RUBRIC = (  # verbatim from avrl/exp/E09_judge_task.py so gold is defined identically everywhere
    "Rate the assistant's response below for helpfulness and correctness, on an integer scale from 1 "
    "to 10 (1 = unhelpful or incorrect, 10 = extremely helpful and fully correct).\n\n"
    "User request:\n{prompt}\n\nAssistant response:\n{response}\n\n"
    "Respond with only the integer score, nothing else.\nScore:"
)
SCORE_RE = re.compile(r"(?<!\d)(10|[1-9])(?!\d)")
GOLD_THRESHOLD = 7


def parse_score(text):
    """Same parse as avrl/exp/E09_judge_task.py: first bare 1-10 integer in the completion, else
    unparseable (score=None, valid=0)."""
    m = SCORE_RE.search(text)
    return (int(m.group(1)), 1) if m else (None, 0)


def is_gold_positive(score, valid):
    """gold = 1 iff the judge gave a parseable score >= GOLD_THRESHOLD."""
    return int(bool(valid) and score is not None and score >= GOLD_THRESHOLD)


def gold_rewards(scores_and_valids):
    """Per-response training reward for a *verified* group: one judge call per response, reward
    = is_gold_positive(score, valid). list[(score, valid)] -> list[int]."""
    return [is_gold_positive(s, v) for s, v in scores_and_valids]


MIN_JUDGE_SCORE = 1  # RUBRIC's scale floor (integer 1-10); see rank_gold_rewards below


def rank_gold_rewards(scores_and_valids):
    """Per-response training reward for a *verified* group under --gold_reward_shape rank: 1 if
    the response's judge score is strictly above the group median of judge scores, else 0 --
    mirrors proxy_rewards() below exactly, but on judge scores instead of reward-model scores.
    Unparseable scores (valid=0) are treated as MIN_JUDGE_SCORE, the judge's scale floor, so they
    never land in the top half unless every score in the group is unparseable, in which case all
    scores tie at MIN_JUDGE_SCORE and every reward is 0 (none strictly above the median).
    list[(score, valid)] -> list[int]."""
    scores = np.asarray([s if v else MIN_JUDGE_SCORE for s, v in scores_and_valids], dtype=float)
    med = np.median(scores)
    return (scores > med).astype(int).tolist()


def proxy_rewards(rm_scores):
    """Cheap proxy reward per response for an *unverified* group: 1 if its reward-model score is
    above the group median (top half), strict '>' to match E09_judge_task.py's
    `R["proxy"] = (R.rm_score > med)`."""
    rm = np.asarray(rm_scores, dtype=float)
    med = np.median(rm)
    return (rm > med).astype(int).tolist()


def mean_pairwise_cosine(unit_vectors):
    """Mean pairwise cosine among G already-L2-normalized embeddings; mirrors
    E09_judge_task.py's mean_pairwise_cos. NaN for a group of size < 2 (never happens at G>=2)."""
    E = np.asarray(unit_vectors, dtype=float)
    n = len(E)
    if n < 2:
        return float("nan")
    S = E @ E.T
    return float((S.sum() - n) / (n * (n - 1)))


def top_two_gap(rm_scores):
    """Top-two reward-model score gap within a group: this task's analogue of GSM8K's majority
    `agreement` fraction (both are a 'how confident is the cheap signal' scalar). Used both as a
    cheap feature (rm_gap12 below) and, in p1_judge_grpo.py, as the value allocators.py's
    ThresholdAllocator sorts on (smallest gap = verified first, per EXPERIMENT_BACKLOG.md's E09
    section: 'verifies the groups with the smallest top-two reward-model gap')."""
    rm = np.sort(np.asarray(rm_scores, dtype=float))[::-1][:2]
    return float(rm[0] - rm[1]) if len(rm) >= 2 else 0.0


def cheap_feature_vector(rm_scores, lengths, truncated, q_len, consistency, pcs):
    """[rm_max, rm_mean, rm_std, rm_gap12 (top-two gap), mean_len, std_len, trunc_frac, q_len,
    consistency] + the 16 prompt PCA components (pcs, in p1_grpo.py's PC_COLS order). Matches
    EXPERIMENT_BACKLOG.md's E09 cheap-feature list exactly. Gold never appears here -- see the
    module docstring's fairness note."""
    rm = np.asarray(rm_scores, dtype=float)
    cheap = [
        float(rm.max()), float(rm.mean()), float(rm.std()), top_two_gap(rm_scores),
        float(np.mean(lengths)), float(np.std(lengths)), float(np.mean(truncated)),
        float(q_len), float(consistency),
    ]
    return cheap + [float(x) for x in pcs]


class MockRewardModel:
    """Deterministic, model-free stand-in for Skywork-Reward-V2-Qwen3-0.6B: score is a stable
    hash of the response text (sha256, not Python's randomized str hash) so repeated calls on
    the same text agree across processes, with no torch/transformers dependency -- lets
    judge_rewards.py's math be unit-tested on a machine (or in CI) with neither installed.
    Pass score_fn to control scores exactly in a test."""

    def __init__(self, score_fn=None):
        self.score_fn = score_fn

    def score(self, prompt, response):
        if self.score_fn is not None:
            return self.score_fn(prompt, response)
        return (_stable_hash(response) % 1000) / 100.0  # arbitrary but deterministic, no real meaning


class MockJudge:
    """Deterministic, model-free stand-in for the 7B/3B judge: score is 1-10 from a stable hash
    of the response text, always 'valid' (the unparseable path is exercised directly via
    parse_score() unit tests instead). Pass score_fn(prompt, response) -> (score, valid) to
    control scores exactly in a test."""

    name = "mock"

    def __init__(self, score_fn=None):
        self.score_fn = score_fn

    def score_batch(self, prompts, responses):
        out = []
        for p, r in zip(prompts, responses):
            if self.score_fn is not None:
                out.append(self.score_fn(p, r))
            else:
                out.append(((_stable_hash(("judge", r)) % 10) + 1, 1))
        return out


def _stable_hash(obj):
    import hashlib

    return int(hashlib.sha256(repr(obj).encode()).hexdigest(), 16)


class RewardModelScorer:
    """Skywork-Reward-V2-Qwen3-0.6B wrapped for single-pair scoring. bf16 on cuda, float32 on
    cpu -- mirrors avrl/exp/E09_judge_task.py's --stage score_rm exactly, including the
    bos-token-strip-after-chat-template step noted there."""

    def __init__(self, model_path, device):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, dtype=torch.bfloat16 if str(device).startswith("cuda") else torch.float32, num_labels=1
        ).to(device)
        self.model.eval()
        self.tok = AutoTokenizer.from_pretrained(model_path)

    def score(self, prompt, response):
        import torch

        conv = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
        text = self.tok.apply_chat_template(conv, tokenize=False)
        if self.tok.bos_token and text.startswith(self.tok.bos_token):
            text = text[len(self.tok.bos_token):]
        ids = self.tok(text, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
        with torch.no_grad():
            return self.model(**ids).logits[0][0].item()


class HFJudge:
    """Batched HF-generation judge: Qwen2.5-7B-Instruct 4-bit (bitsandbytes NF4) or
    Qwen2.5-{3,7}B-Instruct bf16, scoring with RUBRIC at temperature 0 (greedy), a few
    max_new_tokens (the rubric asks for a bare integer). Mirrors avrl/exp/E09_judge_task.py's
    judge stage, but via transformers.generate in-process (that file uses vLLM in a separate OS
    process) since this harness keeps the judge resident alongside the policy and reward model
    -- see p1_judge_grpo.py's memory-estimate comment."""

    def __init__(self, name, model_path, device, max_new_tokens=6, load_4bit=False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        if load_4bit:
            if not str(device).startswith("cuda"):
                raise ValueError("--judge 4bit-7b requires --judge_device cuda (bitsandbytes 4-bit needs a GPU)")
            from transformers import BitsAndBytesConfig

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"),
                device_map={"": int(str(device).split(":")[-1]) if ":" in str(device) else 0},
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, dtype=torch.bfloat16 if str(device).startswith("cuda") else torch.float32).to(device)
        self.model.eval()

    def score_batch(self, prompts, responses):
        """Scores every (prompt, response) pair. When self.batch_size is a positive int the pairs are
        scored in chunks of that size (a 14B bf16 judge on a 40 GB card cannot take a whole group of
        8 long responses at once); chunking does not change any score."""
        bs = getattr(self, "batch_size", 0) or 0
        if bs and len(prompts) > bs:
            out = []
            for i in range(0, len(prompts), bs):
                out.extend(self._score_chunk(prompts[i:i + bs], responses[i:i + bs]))
            return out
        return self._score_chunk(prompts, responses)

    def _score_chunk(self, prompts, responses):
        import torch

        texts = [
            self.tok.apply_chat_template(
                [{"role": "user", "content": RUBRIC.format(prompt=p, response=r)}],
                tokenize=False, add_generation_prompt=True,
            )
            for p, r in zip(prompts, responses)
        ]
        enc = self.tok(texts, return_tensors="pt", padding=True).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                pad_token_id=self.tok.pad_token_id,
            )
        gen = out[:, enc["input_ids"].shape[1]:]
        decoded = self.tok.batch_decode(gen, skip_special_tokens=True)
        return [parse_score(t) for t in decoded]


class MiniLMConsistency:
    """Loads all-MiniLM-L6-v2 once; .group_consistency(texts_per_group) -> one mean-pairwise-
    cosine float per group, in a single batched encode call. Same model p1_grpo.py's
    compute_prompt_feats() uses for the prompt-embedding cache, loaded as its own instance here
    since this one is called every training step (not once at startup) and stays resident."""

    def __init__(self, device="cpu"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    def group_consistency(self, texts_per_group):
        flat = [t for grp in texts_per_group for t in grp]
        emb = self.model.encode(flat, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
        out, i = [], 0
        for grp in texts_per_group:
            n = len(grp)
            out.append(mean_pairwise_cosine(emb[i:i + n]))
            i += n
        return out


def make_rm(mock, model_path, device):
    return MockRewardModel() if mock else RewardModelScorer(model_path, device)


def make_judge(judge, model_path, device, max_new_tokens=6):
    """judge in {'mock', '4bit-7b', 'bf16-3b', 'bf16-7b'}; see p1_judge_grpo.py's --judge help."""
    if judge == "mock":
        return MockJudge()
    if judge in ("4bit-7b", "bf16-3b", "bf16-7b"):
        return HFJudge(judge, model_path, device, max_new_tokens=max_new_tokens, load_4bit=(judge == "4bit-7b"))
    raise ValueError(f"unknown --judge {judge!r}")
