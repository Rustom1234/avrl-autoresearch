"""Curator: junk filtering + presence-only vocabulary filter against the target sample.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Pipeline:
  1. Drop empty / degenerate documents and exact (normalised) duplicates.
  2. Junk filtering: drop documents whose surface statistics look damaged or
     spammy -- high ratios of symbols / non-alphanumeric characters, control
     or replacement characters, non-ASCII bytes, digits, long runs of one
     repeated character ("!!!!", "......"), URL spam ("www.xxx"), very long
     unbroken tokens, repeated lines, or highly compressible (repetitive)
     text. Each threshold has a conservative default and is relaxed
     automatically if the target sample itself shows high values for that
     feature, so genuine target-like text is never filtered by a rule the
     target domain would fail.
  3. Score each remaining document on the *set* of distinct words in its
     first N_WORDS words (presence only, no frequency ratios):
       - miss fraction: share of its distinct words that never occur in the
         target sample, with a leave-one-out tolerance; excess is penalised.
       - boost: share of its distinct words that are high-frequency target
         words (words occurring in many target documents).
  4. Walk the documents from highest score down, counting tokens with the
     tiktoken "gpt2" encoder, until the budget of T tokens is filled.
  5. Shuffle the chosen documents (seeded by CURATE_SEED) so training sees a
     mixed stream; the unchosen (non-junk) documents follow in score order.
"""
import os
import re
import zlib
import random
import hashlib
from collections import Counter

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830_400

N_WORDS = 400          # words per document used for scoring
CHAR_CAP = 6000        # characters per document scanned for those words
HF_MIN_DOC_FRAC = 0.05  # a word is "high-frequency" if in >= this share of target docs
HF_MIN_DOCS = 3         # ... and in at least this many target docs
TOL_QUANTILE = 0.75     # quantile of leave-one-out target miss rates used as tolerance
TOL_LO, TOL_HI = 0.02, 0.60
EXCESS_PENALTY = 6.0    # per unit of miss fraction beyond the tolerance
BASE_PENALTY = 0.5      # mild slope on miss fraction inside the tolerance
BOOST_WEIGHT = 1.0      # weight of the high-frequency-target-word share
BUDGET_MARGIN = 1.03   # select slightly more than T to absorb count mismatches
CHUNK = 2000           # documents per tokeniser batch

# ---- junk filter settings ----
# Default upper limits per feature (a document above any limit is junk).
JUNK_DEFAULTS = {
    "sym": 0.25,      # symbols (non-alphanumeric, non-space) / non-space chars
    "nonletter": 0.50,  # (symbols + digits) / non-space chars
    "digit": 0.30,    # digits / non-space chars
    "nonascii": 0.30,  # non-ASCII chars / chars
    "ctrl": 0.01,     # control / replacement chars / chars
    "runs": 0.05,     # chars inside long runs of one repeated char / chars
    "url": 0.05,      # URL-like markers per whitespace token
    "longtok": 0.20,  # chars in tokens longer than LONG_TOKEN / chars
    "duplines": 0.40,  # share of (non-trivial) lines that are repeats
    "redund": 0.78,   # 1 - zlib compressed size / raw size
}
JUNK_TARGET_QUANTILE = 0.97   # target-sample quantile used to relax limits
JUNK_TARGET_SLACK = 1.25      # multiplicative slack on that quantile
LONG_TOKEN = 40
MIN_COMPRESS_BYTES = 300

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
_WS_RE = re.compile(r"\s+")
_SYM_RE = re.compile(r"[^\w\s]|_", re.UNICODE)
_DIGIT_RE = re.compile(r"\d", re.UNICODE)
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufffd]")
_RUN_RE = re.compile(r"([^\w\s]|_)\1{2,}|([^\W_])\2{4,}", re.UNICODE)
_URL_RE = re.compile(r"https?://|www\.|\.com\b|\.net\b|\.org\b", re.IGNORECASE)


def _words(text):
    return _WORD_RE.findall(text[:CHAR_CAP].lower())[:N_WORDS]


def _junk_features(text):
    """Surface statistics of a document prefix; higher always means junkier."""
    s = text[:CHAR_CAP]
    n = len(s)
    if n == 0:
        return None
    toks = s.split()
    n_tok = len(toks)
    nonspace = sum(len(t) for t in toks)
    if nonspace == 0:
        return None
    sym = len(_SYM_RE.findall(s))
    dig = len(_DIGIT_RE.findall(s))
    nonascii = n - len(s.encode("ascii", "ignore"))
    ctrl = len(_CTRL_RE.findall(s))
    runs = 0
    for m in _RUN_RE.finditer(s):
        runs += m.end() - m.start()
    urls = len(_URL_RE.findall(s))
    longtok = sum(len(t) for t in toks if len(t) > LONG_TOKEN)

    lines = [l.strip() for l in s.split("\n")]
    lines = [l for l in lines if len(l) >= 10]
    if len(lines) >= 5:
        duplines = 1.0 - len(set(lines)) / len(lines)
    else:
        duplines = 0.0

    raw = s.encode("utf-8", "ignore")
    if len(raw) >= MIN_COMPRESS_BYTES:
        redund = 1.0 - len(zlib.compress(raw, 1)) / len(raw)
    else:
        redund = 0.0

    return {
        "sym": sym / nonspace,
        "nonletter": (sym + dig) / nonspace,
        "digit": dig / nonspace,
        "nonascii": nonascii / n,
        "ctrl": ctrl / n,
        "runs": runs / n,
        "url": urls / max(1, n_tok),
        "longtok": longtok / n,
        "duplines": duplines,
        "redund": redund,
    }


def _junk_limits(target_sample):
    """Default limits, relaxed wherever the target sample itself runs high."""
    limits = dict(JUNK_DEFAULTS)
    feats = []
    for s in target_sample:
        if isinstance(s, str) and s:
            f = _junk_features(s)
            if f is not None:
                feats.append(f)
    if len(feats) >= 5:
        for k in limits:
            vals = sorted(f[k] for f in feats)
            q = vals[min(len(vals) - 1, int(JUNK_TARGET_QUANTILE * len(vals)))]
            relaxed = q * JUNK_TARGET_SLACK + 1e-6
            if relaxed > limits[k]:
                limits[k] = relaxed
    return limits


def _is_junk(text, limits):
    f = _junk_features(text)
    if f is None:
        return True
    for k, lim in limits.items():
        if f[k] > lim:
            return True
    return False


def _make_counter():
    """Return a function mapping a list of strings to a list of token counts."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")

        def count(texts):
            try:
                return [len(x) for x in enc.encode_ordinary_batch(texts, num_threads=8)]
            except Exception:
                out = []
                for t in texts:
                    try:
                        out.append(len(enc.encode_ordinary(t)))
                    except Exception:
                        out.append(max(1, len(t) // 4))
                return out

        return count
    except Exception:
        return lambda texts: [max(1, len(t) // 4) for t in texts]


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)
    if n == 0:
        return []

    # ---- target statistics (presence only: document frequencies) ----
    t_df = Counter()
    t_sets = []
    t_lens = []
    for s in target_sample:
        w = _words(s) if isinstance(s, str) else []
        t_lens.append(len(w))
        if w:
            ws = set(w)
            t_sets.append(ws)
            t_df.update(ws)

    if not t_df:
        idx = list(range(n))
        rng.shuffle(idx)
        return idx

    n_t = len(t_sets)
    t_vocab = set(t_df)

    # high-frequency target words: present in many target documents
    hf_min = max(HF_MIN_DOCS, int(HF_MIN_DOC_FRAC * n_t + 0.999))
    hf_vocab = {w for w, c in t_df.items() if c >= hf_min}
    if not hf_vocab:
        # tiny target sample: fall back to the most widespread words
        hf_vocab = {w for w, _ in t_df.most_common(200)}

    # tolerance: how many unseen words does a genuine target doc have when
    # compared against the rest of the sample (leave-one-out)?
    loo = []
    for ws in t_sets:
        if len(ws) >= 5:
            miss = sum(1 for w in ws if t_df[w] <= 1)
            loo.append(miss / len(ws))
    if loo:
        loo.sort()
        tol = loo[min(len(loo) - 1, int(TOL_QUANTILE * len(loo)))]
    else:
        tol = 0.2
    tol = min(TOL_HI, max(TOL_LO, tol))

    t_lens.sort()
    # minimum words a document needs, adapted to how short target docs are
    min_words = max(3, min(20, (t_lens[len(t_lens) // 10] if t_lens else 20) // 2))

    # junk limits, adapted to the target sample
    try:
        junk_limits = _junk_limits(target_sample)
    except Exception:
        junk_limits = dict(JUNK_DEFAULTS)

    # ---- pool pass: dedupe, junk filter, score ----
    seen = set()
    scored = []
    for i, d in enumerate(docs):
        if not isinstance(d, str) or not d:
            continue
        key = hashlib.md5(
            _WS_RE.sub(" ", d).strip().lower().encode("utf-8", "ignore")
        ).digest()
        if key in seen:
            continue
        seen.add(key)
        w = _words(d)
        if len(w) < min_words:
            continue
        ws = set(w)
        # highly repetitive documents (boilerplate / junk)
        if len(w) >= 50 and len(ws) / len(w) < 0.2:
            continue
        # explicit junk filtering on surface statistics
        try:
            if _is_junk(d, junk_limits):
                continue
        except Exception:
            pass
        nt = len(ws)
        present = ws & t_vocab
        miss_frac = 1.0 - len(present) / nt
        hf_frac = len(present & hf_vocab) / nt
        excess = miss_frac - tol
        s = BOOST_WEIGHT * hf_frac - BASE_PENALTY * miss_frac
        if excess > 0:
            s -= EXCESS_PENALTY * excess
        scored.append((s, i))

    if not scored:
        idx = list(range(n))
        rng.shuffle(idx)
        return idx

    # deterministic ordering: score descending, index as tie-break
    scored.sort(key=lambda x: (-x[0], x[1]))
    ranked = [i for _, i in scored]

    # ---- fill the token budget from the top of the ranking ----
    count = _make_counter()
    budget = int(T * BUDGET_MARGIN)
    chosen = []
    total = 0
    pos = 0
    while pos < len(ranked) and total < budget:
        batch = ranked[pos:pos + CHUNK]
        lens = count([docs[i] for i in batch])
        for i, L in zip(batch, lens):
            if total >= budget:
                break
            chosen.append(i)
            total += L
            pos += 1

    if not chosen:
        idx = list(range(n))
        rng.shuffle(idx)
        return idx

    rest = ranked[len(chosen):]
    rng.shuffle(chosen)
    # If the best documents fall short of T, the remaining ones (still in score
    # order) top the stream up before any cycling happens.
    return chosen + rest
