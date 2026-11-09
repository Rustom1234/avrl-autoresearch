"""Backlog scheduler for arms C and D (DESIGN.md section 3, arm C).

In those arms the loop keeps a backlog of ideas (proposed edits to train.py, written by the planner role) and has to
choose which one gets the next slot (one GPU training run). This module is that choice, as pure functions over plain
data; the engine (arms.py) calls it and owns all I/O.

Items are plain dicts so they live in state.json:
  {id, title, spec, family, alpha, beta, times_tried, status}   status: open | done | dropped
Each item carries a Beta(alpha, beta) belief about "this idea would be accepted": the prior is Beta(1 + yes, 1 + no)
from cheap voter-model votes, and a GPU-evaluated outcome adds one to alpha (accepted) or beta (rejected).
Thompson sampling picks the next idea: draw one sample from every open item's Beta and take the highest, which
tries uncertain ideas sometimes and promising ones often with no exploration parameter to tune. The score is the
sample / (1 + times_tried), so an idea that already failed before reaching the GPU sinks.
Also here: near-duplicate removal (embedding cosine >= 0.90, with a character 3-gram Jaccard fallback >= 0.6),
stagnation widening (after a run of rejections, pick more broadly) and the memo cadence.
Every random draw comes from the caller's random.Random, so a seeded run is deterministic.
"""
import re

# Idea families: a coarse label the planner puts on each idea. Outcomes are also pooled per family (family_stats).
FAMILIES = ("optimizer", "schedule", "architecture", "batch", "regularization", "data", "other")
# Family to linter edit class, used only to route the coder in arm D (the real class is measured after the edit).
FAMILY_CLASS = {"schedule": "schedule", "optimizer": "constant", "batch": "constant", "regularization": "constant",
                "architecture": "structural", "data": "structural", "other": "structural"}
TOP_K_WIDEN = 5         # when widening, pick uniformly among this many top-scored items
STAGNATION_K = 5        # consecutive rejections that count as stagnation and trigger widening
MEMO_EVERY = 5          # the analyst rewrites the memo after every this many decided experiments
COSINE_MIN = 0.90       # embedding cosine at or above this: near duplicate
JACCARD_MIN = 0.6       # same test for the character 3-gram fallback (Jaccard overlap of the two gram sets)
MAX_PRE_GPU_TRIES = 3   # attempts that never reached the GPU before an item is dropped for good


def seed_prior(votes):
    """votes: iterable of "yes"/"no" (case-insensitive) or booleans. Returns (alpha, beta) = (1 + yes, 1 + no)."""
    yes = no = 0
    for v in votes:
        if v is True or str(v).strip().lower() == "yes":
            yes += 1
        else:
            no += 1
    return 1 + yes, 1 + no


def make_item(item_id, title, spec, family, votes=()):
    """New open backlog item with its prior seeded from votes. An unknown family becomes "other"."""
    alpha, beta = seed_prior(votes)
    return {"id": item_id, "title": title, "spec": spec, "family": family if family in FAMILIES else "other",
            "alpha": alpha, "beta": beta, "times_tried": 0, "status": "open"}


def open_items(items):
    """Items still eligible to be picked."""
    return [i for i in items if i["status"] == "open"]


def untried_count(items):
    """Open items never attempted; the engine asks the planner for more when this runs low."""
    return sum(1 for i in items if i["status"] == "open" and i["times_tried"] == 0)


def thompson_pick(items, rng, widen=False):
    """Pick among open items. Score = Beta(alpha, beta) sample / (1 + times_tried); argmax, or with widen=True a
    uniform choice among the top 5 scores. Returns None when no item is open."""
    scored = [(rng.betavariate(i["alpha"], i["beta"]) / (1 + i["times_tried"]), n, i)
              for n, i in enumerate(open_items(items))]
    if not scored:
        return None
    scored.sort(key=lambda t: (-t[0], t[1]))
    if widen:
        return rng.choice(scored[:TOP_K_WIDEN])[2]
    return scored[0][2]


def update(item, family_stats, accepted):
    """Record a GPU-evaluated outcome (accepted or not) on the item and on its family. Infra failures, and ideas that
    never reached the GPU, must not call this."""
    key = "alpha" if accepted else "beta"
    item[key] += 1
    fam = family_stats.setdefault(item["family"], {"alpha": 1, "beta": 1})
    fam[key] += 1
    item["times_tried"] += 1
    item["status"] = "done"


def note_untested(item):
    """The idea was dropped before any GPU run (abort, patch or lint failure, veto). It stays open with a smaller score
    and is dropped for good after MAX_PRE_GPU_TRIES such attempts. No posterior change."""
    item["times_tried"] += 1
    if item["times_tried"] >= MAX_PRE_GPU_TRIES:
        item["status"] = "dropped"


def should_widen(consecutive_rejections, k=STAGNATION_K):
    """True when the loop is stagnating: k or more rejections in a row."""
    return consecutive_rejections >= k


def memo_due(n_experiments, every=MEMO_EVERY):
    """True after every `every`-th decided experiment (never at zero)."""
    return n_experiments > 0 and n_experiments % every == 0


class Dedup:
    """similar(a, b): True when two idea texts are near duplicates. mode is "embedding" when sentence-transformers
    (all-MiniLM-L6-v2, loaded from the local cache only) works, else "ngram"; the mode actually used is in .mode.
    Force a mode with mode="ngram" (tests do) or mode="embedding" (raises if unavailable)."""

    def __init__(self, mode="auto"):
        self.mode, self._model, self._cache = "ngram", None, {}
        if mode in ("auto", "embedding"):
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
                self.mode = "embedding"
            except Exception:
                if mode == "embedding":
                    raise

    @staticmethod
    def _grams(text):
        """Set of character 3-grams of the lower-cased, whitespace-collapsed text."""
        t = re.sub(r"\s+", " ", text.lower()).strip()
        return {t[i:i + 3] for i in range(max(1, len(t) - 2))}

    def _vec(self, text):
        """Unit-length embedding of a text, cached, so a dot product of two of them is their cosine."""
        if text not in self._cache:
            self._cache[text] = self._model.encode(text, normalize_embeddings=True)
        return self._cache[text]

    def similarity(self, a, b):
        """Cosine of the two embeddings, or in ngram mode the Jaccard overlap of the 3-gram sets (shared grams over
        all grams). Higher means more alike."""
        if self.mode == "embedding":
            return float(self._vec(a) @ self._vec(b))
        ga, gb = self._grams(a), self._grams(b)
        return len(ga & gb) / len(ga | gb)

    def similar(self, a, b):
        """True at or above the threshold of the active mode."""
        return self.similarity(a, b) >= (COSINE_MIN if self.mode == "embedding" else JACCARD_MIN)


def item_text(item):
    """The text an item is compared by: title plus spec."""
    return "%s. %s" % (item["title"], item["spec"])


def dedup_new(existing, candidates, dedup):
    """Split candidate item dicts (title, spec, family) into (kept, dropped). An idea is dropped when it is similar to
    anything already in `existing` (any status, so tried ideas are not re-proposed) or to an earlier kept candidate."""
    texts = [item_text(i) for i in existing]
    kept, dropped = [], []
    for c in candidates:
        t = item_text(c)
        if any(dedup.similar(t, other) for other in texts):
            dropped.append(c)
        else:
            kept.append(c)
            texts.append(t)
    return kept, dropped
