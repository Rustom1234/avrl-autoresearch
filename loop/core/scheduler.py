"""Backlog scheduler for arms C and D (DESIGN.md section 3, arm C).

Items are plain dicts so they live in state.json:
  {id, title, spec, family, alpha, beta, times_tried, status}   status: open | done | dropped
Beta(1 + yes, 1 + no) prior from cheap votes, Thompson sampling with score sample / (1 + times_tried), embedding
dedup (cosine >= 0.90) with a character 3-gram Jaccard fallback (>= 0.6), stagnation widening, memo cadence.
Every random draw comes from the caller's random.Random, so a seeded run is deterministic.
"""
import re

FAMILIES = ("optimizer", "schedule", "architecture", "batch", "regularization", "data", "other")
# Family to linter edit class, used only to route the coder in arm D (the real class is measured after the edit).
FAMILY_CLASS = {"schedule": "schedule", "optimizer": "constant", "batch": "constant", "regularization": "constant",
                "architecture": "structural", "data": "structural", "other": "structural"}
TOP_K_WIDEN = 5
STAGNATION_K = 5
MEMO_EVERY = 5
COSINE_MIN = 0.90
JACCARD_MIN = 0.6
MAX_PRE_GPU_TRIES = 3


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
    alpha, beta = seed_prior(votes)
    return {"id": item_id, "title": title, "spec": spec, "family": family if family in FAMILIES else "other",
            "alpha": alpha, "beta": beta, "times_tried": 0, "status": "open"}


def open_items(items):
    return [i for i in items if i["status"] == "open"]


def untried_count(items):
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
    return consecutive_rejections >= k


def memo_due(n_experiments, every=MEMO_EVERY):
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
        t = re.sub(r"\s+", " ", text.lower()).strip()
        return {t[i:i + 3] for i in range(max(1, len(t) - 2))}

    def _vec(self, text):
        if text not in self._cache:
            self._cache[text] = self._model.encode(text, normalize_embeddings=True)
        return self._cache[text]

    def similarity(self, a, b):
        if self.mode == "embedding":
            return float(self._vec(a) @ self._vec(b))
        ga, gb = self._grams(a), self._grams(b)
        return len(ga & gb) / len(ga | gb)

    def similar(self, a, b):
        return self.similarity(a, b) >= (COSINE_MIN if self.mode == "embedding" else JACCARD_MIN)


def item_text(item):
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
