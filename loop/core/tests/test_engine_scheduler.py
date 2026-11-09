"""scheduler.py: priors, Thompson sampling, posterior updates, dedup, stagnation and memo cadence."""
import random
import sys

import pytest

import scheduler as S


def item(i, alpha=1, beta=1, tried=0, status="open", family="optimizer", title=None, spec="s"):
    return {"id": "i%d" % i, "title": title or "t%d" % i, "spec": spec, "family": family, "alpha": alpha,
            "beta": beta, "times_tried": tried, "status": status}


def test_seed_prior_is_one_plus_votes():
    assert S.seed_prior(["yes", "yes", "no"]) == (3, 2)
    assert S.seed_prior(["no", "no", "no"]) == (1, 4)
    assert S.seed_prior(["Yes", True, False]) == (3, 2)
    assert S.seed_prior([]) == (1, 1)


def test_make_item_defaults():
    it = S.make_item("i001", "t", "spec", "schedule", ["yes", "no", "no"])
    assert (it["alpha"], it["beta"], it["times_tried"], it["status"]) == (2, 3, 0, "open")
    assert S.make_item("i002", "t", "s", "not-a-family")["family"] == "other"


def test_thompson_is_deterministic_for_a_seed_and_prefers_strong_posteriors():
    items = [item(1, 1000, 1), item(2, 1, 1000), item(3, 5, 5)]
    picks = [S.thompson_pick(items, random.Random(s))["id"] for s in range(20)]
    assert picks == [S.thompson_pick(items, random.Random(s))["id"] for s in range(20)]
    assert set(picks) == {"i1"}                       # score ~0.999 against ~0.001 and ~0.5


def test_times_tried_discounts_the_score():
    # equal posteriors (~0.999), but the tried one is divided by 1 + 3
    items = [item(1, 1000, 1, tried=3), item(2, 1000, 1, tried=0)]
    assert {S.thompson_pick(items, random.Random(s))["id"] for s in range(20)} == {"i2"}


def test_only_open_items_are_picked():
    items = [item(1, status="done"), item(2, status="dropped")]
    assert S.thompson_pick(items, random.Random(0)) is None
    items.append(item(3))
    assert S.thompson_pick(items, random.Random(0))["id"] == "i3"
    assert S.untried_count(items) == 1


def test_widen_samples_uniformly_from_the_top_five():
    items = [item(i, 1000, 1) for i in range(1, 6)] + [item(i, 1, 1000) for i in range(6, 9)]
    picks = {S.thompson_pick(items, random.Random(s), widen=True)["id"] for s in range(60)}
    assert picks <= {"i1", "i2", "i3", "i4", "i5"}
    assert len(picks) >= 4                            # not the argmax every time
    few = [item(1), item(2)]
    assert S.thompson_pick(few, random.Random(0), widen=True)["id"] in ("i1", "i2")


def test_update_moves_item_and_family_posteriors():
    fam, it = {}, item(1, 3, 2, family="schedule")
    S.update(it, fam, accepted=True)
    assert (it["alpha"], it["beta"], it["times_tried"], it["status"]) == (4, 2, 1, "done")
    assert fam == {"schedule": {"alpha": 2, "beta": 1}}
    it2 = item(2, family="schedule")
    S.update(it2, fam, accepted=False)
    assert (it2["alpha"], it2["beta"]) == (1, 2)
    assert fam["schedule"] == {"alpha": 2, "beta": 2}


def test_untested_ideas_stay_open_until_three_tries():
    it = item(1)
    S.note_untested(it)
    S.note_untested(it)
    assert it["status"] == "open" and it["times_tried"] == 2 and (it["alpha"], it["beta"]) == (1, 1)
    S.note_untested(it)
    assert it["status"] == "dropped"


def test_ngram_dedup_catches_near_duplicates_only():
    d = S.Dedup("ngram")
    assert d.mode == "ngram"
    a = "Raise matrix learning rate. Increase MATRIX_LR from 0.04 to 0.05 for faster early progress."
    b = "Raise matrix learning rate. Increase MATRIX_LR from 0.04 to 0.06 for faster early progress."
    c = "Add value embeddings. Give every layer its own value embedding table to carry token identity."
    assert d.similar(a, b) and not d.similar(a, c)
    existing = [{"title": "Raise matrix learning rate", "spec": a.split(". ", 1)[1]}]
    cands = [{"title": "Raise matrix learning rate", "spec": b.split(". ", 1)[1], "family": "optimizer"},
             {"title": "Add value embeddings", "spec": c.split(". ", 1)[1], "family": "architecture"},
             {"title": "Add value embeddings", "spec": c.split(". ", 1)[1], "family": "architecture"}]
    kept, dropped = S.dedup_new(existing, cands, d)
    assert len(kept) == 1 and len(dropped) == 2        # duplicate of existing, and duplicate within the batch


def test_dedup_auto_falls_back_to_ngram_when_embeddings_are_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)    # import raises ImportError
    d = S.Dedup("auto")
    assert d.mode == "ngram"
    with pytest.raises(Exception):
        S.Dedup("embedding")


def test_stagnation_and_memo_cadence():
    assert not S.should_widen(4) and S.should_widen(5) and S.should_widen(9)
    assert not S.memo_due(0) and not S.memo_due(4) and S.memo_due(5) and not S.memo_due(6) and S.memo_due(10)
