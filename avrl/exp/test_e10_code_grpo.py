"""Standalone CPU test for p1_code_grpo.py / code_rewards.py -- no pytest dependency, plain
asserts, same lightweight convention as the rest of avrl/ (no test framework in requirements.txt).
Three parts:
  1. A unit test of CodeRewardState.__call__ with a hand-written mock policy (fixed completion
     strings standing in for what GRPOTrainer would have generated), checking FAIRNESS RULES 1-3
     (see p1_grpo.py/allocators.py module docstrings) and that gold (the real-asserts `correct`
     flag) never reaches an unverified group's training reward, only its logging.
  2. The sandbox smoke test (three known-good, three known-bad programs) via
     code_rewards.verify_program -- already exercised interactively during development; re-run
     here so it is part of the one command that checks everything.
  3. A reminder of FAIRNESS RULES 4 and 5, which live in p1_code_grpo.py's main() (separate
     --train_data/--test_data files; one seeded shuffle independent of --arm) rather than in
     CodeRewardState, so they are not exercised by this file's mocked reward-function call --
     verified instead by direct inspection of main()'s source, same as p1_grpo.py's own rules.

Run: /homes/rdubash/active-audit/.venv/bin/python avrl/exp/test_e10_code_grpo.py   (cwd=avrl/)
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # avrl/

import numpy as np
import pandas as pd

import code_rewards as cr
from allocators import ALLOCATORS
from p1_grpo import PC_COLS
from p1_code_grpo import CodeRewardState


def make_feats(pids):
    """Minimal stand-in for p1_grpo.compute_prompt_feats()'s cached output: one row per pid with
    cell/q_len/pc_0..pc_15, indexed by pid -- the exact shape CodeRewardState.__init__ expects."""
    rows = []
    for i, pid in enumerate(pids):
        row = {"pid": pid, "cell": i % 3, "q_len": 40.0 + i}
        for j, c in enumerate(PC_COLS):
            row[c] = float(i + j) / 10.0
        rows.append(row)
    return pd.DataFrame(rows).set_index("pid")


def fenced(code):
    return code  # completions already continue an open fence; no closing ``` needed (extract_code handles both)


def test_reward_state_fairness():
    print("=== CodeRewardState unit test (hand-written mock policy, 2 groups x G=4) ===")
    G = 4
    pid0_tests = ["assert add(1, 2) == 3", "assert add(-1, 1) == 0"]
    pid1_tests = ["assert double(3) == 6", "assert double(0) == 0"]

    good_add = "def add(a, b):\n    return a + b"
    bad_add = "def add(a, b):\n    return a - b"   # wrong logic
    good_double = "def double(x):\n    return x * 2"
    bad_double = "def double(x):\n    return x + 2"  # wrong logic, but happens to be the MAJORITY signature

    # Group 0 (pid=0): 3-of-4 correct majority -> majority_sig is the CORRECT signature.
    group0 = [good_add, good_add, good_add, bad_add]
    # Group 1 (pid=1): 3-of-4 WRONG majority -> majority_sig is an INCORRECT signature
    # (exercises "corrupted_reward_groups" / gold-for-logging-only when this group is unverified).
    group1 = [bad_double, bad_double, bad_double, good_double]

    completions = [fenced(c) for c in group0 + group1]
    completion_ids = [list(range(len(c.split()))) or [0] for c in completions]  # length is all that matters
    pid_col = [0] * G + [1] * G
    test_list_col = [pid0_tests] * G + [pid1_tests] * G
    setup_col = [""] * (2 * G)

    feats = make_feats([0, 1])
    call_kwargs = dict(prompts=None, completions=completions, completion_ids=completion_ids,
                        pid=pid_col, test_list=test_list_col, test_setup_code=setup_col)

    with tempfile.TemporaryDirectory() as tmp:
        # --- fully-unverified branch (ProxyAllocator, n_verify always 0): reward is pure
        # sig-match-to-majority for every completion, including group 1, whose majority
        # signature is WRONG (bad_double matches 3-of-4, but fails its own asserts). This is the
        # "gold never reaches the (unverified) training reward" check -- bad_double still gets
        # reward 1.0 -- and corrupted_reward_groups must count it (gold used for logging only).
        log_path = os.path.join(tmp, "proxy.jsonl")
        state = CodeRewardState(ALLOCATORS["proxy"](), 0.25, G, feats, log_path, seed=0,
                                  max_completion_length=512, timeout=5.0)
        rewards = state(**call_kwargs)
        log_line = json.loads(open(log_path).readlines()[-1])
        assert log_line["expensive_calls"] == 0
        assert rewards[:G] == [1.0, 1.0, 1.0, 0.0], rewards[:G]      # good_add is majority & correct
        assert rewards[G:] == [1.0, 1.0, 1.0, 0.0], rewards[G:]      # bad_double is majority & WRONG
        assert log_line["corrupted_reward_groups"] == 1, log_line    # group 1 only
        assert log_line["proxy_error_rate_among_unverified"] == 0.5, log_line
        print(f"proxy (unverified) branch: rewards={rewards} corrupted={log_line['corrupted_reward_groups']} -- OK")

        # --- fully-verified branch (FullAllocator, n_verify always n_groups): reward is the
        # REAL correctness flag, observably different from the proxy case for group 1 (bad_double
        # x3 now correctly get reward 0.0, good_double gets 1.0) -- the "verified branch actually
        # uses gold" check, and its mirror image of the proxy-branch check above.
        log_path = os.path.join(tmp, "full.jsonl")
        state = CodeRewardState(ALLOCATORS["full"](), 0.25, G, feats, log_path, seed=0,
                                  max_completion_length=512, timeout=5.0)
        rewards = state(**call_kwargs)
        log_line = json.loads(open(log_path).readlines()[-1])
        assert log_line["expensive_calls"] == 2
        assert rewards[:G] == [1.0, 1.0, 1.0, 0.0], rewards[:G]      # same as proxy here: majority IS correct
        assert rewards[G:] == [0.0, 0.0, 0.0, 1.0], rewards[G:]      # DIFFERENT from proxy: majority was WRONG
        assert log_line["corrupted_reward_groups"] == 0, log_line    # nothing left unverified to flag
        assert log_line["proxy_error_rate_among_unverified"] is None, log_line
        print(f"full (verified) branch: rewards={rewards} -- OK")

        # --- FAIRNESS RULE 1 (safe_records key whitelist, no gold) and RULE 2 (n_verify is a
        # pure function of (n_groups, budget): calling the SAME state twice in a row -- step 0
        # and step 1 -- must give the identical expensive-call count, no carry-over/drift).
        log_path = os.path.join(tmp, "threshold.jsonl")
        allocator = ALLOCATORS["threshold"]()
        real_select = allocator.select
        seen_k = []

        def spying_select(records, k, rng, _real=real_select):
            for r in records:
                assert set(r.keys()) == {"idx", "pid", "majority_sig", "agreement", "n_distinct", "cell", "feat_vec"}, (
                    f"FAIRNESS RULE 1 violated: safe_record leaked a gold-derived key: {sorted(r.keys())}"
                )
            seen_k.append(k)
            return _real(records, k, rng)

        allocator.select = spying_select
        state = CodeRewardState(allocator, 0.5, G, feats, log_path, seed=0,
                                  max_completion_length=512, timeout=5.0)
        state(**call_kwargs)  # step 0
        state(**call_kwargs)  # step 1
        assert seen_k == [1, 1], (
            f"FAIRNESS RULE 2 violated: n_verify should be identical across steps at a fixed "
            f"(n_groups, budget); got {seen_k}"
        )
        print(f"FAIRNESS RULE 1 (no gold in safe_records) and RULE 2 (n_verify={seen_k}, no carry-over): OK")

        # --- FAIRNESS RULE 3 (LearnedAllocator.update() only ever sees groups verified THIS
        # step): spy on .update() across two steps and confirm each call gets exactly n_verify
        # items, never anything from the other step.
        log_path = os.path.join(tmp, "learned.jsonl")
        allocator = ALLOCATORS["learned"]()
        real_update = allocator.update
        seen_update_sizes = []

        def spying_update(verified, _real=real_update):
            seen_update_sizes.append(len(verified))
            return _real(verified)

        allocator.update = spying_update
        state = CodeRewardState(allocator, 0.5, G, feats, log_path, seed=0,
                                  max_completion_length=512, timeout=5.0)
        state(**call_kwargs)  # step 0: n_verify=1
        state(**call_kwargs)  # step 1: n_verify=1
        assert seen_update_sizes == [1, 1], (
            f"FAIRNESS RULE 3 violated: update() should see exactly this step's verified groups "
            f"each call, no accumulation of other steps' groups passed in by the caller; got {seen_update_sizes}"
        )
        print(f"FAIRNESS RULE 3 (update() sees only this step's verified groups: {seen_update_sizes}): OK")

        print("CodeRewardState fairness checks: OK")


def test_sandbox_known_good_bad():
    print("\n=== sandbox: 3 known-good + 3 known-bad programs ===")
    tests = ["assert add(1, 2) == 3", "assert add(-1, 1) == 0", "assert add(0, 0) == 0"]
    calls = [cr.parse_call(t) for t in tests]
    good = {
        "exact": "def add(a, b):\n    return a + b",
        "alt_var": "def add(a, b):\n    result = a + b\n    return result",
        "lambda": "add = lambda a, b: a + b",
    }
    bad = {
        "wrong_logic": "def add(a, b):\n    return a - b",
        "syntax_error": "def add(a, b):\n    return a +",
        "timeout": "def add(a, b):\n    while True:\n        pass",
    }
    for name, code in good.items():
        r = cr.verify_program(code, "", tests, calls, timeout=5.0)
        assert r["correct"] is True, f"expected {name!r} to pass all asserts, got {r}"
        print(f"good/{name}: correct={r['correct']} sig={r['sig']}")
    for name, code in bad.items():
        r = cr.verify_program(code, "", tests, calls, timeout=5.0)
        assert r["correct"] is False, f"expected {name!r} to fail, got {r}"
        print(f"bad/{name}: correct={r['correct']} sig={r['sig']} syntax_error={r['syntax_error']} exec_exception={r['exec_exception']}")
    print("sandbox good/bad checks: OK")


if __name__ == "__main__":
    test_reward_state_fairness()
    test_sandbox_known_good_bad()
    print("\nALL TESTS PASSED")
