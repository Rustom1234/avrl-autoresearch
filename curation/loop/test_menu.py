"""The no-model menu loop: the moves, the generated file, and the search on a fake trainer."""
import json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine as E
import menu as M


class CfgRunner:
    """Fake trainer: dev bpb 2.00, and 0.05 better for every setting in `good` that the file has."""
    def __init__(self, good=()):
        self.good, self.log = dict(good), []

    def __call__(self, path, seed, tag):
        src = Path(path).read_text()
        m = re.search(r"^CFG = (.*)$", src, re.M)
        cfg = eval(m.group(1)) if m else dict(M.START)
        self.log.append((tag, cfg))
        v = 2.00 - 0.05 * sum(cfg[k] == val for k, val in self.good.items())
        return {"dev_bpb": v, "sealed_bpb": v + 0.02, "n_selected": 5, "curator_tokens_before_cycle": E.T}, None


def loop(tmp_path, slots=12, runner=None, seed=1):
    cfg = {"loop_id": "N", "run_dir": str(tmp_path), "cell": "N", "pool": "v2", "slots": slots, "seed": seed,
           "sigma": 0.02, "gpu_host": None, "policy": "menu", "routing": None}
    return M.MenuLoop(cfg, runner=runner or CfgRunner())


def ledger(tmp_path):
    return [json.loads(l) for l in open(tmp_path / "ledger.jsonl")]


def test_moves():
    assert M.moves_of(M.START) == [["dedup", True], ["junk", True], ["rank", "word"], ["rank", "char"]]
    mv = M.moves_of(dict(M.START, rank="word", keep=1.5))
    assert ["keep", 1.0] in mv and len([m for m in mv if m[0] == "keep"]) == 1 and len(mv) == 5


def test_every_generated_file_passes_the_lint_and_selects():
    docs = ["alpha beta gamma delta " * 40, "alpha beta gamma delta " * 40, "x" * 100, "the cat sat on the mat " * 30,
            "wiki text about history and rivers " * 30, "rivers of history, alpha and gamma " * 30,
            "a cat, a mat, a delta and some wiki text " * 30]
    for rank in M.RANKS:
        for flag in (False, True):
            src = M.render({"dedup": flag, "junk": flag, "rank": rank, "keep": 0.5})
            assert E.lint(src) is None
            ns = {}
            exec(compile(src, "curate.py", "exec"), ns)
            idx = ns["select"](docs, ["history of rivers and wiki text", "the mat and the cat"])
            assert idx and set(idx) <= set(range(len(docs)))
            if flag:
                assert 2 not in idx and not {0, 1} <= set(idx)        # junk dropped, one of the duplicates dropped


def test_no_move_helps_nothing_repeats_until_the_list_is_exhausted(tmp_path):
    lp = loop(tmp_path, slots=6)
    lp.run()
    recs = ledger(tmp_path)
    assert [r["decision"] for r in recs] == ["reject"] * 6
    ideas = [r["idea"] for r in recs]
    assert len(set(ideas[:4])) == 4                                # the four moves of the start, each once
    assert all(c["usd"] == 0 and c["model"] == "menu" for r in recs for c in r["calls"])
    assert json.loads((tmp_path / "final.json").read_text())["model_usd"] == 0


def test_good_moves_are_kept_and_stack(tmp_path):
    lp = loop(tmp_path, slots=12, runner=CfgRunner(good={"junk": True, "rank": "char"}))
    lp.run()
    cfg = lp.s["menu"]["config"]
    assert cfg["junk"] is True and cfg["rank"] == "char"
    assert sum(r["decision"] == "accept" for r in ledger(tmp_path)) >= 2
    assert "CFG = " in Path(lp.s["incumbent"]).read_text()


def test_resume_gives_the_same_search(tmp_path):
    ref = loop(tmp_path / "ref", slots=8, runner=CfgRunner(good={"dedup": True}))
    ref.run()
    a = loop(tmp_path / "cut", slots=8, runner=CfgRunner(good={"dedup": True}))
    for _ in range(3):
        a.run_slot()
    b = loop(tmp_path / "cut", slots=8, runner=CfgRunner(good={"dedup": True}))
    b.run()
    assert [r["idea"] for r in ledger(tmp_path / "cut")] == [r["idea"] for r in ledger(tmp_path / "ref")]
    assert b.s["menu"]["config"] == ref.s["menu"]["config"]
