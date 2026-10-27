"""calibration.py tests. No model is ever called: run() gets a fake call function and a fake health check."""
import json
import re
from dataclasses import dataclass, field

import pytest

import calibration as cal
import prompts

BASE = cal.BASE_PATH.read_text()
SPECS = cal.load_specs()
CODER = {s["id"]: s for s in SPECS["coder"]}


def sr(*pairs):
    """SEARCH/REPLACE reply from (old, new) pairs."""
    return "\n".join("<<<<<<< SEARCH\n%s\n=======\n%s>>>>>>> REPLACE" % (o, n + "\n" if n else "") for o, n in pairs)


def line(prefix):
    (hit,) = [x for x in BASE.splitlines() if x.startswith(prefix)]
    return hit


def const(prefix, old, new):
    ln = line(prefix)
    assert old in ln
    return ln, ln.replace(old, new, 1)


def plus(prefix, *added):
    """Keep the base line that starts with prefix and add lines after it."""
    ln = line(prefix)
    return ln, "\n".join((ln,) + added)


Q_INIT = "        self.ve_gate = nn.Linear("
ASSERT = ("        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +\n"
          "            len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params))")
ASSERT_PLUS = lambda extra: ASSERT[:-1] + " + len(%s))" % extra
X0_GROUP = line("            dict(kind='adamw', params=x0_params")

# One correct reply per coder spec: proves every spec is satisfiable under its own scoring and the linter.
REFERENCE = {
    "const-1": sr(const("EMBEDDING_LR = ", "0.6", "0.8")),
    "const-2": sr(const("DEPTH = ", "8", "9"), const("ASPECT_RATIO = ", "64", "57")),
    "const-3": sr(const("WINDOW_PATTERN = ", "SSSL", "SSSSL")),
    "const-4": sr(const("TOTAL_BATCH_SIZE = ", "2**19", "2**18")),
    "const-5": sr(const("WEIGHT_DECAY = ", "0.2", "0.1")),
    "const-6": sr(const("MATRIX_LR = ", "0.04", "0.05")),
    "const-7": sr(const("ADAM_BETAS = ", "(0.8, 0.95)", "(0.9, 0.95)")),
    "const-8": sr(const("FINAL_LR_FRAC = ", "0.0", "0.05")),
    "sched-1": sr(const("WARMUP_RATIO = ", "0.0", "0.05")),
    "sched-2": sr(("        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC",
                   "        t = 1.0 - cooldown\n"
                   "        return FINAL_LR_FRAC + (1 - FINAL_LR_FRAC) * 0.5 * (1 + math.cos(math.pi * t))")),
    "sched-3": sr(const("WARMUP_RATIO = ", "0.0", "0.02"), const("WARMDOWN_RATIO = ", "0.5", "0.0")),
    "sched-4": sr(("    frac = min(step / 300, 1)", "    frac = min(step / 500, 1)")),
    "sched-5": sr(("        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC",
                   "        if cooldown > 0.5:\n"
                   "            return 0.4 + 0.6 * (cooldown - 0.5) / 0.5\n"
                   "        return FINAL_LR_FRAC + (0.4 - FINAL_LR_FRAC) * cooldown / 0.5")),
    "sched-6": sr(const("FINAL_LR_FRAC = ", "0.0", "0.1")),
    "sched-7": sr(("    if progress < WARMUP_RATIO:\n        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0",
                   "    if step < 100:\n        return step / 100")),
    "sched-8": sr(const("WARMDOWN_RATIO = ", "0.5", "0.4")),
    "struct-1": sr(plus("        self.c_fc = nn.Linear(config.n_embd, 4", "        self.c_gate = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)"),
                   ("        x = self.c_fc(x)\n        x = F.relu(x).square()", "        x = F.silu(self.c_gate(x)) * self.c_fc(x)"),
                   plus("            torch.nn.init.uniform_(block.mlp.c_fc.weight", "            torch.nn.init.uniform_(block.mlp.c_gate.weight, -s, s)")),
    "struct-2": sr(("        q, k = norm(q), norm(k)", "")),
    "struct-3": sr(plus(Q_INIT, "        self.q_scale = nn.Parameter(torch.ones(self.n_head))"),
                   plus("        q, k = norm(q), norm(k)", "        q = q * self.q_scale.view(1, 1, -1, 1)"),
                   plus("            torch.nn.init.zeros_(block.attn.c_proj.weight)", "            block.attn.q_scale.fill_(1.0)"),
                   ("        matrix_params = list(self.transformer.h.parameters())",
                    "        matrix_params = [p for n, p in self.transformer.h.named_parameters() if not n.endswith(\"q_scale\")]\n"
                    "        q_scale_params = [b.attn.q_scale for b in self.transformer.h]"),
                   (ASSERT, ASSERT_PLUS("q_scale_params")),
                   (X0_GROUP, X0_GROUP + "\n            dict(kind='adamw', params=q_scale_params, lr=scalar_lr, betas=adam_betas, eps=1e-10, weight_decay=0.0),")),
    "struct-4": sr(("        x = F.relu(x).square()", "        x = F.gelu(x)")),
    "struct-5": sr(const("            dict(kind='adamw', params=value_embeds_params", "weight_decay=0.0", "weight_decay=0.01")),
    "struct-6": sr(("        softcap = 15", "        softcap = 12")),
    "struct-7": sr(plus("        self.lm_head = nn.Linear(config.n_embd, config.vocab_size", "        self.smear_gate = nn.Linear(24, 1, bias=False)"),
                   plus("        self.x0_lambdas.fill_(0.1)", "        torch.nn.init.zeros_(self.smear_gate.weight)"),
                   ("        x = norm(x)\n        x0 = x",
                    "        x = norm(x)\n        gate = torch.sigmoid(self.smear_gate(x[..., :24]))\n"
                    "        x = x + gate * F.pad(x[:, :-1], (0, 0, 1, 0))\n        x0 = x"),
                   plus("        x0_params = [self.x0_lambdas]", "        smear_params = list(self.smear_gate.parameters())"),
                   (ASSERT, ASSERT_PLUS("smear_params")),
                   (X0_GROUP, X0_GROUP + "\n            dict(kind='adamw', params=smear_params, lr=scalar_lr, betas=adam_betas, eps=1e-10, weight_decay=0.0),")),
    "struct-8": sr(("    return layer_idx % 2 == (n_layer - 1) % 2", "    return layer_idx >= 2 and layer_idx % 2 == (n_layer - 1) % 2")),
}


# ---------------------------------------------------------------- specs

def test_specs_shape():
    assert len(SPECS["coder"]) == 24
    for c in cal.CODER_CLASSES:
        assert sum(s["edit_class"] == c for s in SPECS["coder"]) == 8
    assert len(SPECS["repair"]) == len(SPECS["planner"]) == len(SPECS["analyst"]) == 6
    ids = [s["id"] for k in ("coder", "repair", "planner", "analyst") for s in SPECS[k]]
    assert len(ids) == len(set(ids))
    for s in SPECS["coder"] + SPECS["repair"]:
        assert s["spec"] and s["target_symbols"] and isinstance(s["must_not_touch"], list)
        for e in s["target_symbols"] + s["must_not_touch"] + s["expect"]:
            re.compile(cal._rx(e) if not e.startswith("!") else e[1:])
    assert chr(0x2014) not in (cal.HERE / "calib_specs.json").read_text() + (cal.HERE / "calibration.py").read_text()


def test_every_spec_has_a_reference_reply():
    assert set(REFERENCE) == set(CODER)


@pytest.mark.parametrize("spec_id", sorted(CODER))
def test_reference_reply_passes_its_spec(spec_id):
    reasons, _ = cal.score_edit(BASE, BASE, REFERENCE[spec_id], CODER[spec_id])
    assert reasons == []


@pytest.mark.parametrize("spec_id", sorted(CODER))
def test_noop_reply_fails_every_spec(spec_id):
    reasons, _ = cal.score_edit(BASE, BASE, sr(("import gc", "import gc")), CODER[spec_id])
    assert "noop" in reasons


def test_reference_replies_get_the_linters_edit_class():
    got = {s["id"]: cal.score_edit(BASE, BASE, REFERENCE[s["id"]], s)[1] for s in SPECS["coder"]}
    assert all(got[i] == "structural" for i in got if i.startswith("struct-"))
    assert all(got[i] in ("constant", "schedule") for i in got if i.startswith(("const-", "sched-")))
    assert got["sched-2"] == got["sched-7"] == "schedule"


# ---------------------------------------------------------------- score_edit negatives

def test_wrong_value_is_caught_by_expect():
    reasons, _ = cal.score_edit(BASE, BASE, sr(const("EMBEDDING_LR = ", "0.6", "0.5")), CODER["const-1"])
    assert reasons == ["expect:" + CODER["const-1"]["expect"][0]]


def test_sched8_inverted_ratio_fails():
    reasons, _ = cal.score_edit(BASE, BASE, sr(const("WARMDOWN_RATIO = ", "0.5", "0.6")), CODER["sched-8"])
    assert reasons and reasons[0].startswith("expect:")


def test_touching_a_forbidden_symbol_is_caught_ast_level():
    reply = REFERENCE["const-1"] + "\n" + sr(const("MATRIX_LR = ", "0.04", "0.045"))
    assert "touched:MATRIX_LR" in cal.score_edit(BASE, BASE, reply, CODER["const-1"])[0]
    # a body edit inside a protected function that never mentions its name is still a touch
    reply = REFERENCE["const-5"] + "\n" + sr(("    return (1 - frac) * 0.85 + frac * 0.95", "    return (1 - frac) * 0.80 + frac * 0.95"))
    spec = dict(CODER["const-5"], must_not_touch=["get_muon_momentum"])
    assert cal.score_edit(BASE, BASE, reply, spec)[0] == ["touched:get_muon_momentum"]


def test_comment_only_edit_to_protected_symbol_is_not_a_touch():
    reply = REFERENCE["const-1"] + "\n" + sr(const("MATRIX_LR = ", "(Muon)", "(Muon optimizer)"))
    assert cal.score_edit(BASE, BASE, reply, CODER["const-1"])[0] == []


def test_lint_parse_apply_failures():
    bad_import = REFERENCE["const-5"] + "\n" + sr(plus("import time", "import subprocess"))
    assert "lint:forbidden-import" in cal.score_edit(BASE, BASE, bad_import, CODER["const-5"])[0]
    assert cal.score_edit(BASE, BASE, sr(("WEIGHT_DECAY = 0.2", "WEIGHT_DECAY = (")), CODER["const-5"])[0][0].startswith("apply:")
    syntax = sr(("import gc", "import gc ("))
    assert cal.score_edit(BASE, BASE, syntax, CODER["const-5"])[0] == ["parse:SyntaxError"]
    assert cal.score_edit(BASE, BASE, "I would set WEIGHT_DECAY to 0.1.", CODER["const-5"])[0][0].startswith("apply:")


def test_abort_and_missing_edits_text_fail():
    assert cal.score_coder({"abort": True, "edits_text": REFERENCE["const-5"]}, BASE, BASE, CODER["const-5"])[0] == ["abort"]
    assert cal.score_coder({"description": "x"}, BASE, BASE, CODER["const-5"])[0][0].startswith("apply:")
    assert cal.score_coder({"edits_text": REFERENCE["const-5"]}, BASE, BASE, CODER["const-5"])[0] == []


# ---------------------------------------------------------------- repair specs

REPAIRS = {s["id"]: s for s in SPECS["repair"]}


@pytest.mark.parametrize("rid", sorted(REPAIRS))
def test_repair_spec_is_a_real_lint_failure_with_a_clean_fix(rid):
    s = REPAIRS[rid]
    broken, hard = cal.repair_inputs(BASE, s)
    assert s["expect_rule"] in [h["rule"] for h in hard]
    legit = cal.apply_edits(BASE, s["legit_edits"])
    assert cal.linter.lint(legit, BASE).ok
    (old, new), = s["violation_edits"]
    fix = sr((new.rstrip("\n"), old.rstrip("\n")))
    assert cal.score_edit(broken, BASE, fix, s)[0] == []
    # not repairing, or repairing by throwing the intended change away, both fail
    assert cal.score_edit(broken, BASE, "no change needed", s)[0][0].startswith("apply:")
    (lo, ln), = s["legit_edits"]
    undo = sr((new.rstrip("\n"), old.rstrip("\n")), (ln, lo))
    assert cal.score_edit(broken, BASE, undo, s)[0]


def test_repair_prompt_is_the_production_one_and_embeds_the_lint_output():
    s = REPAIRS["repair-1"]
    (rid, plan_spec, user, schema), = [(c, sp, u, sc) for c, sp, u, sc in cal.tasks(SPECS, ["repair"], BASE, 1)]
    assert plan_spec["id"] == "repair-1" and schema is prompts.SCHEMAS["repairer"]
    assert "forbidden-import" in user and "import subprocess" in user and "Static check failed" in user


# ---------------------------------------------------------------- planner / analyst scoring

def test_render_table_matches_the_ledger_layout_and_numbers():
    t = cal.render_table(SPECS["planner"][0])
    assert t.splitlines()[0] == "incumbent now: mean_adjusted 1.0689 over 1 run(s)"
    assert t.splitlines()[2] == "| slot | description | official | delta vs incumbent | decision | runs | flags |"
    assert "| 3 | WEIGHT_DECAY 0.2 -> 0.1 | 1.0689 | -0.0153 | accept | 1 | none |" in t
    assert cal.numbers("1. gain of -0.0153, runs 2, id e003, bf16, 2**19.") == {0.0153, 2.0, 19.0}
    assert cal.numbers("0.05 and 0.0153", min_decimals=4) == {0.0153}
    assert cal.check_numbers("a 0.0153 gain and 0.9 else", "table has 0.0153") == [0.9]


def good_planner(inp):
    return {"items": [{"title": f"idea {i}", "family": "optimizer", "spec": "set MATRIX_LR to 0.07, up from 0.04"}
                      for i in range(inp["n_items"])]}


def test_planner_scoring():
    (_, inp, prompt, _), = cal.tasks(SPECS, ["planner"], BASE, 1)
    assert cal.score_planner(good_planner(inp), inp, prompt) == []
    short = good_planner(inp)
    short["items"].pop()
    assert cal.score_planner(short, inp, prompt)[0].startswith("count:")
    invented = good_planner(inp)
    invented["items"][0]["spec"] = "since slot 3 reached 1.0612, push further"   # a result that is not in the table
    assert cal.score_planner(invented, inp, prompt) == ["numbers:1.0612"]
    invented["items"][0]["spec"] = "since slot 3 reached 1.0689, push further"   # a result that is
    assert cal.score_planner(invented, inp, prompt) == []
    dup = good_planner(inp)
    dup["items"][1]["title"] = dup["items"][0]["title"].upper()
    assert "duplicate:title" in cal.score_planner(dup, inp, prompt)
    empty = good_planner(inp)
    empty["items"][2]["spec"] = " "
    assert "empty:spec" in cal.score_planner(empty, inp, prompt)
    assert cal.score_planner({}, inp, prompt)[0].startswith("count:0")


def test_analyst_scoring():
    (_, inp, prompt, schema), = cal.tasks(SPECS, ["analyst"], BASE, 1)
    assert schema is prompts.SCHEMAS["analyst"]
    shown = inp["rows"][1]["official"]
    assert cal.score_analyst({"memo": f"Slot 2 reached {shown}. Two of {len(inp['rows'])} slots helped, runs 1."}, prompt) == []
    assert cal.score_analyst({"memo": "The delta is 0.0777 overall."}, prompt) == ["numbers:0.0777"]
    assert cal.score_analyst({"memo": "  "}, prompt) == ["empty:memo"]
    assert cal.score_analyst({}, prompt) == ["empty:memo"]
    assert cal.score_analyst({"memo": "word " * 400}, prompt) == ["long:memo"]
    assert cal.score_analyst({"memo": "1. best is %s\n2. next is %s" % (shown, shown)}, prompt) == []


# ---------------------------------------------------------------- Wilson, qualification

def test_wilson_values_and_edges():
    assert cal.wilson_lower(0, 0) == 0.0
    assert cal.wilson_lower(16, 16) == pytest.approx(0.806, abs=0.002)
    assert cal.wilson_lower(0, 16) == 0.0
    assert cal.wilson_lower(8, 16) < cal.wilson_lower(12, 16) < cal.wilson_lower(16, 16) < 1.0
    assert cal.wilson_lower(16, 16) < cal.wilson_lower(32, 32)


def test_thresholds_at_default_sample_counts():
    # constant and schedule: 8 specs x 2 samples = 16 trials, bar 0.6 needs 14/16
    assert cal.wilson_lower(14, 16) >= 0.6 > cal.wilson_lower(13, 16)
    # structural bar 0.75 needs a clean sweep of 16/16
    assert cal.wilson_lower(16, 16) >= 0.75 > cal.wilson_lower(15, 16)
    # repair, planner, analyst: 6 x 2 = 12 trials, bar 0.6 needs 11/12
    assert cal.wilson_lower(11, 12) >= 0.6 > cal.wilson_lower(10, 12)


def trials(model, cls, k, n, cost=0.01, infra=0):
    out = [dict(model=model, spec_id=f"{cls}-{i}", sample=0, edit_class=cls, ok=i < k, infra=False,
                reasons=[] if i < k else ["expect:x"], cost=cost, cost_plan=cost, latency=1.0) for i in range(n)]
    out += [dict(model=model, spec_id=f"{cls}-i{i}", sample=0, edit_class=cls, ok=False, infra=True,
                 reasons=["call:down"], cost=0.0, cost_plan=0.0, latency=0.0) for i in range(infra)]
    return out


def test_summarize_applies_bars_and_ignores_infra():
    ts = trials("a", "constant", 14, 16) + trials("a", "structural", 15, 16) + trials("b", "constant", 5, 5, infra=11)
    s = cal.summarize(ts)
    assert s["a"]["constant"]["qualified"] and not s["a"]["structural"]["qualified"]
    assert s["a"]["constant"]["bar"] == 0.6 and s["a"]["structural"]["bar"] == 0.75
    assert s["b"]["constant"]["n"] == 5 and s["b"]["constant"]["n_infra"] == 11 and not s["b"]["constant"]["qualified"]
    assert s["a"]["structural"]["fails"] == {"expect": 1}


PRICES = {"haiku": {"input": 1, "output": 5}, "sonnet": {"input": 3, "output": 15}, "opus": {"input": 15, "output": 75},
          "qwen35-9b": {"input": 0.1, "output": 0.3}, "qwen35-4b": {"input": 0.05, "output": 0.2},
          "gemma4-12b": {"input": 0.15, "output": 0.6}, "r1-qwen3-8b": {"input": 0.1, "output": 0.3}}


def table_for(passes):
    """passes: {(model, class): (k, n)}"""
    ts = [t for (m, c), (k, n) in passes.items() for t in trials(m, c, k, n)]
    res = {"trials": ts, "summary": cal.summarize(ts)}
    return cal.build_routing_table(res, PRICES)


def test_routing_cheapest_qualifying_and_sonnet_fallback():
    t = table_for({("qwen35-4b", "constant"): (16, 16), ("qwen35-9b", "constant"): (16, 16),
                   ("haiku", "schedule"): (16, 16), ("qwen35-9b", "schedule"): (12, 16),
                   ("sonnet", "structural"): (16, 16)})
    assert t["coder_by_class"] == {"constant": "qwen35-4b", "schedule": "haiku", "structural": "sonnet"}
    assert t["critic"] == "opus" and t["fallback"] == "sonnet"
    assert t["planner"] == t["planner_cheap"]
    # arms.py health-checks every top-level string value as a model name
    reg = cal.call_mod.load_registry()
    assert all(v in reg for v in t.values() if isinstance(v, str))
    assert t["repairer"] == "sonnet" and t["analyst"] == "sonnet" and t["planner_cheap"] == "sonnet"
    assert t["qualified"]["qwen35-9b"] == ["constant"]


def test_routing_structural_bar_is_stricter():
    t = table_for({("qwen35-9b", "structural"): (15, 16), ("qwen35-9b", "constant"): (15, 16),
                   ("haiku", "structural"): (16, 16)})
    assert t["coder_by_class"]["constant"] == "qwen35-9b"   # 15/16 clears 0.6
    assert t["coder_by_class"]["structural"] == "haiku"     # 15/16 misses 0.75
    assert not t["qualified"]["qwen35-9b"].count("structural")


def test_routing_ties_break_on_measured_cost_then_name():
    ts = trials("qwen35-9b", "repair", 12, 12, cost=0.01) + trials("r1-qwen3-8b", "repair", 12, 12, cost=0.05)
    t = cal.build_routing_table({"trials": ts, "summary": cal.summarize(ts)}, PRICES)
    assert t["repairer"] == "qwen35-9b"   # same nominal price, cheaper measured


def test_routing_voter_family_rule():
    # planner = qwen35-4b (cheapest); voter must not be qwen: the cheapest other-family qualifier is gemma, not qwen35-9b
    t = table_for({("qwen35-4b", "planner"): (12, 12), ("qwen35-9b", "planner"): (12, 12),
                   ("gemma4-12b", "planner"): (12, 12), ("r1-qwen3-8b", "planner"): (12, 12)})
    assert t["planner_cheap"] == "qwen35-4b"
    assert t["voter"] == "r1-qwen3-8b"   # deepseek family, cheaper than gemma (0.4 vs 0.75), not qwen
    t = table_for({("qwen35-4b", "planner"): (12, 12), ("qwen35-9b", "planner"): (12, 12), ("gemma4-12b", "planner"): (12, 12)})
    assert t["voter"] == "gemma4-12b"
    # only same-family qualifiers: sonnet (always a candidate, claude family) becomes the cross-family voter
    t = table_for({("qwen35-4b", "planner"): (12, 12), ("qwen35-9b", "planner"): (12, 12)})
    assert t["voter"] == "sonnet"


def test_routing_voter_without_cross_family_candidate_is_flagged():
    t = table_for({})   # nothing qualified: planner is sonnet, the only candidate
    assert t["planner_cheap"] == t["voter"] == "sonnet"
    assert any("another family" in n for n in t["notes"])


def test_family_map():
    assert [cal.family(m) for m in ("qwen35-9b", "gemma4-12b", "r1-qwen3-8b", "haiku", "sonnet", "opus")] == \
        ["qwen", "gemma", "deepseek", "claude", "claude", "claude"]


def test_routing_markdown_mentions_every_role():
    t = table_for({("qwen35-4b", "constant"): (16, 16)})
    ts = trials("qwen35-4b", "constant", 16, 16)
    md = cal.routing_markdown(t, cal.summarize(ts))
    for role in ("coder constant", "repairer", "planner_cheap", "analyst", "voter", "critic"):
        assert role in md
    assert "qwen35-4b | constant | 16/16" in md


# ---------------------------------------------------------------- run(): fake call, resume

@dataclass
class Fake:
    ok: bool = True
    obj: object = None
    text: str = ""
    error: str = None
    error_class: str = None
    served_model: str = ""
    cost_usd_plan: float = 0.0
    cost_usd_nominal: float = 0.01
    latency_s: float = 0.5


def edit_reply(model, edits_text):
    obj = {"description": "d", "edits_text": edits_text}
    return Fake(obj=obj, text=json.dumps(obj), served_model=model)


class Caller:
    def __init__(self, replies=None, overrides=None):
        self.calls, self.replies, self.overrides, self.kwargs = [], replies or {}, overrides or {}, []

    def __call__(self, model, user, **kw):
        self.calls.append((model, kw["tag"]))
        self.kwargs.append((model, kw))
        _, _, spec_id, _ = kw["tag"].split(":")
        r = self.overrides.get((model, spec_id))
        return r if r is not None else edit_reply(model, self.replies.get(spec_id, "no idea"))


def up(_):
    return True


def test_run_scores_records_and_resumes(tmp_path):
    out = tmp_path / "cal"
    fake = Caller(replies={"const-1": REFERENCE["const-1"], "const-2": REFERENCE["const-2"]})
    res = cal.run(["haiku", "qwen35-9b"], ["constant"], 2, out, call_fn=fake, healthy_fn=up, limit=3)
    assert len(fake.calls) == 2 * 3 * 2
    s = res["summary"]
    assert s["haiku"]["constant"]["k"] == 4 and s["haiku"]["constant"]["n"] == 6
    lines = [json.loads(x) for x in (out / "results.jsonl").read_text().splitlines()]
    assert len(lines) == 12
    assert set(lines[0]) >= {"model", "spec_id", "edit_class", "ok", "reasons", "cost", "latency", "served_model"}
    assert json.loads((out / "replies" / "haiku" / "const-1.0.txt").read_text())["edits_text"] == REFERENCE["const-1"]
    # production prompts: card as system, coder schema, one attempt; open model temperature 0.7 with thinking off
    kw = dict((m, k) for m, k in fake.kwargs)
    assert kw["haiku"]["timeout_s"] == 600 and kw["haiku"]["max_tokens"] == 4096
    assert kw["haiku"]["system"] == prompts.program_card(cal.N_SLOTS) and kw["haiku"]["schema"] is prompts.SCHEMAS["coder"]
    assert kw["haiku"]["attempts"] == 1
    assert kw["qwen35-9b"]["temperature"] == 0.7 and kw["qwen35-9b"]["extra"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert kw["haiku"]["temperature"] is None and kw["haiku"]["extra"] is None
    assert cal.model_cfg("gemma4-12b", cal.call_mod.load_registry()) == {"temperature": 0.7, "extra": cal.OPEN_EXTRA, "max_tokens": 4096}
    # resume: nothing left to do
    again = Caller()
    cal.run(["haiku", "qwen35-9b"], ["constant"], 2, out, call_fn=again, healthy_fn=up, limit=3)
    assert again.calls == []
    # a third sample is the only new work; adding a model only runs that model
    third = Caller()
    cal.run(["haiku", "qwen35-9b"], ["constant"], 3, out, call_fn=third, healthy_fn=up, limit=3)
    assert len(third.calls) == 2 * 3
    fourth = Caller()
    cal.run(["haiku", "qwen35-9b", "qwen35-4b"], ["constant"], 3, out, call_fn=fourth, healthy_fn=up, limit=3)
    assert {m for m, _ in fourth.calls} == {"qwen35-4b"}


def test_infra_failures_are_excluded_and_retried_on_resume(tmp_path):
    out = tmp_path / "cal"
    down = Fake(ok=False, error="refused", error_class="infra", served_model="qwen35-9b")
    hopped = edit_reply("haiku", REFERENCE["const-2"])   # call() fell back to haiku
    fake = Caller(overrides={("qwen35-9b", "const-1"): down, ("qwen35-9b", "const-2"): hopped})
    res = cal.run(["qwen35-9b"], ["constant"], 1, out, call_fn=fake, healthy_fn=up, limit=2)
    c = res["summary"]["qwen35-9b"]["constant"]
    assert (c["n"], c["n_infra"], c["k"]) == (0, 2, 0) and not c["qualified"]
    by_id = {t["spec_id"]: t for t in res["trials"]}
    assert by_id["const-2"]["reasons"] == ["fallback:haiku"] and by_id["const-2"]["infra"]
    again = Caller(replies={"const-1": REFERENCE["const-1"], "const-2": REFERENCE["const-2"]})
    res = cal.run(["qwen35-9b"], ["constant"], 1, out, call_fn=again, healthy_fn=up, limit=2)
    assert len(again.calls) == 2   # both infra trials retried
    assert res["summary"]["qwen35-9b"]["constant"]["k"] == 2 and len(res["trials"]) == 2   # latest line wins


def test_model_failure_counts_and_context_error_is_infra(tmp_path):
    bad = Fake(ok=False, error="response is not parseable JSON", error_class="model", served_model="qwen35-4b")

    def caller(model, user, **kw):
        if kw["tag"].endswith(":ana-1:0"):
            raise cal.call_mod.ContextTooLong("too long")
        return bad
    res = cal.run(["qwen35-4b"], ["analyst"], 1, tmp_path, call_fn=caller, healthy_fn=up, limit=2)
    by_id = {t["spec_id"]: t for t in res["trials"]}
    assert by_id["ana-1"]["infra"] and by_id["ana-1"]["reasons"][0].startswith("context:")
    assert not by_id["ana-2"]["infra"] and by_id["ana-2"]["reasons"][0].startswith("schema:")


def test_unhealthy_open_model_is_skipped_without_calls(tmp_path):
    fake = Caller()
    res = cal.run(["haiku", "gemma4-12b"], ["constant"], 1, tmp_path, call_fn=fake,
                  healthy_fn=lambda m: m != "gemma4-12b", limit=1)
    assert res["skipped"] == {"gemma4-12b": "unhealthy"}
    assert {m for m, _ in fake.calls} == {"haiku"}


def test_repair_planner_analyst_plumbing(tmp_path):
    plan = {s["id"]: s for s in SPECS["planner"]}["plan-1"]
    (_, repair_spec, _, _), = cal.tasks(SPECS, ["repair"], BASE, 1)
    (old, new), = repair_spec["violation_edits"]

    def caller(model, user, **kw):
        spec_id = kw["tag"].split(":")[2]
        if spec_id.startswith("repair"):
            return edit_reply(model, sr((new.rstrip("\n"), old.rstrip("\n"))))
        obj = good_planner(plan) if spec_id.startswith("plan") else {"memo": "Nothing helped in 1 run(s)."}
        return Fake(obj=obj, text=json.dumps(obj), served_model=model)
    res = cal.run(["haiku"], ["repair", "planner", "analyst"], 1, tmp_path, call_fn=caller, healthy_fn=up, limit=1)
    assert {t["spec_id"]: t["ok"] for t in res["trials"]} == {"repair-1": True, "plan-1": True, "ana-1": True}


def test_task_prompts_use_the_production_builders():
    ts = {c: [t for t in cal.tasks(SPECS, [c], BASE) if t[0] == c] for c in cal.ALL_CLASSES}
    assert [len(v) for v in ts.values()] == [8, 8, 8, 6, 6, 6]
    cls, spec, user, schema = ts["constant"][0]
    assert user.startswith("## Current train.py") and "EMBEDDING_LR = 0.6" in user and spec["spec"] in user
    assert "Title: Raise EMBEDDING_LR to 0.8" in user and "Family: optimizer" in user
    assert schema is prompts.SCHEMAS["coder"]
    assert ts["planner"][2][2].count("| crash |") == 3 and "Propose exactly" not in ts["planner"][0][2]
    assert "Propose 5 distinct ideas" in ts["planner"][0][2]
    assert chr(0x2014) not in prompts.program_card(cal.N_SLOTS)


def test_cli_rebuild_makes_table_files_without_calls(tmp_path, capsys):
    (tmp_path / "results.jsonl").write_text("".join(json.dumps(t) + "\n" for t in trials("haiku", "constant", 16, 16)))
    cal.main(["--models", "haiku", "--out", str(tmp_path), "--rebuild"])
    table = json.loads((tmp_path / "routing_table.json").read_text())
    assert table["coder_by_class"]["constant"] == "haiku"
    assert (tmp_path / "routing_table.md").read_text().startswith("# Routing table")
    assert json.loads((tmp_path / "summary.json").read_text())["haiku"]["constant"]["qualified"]


def test_build_routing_table_writes_files_when_given_out_dir(tmp_path):
    ts = trials("haiku", "repair", 12, 12)
    cal.build_routing_table({"trials": ts, "summary": cal.summarize(ts)}, PRICES, out_dir=tmp_path)
    assert json.loads((tmp_path / "routing_table.json").read_text())["repairer"] == "haiku"
    assert "| repairer | haiku |" in (tmp_path / "routing_table.md").read_text()
