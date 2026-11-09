"""Validate the requalification items (specs_requal.json + references.json) against the real scorer. No model calls.

    .venv/bin/python -B loop/runs/calibration_requal/validate_specs.py

Exit code 0 only if every check passes. Checks:
  1. The specs load with calibration.load_specs and have the shape promised in SPECS_README.md: 30 structural coder specs
     (rstruct-01..15 single block, rstruct-16..30 multi block) and 20 repair specs, the same keys as calib_specs.json,
     valid regexes, and prompts that build with calibration.tasks.
  2. Ids are unique, new (absent from calib_specs.json) and each has a reference.
  3. Coder: the reference passes score_edit(BASE, BASE, ref, spec) with no reasons and is classed "structural" by the
     linter; a JSON reply passes score_coder; a no-op reply fails; every hand-written wrong edit fails and every
     hand-written valid alternative passes; dropping any one SEARCH/REPLACE block of a reference makes it fail.
  4. Repair: the broken file triggers expect_rule; the legitimate edit alone lints clean; the reference passes
     score_edit(broken, BASE, ref, spec); "no change" fails; undoing the legitimate edit fails; dropping any one block of a
     multi-block reference fails; hand-written wrong repairs fail and valid alternative repairs pass; the 20 primary
     rules are distinct.
  5. No spec text, title or repair edit duplicates one in calib_specs.json or another new spec.
  6. No em dash in the new files; specs.sha256 matches specs_requal.json.
"""
import difflib
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "loop" / "core"))
import calibration as cal  # noqa: E402
import linter  # noqa: E402
import patch  # noqa: E402

SPECS_FILE, REFS_FILE, SHA_FILE = HERE / "specs_requal.json", HERE / "references.json", HERE / "specs.sha256"
BASE = cal.BASE_PATH.read_text()
SIMILAR = 0.8   # difflib ratio on normalised text at or above which two coder specs count as near duplicates


def norm_text(s):
    return re.sub(r"\s+", " ", s.lower()).strip()


def sr_blocks(blocks):
    return "\n".join("<<<<<<< SEARCH\n%s=======\n%s>>>>>>> REPLACE" % b for b in blocks)


def drop_one(reply):
    """Every reply that keeps all SEARCH/REPLACE blocks but one."""
    blocks = patch.parse_blocks(reply)
    return [(k, sr_blocks(blocks[:k] + blocks[k + 1:])) for k in range(len(blocks))] if len(blocks) > 1 else []


def split_reasons(reasons):
    return ",".join(reasons) if reasons else "pass"


def main():
    fails = []

    def check(cond, msg):
        if not cond:
            fails.append(msg)

    specs, old = cal.load_specs(SPECS_FILE), cal.load_specs()
    refs = json.loads(REFS_FILE.read_text())
    coder, repair = specs["coder"], specs["repair"]

    # 1. shape
    check(set(specs) == {"version", "base", "coder", "repair"}, "top-level keys are version, base, coder, repair")
    check(len(coder) == 30 and len(repair) == 20, "need 30 coder and 20 repair specs, got %d and %d" % (len(coder), len(repair)))
    check([s["id"] for s in coder] == ["rstruct-%02d" % i for i in range(1, 31)], "coder ids must be rstruct-01..30 in order")
    check([s["id"] for s in repair] == ["rrep-%02d" % i for i in range(1, 21)], "repair ids must be rrep-01..20 in order")
    old_keys = {k: set(old[k][0]) for k in ("coder", "repair")}
    for k, group in (("coder", coder), ("repair", repair)):
        for s in group:
            check(set(s) == old_keys[k], "%s: keys differ from calib_specs.json (%s)" % (s["id"], sorted(set(s) ^ old_keys[k])))
            check(s["spec"] and s["target_symbols"] and isinstance(s["must_not_touch"], list), "%s: empty spec or target_symbols" % s["id"])
            try:
                for e in s["target_symbols"] + s["must_not_touch"] + s["expect"]:
                    re.compile(cal._rx(e) if not e.startswith("!") else e[1:])
            except re.error as ex:
                fails.append("%s: bad regex (%s)" % (s["id"], ex))
    check(all(s["edit_class"] == "structural" for s in coder) and all(s["edit_class"] == "repair" for s in repair), "edit_class values")
    plan = cal.tasks(specs, ["structural", "repair"], BASE)
    check(len(plan) == 50, "calibration.tasks should build 50 prompts, got %d" % len(plan))
    for cls, s, user, _ in plan:
        check(s["id"] in {x["id"] for x in coder + repair}, "tasks() returned a foreign spec")
        if cls == "structural":
            check(s["spec"] in user and s["title"] in user, "%s: prompt misses the spec or title" % s["id"])
        else:
            check(s["expect_rule"] in user, "%s: repair prompt does not list the %s violation" % (s["id"], s["expect_rule"]))

    # 2. ids
    ids = [s["id"] for s in coder + repair]
    old_ids = {s["id"] for k in ("coder", "repair", "planner", "analyst") for s in old[k]}
    check(len(ids) == len(set(ids)), "duplicate ids")
    check(not (set(ids) & old_ids), "ids collide with calib_specs.json: %s" % sorted(set(ids) & old_ids))
    check(set(refs["coder"]) == {s["id"] for s in coder} and set(refs["repair"]) == {s["id"] for s in repair}, "references.json ids differ from the specs")

    if fails:   # the later checks look references up by id, so stop here
        for f in fails:
            print("FAIL", f)
        return 1

    # 3. coder
    report = []
    for s in coder:
        sid, ref = s["id"], refs["coder"][s["id"]]
        reasons, klass = cal.score_edit(BASE, BASE, ref, s)
        check(not reasons, "%s: reference fails: %s" % (sid, split_reasons(reasons)))
        check(klass == "structural", "%s: linter classes the reference as %r, not structural" % (sid, klass))
        check(cal.score_coder({"description": "reference", "edits_text": ref}, BASE, BASE, s)[0] == [], "%s: JSON reply fails" % sid)
        check("noop" in cal.score_edit(BASE, BASE, "<<<<<<< SEARCH\nimport gc\n=======\nimport gc\n>>>>>>> REPLACE", s)[0], "%s: a no-op reply is not rejected" % sid)
        for i, wrong in enumerate(refs.get("negative_coder", {}).get(sid, [])):
            check(cal.score_edit(BASE, BASE, wrong, s)[0], "%s: wrong edit #%d passes" % (sid, i))
        for i, alt in enumerate(refs.get("alternative_coder", {}).get(sid, [])):
            r = cal.score_edit(BASE, BASE, alt, s)[0]
            check(not r, "%s: valid alternative #%d fails: %s" % (sid, i, split_reasons(r)))
        survivors = [k for k, rest in drop_one(ref) if not cal.score_edit(BASE, BASE, rest, s)[0]]
        check(not survivors, "%s: dropping block(s) %s of the reference still passes" % (sid, survivors))
        report.append((sid, len(patch.parse_blocks(ref)), len(refs.get("negative_coder", {}).get(sid, []))))
    single, multi = report[:15], report[15:]
    check(all(n == 1 for _, n, _ in single), "rstruct-01..15 references must each be one block")
    check(all(n >= 2 for _, n, _ in multi), "rstruct-16..30 references must each have at least two blocks")
    check(all(neg >= 1 for _, _, neg in single), "rstruct-01..15 each need a hand-written wrong edit")

    # 4. repair
    primary, seen_rules = [], set()
    for s in repair:
        rid, ref = s["id"], refs["repair"][s["id"]]
        broken, hard = cal.repair_inputs(BASE, s)
        rules = sorted({h["rule"] for h in hard})
        seen_rules |= set(rules)
        primary.append(s["expect_rule"])
        check(s["expect_rule"] in rules, "%s: broken file does not trigger %s (got %s)" % (rid, s["expect_rule"], rules))
        legit = cal.apply_edits(BASE, s["legit_edits"])
        check(linter.lint(legit, BASE).ok, "%s: the legitimate edit alone does not lint clean" % rid)
        check(broken != BASE and legit != BASE, "%s: broken or legit file equals the base" % rid)
        reasons, _ = cal.score_edit(broken, BASE, ref, s)
        check(not reasons, "%s: reference repair fails: %s" % (rid, split_reasons(reasons)))
        check(cal.score_coder({"description": "reference", "edits_text": ref}, broken, BASE, s)[0] == [], "%s: JSON repair reply fails" % rid)
        check(cal.score_edit(broken, BASE, "no change needed", s)[0], "%s: an empty repair is not rejected" % rid)
        undo = ref + "\n" + sr_blocks([(new + "\n", old + "\n") for old, new in s["legit_edits"]])
        check(cal.score_edit(broken, BASE, undo, s)[0], "%s: repair that throws the legitimate edit away passes" % rid)
        for i, wrong in enumerate(refs.get("negative_repair", {}).get(rid, [])):
            check(cal.score_edit(broken, BASE, wrong, s)[0], "%s: wrong repair #%d passes" % (rid, i))
        for i, alt in enumerate(refs.get("alternative_repair", {}).get(rid, [])):
            r = cal.score_edit(broken, BASE, alt, s)[0]
            check(not r, "%s: valid alternative repair #%d fails: %s" % (rid, i, split_reasons(r)))
        survivors = [k for k, rest in drop_one(ref) if not cal.score_edit(broken, BASE, rest, s)[0]]
        check(not survivors, "%s: dropping block(s) %s of the repair still passes" % (rid, survivors))
    check(len(set(primary)) == 20, "the 20 repair specs must have 20 distinct primary rules, got %d" % len(set(primary)))
    check(not (set(primary) & {"parse", "file-size"}), "parse and file-size are out of scope")

    # 5. duplicates
    old_specs = old["coder"] + old["repair"]
    for s in coder + repair:
        for o in old_specs:
            check(norm_text(s["spec"]) != norm_text(o["spec"]), "%s: spec text equals %s" % (s["id"], o["id"]))
            if "title" in s and "title" in o:
                check(s["title"].lower() != o["title"].lower(), "%s: title equals %s" % (s["id"], o["id"]))
    worst = (0.0, "", "")
    for i, s in enumerate(coder):
        for o in old["coder"] + coder[:i]:
            r = difflib.SequenceMatcher(None, norm_text(s["spec"]), norm_text(o["spec"])).ratio()
            worst = max(worst, (r, s["id"], o["id"]))
            check(r < SIMILAR, "%s: spec text is %.2f similar to %s" % (s["id"], r, o["id"]))
    old_legit = {tuple(e[0] for e in o["legit_edits"]) for o in old["repair"]}
    old_vio = {json.dumps(o["violation_edits"]) for o in old["repair"]}
    new_legit = [tuple(e[0] for e in s["legit_edits"]) for s in repair]
    check(len(set(new_legit)) == len(new_legit), "two new repair specs change the same base line")
    for s, lg in zip(repair, new_legit):
        check(lg not in old_legit, "%s: legitimate edit is on a base line already used by an old repair spec" % s["id"])
        check(json.dumps(s["violation_edits"]) not in old_vio, "%s: violation equals an old repair violation" % s["id"])

    # 6. files
    for f in (SPECS_FILE, REFS_FILE, HERE / "SPECS_README.md", HERE / "validate_specs.py", HERE / "smoke_references.py"):
        if f.exists():
            check(chr(0x2014) not in f.read_text(), "%s contains an em dash" % f.name)
    digest = hashlib.sha256(SPECS_FILE.read_bytes()).hexdigest()
    check(SHA_FILE.exists() and SHA_FILE.read_text().split()[0] == digest, "specs.sha256 is missing or does not match specs_requal.json")

    print("coder: %d specs (%d single block, %d multi block, %d to %d blocks), %d hand-written wrong edits"
          % (len(coder), len(single), len(multi), min(n for _, n, _ in multi), max(n for _, n, _ in multi), sum(x[2] for x in report)))
    print("repair: %d specs, %d distinct primary rules, %d distinct rules triggered in all" % (len(repair), len(set(primary)), len(seen_rules)))
    print("hand-written checks: %s" % ", ".join("%d %s" % (sum(map(len, refs.get(k, {}).values())), k)
                                                for k in ("alternative_coder", "negative_repair", "alternative_repair")))
    print("most similar coder spec pair: %.2f (%s vs %s), limit %.2f" % (worst[0], worst[1], worst[2], SIMILAR))
    print("specs sha256:", digest)
    for f in fails:
        print("FAIL", f)
    print("ALL CHECKS PASS" if not fails else "%d CHECK(S) FAILED" % len(fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
