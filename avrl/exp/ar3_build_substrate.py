"""AR3: build the critic-cascade substrate offline from the research ledger (design: exp/AR3_DESIGN.md section 3).

Splits avrl/RESEARCH_LOG.md into entries, strips the real critic paragraphs, applies the mechanical pool rule,
renders each entry with line numbers (L001| ...), and writes one untouched control plus up to seven planted
defects per entry (five kinds: N-big, N-mid, N-stale, V-body, V-coh). Six natural-probe entries are rendered as
controls only. Entries are split into burn and test by hash of the entry key (never by variant).

Writes avrl/runs/AR3/{ledger_snapshot.md, items.jsonl, split.json, pool_report.json, items.sha256}.
Offline, no model calls. Exposes render_entry(section) and build_items_for_entries(keys) for the runner's smoke mode.
Usage: python exp/ar3_build_substrate.py [--print-trips]
"""
import argparse
import hashlib
import json
import math
import random
import re
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "RESEARCH_LOG.md"
ORACLE = ROOT / "runs" / "AR1" / "oracle.json"
OUT = ROOT / "runs" / "AR3"
WHITELIST = Path(__file__).resolve().parent / "ar3_asis_whitelist.json"

SALT = "ar3-2026-10-17"
SHUFFLE_SEED = 20261017
BURN_FRAC = 0.25
DATE_CUTOFF = "2026-10-16"
EXCLUDE = {  # header id -> reason
    "M01": "interleaved corrections",
    "P1-judge-lr5-s2": "no tables",
    "P1-judge-5090b-s0": "no tables",
    "M06b": "no tables",
}
PROBES = ["E04", "L5a", "E05", "L7b", "L12b", "L13a"]
SMOKE_IDS = ["M01", "P1-judge-lr5-s2", "P1-judge-5090b-s0", "M06b"]
TRIP_IDS = ["E01", "E06", "E09b", "E09c", "L9d", "P1-judge-s1", "L10a", "E10", "M14", "M15"]
KINDS = ["N-big", "N-mid", "N-stale", "V-body", "V-coh"]
N_SEEDS = {"N-big": 1, "N-mid": 2, "N-stale": 2, "V-body": 1, "V-coh": 1}

DATE = re.compile(r"2026-\d\d-\d\d")
NUM = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+\.\d+|\d+)(?![\w.%]*\w)")
DATE_SPAN = re.compile(r"\d{4}-\d\d-\d\d")
TRIP = re.compile(r"critic|corrected verdict|review note", re.I)
VTOK = re.compile(r"\b(pass|kill)\b", re.I)
SEP = re.compile(r"^\|[\s:|-]+\|?$")
UNITS = {"seeds", "seed", "min", "minutes", "steps", "step", "epochs", "gb", "mb", "tokens", "s", "h", "utc", "job"}
NOT_PROSE = ("Run:", "Files:", "Script", "Hypothesis:")

# Critic continuation openers. The first block is the design's list; the second block extends it to the critic
# paragraphs actually present in the ledger (deviation, see the report): without it residual critic prose stays in.
CONT = ["Corrected verdict", "Everything else checked out", "One real problem", "One claim in the entry", "Four things",
        "Three real issues", "That said"]
CONT_EXTRA = ["One number in the entry", "One correction to", "One defect found", "This does not", "The entry's",
              "The confident-subset", "The sim replay numbers"]

CHECKS = Counter()  # assertion tallies for the unit-test report


def check(cond, msg):
    CHECKS["total"] += 1
    if not cond:
        raise AssertionError(msg)


def sha1(s):
    return hashlib.sha1(s.encode()).hexdigest()


# ---------------------------------------------------------------- 3.1 sections
def parse_sections(text):
    lines = text.split("\n")
    hs = [i for i, l in enumerate(lines) if l.startswith("## ")]
    secs = []
    for n, i in enumerate(hs):
        j = hs[n + 1] if n + 1 < len(hs) else len(lines)
        body = lines[i + 1:j]
        while body and not body[-1].strip():
            body.pop()
        h = lines[i]
        m = DATE.search(h) or next((DATE.search(b) for b in body if DATE.search(b)), None)
        hid = h[3:m.start()].strip() if DATE.search(h) else (h[3:].split() or [""])[0]
        secs.append({"key": sha1(h)[:8], "header": h, "header_id": hid, "date": m.group(0) if m else None,
                     "body": body, "line_no": i + 1})
    return secs


# ---------------------------------------------------------------- 3.2 strip critic text
def paragraphs(body):
    paras, cur = [], []
    for l in body + [""]:
        if l.strip():
            cur.append(l)
        elif cur:
            paras.append(cur)
            cur = []
    return paras


def _is_cont(txt):
    low = txt.lower()
    if any(low.startswith(s.lower()) for s in CONT + CONT_EXTRA):
        return True
    return low.startswith("verdict") and "confirmed" in low


def strip_critic(body):
    lead = bool(body) and not body[0].strip()
    kept, dropping, ndrop = [], False, 0
    for p in paragraphs(body):
        txt = " ".join(x.strip() for x in p).strip()
        if re.match(r"critic note", txt, re.I):
            dropping = True
        elif dropping and (_is_cont(txt) or re.match(r"\d+\.", txt)):
            pass
        else:
            dropping = False
        if dropping:
            ndrop += 1
        else:
            kept.append(p)
    out = [""] if lead else []
    for k, p in enumerate(kept):
        if k:
            out.append("")
        out += p
    return out, ndrop


def trip_paragraphs(lines):
    """Paragraphs left after stripping that still mention critic text: list of (sha1, text)."""
    res = []
    for p in paragraphs(lines):
        t = "\n".join(p).strip()
        if TRIP.search(t):
            res.append((sha1(t), t))
    return res


def load_whitelist():
    return json.loads(WHITELIST.read_text()) if WHITELIST.exists() else {}


def prepare(sec, whitelist):
    """Strip critic text, find residual tripping paragraphs. Adds lines/trips/whitelist_hits to the section."""
    body, ndrop = strip_critic(sec["body"])
    trips = trip_paragraphs(body)
    sec["lines"] = [sec["header"]] + body
    sec["n_dropped_paras"] = ndrop
    sec["trips"] = trips
    sec["wl_hits"] = [h for h, _ in trips if h in whitelist]
    sec["residual"] = [h for h, _ in trips if h not in whitelist]
    return sec


# ---------------------------------------------------------------- 3.5 rendering
def render_lines(lines):
    return "\n".join("L%03d| %s" % (i + 1, l) for i, l in enumerate(lines))


def render_entry(section, whitelist=None):
    """Rendered text of one parsed section (stripped of critic text, header first, L%03d| prefixes)."""
    if "lines" not in section:
        prepare(section, whitelist if whitelist is not None else load_whitelist())
    return render_lines(section["lines"])


# ---------------------------------------------------------------- 3.6 numbers
def is_table(l):
    s = l.strip()
    return s.startswith("|") and not SEP.match(s)


def is_sep(l):
    return bool(SEP.match(l.strip()))


def tokens(line):
    """Numeric tokens outside dates: list of (start, end, text)."""
    spans = [m.span() for m in DATE_SPAN.finditer(line)]
    return [(m.start(1), m.end(1), m.group(1)) for m in NUM.finditer(line)
            if not any(a <= m.start(1) < b for a, b in spans)]


LABEL = re.compile(r"(?:claims?|seeds?|steps?|points?|arms?|jobs?|runs?|sections?|tests?|rounds?|figures?|tables?|panels?|parts?|"
                   r"phases?|stages?|versions?|reads?|readings?|olmo|top)\s*(?:[\d.]+\s*(?:,|to|and|or|-|/|&)\s*)*$", re.I)


def prose_ok(line, s, e, tok):
    if line.lstrip().startswith(NOT_PROSE):
        return False
    # guards beyond the design's list: identifiers and labels are not quantities (documented deviation)
    if num_val(tok) == 0 or line[s - 1:s] == "@" or (s >= 2 and line[s - 1] == "-" and line[s - 2].isalpha()):
        return False
    if re.match(r"-[A-Za-z]", line[e:e + 2]):
        return False
    if re.fullmatch(r"\d+", tok):
        if LABEL.search(line[:s]) or s == len(line) - len(line.lstrip()) and re.match(r"\d+[.)]\s", line.lstrip()):
            return False
        if line[s - 1:s] == "(" and line[e:e + 1] == ")" and int(tok) < 10:
            return False
    if line[:s].count("`") % 2:
        return False
    if (s > 0 and line[s - 1] == ":" and s > 1 and line[s - 2].isdigit()) or (e < len(line) - 1 and line[e] == ":" and line[e + 1].isdigit()):
        return False
    if re.fullmatch(r"(19|20)\d\d", tok):
        return False
    nxt = re.findall(r"\w+", line[e:])[:2]
    return not any(w.lower() in UNITS for w in nxt)


def fmt_class(t):
    return (("," in t), len(t.split(".")[1]) if "." in t else 0)


def table_blocks(lines):
    """Contiguous table blocks: list of dict(rows=[(line_idx, cells)], data=[row idx])."""
    blocks, cur = [], []
    for i, l in enumerate(lines + [""]):
        if l.strip().startswith("|"):
            cur.append(i)
        elif cur:
            blocks.append(cur)
            cur = []
    res = []
    for b in blocks:
        rows = [(i, [c for c in lines[i].strip().strip("|").split("|")]) for i in b if not is_sep(lines[i])]
        has_sep = any(is_sep(lines[i]) for i in b)
        data = rows[1:] if has_sep else rows
        res.append({"rows": rows, "data": data})
    return res


def entry_numbers(lines):
    """Plantable occurrences: list of dict(line, s, e, tok), prose lines only, token also in a table line."""
    table_toks = set()
    for l in lines:
        if is_table(l):
            table_toks.update(t for _, _, t in tokens(l))
    out = []
    for i, l in enumerate(lines):
        if i == 0 or not l.strip() or l.strip().startswith("|"):
            continue
        for s, e, t in tokens(l):
            if t in table_toks and prose_ok(l, s, e, t):
                out.append({"line": i, "s": s, "e": e, "tok": t})
    return out


def all_tokens(lines):
    return {t for l in lines for _, _, t in tokens(l)}


def num_val(t):
    return float(t.replace(",", ""))


def fmt_like(v, old):
    if "." in old:
        d = len(old.split(".")[1])
        s = f"{v:.{d}f}"
    else:
        s = f"{int(round(v))}"
    if "," in old:
        ip, _, fp = s.partition(".")
        s = f"{int(ip):,}" + ("." + fp if fp else "")
    return s


# ---------------------------------------------------------------- verdict helpers
def find_vtok(line, start=0):
    """First pass|kill token not inside parentheses or brackets: (s, e, word) or None."""
    for m in VTOK.finditer(line, start):
        pre = line[:m.start()]
        if pre.count("(") <= pre.count(")") and pre.count("[") <= pre.count("]"):
            return m.start(), m.end(), m.group(1)
    return None


def flip_word(w):
    n = "kill" if w.lower() == "pass" else "pass"
    return n.upper() if w.isupper() else n.capitalize() if w[0].isupper() else n


def flip_at(line, s, e):
    return line[:s] + flip_word(line[s:e]) + line[e:]


def verdict_para(lines):
    """(first line idx, last line idx) of the paragraph starting at the first line that begins with 'Verdict'."""
    for i, l in enumerate(lines):
        if i > 0 and l.strip().startswith("Verdict"):
            j = i
            while j + 1 < len(lines) and lines[j + 1].strip():
                j += 1
            return i, j
    return None


def threshold_language(lines):
    vp = verdict_para(lines)
    if not vp:
        return False
    t = " ".join(lines[vp[0]:vp[1] + 1])
    low = t.lower()
    both = bool(re.search(r"\bpass\b", low)) and bool(re.search(r"\bkill\b", low))
    return (both or "threshold" in low) and bool(NUM.search(t))


# ---------------------------------------------------------------- 3.7 plants
def plant_rng(key, kind, idx):
    return random.Random(int(sha1(key + kind + str(idx)), 16) % 2**32)


def apply_plant(lines, i, s, e, new):
    out = list(lines)
    out[i] = out[i][:s] + new + out[i][e:]
    return out


def table_lines_with(lines, tok):
    return [i for i, l in enumerate(lines) if is_table(l) and any(t == tok for _, _, t in tokens(l))]


def plant_n(lines, key, kind, idx, used):
    cands = entry_numbers(lines)
    cands = [c for c in cands if (c["line"], c["s"]) not in used]
    if not cands:
        return None
    rng = plant_rng(key, kind, idx)
    present = all_tokens(lines)
    order = list(cands)
    rng.shuffle(order)
    blocks = table_blocks(lines)
    for c in order:
        old = c["tok"]
        v = num_val(old)
        if kind == "N-stale":
            news = []
            for ti in table_lines_with(lines, old):
                for b in blocks:
                    for ri, cells in b["rows"]:
                        if ri != ti:
                            continue
                        for ci, cell in enumerate(cells):
                            if not any(t == old for _, _, t in tokens(cell)):
                                continue
                            for rj, cells2 in b["data"]:
                                if rj == ri or ci >= len(cells2):
                                    continue
                                for _, _, t in tokens(cells2[ci]):
                                    if t != old and fmt_class(t) == fmt_class(old) and 0.25 <= num_val(t) / v <= 4 and abs(num_val(t) - v) / v >= 0.04:
                                        news.append((t, ci))
            ok = []
            for t, ci in news:
                good = True
                for l in lines:
                    if is_table(l):
                        cells = l.strip().strip("|").split("|")
                        for cj, cell in enumerate(cells):
                            if any(x == t for _, _, x in tokens(cell)) and cj != ci:
                                good = False
                    elif any(x == t for _, _, x in tokens(l)):
                        good = False
                if good:
                    ok.append(t)
            ok = sorted(set(ok))
            if not ok:
                continue
            new = rng.choice(ok)
        else:
            if kind == "N-big":
                fs = [0.4, 0.6, 1.4, 1.6]
                rng.shuffle(fs)
                vals = [v * f for f in fs]
            elif "." not in old and "," not in old and v < 20:
                vals = [v - 2, v + 2]
                rng.shuffle(vals)
                vals = [x for x in vals if x >= 0]
            else:
                fs = [0.85, 0.90, 1.10, 1.15]
                rng.shuffle(fs)
                vals = [v * f for f in fs]
            new = None
            for x in vals:
                t = fmt_like(x, old)
                if t != old and t not in present and num_val(t) > 0 and abs(num_val(t) - v) / v >= 0.04:
                    new = t
                    break
            if new is None:
                continue
        i = c["line"]
        text = apply_plant(lines, i, c["s"], c["e"], new)
        copy = any(j != i and not is_table(l) and any(t == old for _, _, t in tokens(l)) for j, l in enumerate(lines))
        tl = table_lines_with(lines, old)
        check(len(tl) > 0, "old value not in a table line")
        acc = sorted({j for j in (i - 1, i, i + 1) if 0 <= j < len(lines)} | set(tl))
        return {"text": text, "changed": [i], "old": old, "new": new, "accept": acc, "pos": (i, c["s"]), "copy": copy}
    return None


def plant_vbody(lines):
    vp = verdict_para(lines)
    if not vp:
        return None
    i = vp[0]
    t = find_vtok(lines[i])
    if not t:
        return None
    s, e, w = t
    text = list(lines)
    text[i] = flip_at(lines[i], s, e)
    acc = sorted({j for j in (i - 1, i, i + 1) if 0 <= j < len(lines)} | {0})
    return {"text": text, "changed": [i], "old": w, "new": flip_word(w), "accept": acc, "pos": (i, s), "copy": None}


def plant_vcoh(lines):
    vp = verdict_para(lines)
    if not vp or not threshold_language(lines):
        return None
    ht = None
    m = DATE.search(lines[0])
    ht = find_vtok(lines[0], m.end() if m else 0)
    vt = find_vtok(lines[vp[0]])
    if not ht or not vt:
        return None
    text = list(lines)
    changed = [0, vp[0]]
    text[0] = flip_at(lines[0], ht[0], ht[1])
    text[vp[0]] = flip_at(lines[vp[0]], vt[0], vt[1])
    # the "What it changes" paragraph: first pass|kill token in it
    for i, l in enumerate(lines):
        if l.strip().startswith("What it changes"):
            j = i
            while True:
                t = find_vtok(lines[j])
                if t:
                    text[j] = flip_at(lines[j], t[0], t[1])
                    changed.append(j)
                    break
                if j + 1 >= len(lines) or not lines[j + 1].strip():
                    break
                j += 1
            break
    changed = sorted(set(changed))
    acc = set()
    for j in changed:
        acc |= {x for x in (j - 1, j, j + 1) if 0 <= x < len(lines)}
    acc |= {k for k in range(vp[0], vp[1] + 1) if NUM.search(lines[k])}
    return {"text": text, "changed": changed, "old": ht[2], "new": flip_word(ht[2]), "accept": sorted(acc),
            "pos": (0, ht[0]), "copy": None}


def make_plants(lines, key):
    """List of (kind, seed_idx, plant dict) for one entry; asserts per-plant invariants."""
    res, used, seen = [], set(), {sha1("\n".join(lines))}
    for kind in KINDS:
        for idx in range(N_SEEDS[kind]):
            if kind.startswith("N"):
                p = plant_n(lines, key, kind, idx, used)
            elif kind == "V-body":
                p = plant_vbody(lines)
            else:
                p = plant_vcoh(lines)
            if not p:
                continue
            h = sha1("\n".join(p["text"]))
            if h in seen:
                continue
            seen.add(h)
            used.add(p["pos"])
            # per-plant assertions
            diff = [i for i, (a, b) in enumerate(zip(lines, p["text"])) if a != b]
            check(diff == sorted(p["changed"]), f"{key} {kind}: differing lines {diff} != planted {p['changed']}")
            check(len(p["text"]) == len(lines), "line count changed")
            check(len(p["accept"]) > 0, "empty accept set")
            check(all(0 <= j < len(lines) for j in p["accept"]), "accept line out of range")
            check(p["old"] != p["new"], "old == new")
            res.append((kind, idx, p))
    return res


# ---------------------------------------------------------------- items
def covariates(lines, text, copy):
    """Entry-level covariates come from the untouched control lines, so a verdict flip cannot change them."""
    return {"second_prose_copy": copy, "n_table_lines": sum(is_table(l) for l in lines),
            "n_numeric_tokens": sum(len(tokens(l)) for l in lines), "has_threshold_language": threshold_language(lines),
            "n_chars": len(text)}


def item(sec, kind, idx, lines, plant, split, stratum):
    text = render_lines(lines)
    return {"item_id": sha1(sec["key"] + kind + str(idx))[:12], "entry_key": sec["key"], "header_id": sec["header_id"],
            "date": sec["date"], "stratum": stratum, "split": split, "kind": kind, "seed_idx": idx, "text": text,
            "n_lines": len(lines),
            "plant": None if plant is None else {"lines": [j + 1 for j in plant["changed"]], "old": plant["old"],
                                                 "new": plant["new"], "accept_lines": [j + 1 for j in plant["accept"]]},
            "covariates": dict(covariates(sec["lines"], text, None if plant is None else plant["copy"]), n_chars=len(text))}


def build_items_for_entries(keys, sections=None, with_plants=False):
    """Items for the given entry keys or header ids (controls only unless with_plants), same rendering as the pool.
    Used by the runner's --smoke mode for the excluded entries. No pool or critic-text assertions are applied."""
    if isinstance(keys, str):
        keys = [keys]
    if sections is None:
        sections = parse_sections(LEDGER.read_text())
    wl = load_whitelist()
    out = []
    for sec in sections:
        if sec["key"] in keys or sec["header_id"] in keys:
            prepare(sec, wl)
            out.append(item(sec, "control", 0, sec["lines"], None, "smoke", "excluded"))
            if with_plants:
                for kind, idx, p in make_plants(sec["lines"], sec["key"]):
                    out.append(item(sec, kind, idx, p["text"], p, "smoke", "excluded"))
    return out


# ---------------------------------------------------------------- main
def pool_status(sec, oracle_bodies, whitelist):
    """Return None if the section is in the pool, else the reason it is not."""
    hid = sec["header_id"]
    if hid.startswith("AR"):
        return "header id starts with AR"
    if sec["date"] is None or sec["date"] >= DATE_CUTOFF:
        return f"date {sec['date']} is not before {DATE_CUTOFF}"
    if hid in EXCLUDE:
        return "exclusion list: " + EXCLUDE[hid]
    if hid in PROBES:
        return "natural probe"
    prepare(sec, whitelist)
    if sec["residual"]:
        return f"critic text residue after stripping ({len(sec['residual'])} paragraph(s) not whitelisted)"
    nb = len(sec["lines"]) - 1
    if not 6 <= nb <= 80:
        return f"{nb} body lines outside 6..80"
    nt = sum(is_table(l) for l in sec["lines"])
    if nt < 2:
        return f"{nt} table lines (<2)"
    if not entry_numbers(sec["lines"]):
        return "no plantable number"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--print-trips", action="store_true", help="print tripping paragraphs of the whitelist-review entries and exit")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out)
    text = LEDGER.read_text()
    secs = parse_sections(text)
    oracle = json.loads(ORACLE.read_text())
    ob = {"\n".join(r["full_entry"].strip().split("\n")).strip() for r in oracle}
    for s in secs:
        s["stratum"] = "oracle" if "\n".join(s["body"]).strip() in ob else "extra"
    check(sum(s["stratum"] == "oracle" for s in secs) == len(oracle), "oracle entries not all matched to ledger sections")
    check(len({s["key"] for s in secs}) == len(secs), "entry_key collision")

    wl = load_whitelist()
    if a.print_trips:
        for s in secs:
            if s["stratum"] == "oracle":
                prepare(s, {})
                for h, t in s["trips"]:
                    mark = "IN-LIST" if s["header_id"] in TRIP_IDS else "other"
                    print(f"--- {s['header_id']} [{mark}] sha1={h}\n{t[:1500]}\n")
        return

    out.mkdir(parents=True, exist_ok=True)
    snap = out / "ledger_snapshot.md"
    shutil.copyfile(LEDGER, snap)
    led_sha = hashlib.sha256(snap.read_bytes()).hexdigest()
    led_lines = len(text.split("\n")) - (1 if text.endswith("\n") else 0)

    stages, excluded, pool = Counter(), [], []
    for s in secs:
        r = pool_status(s, ob, wl)
        stages[re.sub(r"\d+ (body|table)", r"N \1", r.split(" (")[0]) if r else "in pool"] += 1
        if r:
            excluded.append({"entry_key": s["key"], "header_id": s["header_id"], "header": s["header"][:100], "reason": r})
        else:
            pool.append(s)
    probes = [s for s in secs if s["header_id"] in PROBES and not s["header_id"].startswith("AR")]
    check(len(probes) == len(PROBES), f"expected {len(PROBES)} probe entries, found {len(probes)}")
    for s in probes:
        prepare(s, wl)
        check(not s["residual"], f"probe {s['header_id']} still has critic text residue")

    # split by entry
    order = sorted(pool, key=lambda s: hashlib.sha256((s["key"] + SALT).encode()).hexdigest())
    nburn = math.ceil(BURN_FRAC * len(pool))
    split = {s["key"]: ("burn" if k < nburn else "test") for k, s in enumerate(order)}
    split.update({s["key"]: "probe" for s in probes})

    items = []
    for s in pool + probes:
        sp = split[s["key"]]
        items.append(item(s, "control", 0, s["lines"], None, sp, s["stratum"]))
        if s in probes:
            continue
        for kind, idx, p in make_plants(s["lines"], s["key"]):
            items.append(item(s, kind, idx, p["text"], p, sp, s["stratum"]))
    check(len({i["item_id"] for i in items}) == len(items), "item_id collision")
    # residual critic-text paragraphs over everything written (whitelisted narration excepted)
    resid = 0
    for i in items:
        lines = [re.sub(r"^L\d+\| ?", "", l) for l in i["text"].split("\n")]
        resid += sum(1 for h, _ in trip_paragraphs(lines) if h not in wl and i["kind"] == "control")
    check(resid == 0, f"{resid} residual critic-text paragraphs")

    items.sort(key=lambda i: i["item_id"])
    random.Random(SHUFFLE_SEED).shuffle(items)
    ip = out / "items.jsonl"
    ip.write_text("".join(json.dumps(i) + "\n" for i in items))
    (out / "items.sha256").write_text(hashlib.sha256(ip.read_bytes()).hexdigest() + "  items.jsonl\n")
    (out / "split.json").write_text(json.dumps(split, indent=1, sort_keys=True))

    per_kind = Counter(i["kind"] for i in items if i["split"] != "probe")
    per_split_kind = {sp: dict(Counter(i["kind"] for i in items if i["split"] == sp)) for sp in ("burn", "test", "probe")}
    per_split_entries = Counter(split.values())
    wl_hits = [{"entry": s["header_id"], "sha1": h} for s in pool + probes for h in s["wl_hits"]]
    report = {
        "ledger_sha256": led_sha, "ledger_lines": led_lines, "n_sections": len(secs), "n_oracle": sum(s["stratum"] == "oracle" for s in secs),
        "stage_counts": dict(stages), "pool_entries": len(pool), "pool_oracle": sum(s["stratum"] == "oracle" for s in pool),
        "pool_extra": sum(s["stratum"] == "extra" for s in pool), "entries_per_split": dict(per_split_entries),
        "items_total": len(items), "items_per_kind_pool": dict(per_kind), "items_per_split_kind": per_split_kind,
        "controls_with_plants_by_kind_entries": {k: len({i["entry_key"] for i in items if i["kind"] == k and i["split"] != "probe"}) for k in KINDS},
        "excluded": excluded, "whitelist_hits": wl_hits, "whitelist_size": len(wl),
        "assertions_passed": CHECKS["total"], "residual_critic_paragraphs": resid,
        "pool_keys": sorted((s["key"], s["header_id"]) for s in pool), "probe_keys": [(s["key"], s["header_id"]) for s in probes],
    }
    (out / "pool_report.json").write_text(json.dumps(report, indent=1))

    # unit-test report
    print("== AR3 substrate unit-test report ==")
    print(f"ledger {led_lines} lines, sha256 {led_sha[:16]}, {len(secs)} sections, {report['n_oracle']} oracle")
    for k, v in stages.items():
        print(f"  {v:4d}  {k}")
    print(f"pool {len(pool)} entries ({report['pool_oracle']} oracle, {report['pool_extra']} extra); split {dict(per_split_entries)}")
    print("plants per kind (pool): " + ", ".join(f"{k}={per_kind.get(k, 0)}" for k in ["control"] + KINDS))
    for sp in ("burn", "test", "probe"):
        print(f"  {sp}: {per_split_kind[sp]}")
    print(f"probes: {[h for _, h in report['probe_keys']]}")
    print(f"residual critic-text paragraphs: {resid}")
    print(f"assertions passed: {CHECKS['total']} (0 failed)")
    print(f"items {len(items)} -> {ip}")


if __name__ == "__main__":
    main()
