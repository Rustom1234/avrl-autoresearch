"""Build a pool of real web pages from Common Crawl text (WET) files, in the format run.py reads.

    curation/.venv/bin/python curation/skills/build_pool_web.py --name web1 --files 3 --n 120000 --seed 0

docs.json holds two views of every page: index i is the raw page text, index n + i is the same page after line-level
cleaning (ops.clean_lines). Both are cut to MAX_CHARS. The target sample and the dev and sealed sets are copied from
an existing pool (clean Wikipedia), so scores stay in the same unit as the earlier studies.
"""
import argparse, collections, gzip, json, random, shutil, subprocess, sys
from pathlib import Path
import numpy as np
import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "task"))
from common import DATA, T, MAX_CHARS          # noqa: E402
from build_pool import trunc                   # noqa: E402
from ops import clean_lines, norm_line         # noqa: E402

CC, BASE_URL = DATA / "raw/cc", "https://data.commoncrawl.org/"


def wet_records(path):
    """(url, language header, text) of every page in one WET file."""
    with gzip.open(path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                return
            if not line.startswith(b"WARC/"):
                continue
            head = {}
            while True:
                ln = f.readline().strip()
                if not ln:
                    break
                k, _, v = ln.partition(b":")
                head[k.decode().lower()] = v.strip().decode("utf-8", "replace")
            body = f.read(int(head.get("content-length", 0)))
            if head.get("warc-type") == "conversion":
                yield (head.get("warc-target-uri", ""), head.get("warc-identified-content-language", ""),
                       body.decode("utf-8", "replace"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--files", type=int, default=3)
    ap.add_argument("--n", type=int, default=120000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-chars", type=int, default=200, help="pages shorter than this are not part of the pool")
    ap.add_argument("--splits-from", default="v2b")
    a = ap.parse_args()
    out = DATA / ("pool_%s" % a.name)
    if out.exists():
        sys.exit("%s exists; pools are never overwritten" % out)
    rng = random.Random(a.seed)
    paths = gzip.open(CC / "wet.paths.gz", "rt").read().split()
    pages = []
    for p in rng.sample(paths, a.files):
        f = CC / Path(p).name
        if not f.exists():
            subprocess.run(["curl", "-s", "-f", "-m", "900", "-o", str(f) + ".part", BASE_URL + p], check=True)
            Path(str(f) + ".part").rename(f)
        got = [(u, lang, t) for u, lang, t in wet_records(f) if len(t) >= a.min_chars]
        print("%s: %d pages" % (f.name, len(got)), flush=True)
        pages += got
    pages = rng.sample(pages, min(a.n, len(pages)))
    n = len(pages)
    seen = collections.Counter()
    for _, _, t in pages:
        seen.update({norm_line(ln).lower() for ln in t.split("\n")})
    common = frozenset(ln for ln, c in seen.items() if c >= 3 and ln)
    docs = [trunc(t) for _, _, t in pages] + [trunc(clean_lines(t, common)) for _, _, t in pages]
    out.mkdir(parents=True)
    json.dump(docs, open(out / "docs.json", "w"))
    json.dump([dict(url=u, cc_lang=lang) for u, lang, _ in pages], open(out / "meta_HIDDEN.json", "w"))
    enc = tiktoken.get_encoding("gpt2")
    arrs = []
    for i in range(0, len(docs), 4000):                               # in pieces: token lists of the whole pool need 15 GB
        arrs += [np.array(t + [enc.eot_token], dtype=np.uint16)       # EOT is appended in training
                 for t in enc.encode_ordinary_batch(docs[i:i + 4000], num_threads=8)]
    lens = np.array([len(x) for x in arrs], dtype=np.int64)
    np.save(out / "pool_tokens.npy", np.concatenate(arrs))
    np.save(out / "pool_offsets.npy", np.concatenate([[0], np.cumsum(lens)]))
    src = DATA / ("pool_%s" % a.splits_from)
    for f in ("target_sample.json", "dev_tokens.npy", "sealed_tokens.npy"):
        shutil.copy(src / f, out / f)
    summ = dict(name=a.name, seed=a.seed, crawl=paths[0].split("/")[1], n_pages=n, n_docs=len(docs),
                pool_tokens=int(lens[:n].sum()), raw_over_T=float(lens[:n].sum() / T),
                cleaned_over_T=float(lens[n:].sum() / T), common_lines=len(common), splits_from=a.splits_from)
    json.dump(summ, open(out / "summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
