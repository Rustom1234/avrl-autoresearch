"""Acceptance rules (DESIGN.md sections 5 and 8, PREREG.md decision rules). Pure functions, no I/O.

All values passed to the z rules are GPU-offset adjusted (see adjust). Arm A uses decide_base on raw values.
"""
import math
from statistics import fmean

Z_REJECT = 0.75        # z1 below this: reject
Z_ACCEPT = 2.0         # z1 at or above this: accept
Z_ACCEPT_REPLICATED = 1.5   # z over two or more runs at or above this: accept
TRIPWIRE_SE = 5.0      # single-step gain limit, in units of sigma_used * sqrt(2)
TRIPWIRE_CUMULATIVE = 0.03   # gain over the anchor that forces an audit


def adjust(value, gpu, gpu_offsets):
    """value minus the frozen per-GPU offset. gpu_offsets maps gpu id (int or str key) to offset; unknown gpu: 0."""
    off = gpu_offsets.get(gpu)
    if off is None:
        off = gpu_offsets.get(str(gpu), 0.0)
    return value - off


def z_score(inc_mean, inc_n, cand_values, sigma_used):
    """(incumbent_mean - candidate_mean) / (sigma_used * sqrt(1/n_c + 1/n_i)). Positive means the candidate is better."""
    n_c = len(cand_values)
    return (inc_mean - fmean(cand_values)) / (sigma_used * math.sqrt(1.0 / n_c + 1.0 / inc_n))


def decide_verified(incumbent, cand_adjusted_values, sigma_used):
    """Arms B, C, D. incumbent is the state.json incumbent dict (mean_adjusted, n).

    One run: reject if z1 < 0.75, accept if z1 >= 2.0, else "replicate". Two or more runs (candidate plus the
    replicate and/or audit): accept if z >= 1.5 over all of them, else reject. Returns (decision, z)."""
    z = z_score(incumbent["mean_adjusted"], incumbent["n"], cand_adjusted_values, sigma_used)
    if len(cand_adjusted_values) == 1:
        return ("reject" if z < Z_REJECT else "accept" if z >= Z_ACCEPT else "replicate"), z
    return ("accept" if z >= Z_ACCEPT_REPLICATED else "reject"), z


def decide_base(inc_raw_bar, cand_raw):
    """Arm A: keep only if the raw official value is strictly lower than the incumbent's raw bar."""
    return "accept" if cand_raw < inc_raw_bar else "reject"


def tripwire(inc_mean, cand_adjusted, anchor_mean, sigma_used):
    """True when an accept must first pass an audit: single-step gain over the incumbent above
    5 * sigma_used * sqrt(2), or cumulative gain over the anchor above 0.03 (both adjusted, lower is better)."""
    return ((inc_mean - cand_adjusted) > TRIPWIRE_SE * sigma_used * math.sqrt(2)
            or (anchor_mean - cand_adjusted) > TRIPWIRE_CUMULATIVE)


def _field(result, key):
    """A result.json field, looked up at the top level first and then in the nested harness dict."""
    if key in result:
        return result[key]
    h = result.get("harness")
    return h.get(key) if isinstance(h, dict) else None


def integrity_flags(result):
    """Integrity flags from a worker result dict (DESIGN section 4). Any flag auto-rejects the candidate."""
    f = []
    if result.get("mismatch_flag"):
        f.append("mismatch")
    if _field(result, "target_leak_flag") is True:
        f.append("target_leak")
    if _field(result, "canary_ok") is False:
        f.append("canary_failed")
    h = result.get("harness")
    if result.get("harness_missing") or (isinstance(h, dict) and h.get("error")):
        f.append("harness_missing")
    for name in result.get("integrity_failed") or []:
        tag = "worker:%s" % name
        if tag not in f:
            f.append(tag)
    if _field(result, "constants_ok") is False:
        f.append("constants")
    if any(_field(result, k) is False for k in ("prepare_code_hash_ok", "pipeline_code_hash_ok")):
        f.append("hash")
    status = result.get("status")
    if status == "overbudget" or result.get("overbudget"):
        f.append("overbudget")
    seed, matched = result.get("seed"), result.get("seed_lines_matched")
    if status == "invalid_seed" or (seed is not None and matched is not None and matched < 2) \
            or (seed is not None and result.get("seed_patched") is False):
        f.append("invalid_seed")
    if result.get("official_val_bpb") is None:
        f.append("no_official")
    return f
