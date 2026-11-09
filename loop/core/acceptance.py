"""Acceptance rules: does a candidate train.py replace the incumbent?

Where it sits. After a slot (one 300 s training run on one GPU) finishes, arms.py passes the worker's
result through these functions and writes the decision to the ledger. Pure functions, no I/O. The rules
and thresholds are frozen in loop/design/DESIGN.md (sections 5 and 8) and PREREG.md (decision rules).

Terms:
  val_bpb     validation bits per byte of the trained model, lower is better.
  incumbent   the best train.py accepted so far; a candidate is compared with the mean of its runs.
  anchor      the unmodified baseline measured before launch; it is the first incumbent.
  sigma_used  run-to-run noise of val_bpb for an unchanged file (fixed before launch), so a z value
              reads as "gain in units of noise".
  arm         one loop configuration. A is the one-shot control; B, C and D add verification.

Entry points:
  adjust           remove the frozen per-GPU offset from a raw value.
  decide_verified  arms B, C, D: z thresholds with a "replicate" zone in the middle.
  decide_base      arm A: strict comparison of raw values, no noise model.
  tripwire         gains too large to accept without an audit run.
  integrity_flags  harness checks that reject a run whatever its number.

All values passed to the z rules are GPU-offset adjusted (see adjust). Arm A uses decide_base on raw values.
"""
import math
from statistics import fmean

# Thresholds in z units (see z_score). Between Z_REJECT and Z_ACCEPT one run cannot separate a real gain
# from noise, so the candidate is run again and judged on all of its runs.
Z_REJECT = 0.75        # z1 below this: reject
Z_ACCEPT = 2.0         # z1 at or above this: accept
Z_ACCEPT_REPLICATED = 1.5   # z over two or more runs at or above this: accept
TRIPWIRE_SE = 5.0      # single-step gain limit, in units of sigma_used * sqrt(2)
TRIPWIRE_CUMULATIVE = 0.03   # gain over the anchor that forces an audit


def adjust(value, gpu, gpu_offsets):
    """value minus the frozen per-GPU offset. gpu_offsets maps gpu id (int or str key) to offset; unknown gpu: 0.

    The offset of a GPU is its mean anchor val_bpb minus the mean over all GPUs, so adjusted values from
    different cards are comparable. Keys may be str because the offsets are stored in a JSON config."""
    off = gpu_offsets.get(gpu)
    if off is None:
        off = gpu_offsets.get(str(gpu), 0.0)
    return value - off


def z_score(inc_mean, inc_n, cand_values, sigma_used):
    """(incumbent_mean - candidate_mean) / (sigma_used * sqrt(1/n_c + 1/n_i)). Positive means the candidate is better.

    A two-sample z statistic with known noise: the denominator is the standard error of the difference of
    two means, n_i runs behind the incumbent (inc_n) and n_c candidate runs (len(cand_values))."""
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
    5 * sigma_used * sqrt(2), or cumulative gain over the anchor above 0.03 (both adjusted, lower is better).

    sigma_used * sqrt(2) is the standard error of a difference of two single runs, so the first test reads
    "more than 5 standard errors in one step". A gain that large is more often a measurement fault or metric
    gaming than a real improvement, hence the extra audit run (a fresh seed with sealed evaluation)."""
    return ((inc_mean - cand_adjusted) > TRIPWIRE_SE * sigma_used * math.sqrt(2)
            or (anchor_mean - cand_adjusted) > TRIPWIRE_CUMULATIVE)


def _field(result, key):
    """A result.json field, looked up at the top level first and then in the nested harness dict."""
    if key in result:
        return result[key]
    h = result.get("harness")
    return h.get(key) if isinstance(h, dict) else None


def integrity_flags(result):
    """Integrity flags from a worker result dict (DESIGN section 4). Any flag auto-rejects the candidate.

    Returns a list of short tags, empty when the run is clean. The worker and the harness tail (the scoring
    code the harness appends to every train.py) set the underlying fields; this function only names them:
      mismatch         printed val_bpb differs from the value the harness recomputed
      target_leak      the model's forward pass peeks at the targets (loss with targets given differs
                       from the loss recomputed from its logits)
      canary_failed    scoring a uniform-logits stand-in model did not give the known exact value, so
                       the scoring path itself was altered
      harness_missing  no harness result, or the harness reported an error
      worker:<name>    a named integrity check failed in the worker
      constants, hash  evaluation constants or the pinned prepare.py / pipeline code changed
      overbudget       startup plus training ran past the wall-time limit
      invalid_seed     the two harness-owned seed lines were not found or not patched
      no_official      the run produced no official val_bpb"""
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
