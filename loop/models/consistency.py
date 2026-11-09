"""Self-consistency sampling: k independent calls, agreement on one key of the returned object.

The same prompt is sent k times and the answers are compared on a single field (`key`) of the JSON reply. The modal
value is the consensus answer and its share among the usable replies (`agreement`) says how stable that answer is.
The loop engine uses this for its voter role: k votes on one item, read from the "vote" key (loop/core/arms.py).

Entry point: `sample_k()`. Each of the k calls goes through call.call(), so each is logged and costed on its own.
"""
from __future__ import annotations

import json
from collections import Counter

try:
    from .call import call
except ImportError:  # run with this directory on sys.path
    from call import call


def sample_k(model_name, user, k, *, key, **call_kwargs) -> dict:
    """Make k sequential independent calls (call_kwargs go to call(), e.g. schema, temperature, tag).

    key names the field of each reply object to compare, e.g. "label" or "vote".

    Returns {"results": [CallResult]*k, "values": [obj[key] for calls that are ok and contain key],
             "agreement": share of `values` equal to the modal value, "modal_value": ..., "n_ok": len(values)}.
    Failed calls are left out of the denominator, so read agreement together with n_ok.
    Ties go to the value seen first. With n_ok == 0: agreement 0.0, modal_value None.
    For vllm pass temperature > 0, otherwise the k samples are near-identical. Claude temperature is not controllable.
    """
    results = [call(model_name, user, **call_kwargs) for _ in range(k)]
    values = [r.obj[key] for r in results if r.ok and isinstance(r.obj, dict) and key in r.obj]
    canon = [json.dumps(v, sort_keys=True) for v in values]  # values may be unhashable (lists, dicts)
    if not values:
        return {"results": results, "values": values, "agreement": 0.0, "modal_value": None, "n_ok": 0}
    top, count = Counter(canon).most_common(1)[0]  # most_common keeps first-seen order among equal counts
    return {"results": results, "values": values, "agreement": count / len(values),
            "modal_value": values[canon.index(top)], "n_ok": len(values)}
