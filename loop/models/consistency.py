"""Self-consistency sampling: k independent calls, agreement on one key of the returned object."""
from __future__ import annotations

import json
from collections import Counter

try:
    from .call import call
except ImportError:  # run with this directory on sys.path
    from call import call


def sample_k(model_name, user, k, *, key, **call_kwargs) -> dict:
    """Make k sequential independent calls (call_kwargs go to call(), e.g. schema, temperature, tag).

    Returns {"results": [CallResult]*k, "values": [obj[key] for calls that are ok and contain key],
             "agreement": share of `values` equal to the modal value, "modal_value": ..., "n_ok": len(values)}.
    Ties go to the value seen first. With n_ok == 0: agreement 0.0, modal_value None.
    For vllm pass temperature > 0, otherwise the k samples are near-identical. Claude temperature is not controllable.
    """
    results = [call(model_name, user, **call_kwargs) for _ in range(k)]
    values = [r.obj[key] for r in results if r.ok and isinstance(r.obj, dict) and key in r.obj]
    canon = [json.dumps(v, sort_keys=True) for v in values]  # values may be unhashable (lists, dicts)
    if not values:
        return {"results": results, "values": values, "agreement": 0.0, "modal_value": None, "n_ok": 0}
    top, count = Counter(canon).most_common(1)[0]
    return {"results": results, "values": values, "agreement": count / len(values),
            "modal_value": values[canon.index(top)], "n_ok": len(values)}
