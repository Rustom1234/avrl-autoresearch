"""analysis_p2b on synthetic loop tables: the sign-flip test, the endpoint and the verdict logic."""
import numpy as np
import pandas as pd

import analysis_p2b as B


def table(gains, costs, n=12, sd=0.002, seed=0):
    """One loop per cell per block with the given cell mean gains and model dollars."""
    rng = np.random.default_rng(seed)
    rows = [dict(loop_id="%s_%02d" % (c, b), cell=c, block=b, status="done", sealed_gain_global=g + rng.normal(0, sd),
                 model_usd=costs[c], total_usd_gpu_2=costs[c] + 4.0) for c, g in gains.items() for b in range(1, n + 1)]
    return pd.DataFrame(rows)


GAINS = dict(SS=0.055, QT=0.012, ST=0.056, H=0.033, F=0.040)
COSTS = dict(SS=1.43, QT=0.15, ST=0.25, H=1.45, F=0.20)


def test_signflip_exact():
    assert B.signflip([1.0] * 12, "greater") == 1 / 4096
    assert B.signflip([1.0] * 12, "less") == 1.0
    assert abs(B.signflip([1.0, -1.0] * 6, "greater") - 0.6128) < 0.001   # P(at least 6 of 12 positive)


def test_endpoint_reading():
    g, c = pd.Series(GAINS), pd.Series(COSTS)
    assert np.allclose(B.endpoint(g, c, "SS"), (1.0, 1.0, 0.0)) and np.allclose(B.endpoint(g, c, "QT"), (0.0, 0.0, 0.0))
    rec, ext, adv = B.endpoint(g, c, "ST")
    assert rec > 1.0 and ext < 0.1 and adv > 0.9


def test_verdicts():
    assert B.analyze(table(GAINS, COSTS))["verdict"] == "REPLICATED"
    res = B.analyze(table(dict(GAINS, H=0.055), COSTS))            # damaged loop as good as SS: validity check fails
    assert res["verdict"] == "KILL" and res["claims"]["C1"]["status"] == "not tested"
    assert B.analyze(table(dict(GAINS, ST=0.012), COSTS))["verdict"] == "NOT REPLICATED"
    res = B.analyze(table(dict(GAINS, ST=0.035), COSTS))           # beats the line, loses to SS by 0.02
    assert res["verdict"] == "PARTIAL (C3 not held)" and res["claims"]["C4"]["status"] == "not tested"


def test_cell_n_is_descriptive_only():
    """Cell N is reported on its own and cannot change the verdict or drop a block, even with loops missing."""
    base = B.analyze(table(GAINS, COSTS))
    n = table(dict(N=0.020), dict(N=0.0), n=10, seed=1)        # two N loops missing
    res = B.analyze(pd.concat([table(GAINS, COSTS), n], ignore_index=True))
    assert res["verdict"] == base["verdict"] == "REPLICATED" and res["claims"] == base["claims"]
    cn = res["cell_n"]
    assert cn["n_blocks"] == 10 and abs(cn["gain"] - 0.020) < 0.002
    assert cn["p_ss_above_n"] < 0.05 and cn["p_n_above_qt"] < 0.05
    assert "cell_n" not in base and "Cell N" in B.build_report(res) and "Cell N" not in B.build_report(base)
