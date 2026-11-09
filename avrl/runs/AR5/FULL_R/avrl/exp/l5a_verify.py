"""Independent re-derivation of the L5a verdict numbers from the 32 raw per-seed JSONs (own bootstrap seed 424242, not the summarizer's code path).
Prints, per cell: lam chosen on order seeds 0-3, fold A (4-7) / fold B (0-3) / pooled 8 ratios to thr_agreement over batches 4-12 with 10,000-resample percentile CIs,
window / carry_offset ratios, gap captured, plus two post hoc extras: cumulative ratio over ALL 12 batches (equal total budget, warm-up included) and per-order-seed counts above 1."""
import json, numpy as np
R = 'runs/L5a/'; judges = ['Qwen2.5-3B-Instruct', 'Qwen2.5-7B-Instruct-AWQ']; rng = np.random.default_rng(424242)
def boot(num, den, nb=10000):
    n = len(num); idx = rng.integers(0, n, (nb, n)); return np.percentile(num[idx].sum(1) / den[idx].sum(1), [2.5, 97.5])
for j in judges:
    for B in (62, 125):
        D = [json.load(open(f'{R}carry_{j}_B{B}_o{s}.json')) for s in range(8)]
        A = lambda arm: np.array([d['arms'][arm] for d in D], float)
        pool = lambda arm, seeds=range(8): A(arm)[list(seeds), 3:12].sum(1)
        lam = 'lam1' if pool('carry_offset_lam1', range(4)).mean() > pool('carry_offset_lam10', range(4)).mean() else 'lam10'
        print(f'== {j} B={B} lam(sel on seeds 0-3)={lam} thr mean errors per batch 4-12 = {pool("thr_agreement").mean() / 9:.1f}')
        for fold, sd in (('A', range(4, 8)), ('B', range(0, 4)), ('P8', range(8))):
            sd = list(sd); tt = pool('thr_agreement', sd); r = {}
            for arm in ['carry_plain', 'carry_offset_lam10', 'carry_offset_lam1', 'restart_offset', 'restart_plain', 'calibrate_once', 'static_out']:
                x = pool(arm, sd); r[arm] = (x.sum() / tt.sum(),) + tuple(boot(x, tt))
            for W in (125, 250, 500):
                x = pool(f'carry_window{W}_{lam}', sd); y = pool('carry_offset_' + lam, sd); r[f'win{W}/carry'] = (x.sum() / y.sum(),) + tuple(boot(x, y))
            print(' fold', fold, ' '.join(f'{k}={v[0]:.3f}({v[1]:.3f},{v[2]:.3f})' for k, v in r.items()))
        tt = pool('thr_agreement'); so = pool('static_out')
        print('  gap captured (batches 4-12, all 8):', ' '.join(f'{a}={(pool(a).sum() - tt.sum()) / (so.sum() - tt.sum()):.2f}' for a in ['carry_plain', 'carry_offset_lam1', 'carry_offset_lam10', 'calibrate_once']))
        thr = A('thr_agreement')
        for arm in ['carry_plain', 'carry_offset_lam1', 'carry_offset_lam10', 'calibrate_once', 'restart_offset']:
            x = A(arm); per = x[:, 3:].sum(1) / thr[:, 3:].sum(1)
            print(f'  [post hoc] {arm}: all-12-batch ratio {x.sum() / thr.sum():.3f}; order seeds above 1 over batches 4-12: {int((per > 1).sum())}/8 (min {per.min():.3f})')
        print('  batch-1 equality checks pass in all 8 files:', all(all(d['checks'].values()) for d in D))
