"""L10a recorder check (CPU only, stdout only). Separately written recomputation from the saved JSONs:
(1) CH from each half's selection table (first listed wins ties), per-cell and pooled C = mean over 6 budgets of log(G/CH),
    percentile CI, basic CI, bias-centred CI, P(boot > bar);
(2) bootstrap bias (bootstrap mean minus plug-in) for each pooled contrast, learner and fixed-vs-fixed;
(3) overlap of the judge-error sets of the four pools.
Usage: python exp/l10a_recorder_check.py   (run from /homes/rdubash/avrl_live/FULL/avrl)
"""
import json, math
import numpy as np
import pandas as pd

R = 'runs/L10a-xfit-fixed-comparator/'
CELLS = [('0.5B q3_4b', 'L9c-weak-policy-tiebreak', 'q3_4b'), ('0.5B rm', 'L9c-weak-policy-tiebreak', 'rm'),
         ('1.5B q3_4b', 'L8c-A-fresh-tiebreak', 'q3_4b'), ('1.5B rm', 'L8c-A-fresh-tiebreak', 'rm')]
F = ['0.1', '0.15', '0.2', '0.25', '0.3', '0.4']
FIXED = ['agree_thr', 'judge_thr', 'blend50', 'agree_acc', 'blend50_acc', 'agree_acc_soth', 'agree_acc_sdiff']
B = 200


def load(pool, p, f):
    return json.load(open(f'{R}{pool}/{p}/xfit_f{f}.json'))


def ch_of(h):
    sel = h['selection']['mean_caught']
    best = max(sel.values())
    tied = [r for r in FIXED if sel[r] == best]
    return tied[0], len(tied) > 1


def arm_name(a, ch, frozen):
    return {'G': 'greedy_blend7', 'CH': ch, 'FROZEN': frozen}.get(a, a)


# ---- part 1
mis = ties = n = 0
C, plug = {}, {}
for name, pool, p in CELLS:
    per_b, pl = [], []
    for f in F:
        x = load(pool, p, f)
        G = np.zeros(B); K = np.zeros(B); Gp = Kp = 0.0
        for h in x['halves']:
            ch, tie = ch_of(h)
            n += 1; ties += tie; mis += (ch != h['full_half']['ch'])
            Gp += np.mean(h['full_half']['greedy_blend7']); Kp += np.mean(h['full_half'][ch])
            G += np.array([np.mean(r['greedy_blend7']) for r in h['resamples']])
            K += np.array([np.mean(r[ch]) for r in h['resamples']])
        per_b.append(np.log(G / K)); pl.append(math.log(Gp / Kp))
    C[name] = np.mean(per_b, axis=0); plug[name] = float(np.mean(pl))
print('CH mismatches', mis, 'of', n, '; exact ties at the top', ties)
for k in C:
    print('%-11s plug %+.4f boot_mean %+.4f pct CI [%+.4f, %+.4f] SD %.4f' % (
        k, plug[k], C[k].mean(), *np.percentile(C[k], [2.5, 97.5]), C[k].std(ddof=1)))
P = np.mean([C[k] for k in C], axis=0); pp = float(np.mean(list(plug.values())))
lo, hi = np.percentile(P, [2.5, 97.5])
print('POOLED plug %+.4f boot_mean %+.4f pct CI [%+.4f, %+.4f] half-width %.4f SD %.4f' % (
    pp, P.mean(), lo, hi, (hi - lo) / 2, P.std(ddof=1)))
print('basic CI [%+.4f, %+.4f]; bias-centred CI [%+.4f, %+.4f]' % (
    2 * pp - hi, 2 * pp - lo, lo - (P.mean() - pp), hi - (P.mean() - pp)))
print('plug-in percentile in bootstrap dist %.3f; P(boot>0) %.3f; P(boot>0.02) %.3f; P(boot<0.01) %.3f' % (
    np.mean(P < pp), np.mean(P > 0), np.mean(P > 0.02), np.mean(P < 0.01)))
print('PASS needs plug>=0.02:', pp >= 0.02, '| KILL needs CI upper<0.01: pct', hi < 0.01,
      'basic', (2 * pp - lo) < 0.01, 'bias-centred', (hi - (P.mean() - pp)) < 0.01)

# ---- part 2
pairs = [('G', 'CH'), ('CH', 'agree_acc'), ('G', 'agree_acc'), ('G', 'FROZEN'), ('FROZEN', 'agree_acc'),
         ('blend50_acc', 'agree_acc')]
acc = {pr: {c[0]: ([], []) for c in CELLS} for pr in pairs}
for name, pool, p in CELLS:
    for f in F:
        x = load(pool, p, f); frozen = x['frozen']
        num = {pr: np.zeros(B) for pr in pairs}; den = {pr: np.zeros(B) for pr in pairs}
        npl = {pr: 0.0 for pr in pairs}; dpl = {pr: 0.0 for pr in pairs}
        for h in x['halves']:
            ch, _ = ch_of(h)
            for pr in pairs:
                a, b = arm_name(pr[0], ch, frozen), arm_name(pr[1], ch, frozen)
                npl[pr] += np.mean(h['full_half'][a]); dpl[pr] += np.mean(h['full_half'][b])
                num[pr] += np.array([np.mean(r[a]) for r in h['resamples']])
                den[pr] += np.array([np.mean(r[b]) for r in h['resamples']])
        for pr in pairs:
            acc[pr][name][0].append(math.log(npl[pr] / dpl[pr])); acc[pr][name][1].append(np.log(num[pr] / den[pr]))
print('\ncontrast: pooled plug-in C | bootstrap mean C | bias (boot minus plug-in)   [FROZEN = agree_acc_soth for rm, agree_acc_sdiff for q3_4b]')
for pr in pairs:
    pl = np.mean([np.mean(acc[pr][c[0]][0]) for c in CELLS]); bm = np.mean([np.mean(acc[pr][c[0]][1]) for c in CELLS])
    print('%-22s %+.4f | %+.4f | %+.4f' % (pr[0] + '/' + pr[1], pl, bm, bm - pl))

# ---- part 3
S = {}
for name, pool, p in CELLS:
    d = pd.read_parquet(f'runs/{pool}/pool/{p}/feats_J.parquet')
    S[name] = set(d.pid[d['judge_error'] == 1]) if 'pid' in d else set(d.index[d['judge_error'] == 1])


def ov(a, b):
    return len(S[a] & S[b]) / min(len(S[a]), len(S[b]))


print('\nerror-set overlap (intersection / smaller set): within 0.5B %.2f, within 1.5B %.2f, across policy same proxy %.2f (q3_4b) %.2f (rm), cross %.2f %.2f' % (
    ov('0.5B q3_4b', '0.5B rm'), ov('1.5B q3_4b', '1.5B rm'), ov('0.5B q3_4b', '1.5B q3_4b'), ov('0.5B rm', '1.5B rm'),
    ov('0.5B q3_4b', '1.5B rm'), ov('0.5B rm', '1.5B q3_4b')))
