"""L5a critic: independent re-derivation of verdict numbers from raw per-seed JSONs (bootstrap seed 777, 20000 resamples). Prints unrounded numbers."""
import json, numpy as np, glob, os
R='runs/L5a/'; J={'3B':'Qwen2.5-3B-Instruct','7B':'Qwen2.5-7B-Instruct-AWQ'}; rng=np.random.default_rng(777)
def boot(n,d,nb=20000):
    k=len(n); idx=rng.integers(0,k,(nb,k)); r=n[idx].sum(1)/d[idx].sum(1); return np.percentile(r,[2.5,97.5])
tot={}
for lab,j in J.items():
    for B in (62,125):
        D=[json.load(open(f'{R}carry_{j}_B{B}_o{s}.json')) for s in range(8)]
        for d in D:
            assert d['budget']==B and d['n_batches']==12 and d['m']==250 and d['features'][0]=='agreement'
            for k,v in d['arms'].items(): assert len(v)==12, (k,len(v))
        A=lambda a: np.array([d['arms'][a] for d in D],float)
        P=lambda a,sd=range(8): A(a)[list(sd),3:].sum(1)
        thr=P('thr_agreement')
        lamA='lam1' if P('carry_offset_lam1',range(4)).mean()>P('carry_offset_lam10',range(4)).mean() else 'lam10'
        lamB='lam1' if P('carry_offset_lam1',range(4,8)).mean()>P('carry_offset_lam10',range(4,8)).mean() else 'lam10'
        print(f'== {lab} B={B} lamA={lamA} lamB={lamB}  hits lam10/lam1 seeds0-3: {P("carry_offset_lam10",range(4)).mean():.2f}/{P("carry_offset_lam1",range(4)).mean():.2f}  seeds4-7: {P("carry_offset_lam10",range(4,8)).mean():.2f}/{P("carry_offset_lam1",range(4,8)).mean():.2f}')
        sd=list(range(4,8)); t=P('thr_agreement',sd)
        for arm in (f'carry_offset_{lamA}','carry_plain'):
            x=P(arm,sd); lo,hi=boot(x,t); print(f'  verdictfold {arm}/thr = {x.sum()/t.sum():.6f} CI {lo:.4f} {hi:.4f}')
        x=P(f'carry_window250_{lamA}',sd); y=P(f'carry_offset_{lamA}',sd); print(f'  win250/carry = {x.sum()/y.sum():.6f}  {boot(x,y)}')
        # per-seed ratios in verdict fold
        print('  per-seed carry_offset/thr (seeds 4-7):',np.round(P(f'carry_offset_{lamA}',sd)/t,4), ' carry_plain:',np.round(P('carry_plain',sd)/t,4))
        # mean-of-ratio alternative
        print('  mean-of-seed-ratios carry_offset', (P(f'carry_offset_{lamA}',sd)/t).mean())
        # both lam in verdict fold
        for l in ('lam10','lam1'):
            x=P(f'carry_offset_{l}',sd); print(f'  verdictfold carry_offset_{l}/thr={x.sum()/t.sum():.6f}')
        # pooled 8 with chosen lam
        tot[(lab,B)]=D
