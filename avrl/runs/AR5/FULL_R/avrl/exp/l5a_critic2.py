import json, numpy as np
R='runs/L5a/'; J={'3B':'Qwen2.5-3B-Instruct','7B':'Qwen2.5-7B-Instruct-AWQ'}; rng=np.random.default_rng(31337)
def boot(n,d,nb=20000):
    k=len(n); idx=rng.integers(0,k,(nb,k)); r=n[idx].sum(1)/d[idx].sum(1); return np.percentile(r,[2.5,97.5])
for lab,j in J.items():
    for B in (62,125):
        D=[json.load(open(f'{R}carry_{j}_B{B}_o{s}.json')) for s in range(8)]
        A=lambda a: np.array([d['arms'][a] for d in D],float)
        P=lambda a,sd=range(8): A(a)[list(sd),3:].sum(1)
        thr=A('thr_agreement'); t=P('thr_agreement')
        print(f'== {lab} B={B}  mean errors/batch {np.mean([d["batch_base_errors"] for d in D]):.1f}  thr mean 4-12 {t.mean()/9:.1f}')
        g=lambda a: (P(a).sum()-t.sum())/(P('static_out').sum()-t.sum())
        for a in ('static_out','carry_plain','carry_offset_lam1','carry_offset_lam10','carry_window250_lam1','carry_window500_lam1','carry_window125_lam1','calibrate_once','restart_offset','restart_plain','uniform'):
            x=P(a); lo,hi=boot(x,t); print(f'  {a:22s} {x.sum()/t.sum():.3f} ({lo:.3f},{hi:.3f})' + ('' if a in('static_out','uniform') else f'  gap {g(a):.2f}'))
        for W in (125,250,500):
            print(f'  W{W}/lam1 {P(f"carry_window{W}_lam1").sum()/P("carry_offset_lam1").sum():.3f}  lam10 {P(f"carry_window{W}_lam10").sum()/P("carry_offset_lam10").sum():.3f}')
        # per batch ratios (ratio of means across seeds)
        for a in ('restart_plain','restart_offset','carry_plain','carry_offset_lam1'):
            r=A(a).sum(0)/thr.sum(0); print(f'  per-batch {a} min {r.min():.3f} max {r.max():.3f}  b1-3 cum {A(a)[:,:3].sum()/thr[:,:3].sum():.3f}  first>=1.03 raw {next((i+1 for i,v in enumerate(r) if v>=1.03),None)}; trailing3 {next((i+1 for i in range(2,12) if A(a)[:,max(0,i-2):i+1].sum()/thr[:,max(0,i-2):i+1].sum()>=1.03),None)}')
        # all12 ratio
        for a in ('carry_plain','carry_offset_lam1'): print(f'  all12 {a} {A(a).sum()/thr.sum():.3f}')
        # diag bias
        for a in ('carry_plain','carry_offset_lam1'):
            e=np.mean([[b['err_rate'] for b in d['diag'][a]] for d in D],0); z=np.mean([[b['mean_z_agree'] for b in d['diag'][a]] for d in D],0); n=np.mean([[b['n'] for b in d['diag'][a]] for d in D],0)
            print(f'  bias {a} n {n[0]:.0f}/{n[3]:.0f}/{n[11]:.0f} err {e[0]:.3f}/{e[3]:.3f}/{e[11]:.3f} z {z[0]:.2f}/{z[3]:.2f}/{z[11]:.2f}')
        # label count budget check: n labels after 12 batches
        print('  n labels after b12 (carry_plain):', {d['diag']['carry_plain'][-1]['n'] for d in D})
        # total hits per batch cannot exceed B and cannot exceed base errors
        for d in D:
            for a,v in d['arms'].items():
                assert max(v)<=B, (a,v)
            for a in ('thr_agreement','carry_plain'):
                assert all(h<=min(B,e) for h,e in zip(d['arms'][a],d['batch_base_errors']))
        # Check batch_base_errors identical across arms? same across order seeds sums
        print('  sum base errors per order seed:', sorted({sum(d['batch_base_errors']) for d in D}))
