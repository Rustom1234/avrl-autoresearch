import json, numpy as np
R='runs/L5a/'; J={'3B':'Qwen2.5-3B-Instruct','7B':'Qwen2.5-7B-Instruct-AWQ'}
for lab,j in J.items():
    for B in (62,125):
        D=[json.load(open(f'{R}carry_{j}_B{B}_o{s}.json')) for s in range(8)]
        A=lambda a: np.array([d['arms'][a] for d in D],float)
        thr=A('thr_agreement'); sl=lambda a,lo,hi: A(a)[:,lo:hi].sum()/thr[:,lo:hi].sum()
        print(f'== {lab} B={B}')
        for a in ('carry_plain','carry_offset_lam1','restart_plain','restart_offset','static_out'):
            print(f'  {a:18s} b1-3 {sl(a,0,3):.3f} b4-6 {sl(a,3,6):.3f} b7-9 {sl(a,6,9):.3f} b10-12 {sl(a,9,12):.3f}  b7-12 {sl(a,6,12):.3f}  slope/batch (per-batch ratio OLS) {np.polyfit(np.arange(12),A(a).sum(0)/thr.sum(0),1)[0]:+.4f}')
        # gain retained by window relative to unbounded, over thr
        g=lambda a: A(a)[:,3:].sum()/thr[:,3:].sum()-1
        for lam in ('lam1','lam10'):
            print(f'  {lam}: unbounded gain {g("carry_offset_"+lam):+.4f}  retained share W125/W250/W500 = '+'/'.join(f'{g("carry_window%d_%s"%(W,lam))/g("carry_offset_"+lam):.2f}' for W in (125,250,500)))
        # restart_plain per-batch range
        r=A('restart_plain').sum(0)/thr.sum(0); print('  restart_plain per-batch',np.round(r,3))
