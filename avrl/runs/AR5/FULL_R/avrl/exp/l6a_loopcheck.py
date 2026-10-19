import json, numpy as np, glob
R='/homes/rdubash/avrl_live/FULL_R/avrl/runs/'
J={'3B':'Qwen2.5-3B-Instruct','7B':'Qwen2.5-7B-Instruct-AWQ'}
ref=['thr_agreement','uniform','restart_offset','carry_plain','carry_offset_lam1','static_out']
fam=[f'amort_{s}_{n}' for s in ('seed','rand','split') for n in (125,250,500)]
# 1. regression gate
diff=0; nv=0
for j in J.values():
    for B in (62,125):
        for o in range(8):
            a=json.load(open(f'{R}L6a/amort_{j}_B{B}_o{o}.json')); b=json.load(open(f'{R}L5a/carry_{j}_B{B}_o{o}.json'))
            for r in ref:
                x=a['arms'][r]; y=b['arms'][r]
                x=x['hits'] if isinstance(x,dict) else x; y=y['hits'] if isinstance(y,dict) else y
                nv+=len(x); diff+=int(list(x)!=list(y))
            diff+=int(a['batch_base_errors']!=b['batch_base_errors'])
print('gate vectors diff',diff,'values',nv)
# 2. headline
def get(j,B,o,arm):
    a=json.load(open(f'{R}L6a/amort_{j}_B{B}_o{o}.json'))['arms'][arm]
    return np.array(a['hits'] if isinstance(a,dict) else a,float)
rng=np.random.default_rng(7)
for name,j in J.items():
    for B in (62,125):
        tot={arm:np.array([get(j,B,o,arm).sum() for o in range(16)]) for arm in fam+['carry_plain','thr_agreement']}
        sel=max(fam,key=lambda a: tot[a][:8].mean())
        c=tot['carry_plain'][8:]; b=tot[sel][8:]
        bs=[c[i].mean()/b[i].mean() for i in (rng.integers(0,8,8) for _ in range(10000))]
        # post hoc best on all 16
        ph=max(fam,key=lambda a: tot[a].mean())
        print(name,B,sel,'ratio %.3f CI %.3f-%.3f'%(c.mean()/b.mean(),*np.percentile(bs,[2.5,97.5])),'best/carry %.3f'%(b.mean()/c.mean()),'| posthoc16',ph,'%.3f'%(tot[ph].mean()/tot['carry_plain'].mean()), '| carry/thr %.3f'%(tot['carry_plain'].mean()/tot['thr_agreement'].mean()))
        # seed-level: fraction of seeds where carry_plain > best (verdict-fold arm)
        print('   per-seed carry>sel on 16:', int((tot['carry_plain']>tot[sel]).sum()),'/16; min per-seed ratio %.3f'%((tot['carry_plain']/tot[sel]).min()))
