import json, numpy as np, math
R='/homes/rdubash/avrl_live/FULL_R/avrl/runs/L6a/'
J={'3B':'Qwen2.5-3B-Instruct','7B':'Qwen2.5-7B-Instruct-AWQ'}
fam=[f'amort_{s}_{n}' for s in ('seed','rand','split') for n in (125,250,500)]
def ld(j,B,arm):
    out=[]
    for o in range(16):
        a=json.load(open(f'{R}amort_{j}_B{B}_o{o}.json'))['arms'][arm]
        out.append(a['hits'] if isinstance(a,dict) else a)
    return np.array(out,float).mean(0)
for name,j in J.items():
    for B in (62,125):
        c=ld(j,B,'carry_plain'); cr=c[6:].mean()
        res=[]
        for H in (12,24,36,60,120):
            best=None
            for arm in fam:
                n=int(arm.split('_')[-1]); k=math.ceil(n/B); f=ld(j,B,arm)
                lo=max(k,6)  # batches k+1.. (0-index k), at least batch 7
                sr=f[lo:].mean()
                tot=f.sum()+(H-12)*sr
                if best is None or tot>best[0]: best=(tot,arm,sr)
            ct=c.sum()+(H-12)*cr
            res.append('H=%d %.3f(%s)'%(H,ct/best[0],best[1].replace('amort_','')))
        print(name,B,' | '.join(res))
