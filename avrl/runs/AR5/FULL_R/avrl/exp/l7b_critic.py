import json, os, numpy as np, sys
R='/homes/rdubash/avrl_live/FULL_R/avrl/runs'
J={'3B':'Qwen2.5-3B-Instruct','7B':'Qwen2.5-7B-Instruct-AWQ'}
DES=[f'amort_{f}_{n}' for f in ('strat','bnd') for n in (125,250,500)]
GATE=['thr_agreement','carry_plain']+[f'amort_{f}_{n}' for f in ('seed','rand','split') for n in (125,250,500)]+[f'freeze_plain_{n}' for n in (125,250,500)]
res=json.load(open(f'{R}/L7b/result.json'))
print('outcome',res['verdict']['outcome'],'gate',res['regression_gate_vs_L6a']['passed'],res['regression_gate_vs_L6a']['batch_value_differences'])
# gate recompute over all L6a arms
diffs=0;nv=0;missing=0;chkfail=0;badlen=0;oor=0
cells={}
for lab,j in J.items():
  for B in (62,125):
    A={}
    for o in range(16):
      f=f'{R}/L7b/design_{j}_B{B}_o{o}.json'; g=f'{R}/L6a/amort_{j}_B{B}_o{o}.json'
      d=json.load(open(f)); r=json.load(open(g))
      chkfail+=sum(1 for v in d['checks'].values() if not v)
      assert d['budget']==B and d['order_seed']==o and d['policy_seed']==o
      for k,v in d['arms'].items():
        if len(v)!=12: badlen+=1
        if min(v)<0 or max(v)>B: oor+=1
        A.setdefault(k,[]).append(v)
      for k in r['arms']:
        a=d['arms'][k]; b=r['arms'][k]; nv+=len(b); diffs+=sum(x!=y for x,y in zip(a,b))
      nv+=12; diffs+=sum(x!=y for x,y in zip(d['batch_base_errors'],r['batch_base_errors']))
    cells[(lab,B)]={k:np.array(v,float) for k,v in A.items()}
print('gate: values',nv,'diffs',diffs,'chkfail',chkfail,'badlen',badlen,'out-of-range',oor)
rng=np.random.default_rng(999)
def boot(num,den,nb=10000):
    n=len(num); idx=rng.integers(0,n,(nb,n)); rb=num[idx].mean(1)/den[idx].mean(1); return num.mean()/den.mean(),np.percentile(rb,2.5),np.percentile(rb,97.5)
for (lab,B),A in cells.items():
    T=lambda k,sl=slice(None):A[k][sl].sum(1)
    bA=max(DES,key=lambda k:T(k,slice(0,8)).mean())
    r,lo,hi=boot(T('carry_plain',slice(8,16)),T(bA,slice(8,16)))
    # oracle best on verdict fold
    orc=max(DES,key=lambda k:T(k,slice(8,16)).mean())
    r2=T('carry_plain',slice(8,16)).mean()/T(orc,slice(8,16)).mean()
    # best of all 16 seeds union incl L6a arms
    allarms=[k for k in A if k.startswith('amort_')]
    ub=max(allarms,key=lambda k:T(k).mean())
    # batch-level bootstrap on verdict fold (8 seeds x12 batches)
    nb=A['carry_plain'][8:16].reshape(-1); bb=A[bA][8:16].reshape(-1)
    idx=rng.integers(0,len(nb),(10000,len(nb))); rbb=nb[idx].mean(1)/bb[idx].mean(1)
    print(lab,B,'best',bA,'cp/best %.3f (%.3f-%.3f)'%(r,lo,hi),'oracle-on-fold',orc,'%.3f'%r2,'| union best16',ub,'%.3f'%(T(ub).mean()/T('carry_plain').mean()),'| batchboot CI %.3f-%.3f P>=1.03 %.2f'%(np.percentile(rbb,2.5),np.percentile(rbb,97.5),(rbb>=1.03).mean()),'| cp16 %.1f thr16 %.1f'%(T('carry_plain').mean(),T('thr_agreement').mean()))
    rc=res['cells'][f'{lab if lab=="3B" else "7B-AWQ"} B={B}']['verdict_folds']['A']
    print('   result.json:',rc['best_arm'],round(rc['carry_plain_over_best']['ratio'],3),round(rc['carry_plain_over_best']['ci_lo'],3),round(rc['carry_plain_over_best']['ci_hi'],3))
    # best/thr
    print('   best/thr %.3f best/cp %.3f'%(T(bA,slice(8,16)).mean()/T('thr_agreement',slice(8,16)).mean(),1/r))
