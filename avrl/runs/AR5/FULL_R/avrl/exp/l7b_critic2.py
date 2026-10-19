import json, numpy as np
R='/homes/rdubash/avrl_live/FULL_R/avrl/runs'
J={'3B':'Qwen2.5-3B-Instruct','7B':'Qwen2.5-7B-Instruct-AWQ'}
cells={}
infos={}
for lab,j in J.items():
  for B in (62,125):
    A={};I=[]
    for o in range(16):
      d=json.load(open(f'{R}/L7b/design_{j}_B{B}_o{o}.json'))
      for k,v in d['arms'].items(): A.setdefault(k,[]).append(v)
      I.append(d['arm_info'])
    cells[(lab,B)]=({k:np.array(v,float) for k,v in A.items()},I)
order=[(l,B) for l in ('3B','7B') for B in (62,125)]
rows=['amort_strat_125','amort_strat_250','amort_strat_500','amort_bnd_125','amort_bnd_250','amort_bnd_500','freeze_plain_125','freeze_plain_250','freeze_plain_500','thr_agreement']
print('TABLE total/cp (post-cal/cp) , total/thr')
for r in rows+['static_out','amort_rand_125','amort_seed_500','amort_rand_250','amort_rand_500']:
  out=[]
  for c in order:
    A,_=cells[c];B=c[1]
    t=r.rpartition('_')[2]
    k=-(-int(t)//B) if t.isdigit() and r.startswith(('amort_','freeze_')) else 6
    tot=A[r].sum()/A['carry_plain'].sum(); pc=A[r][:,k:].sum()/A['carry_plain'][:,k:].sum(); th=A[r].sum()/A['thr_agreement'].sum()
    out.append('%.3f (%.3f) /%.3f'%(tot,pc,th))
  print(r.ljust(18),' | '.join(out))
print('cp, thr abs means')
for c in order:
  A,_=cells[c]; print(c,'%.1f %.1f'%(A['carry_plain'].sum(1).mean(),A['thr_agreement'].sum(1).mean()), 'base err/batch %.1f'%0)
# deficit decomposition
print('deficit cal/post (share of cp total)')
for r in ['amort_strat_125','amort_strat_500','amort_bnd_125','amort_bnd_500']:
  out=[]
  for c in order:
    A,_=cells[c];B=c[1];k=-(-int(r.rpartition('_')[2])//B)
    tot=A['carry_plain'].sum()
    out.append('%.1f/%.1f'%(100*(A['carry_plain'][:,:k].sum()-A[r][:,:k].sum())/tot,100*(A['carry_plain'][:,k:].sum()-A[r][:,k:].sum())/tot))
  print(r,' '.join(out))
# strat vs rand
print('strat_125 vs rand_125 /cp',[('%.3f'%(cells[c][0]['amort_strat_125'].sum()/cells[c][0]['carry_plain'].sum()),'%.3f'%(cells[c][0]['amort_rand_125'].sum()/cells[c][0]['carry_plain'].sum())) for c in order])
# calibration batch hit-rate
print('calib hit rate strat_125 / carry (batches 1..k, per B)')
for c in order:
  A,_=cells[c];B=c[1];k=-(-125//B)
  print(c,'%.2f %.2f'%(A['amort_strat_125'][:,:k].mean()/B,A['carry_plain'][:,:k].mean()/B))
# new finding: post-cal hits ratio to freeze_plain_n at B=125
print('B=125 post-cal ratio to freeze')
for lab in ('3B','7B'):
  A,_=cells[(lab,125)]
  for f in ('strat','bnd'):
    out=[]
    for n in (125,250,500):
      k=-(-n//125); out.append('%.3f'%(A[f'amort_{f}_{n}'][:,k:].sum()/A[f'freeze_plain_{n}'][:,k:].sum()))
    print(lab,f,out)
print('B=62 post-cal ratio to freeze')
for lab in ('3B','7B'):
  A,_=cells[(lab,62)]
  for f in ('strat','bnd'):
    out=[]
    for n in (125,250,500):
      k=-(-n//62); out.append('%.3f'%(A[f'amort_{f}_{n}'][:,k:].sum()/A[f'freeze_plain_{n}'][:,k:].sum()))
    print(lab,f,out)
# post-cal vs carry_plain ahead percent
for lab in ('3B','7B'):
 for B in (62,125):
  A,_=cells[(lab,B)]
  print(lab,B,'carry ahead after cal vs strat_125/250/500:',['%.3f'%(A['carry_plain'][:,-(-n//B):].sum()/A[f'amort_strat_{n}'][:,-(-n//B):].sum()) for n in (125,250,500)],'vs freeze',['%.3f'%(A['carry_plain'][:,-(-n//B):].sum()/A[f'freeze_plain_{n}'][:,-(-n//B):].sum()) for n in (125,250,500)])
# per-seed oracle
cnt=0
for c in order:
  A,_=cells[c]
  for o in range(16):
    best=max(A[k][o].sum() for k in ['amort_strat_125','amort_strat_250','amort_strat_500','amort_bnd_125','amort_bnd_250','amort_bnd_500'])
    cnt+= best>=A['carry_plain'][o].sum()
print('per-seed oracle best>=carry count of 64:',cnt)
# chosen arm below carry count of 16 seeds
for c in order:
  A,_=cells[c]
  bA=max(['amort_strat_125','amort_strat_250','amort_strat_500','amort_bnd_125','amort_bnd_250','amort_bnd_500'],key=lambda k:A[k][:8].sum(1).mean())
  print(c,bA,'below cp count',int((A[bA].sum(1)<A['carry_plain'].sum(1)).sum()),'min cp/best %.3f'%(A['carry_plain'].sum(1)/A[bA].sum(1)).min())
