"""L11b critic recheck (CPU): independent recomputation of the join, the n_steps AUC delta, the leak diagnostic, top-k precision, and the replay ratios from raw files. Usage: python exp/l11b_critic.py (run from avrl/)."""
# ---- part 1: join, CV delta, leak diagnostic
import numpy as np, pandas as pd, json, glob, re
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
R='/homes/rdubash/avrl_live/FULL_R/avrl/'
f=pd.read_parquet(R+'runs/p0/feats.parquet')
g=pd.concat([pd.read_parquet(R+f'data/gsm8k_{n}.parquet') for n in ('train','test')])
print('feats rows',len(f),'gsm8k rows',len(g),'dup q in gsm8k',g.question.duplicated().sum())
m=f[['pid','question']].merge(g,on='question',how='inner')
print('matched',len(m),'unique pids',m.pid.nunique())
m['n_steps']=m.answer.str.count(r'<<')  # independent: count opening markers
m2=m.answer.str.count(r'<<.*?>>')
print('n_steps same by two counts',(m.n_steps==m2).all(),'mean',m.n_steps.mean(),'range',m.n_steps.min(),m.n_steps.max())
print('fraction of annotation-free (n_steps=0):',(m.n_steps==0).mean())
CR=['judge_logit','log_tokens','truncated','has_decimal','is_negative','log_abs_ans','empty_answer','log_qlen']
st=m.set_index('pid').n_steps
for tag in ('3B','7B'):
    t=pd.read_parquet(R+f'runs/L1a/tab_{tag}_q_corrupted.parquet')
    y=t.corrupted.values.astype(int); pid=t.pid.values
    Xc=t[CR].values.astype(float); ns=st.reindex(pid).values.astype(float)[:,None]
    res={'c':[], 's':[]}
    for seed in (0,1,2):
        rng=np.random.default_rng(100+seed); up=np.unique(pid); perm=rng.permutation(up); fo=dict(zip(perm,np.arange(len(perm))%5)); fold=np.array([fo[p] for p in pid])
        for name,X in (('c',Xc),('s',np.hstack([Xc,ns]))):
            p=np.zeros(len(y))
            for k in range(5):
                tr,te=fold!=k,fold==k
                mdl=make_pipeline(StandardScaler(),LogisticRegression(class_weight='balanced',max_iter=1000)).fit(X[tr],y[tr]); p[te]=mdl.predict_proba(X[te])[:,1]
            res[name].append(roc_auc_score(y,p))
    print(tag,'cheap_r AUC',np.mean(res['c']).round(4),'+n_steps',np.mean(res['s']).round(4),'delta',(np.mean(res['s'])-np.mean(res['c'])).round(4))
    # leak diagnostic
    gold=pd.to_numeric(m.set_index('pid').answer.str.split('####').str[-1].str.strip().str.replace(',','',regex=False))
    lg=np.log10(gold.abs()+1).reindex(pid).values
    eq=np.abs(t.log_abs_ans.values-lg)<1e-9
    print(tag,'equal share',eq.mean().round(4),'label rate equal',y[eq].mean().round(5),'not equal',y[~eq].mean().round(4))
    # n_steps vs label correlation among rows
    print(tag,'corr(n_steps,label)',np.corrcoef(ns[:,0],y)[0,1].round(3),'corr(n_steps,log_qlen)',np.corrcoef(ns[:,0],t.log_qlen.values)[0,1].round(3))

# ---- part 2: static top-k precision with and without oracle columns
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
R='/homes/rdubash/avrl_live/FULL_R/avrl/runs/L11b/'
CR=['judge_logit','log_tokens','truncated','has_decimal','is_negative','log_abs_ans','empty_answer','log_qlen']
for tag in ('3B','7B'):
    t=pd.read_parquet(R+f'tab_{tag}_q_corrupted.parquet'); y=t.corrupted.values.astype(int); pid=t.pid.values
    sets={'cheap':CR,'orc_ng':CR+['n_steps','n_numbers_q'],'orc':CR+['n_steps','n_numbers_q','log_abs_gold'],'gold_only':CR+['log_abs_gold']}
    up=np.unique(pid); rng=np.random.default_rng(5); perm=rng.permutation(up); fo=dict(zip(perm,np.arange(len(perm))%5)); fold=np.array([fo[p] for p in pid])
    for nm,cols in sets.items():
        X=t[cols].values.astype(float); p=np.zeros(len(y))
        for k in range(5):
            tr,te=fold!=k,fold==k
            p[te]=make_pipeline(StandardScaler(),LogisticRegression(class_weight='balanced',max_iter=2000)).fit(X[tr],y[tr]).predict_proba(X[te])[:,1]
        o=np.argsort(-p); out=[]
        for fr in (0.05,0.10,0.25,0.5):
            k=int(fr*len(y)); out.append(f'top{int(fr*100)}% prec {y[o[:k]].mean():.3f}')
        print(tag,nm,' | '.join(out))
    print(tag,'base rate',y.mean().round(3))

# ---- part 3: replay ratios from the raw per_seed lists
R='/homes/rdubash/avrl_live/FULL_R/avrl/runs/L11b/'
for tag in ('3B','7B'):
  for lay in ('cell100','cell30'):
    for bud in ('0.1','0.25','0.5'):
      row=[]
      for pol in ('greedy_online','ts_online'):
        base=[];
        out={}
        for fs in ('cheap','orc','orc_ng'):
            v=[]
            for sp in range(4):
                v+=json.load(open(R+f'sim_{tag}_q_corrupted_{fs}_{lay}_s{sp}.json'))[f'budget_{bud}'][pol]['per_seed']
            out[fs]=np.array(v)
        # per-split paired
        rt=out['orc_ng'].mean()/out['cheap'].mean(); rf=out['orc'].mean()/out['cheap'].mean()
        # split-level ratios
        sp_ng=[out['orc_ng'][i*8:(i+1)*8].mean()/out['cheap'][i*8:(i+1)*8].mean() for i in range(4)]
        row.append(f'{pol[:6]} ng {rt:.3f} (splits {" ".join(f"{x:.3f}" for x in sp_ng)}) full {rf:.3f} n={len(out["cheap"])}')
      print(tag,lay,bud,' | '.join(row))

# ---- part 4: fixed versus online arms across feature sets, and greedy / best charged recal rule
R2_='/homes/rdubash/avrl_live/FULL_R/avrl/runs/L11b/'
def load(tag, fs, lay, bud, pol):
    v = []
    for sp in range(4): v += json.load(open(R2_ + f'sim_{tag}_q_corrupted_{fs}_{lay}_s{sp}.json'))[f'budget_{bud}'][pol]['per_seed']
    return np.mean(v)
rc = ['recal_charged_100', 'recal_charged_200', 'recal_charged_400']
for tag in ('3B', '7B'):
    for bud in ('0.1', '0.25'):
        print(tag, bud, '(cheap/orc_ng/orc, cell30)', ' | '.join(f"{p}: " + '/'.join(f"{load(tag, fs, 'cell30', bud, p):.0f}" for fs in ('cheap', 'orc_ng', 'orc')) for p in ('static', 'static_matched', 'greedy_online', 'ts_online')))
    for lay in ('cell100', 'cell30'):
        for bud in ('0.1', '0.25', '0.5'):
            r = {}
            for fs in ('cheap', 'orc_ng'):
                P = {p: load(tag, fs, lay, bud, p) for p in ['greedy_online'] + rc}; bk = max(rc, key=lambda k: P[k]); r[fs] = P['greedy_online'] / P[bk]
            print(tag, lay, bud, 'greedy / best charged recal: cheap %.3f -> orc_ng %.3f' % (r['cheap'], r['orc_ng']))
