import json, os, glob, numpy as np
R="/homes/rdubash/avrl_live/FULL_R/avrl/runs/L7a"; R6="/homes/rdubash/avrl_live/FULL_R/avrl/runs/L6a"
files=glob.glob(R+"/l7a_d*.json"); print("n json",len(files))
D={}
for f in files:
    x=json.load(open(f)); b=os.path.basename(f)[:-5].split("_"); d=int(b[1][1:]); B=int(b[2][1:]); o=int(b[3][1:]); m="_".join(b[4:])
    assert x["budget"]==B and x["order_seed"]==o and x["mode"]==m, f
    assert all(x["checks"].values()), (f,x["checks"])
    assert x["n_batches"]==12 and x["m"]==250
    for k,v in x["arms"].items(): assert len(v)==12, (f,k)
    D[(d,B,m,o)]=x
print("combos", len(D)); 
PERSIST=["carry_plain","carry_plain_window_125","carry_plain_window_250","carry_plain_window_500","carry_plain_decay","carry_offset_lam1"]
ARMS=["thr_agreement","restart_offset"]+PERSIST+["freeze_plain_k3","freeze_plain_k6","reset_at_switch"]
def A(d,B,m,arm,seeds=range(16)): return np.array([D[(d,B,m,o)]["arms"][arm] for o in seeds],float)
G={"4-6":slice(3,6),"7-9":slice(6,9),"10-12":slice(9,12),"7-12":slice(6,12)}
print("== arms table: ratio of means to thr by group (4-6/7-9/10-12), drift, 16 seeds")
for arm in ARMS:
    row=[]
    for (d,B) in [(1,62),(1,125),(2,62),(2,125)]:
        a=A(d,B,"drift",arm); t=A(d,B,"drift","thr_agreement")
        row.append(" / ".join(f"{a[:,G[g]].sum()/t[:,G[g]].sum():.3f}" for g in ["4-6","7-9","10-12"]))
    print(f"| {arm} | "+" | ".join(row)+" |")
print("== thr_agreement errors/batch 4-6,7-9,10-12")
for (d,B) in [(1,62),(1,125),(2,62),(2,125)]:
    t=A(d,B,"drift","thr_agreement"); print(d,B,[round(t[:,G[g]].mean(),1) for g in ["4-6","7-9","10-12"]])
# headline
rng=np.random.default_rng(777); NB=10000
def rat(num,den):
    idx=rng.integers(0,len(num),(NB,len(num))); rb=num[idx].mean(1)/den[idx].mean(1); return num.mean()/den.mean(), np.percentile(rb,2.5), np.percentile(rb,97.5)
S=lambda d,B,m,arm,seeds,g="7-12": A(d,B,m,arm,seeds)[:,G[g]].sum(1)
FA=range(0,8); FB=range(8,16)
print("== headline")
for (d,B) in [(1,62),(1,125),(2,62),(2,125)]:
    bA=max(PERSIST,key=lambda k:S(d,B,"drift",k,FA).mean()); bB=max(PERSIST,key=lambda k:S(d,B,"drift",k,FB).mean())
    r=rat(S(d,B,"drift",bA,FB),S(d,B,"drift","freeze_plain_k6",FB))
    r79=S(d,B,"drift",bA,FB,"7-9").sum()/S(d,B,"drift","thr_agreement",FB,"7-9").sum(); r1012=S(d,B,"drift",bA,FB,"10-12").sum()/S(d,B,"drift","thr_agreement",FB,"10-12").sum()
    rB=rat(S(d,B,"drift",bB,FA),S(d,B,"drift","freeze_plain_k6",FA))
    print(d,B,"bestA",bA,"bestB",bB,"foldA best/freeze %.4f (%.4f,%.4f)"%r,"7-9 %.3f 10-12 %.3f"%(r79,r1012),"foldB %.4f"%rB[0], "freeze/best %.4f"%(1/r[0]))
    # selection means
    print("   means 0-7:",{k[:14]:round(S(d,B,"drift",k,FA).mean(),1) for k in PERSIST}); print("   means 8-15:",{k[:14]:round(S(d,B,"drift",k,FB).mean(),1) for k in PERSIST})
# drift excess table
print("== drift excess (arm/freeze_k6 7-12 drift)/(same control_z) 16 seeds paired bootstrap")
for arm in ["carry_plain","carry_plain_window_250","carry_plain_window_500","carry_plain_decay","reset_at_switch"]:
    row=[]
    for (d,B) in [(1,62),(1,125),(2,62),(2,125)]:
        al=range(16)
        n1=S(d,B,"drift",arm,al); d1=S(d,B,"drift","freeze_plain_k6",al); n0=S(d,B,"control_z",arm,al); d0=S(d,B,"control_z","freeze_plain_k6",al)
        pt=(n1.sum()/d1.sum())/(n0.sum()/d0.sum()); idx=rng.integers(0,16,(NB,16))
        rb=(n1[idx].sum(1)/d1[idx].sum(1))/(n0[idx].sum(1)/d0[idx].sum(1)); row.append("%.3f (%.3f to %.3f)"%(pt,np.percentile(rb,2.5),np.percentile(rb,97.5)))
    print(f"| {arm} | "+" | ".join(row)+" |")
print("== extra: window_125 excess + all-arm excess")
for arm in ["carry_plain_window_125","carry_offset_lam1","freeze_plain_k3"]:
    row=[]
    for (d,B) in [(1,62),(1,125),(2,62),(2,125)]:
        al=range(16)
        n1=S(d,B,"drift",arm,al); d1=S(d,B,"drift","freeze_plain_k6",al); n0=S(d,B,"control_z",arm,al); d0=S(d,B,"control_z","freeze_plain_k6",al)
        pt=(n1.sum()/d1.sum())/(n0.sum()/d0.sum()); idx=rng.integers(0,16,(NB,16))
        rb=(n1[idx].sum(1)/d1[idx].sum(1))/(n0[idx].sum(1)/d0[idx].sum(1)); row.append("%.3f (%.3f to %.3f)"%(pt,np.percentile(rb,2.5),np.percentile(rb,97.5)))
    print(f"| {arm} | "+" | ".join(row)+" |")
print("== 16-seed best arm by 7-12 hits (all PERSIST) vs carry_plain")
for (d,B) in [(1,62),(1,125),(2,62),(2,125)]:
    m={k:round(S(d,B,"drift",k,range(16)).mean(),1) for k in PERSIST}; print(d,B,m, "freeze_k6",round(S(d,B,"drift","freeze_plain_k6",range(16)).mean(),1),"thr",round(S(d,B,"drift","thr_agreement",range(16)).mean(),1))
print("== batch-level noise: within-seed SD of (carry_plain - freeze_k6) per batch, batches 7-12; SE of 6-batch-run mean diff, effect size")
for (d,B) in [(1,62),(1,125),(2,62),(2,125)]:
    for arm in ["carry_plain"]:
        a=A(d,B,"drift",arm); f=A(d,B,"drift","freeze_plain_k6"); dif=(a-f)[:,6:12]
        per_seed=dif.sum(1)  # 6-batch diff per seed
        # seed-to-seed SD of per-seed total diff (order-driven)
        within=dif.std(1,ddof=1).mean()  # batch-to-batch SD of diffs within seed
        print(d,B,"mean diff/batch %.2f"%dif.mean(),"within-seed batch SD %.2f"%within,"=> SE of mean over 6 batches %.2f (z=%.2f)"%(within/np.sqrt(6),dif.mean()/(within/np.sqrt(6))),
              "| seed-to-seed SD of 6-batch mean diff %.2f"%dif.mean(1).std(ddof=1))
print("== batch-level bootstrap (seed,batch pairs, verdict fold 8 seeds x 6 batches = 48; also 16 seeds pooled arm = same best arm)")
rng2=np.random.default_rng(5)
bestA={(1,62):"carry_plain",(1,125):"carry_plain_decay",(2,62):"carry_plain",(2,125):"carry_plain"}
for (d,B),arm in bestA.items():
    a=A(d,B,"drift",arm,range(8,16))[:,6:12].ravel(); f=A(d,B,"drift","freeze_plain_k6",range(8,16))[:,6:12].ravel()
    idx=rng2.integers(0,len(a),(10000,len(a))); rb=a[idx].sum(1)/f[idx].sum(1)
    print(d,B,arm,"point %.3f"%(a.sum()/f.sum()),"batch-level CI (%.3f, %.3f)"%(np.percentile(rb,2.5),np.percentile(rb,97.5)),"P(ratio>=1.03)=%.2f"%(rb>=1.03).mean(), "P(>1)=%.3f"%(rb>1).mean())
    # all 16 seeds
    a=A(d,B,"drift",arm,range(16))[:,6:12].ravel(); f=A(d,B,"drift","freeze_plain_k6",range(16))[:,6:12].ravel()
    idx=rng2.integers(0,len(a),(10000,len(a))); rb=a[idx].sum(1)/f[idx].sum(1)
    print("    16-seed point %.3f CI (%.3f, %.3f)"%(a.sum()/f.sum(),np.percentile(rb,2.5),np.percentile(rb,97.5)))
print("== max forgetting arm / carry_plain by half and all 16")
for (d,B) in [(1,62),(1,125),(2,62),(2,125)]:
    out=[]
    for nm,sd in (("0-7",range(8)),("8-15",range(8,16)),("all16",range(16))):
        cp=S(d,B,"drift","carry_plain",sd).mean()
        out.append((nm,{k[12:]:round(S(d,B,"drift",k,sd).mean()/cp,3) for k in PERSIST[1:5]}))
    print(d,B,out)
