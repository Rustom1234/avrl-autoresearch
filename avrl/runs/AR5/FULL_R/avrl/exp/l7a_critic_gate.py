import json, os, numpy as np
R="/homes/rdubash/avrl_live/FULL_R/avrl/runs/L7a"; R6="/homes/rdubash/avrl_live/FULL_R/avrl/runs/L6a"
nd=0; nv=0; nvec=0; miss=0
for d,j in [(1,"Qwen2.5-3B-Instruct"),(2,"Qwen2.5-7B-Instruct-AWQ")]:
    for B in (62,125):
        for o in range(8):
            f=f"{R}/l7a_d{d}_B{B}_o{o}_control_exact.json"; g=f"{R6}/amort_{j}_B{B}_o{o}.json"
            if not (os.path.exists(f) and os.path.exists(g)): miss+=1; continue
            x=json.load(open(f)); r=json.load(open(g))
            for k in ["thr_agreement","restart_offset","carry_plain","carry_offset_lam1"]:
                nvec+=1; nv+=12; nd+=sum(a!=b for a,b in zip(x["arms"][k],r["arms"][k])); assert len(x["arms"][k])==len(r["arms"][k])==12
            nvec+=1; nv+=12; nd+=sum(a!=b for a,b in zip(x["batch_base_errors"],r["batch_base_errors"]))
            if B==62: nvec+=1; nv+=12; nd+=sum(a!=b for a,b in zip(x["arms"]["freeze_plain_k3"],r["arms"]["freeze_plain_125"]))
print("gate: vectors",nvec,"values",nv,"diffs",nd,"missing",miss)
# L6a protocol sanity: same perm seed / batch layout?
r=json.load(open(f"{R6}/amort_Qwen2.5-3B-Instruct_B62_o0.json")); print({k:r[k] for k in r if k not in ("arms","diag","arm_info","checks","batch_base_errors")})
# pre-switch equality drift vs control_z
bad=0
for d in (1,2):
    for B in (62,125):
        for o in range(16):
            a=json.load(open(f"{R}/l7a_d{d}_B{B}_o{o}_drift.json")); c=json.load(open(f"{R}/l7a_d{d}_B{B}_o{o}_control_z.json"))
            for k in a["arms"]:
                if a["arms"][k][:6]!=c["arms"][k][:6]: bad+=1
            if a["batch_base_errors"][:6]!=c["batch_base_errors"][:6]: bad+=1
print("pre-switch diffs",bad)
