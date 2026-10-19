"""L11b second adversarial recheck, replay part (CPU, reads runs/L11b/sim_*.json only). Usage: l11b_critic2_replay.py (cwd = avrl).
Checks: ts_online / greedy_online with orc_ng at 10 / 25 / 50 percent (the entry quotes 0.989 to 1.011 without a budget); whether FIXED rules fit on random labels (recal_charged_n) gain from the gold-magnitude
column the way the online learners do (entry: 'a static fit does not'); split-level (4 clusters) consistency and paired t of orc_ng / cheap_r for greedy_online at 25 percent; pooled means used by the entry's ratios."""
import json, numpy as np
D = "runs/L11b"; J = lambda p: json.load(open(f"{D}/{p}"))
def pooled(t, lay, fs, b, pol):
    return np.array([x for s in range(4) for x in J(f"sim_{t}_q_corrupted_{fs}_{lay}_s{s}.json")[f"budget_{b}"][pol]["per_seed"]], float)
print("ts_online / greedy_online, orc_ng arm (and cheap arm), pooled 4 splits x 8 seeds")
for t in ("3B", "7B"):
    for lay in ("cell100", "cell30"):
        for b in ("0.1", "0.25", "0.5"):
            r = {fs: pooled(t, lay, fs, b, "ts_online").mean() / pooled(t, lay, fs, b, "greedy_online").mean() for fs in ("cheap", "orc_ng", "orc")}
            print(f"  {t} {lay} {b}: cheap {r['cheap']:.3f} | orc_ng {r['orc_ng']:.3f} | orc {r['orc']:.3f}")
print("\nfixed random-label rules (recal_charged_n, linear logistic) vs online learners, mean errors caught, cheap -> orc_ng -> orc (gold magnitude)")
for t in ("3B", "7B"):
    for lay in ("cell30",):
        for b in ("0.1", "0.25"):
            line = []
            for pol in ("static", "recal_charged_100", "recal_charged_200", "recal_charged_400", "greedy_online", "ts_online"):
                v = [pooled(t, lay, fs, b, pol).mean() for fs in ("cheap", "orc_ng", "orc")]
                line.append(f"{pol} {v[0]:.0f}/{v[1]:.0f}/{v[2]:.0f} (orc/orc_ng {v[2] / v[1]:.3f})")
            print(f"  {t} {lay} {b}: " + " | ".join(line))
print("\nsplit-level greedy_online orc_ng / cheap_r at 25 percent (mean over 8 seeds per split), paired t over 4 splits")
for t in ("3B", "7B"):
    for lay in ("cell100", "cell30"):
        ra = np.array([np.mean(J(f"sim_{t}_q_corrupted_orc_ng_{lay}_s{s}.json")["budget_0.25"]["greedy_online"]["per_seed"]) / np.mean(J(f"sim_{t}_q_corrupted_cheap_{lay}_s{s}.json")["budget_0.25"]["greedy_online"]["per_seed"]) for s in range(4)])
        print(f"  {t} {lay}: splits {np.round(ra, 3).tolist()} mean {ra.mean():.3f} t(df3) vs 1 = {(ra.mean() - 1) / (ra.std(ddof=1) / 2):.1f}")
