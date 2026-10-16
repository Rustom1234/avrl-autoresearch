"""
In-training results figure for the AVRL thesis: corrupted-reward admission
and final held-out quality, one panel per setting, one point per seed plus
a seed-mean bar, allocators ordered threshold / greedy / learned (consistency
threshold last in the open-ended panel).

Every number plotted here is computed in this script directly from the raw
per-step logs under avrl/runs/<run>/log.jsonl (train_step rows' corrupted_
reward_groups and expensive_calls, summed over all steps; eval rows' final
metric at step 100), not copied from RESEARCH_LOG.md prose. The sums are
cross-checked against the ledger's own tables for M06, M07, M08a, M09,
P1-judge-lr5-s0/s1/s2 and match exactly (see fig/FIGURE2.md's data table
and the printed summary below). Re-run this script if the underlying logs
ever change; do not hand-edit the PNG/PDF outputs.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS_DIR = "/homes/rdubash/active-audit/avrl/runs"
OUT_DIR = "/homes/rdubash/active-audit/lit/avrl/fig"
N_STEPS_EXPECTED = 100

# Same validated colorblind-safe pair used throughout the thesis figures
# (see make_fig1.py's palette note). Here the two colors mark mechanism,
# not task type: label-free rules (no refitting, rank by the cheap grader's
# own confidence) vs. learned allocators (greedy self-trainer, bandit-
# flavoured LearnedAllocator), both of which refit on bought labels.
COLOR_LABELFREE = "#2a78d6"  # blue
COLOR_LEARNED = "#eb6834"    # orange
COLOR_REFERENCE = "#8a8a8a"  # grey: full (every group verified) and proxy (none) reference arms
GRID_COLOR = "#dddddd"
TEXT_COLOR = "#222222"

# (allocator_key, plain-word label, kind, {seed: run_dir})
SETTINGS = [
    dict(
        key="gsm8k_1p5b",
        title="GSM8K, 1.5B policy (M07)",
        metric="eval_accuracy",
        metric_label="Eval accuracy\n(mean of 300 held-out prompts)",
        allocators=[
            ("threshold", "threshold\n(agreement)", "labelfree",
             {0: "m07_threshold_s0", 1: "m07_threshold_s1"}),
            ("threshold10", "threshold\n10% budget", "labelfree",
             {0: "m14_threshold_b10_s0", 1: "m14_threshold_b10_s1"}),
            ("greedy", "greedy\n(self-train)", "learned",
             {0: "m07_greedy_s0", 1: "m07_greedy_s1"}),
            ("learned", "learned\n(bandit)", "learned",
             {0: "m07_learned_s0", 1: "m07_learned_s1"}),
            ("full", "full\n(800 calls)", "reference",
             {0: "m10_full_s0", 1: "m10_full_s1"}),
            ("proxy", "proxy\n(0 calls)", "reference",
             {0: "m10_proxy_s0", 1: "m10_proxy_s1"}),
        ],
        omitted_note=None,
    ),
    dict(
        key="gsm8k_0p5b",
        title="GSM8K, 0.5B policy (M06)",
        metric="eval_accuracy",
        metric_label="Eval accuracy\n(mean of 300 held-out prompts)",
        allocators=[
            ("threshold", "threshold\n(agreement)", "labelfree",
             {0: "m06_threshold_s0", 1: "m06_threshold_s1"}),
            ("greedy", "greedy\n(self-train)", "learned",
             {0: "m06b_greedy_s0", 1: "m06b_greedy_s1"}),
            ("learned", "learned\n(bandit)", "learned",
             {0: "m06_learned_s0", 1: "m06_learned_s1"}),
        ],
        # Greedy (m06b_greedy_s*): both seeds complete at 100 steps as of
        # 2026-10-02 (previously seed 1 was mid-run and this arm was
        # omitted; now included).
        omitted_note=None,
    ),
    dict(
        key="helpsteer2",
        title="HelpSteer2, RM proxy (open-ended)",
        metric="eval_judge_score_mean",
        metric_label="Eval judge score\n(mean of 200 held-out prompts)",
        allocators=[
            ("threshold_rm", "threshold\n(RM gap)", "labelfree",
             {0: "p1_judge_threshold_lr5", 1: "p1_judge_threshold_lr5_s1",
              2: "p1_judge_threshold_lr5_s2"}),
            ("greedy", "greedy\n(self-train)", "learned",
             {0: "p1_judge_greedy_lr5_s0", 1: "p1_judge_greedy_lr5_s1"}),
            ("learned", "learned\n(bandit)", "learned",
             {0: "p1_judge_learned_lr5", 1: "p1_judge_learned_lr5_s1",
              2: "p1_judge_learned_lr5_s2"}),
            ("threshold_cons", "threshold\n(consistency)", "labelfree",
             {0: "p1_judge_thrcons_lr5_s0", 1: "p1_judge_thrcons_lr5_s1", 3: "p1_judge_thrcons_lr5_s3"}),
            ("random", "random", "reference",
             {0: "p1_judge_random_lr5_s0", 1: "p1_judge_random_lr5_s1"}),
            ("full", "full\n(800 calls)", "reference",
             {0: "p1_judge_full_lr5_s0", 1: "p1_judge_full_lr5_s1", 3: "p1_judge_full_lr5_s3"}),
            ("full_rank", "full, rank\nreward (M12)", "reference",
             {0: "p1_judge_fullrank_lr5_s0", 1: "p1_judge_fullrank_lr5_s1"}),
        ],
        omitted_note=None,
    ),
    dict(
        key="helpsteer2_7b",
        title="HelpSteer2, 7B judge, 150 steps",
        metric="eval_judge_score_mean",
        metric_label="Eval judge score\n(mean of 200 held-out prompts)",
        n_steps=150,
        allocators=[
            ("threshold_rm", "threshold\n(RM gap)", "labelfree",
             {0: "p1_judge5090b_threshold"}),
            ("greedy", "greedy\n(self-train)", "learned",
             {0: "p1_judge5090b_greedy_s0", 1: "p1_judge5090b_greedy_s1", 2: "p1_judge5090b_greedy_s2", 3: "p1_judge5090b_greedy_s3"}),
            ("learned", "learned\n(bandit)", "learned",
             {0: "p1_judge5090b_learned"}),
            ("threshold_cons", "threshold\n(consistency)", "labelfree",
             {0: "p1_judge5090b_thrcons_s0", 1: "p1_judge5090b_thrcons_s1", 2: "p1_judge5090b_thrcons_s2", 3: "p1_judge5090b_thrcons_s3"}),
        ],
        omitted_note=None,
    ),
    dict(
        key="e10b_code",
        title="MBPP code, OLMo 2 1B Instruct,\nvisible-test grader, 300 steps (E10b)",
        metric="eval_mbpp_pass1",
        metric_label="Held-out MBPP pass@1\n(200 test problems, full asserts)",
        n_steps=300,
        allocators=[
            ("threshold", "threshold\n(visible agree.)", "labelfree",
             {0: "hannah_e10b/e10b_threshold", 1: "e10b_threshold_s1", 2: "e10b_threshold_s2"}),
            ("random", "random", "reference",
             {0: "hannah_e10b/e10b_random", 1: "e10b_random_s1", 2: "e10b_random_s2"}),
            ("full", "full\n(2400 calls)", "reference",
             {0: "hannah_e10b/e10b_full", 1: "e10b_full_s1"}),
            ("proxy", "proxy\n(0 calls)", "reference",
             {0: "hannah_e10b/e10b_proxy", 1: "e10b_proxy_s1"}),
            ("auto", "auto, abs 2%\n(E10e, 2125 calls)", "labelfree",
             {1: "e10e_auto_s1"}),
            ("auto_rel", "auto, rel 25%\n(E10e-rel, 475 calls)", "labelfree",
             {1: "e10e_rel_auto_s1"}),
            ("auto_up", "auto, rel 25% upper\n(E10e-up, 1575 calls)", "labelfree",
             {1: "e10e_up_auto_s1"}),
        ],
        omitted_note=None,
    ),
    dict(
        key="e10d_code",
        title="MBPP code, Qwen2.5 1.5B Instruct,\nvisible-test grader, 300 steps (E10d)",
        metric="eval_mbpp_pass1",
        metric_label="Held-out MBPP pass@1\n(200 test problems, full asserts)",
        n_steps=300,
        allocators=[
            ("threshold", "threshold\n(visible agree.)", "labelfree",
             {0: "hannah_e10d/e10d_threshold", 1: "hannah_e10d_s1/e10d_threshold"}),
            ("random", "random", "reference",
             {0: "hannah_e10d/e10d_random", 1: "hannah_e10d_s1/e10d_random"}),
            ("full", "full\n(2400 calls)", "reference",
             {0: "hannah_e10d/e10d_full", 1: "hannah_e10d_s1/e10d_full"}),
            ("proxy", "proxy\n(0 calls)", "reference",
             {0: "hannah_e10d/e10d_proxy", 1: "hannah_e10d_s1/e10d_proxy"}),
        ],
        omitted_note=None,
    ),
    dict(
        key="helpsteer2_7b_bf16",
        title="HelpSteer2, bf16 7B judge, 150 steps\n(hannah, 8 prompts per step)",
        metric="eval_judge_score_mean",
        metric_label="Eval judge score\n(mean of 200 held-out prompts)",
        n_steps=150,
        allocators=[
            ("threshold_cons", "threshold\n(consistency)", "labelfree",
             {0: "hannah_judge7b/thrcons", 1: "hannah_judge7b_s1/thrcons", 2: "hannah_judge7b_s2/thrcons", 3: "hannah_judge7b_s3/thrcons", 4: "hannah_judge7b_s4/thrcons"}),
            ("greedy", "greedy\n(self-train)", "learned",
             {0: "hannah_judge7b/greedy", 1: "hannah_judge7b_s1/greedy", 2: "hannah_judge7b_s2/greedy", 3: "hannah_judge7b_s3/greedy", 4: "hannah_judge7b_s4/greedy"}),
            ("random", "random", "reference",
             {0: "hannah_judge7b/random", 1: "hannah_judge7b_s1/random", 2: "hannah_judge7b_s2/random", 3: "hannah_judge7b_s3/random", 4: "hannah_judge7b_s4/random"}),
            ("full", "full\n(1200 calls)", "reference",
             {0: "hannah_judge7b/full", 1: "hannah_judge7b_s1/full", 2: "hannah_judge7b_s2/full", 3: "hannah_judge7b_s3/full", 4: "hannah_judge7b_s4/full"}),
        ],
        omitted_note=None,
    ),
]


def load_run(run_dir, metric_key, n_steps=N_STEPS_EXPECTED):
    """Recompute corrupted-group admission, expensive calls, and the final
    eval metric directly from one run's log.jsonl. Returns None (and prints
    why) if the run is not a complete 100-step run."""
    path = os.path.join(RUNS_DIR, run_dir, "log.jsonl")
    if not os.path.exists(path):
        print(f"  SKIP {run_dir}: no log.jsonl yet")
        return None
    train_rows = []
    eval_rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("type") == "train_step":
                train_rows.append(row)
            elif row.get("type") == "eval":
                eval_rows.append(row)

    if len(train_rows) != n_steps:
        print(f"  SKIP {run_dir}: {len(train_rows)} train_step rows "
              f"(expected {n_steps}), run is incomplete")
        return None

    final_eval_rows = [r for r in eval_rows if r.get("step") == n_steps]
    if not final_eval_rows:
        print(f"  SKIP {run_dir}: no eval row at step {n_steps}")
        return None

    corrupted = sum(r["corrupted_reward_groups"] for r in train_rows)
    expensive = sum(r["expensive_calls"] for r in train_rows)
    n_unverified = sum(r["n_groups"] - r["expensive_calls"] for r in train_rows)
    final_eval = final_eval_rows[0][metric_key]

    return dict(
        run_dir=run_dir, corrupted=corrupted, expensive=expensive,
        n_unverified=n_unverified, final_eval=final_eval,
    )


def collect(setting):
    """Load every seed for every allocator in a setting; drop allocators
    with zero usable seeds."""
    out_allocators = []
    for key, label, kind, seed_dirs in setting["allocators"]:
        seeds = {}
        for seed, run_dir in sorted(seed_dirs.items()):
            rec = load_run(run_dir, setting["metric"], setting.get("n_steps", N_STEPS_EXPECTED))
            if rec is not None:
                seeds[seed] = rec
        if seeds:
            out_allocators.append(dict(key=key, label=label, kind=kind, seeds=seeds))
    return out_allocators


def style_axes(ax):
    ax.grid(True, axis="y", linewidth=0.6, color=GRID_COLOR, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#888888")
    ax.tick_params(colors=TEXT_COLOR, labelsize=8.5)


def seed_offsets(n_seeds):
    if n_seeds <= 1:
        return [0.0]
    if n_seeds == 2:
        return [-0.09, 0.09]
    return [-0.12, 0.0, 0.12]


def draw_panel(ax, allocators, value_key, y_label):
    x_positions = list(range(len(allocators)))
    for x, alloc in zip(x_positions, allocators):
        color = {"labelfree": COLOR_LABELFREE, "learned": COLOR_LEARNED}.get(alloc["kind"], COLOR_REFERENCE)
        seed_items = sorted(alloc["seeds"].items())
        values = [rec[value_key] for _, rec in seed_items]
        mean_val = sum(values) / len(values)

        ax.bar(
            [x], [mean_val], width=0.46, color=color, alpha=0.32,
            edgecolor=color, linewidth=1.2, zorder=1,
        )
        offsets = seed_offsets(len(seed_items))
        for (seed, rec), dx in zip(seed_items, offsets):
            ax.scatter(
                [x + dx], [rec[value_key]], s=52, facecolors=color,
                edgecolors="#333333", linewidths=0.7, alpha=0.95, zorder=3,
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([a["label"] for a in allocators], fontsize=7.5 if len(allocators) >= 4 else 8.5)
    ax.set_xlim(-0.6, len(allocators) - 0.4)
    ax.set_ylabel(y_label, fontsize=9)
    style_axes(ax)


def legend_handles():
    labelfree_h = plt.Line2D(
        [0], [0], marker="s", linestyle="none", markersize=10,
        markerfacecolor=COLOR_LABELFREE, markeredgecolor=COLOR_LABELFREE,
        alpha=0.8, label="Label-free rule (threshold, no refitting)",
    )
    learned_h = plt.Line2D(
        [0], [0], marker="s", linestyle="none", markersize=10,
        markerfacecolor=COLOR_LEARNED, markeredgecolor=COLOR_LEARNED,
        alpha=0.8, label="Learned allocator (greedy / bandit, refits on bought labels)",
    )
    reference_h = plt.Line2D(
        [0], [0], marker="s", linestyle="none", markersize=10,
        markerfacecolor=COLOR_REFERENCE, markeredgecolor=COLOR_REFERENCE,
        alpha=0.8, label="Reference (full: every group verified; proxy: none)",
    )
    return [labelfree_h, learned_h, reference_h]


def make_figure(value_key, y_label_fn, out_stem, extra_footnote=None):
    plt.rcParams["font.size"] = 9.5
    n_panels = len(SETTINGS)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 4.4), dpi=200)

    collected = []
    for ax, setting in zip(axes, SETTINGS):
        allocators = collect(setting)
        collected.append((setting, allocators))
        draw_panel(ax, allocators, value_key, y_label_fn(setting))
        ax.set_title(setting["title"], fontsize=9.5, color=TEXT_COLOR, pad=8)

    fig.legend(
        handles=legend_handles(), loc="upper center", bbox_to_anchor=(0.5, 1.0),
        ncol=3, fontsize=8.5, frameon=False,
    )
    footnote = "Each marker is one seed; the bar is the seed mean."
    if extra_footnote:
        footnote += "  " + extra_footnote
    fig.text(0.5, 0.01, footnote, ha="center", fontsize=7.8, color=TEXT_COLOR)

    fig.subplots_adjust(top=0.80, bottom=0.22, left=0.06, right=0.98, wspace=0.38)

    png_path = f"{OUT_DIR}/{out_stem}.png"
    fig.savefig(png_path, dpi=200)
    fig.savefig(f"{OUT_DIR}/{out_stem}.pdf")
    plt.close(fig)
    print("wrote", png_path, "(+pdf)")
    return collected


if __name__ == "__main__":
    print("Recomputing corrupted-admission figure (fig2_in_training):")
    collected_corrupted = make_figure(
        value_key="corrupted",
        y_label_fn=lambda s: "Corrupted reward groups admitted\n(of %d unverified; %d expensive calls/arm)" % (6 * s.get("n_steps", N_STEPS_EXPECTED), 2 * s.get("n_steps", N_STEPS_EXPECTED)),
        out_stem="fig2_in_training",
    )

    print("\nRecomputing final-eval figure (fig2b_eval):")
    collected_eval = make_figure(
        value_key="final_eval",
        y_label_fn=lambda s: s["metric_label"],
        out_stem="fig2b_eval",
    )

    # Print the full table once, for transcription into FIGURE2.md.
    print("\nFull data table (setting, allocator, seed, corrupted, expensive, final_eval, run_dir):")
    for setting, allocators in collected_corrupted:
        for alloc in allocators:
            for seed, rec in sorted(alloc["seeds"].items()):
                print(f"{setting['key']}\t{alloc['key']}\t{seed}\t{rec['corrupted']}\t"
                      f"{rec['expensive']}\t{rec['final_eval']}\t{rec['run_dir']}")
        if setting.get("omitted_note"):
            print(f"  NOTE ({setting['key']}): {setting['omitted_note']}")
