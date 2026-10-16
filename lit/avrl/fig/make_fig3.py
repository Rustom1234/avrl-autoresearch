"""
Figure 3 for the AVRL thesis: held-out MBPP pass@1 over training, one panel
per code experiment, one line per allocator arm.

Every number plotted here is read directly from each run's log.jsonl under
avrl/runs/ ("eval" rows' step and eval_mbpp_pass1, plus eval_humaneval_pass1
on the final row where present). The untrained Instruct/base reference
values are fixed constants supplied for this figure (E10 from hannah's
evaluations, E10b from the local evaluations), not read from a run log.
Re-run this script if the underlying logs change; do not hand-edit the
PNG/PDF outputs.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS_DIR = "/homes/rdubash/active-audit/avrl/runs"
OUT_DIR = "/homes/rdubash/active-audit/lit/avrl/fig"

# Same palette as make_fig2.py.
COLOR_LABELFREE = "#2a78d6"  # blue: threshold (label-free rule)
COLOR_LEARNED = "#eb6834"    # orange: learned allocator
COLOR_REFERENCE = "#8a8a8a"  # grey: full / proxy / random reference arms
GRID_COLOR = "#dddddd"
TEXT_COLOR = "#222222"

# Each panel: title, untrained reference values + where they came from, and
# arms. Each arm is (key, legend word, color, linestyle, candidate run dirs
# as (dir relative to RUNS_DIR, seed or None)). Candidates are tried in
# order; a seed is only shown in the legend label if more than one loads.
PANELS = [
    dict(
        key="E10",
        title="E10: OLMo 2 1B base,\nsignature-vote cheap grader",
        instruct_ref=0.130,
        base_ref=0.088,
        ref_source="hannah's evaluations",
        arms=[
            ("full", "full", COLOR_REFERENCE, "solid",
             [("hannah_e10/e10_full", None)]),
            ("proxy", "proxy", COLOR_REFERENCE, "dashed",
             [("hannah_e10/e10_proxy", None)]),
            ("threshold", "threshold", COLOR_LABELFREE, "solid",
             [("hannah_e10/e10_threshold", None)]),
            ("learned", "learned", COLOR_LEARNED, "solid",
             [("hannah_e10/e10_learned", None)]),
        ],
    ),
    dict(
        key="E10b",
        title="E10b: OLMo 2 1B Instruct, visible-test cheap grader,\n108-problem curriculum",
        instruct_ref=0.145,
        base_ref=0.085,
        ref_source="the local evaluations",
        arms=[
            ("full", "full", COLOR_REFERENCE, "solid",
             [("e10b_full_s1", 1), ("hannah_e10b/e10b_full", 0)]),
            ("proxy", "proxy", COLOR_REFERENCE, "dashed",
             [("e10b_proxy_s1", 1), ("hannah_e10b/e10b_proxy", 0)]),
            ("threshold", "threshold", COLOR_LABELFREE, "solid",
             [("e10b_threshold_s1", 1), ("hannah_e10b/e10b_threshold", 0)]),
            ("random", "random", COLOR_REFERENCE, "dotted",
             [("e10b_random_s1", 1), ("hannah_e10b/e10b_random", 0)]),
        ],
    ),
]


def load_run(run_dir, label):
    """Read one run's eval rows from log.jsonl. Always returns a dict (so
    the caller can still build a table row for a missing/incomplete run);
    ok=False means there is nothing to plot."""
    path = os.path.join(RUNS_DIR, run_dir, "log.jsonl")
    result = dict(ok=False, steps_completed=0, final_mbpp=None,
                  final_humaneval=None, steps=[], mbpp=[])
    if not os.path.exists(path):
        print(f"  SKIP {label}: no log.jsonl at {path}")
        return result

    train_step_max = -1
    eval_rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("type") == "train_step":
                train_step_max = max(train_step_max, row["step"])
            elif row.get("type") == "eval":
                eval_rows.append(row)

    if not eval_rows:
        n_train = train_step_max + 1 if train_step_max >= 0 else 0
        print(f"  SKIP {label}: no eval rows yet ({n_train} train steps logged)")
        result["steps_completed"] = n_train
        return result

    eval_rows.sort(key=lambda r: r["step"])
    final = eval_rows[-1]
    result.update(
        ok=True,
        steps=[r["step"] for r in eval_rows],
        mbpp=[r["eval_mbpp_pass1"] for r in eval_rows],
        steps_completed=final["step"],
        final_mbpp=final["eval_mbpp_pass1"],
        final_humaneval=final.get("eval_humaneval_pass1"),
    )
    return result


def style_axes(ax):
    ax.grid(True, axis="y", linewidth=0.6, color=GRID_COLOR, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#888888")
    ax.tick_params(colors=TEXT_COLOR, labelsize=8.5)


def arm_label(word, seed, show_seed, final_humaneval):
    label = word
    if show_seed and seed is not None:
        label += f" (seed {seed})"
    if final_humaneval is not None:
        label += f", HumanEval {final_humaneval:.3f}"
    return label


def draw_panel(ax, panel, table_rows):
    all_y = [panel["instruct_ref"], panel["base_ref"]]
    for arm_key, word, color, linestyle, candidates in panel["arms"]:
        loaded = []
        for run_dir, seed in candidates:
            label_ctx = f"{panel['key']}/{arm_key}" + (f" seed {seed}" if seed is not None else "")
            rec = load_run(run_dir, label_ctx)
            table_rows.append(dict(
                experiment=panel["key"], arm=arm_key,
                seed=seed if seed is not None else "-",
                steps_completed=rec["steps_completed"],
                final_mbpp=rec["final_mbpp"], final_humaneval=rec["final_humaneval"],
                source=f"avrl/runs/{run_dir}",
            ))
            if rec["ok"]:
                loaded.append((seed, rec))

        show_seed = len(loaded) > 1
        for seed, rec in loaded:
            ax.plot(
                rec["steps"], rec["mbpp"], color=color, linestyle=linestyle,
                marker="o", markersize=3, linewidth=1.6, alpha=0.95, zorder=3,
                label=arm_label(word, seed, show_seed, rec["final_humaneval"]),
            )
            all_y.extend(rec["mbpp"])

    ax.axhline(
        panel["instruct_ref"], color=COLOR_REFERENCE, linestyle="dashed",
        linewidth=1.1, alpha=0.7, zorder=1,
        label=f"Instruct untrained, {panel['instruct_ref']:.3f}",
    )
    ax.axhline(
        panel["base_ref"], color=COLOR_REFERENCE, linestyle="dotted",
        linewidth=1.1, alpha=0.7, zorder=1,
        label=f"base untrained, {panel['base_ref']:.3f}",
    )

    ax.set_ylim(0, max(all_y) * 1.15)
    ax.set_xlabel("Training step", fontsize=9)
    ax.set_ylabel("MBPP pass@1\n(200 held-out, full asserts)", fontsize=9)
    ax.set_title(panel["title"], fontsize=9.5, color=TEXT_COLOR, pad=8)
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.46), ncol=2,
        fontsize=7.2, frameon=False, handlelength=2.8,
        columnspacing=1.0, handletextpad=0.4, labelspacing=0.35,
    )
    style_axes(ax)


def make_figure():
    plt.rcParams["font.size"] = 9.5
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.6), dpi=200)

    table_rows = []
    for ax, panel in zip(axes, PANELS):
        print(f"Panel {panel['key']}:")
        draw_panel(ax, panel, table_rows)

    footnote = (
        "Held-out MBPP pass@1 on 200 test problems with the full assert set "
        "(standard error about 3 points); HumanEval pass@1 at the final step "
        "in the legend."
    )
    fig.text(0.5, 0.01, footnote, ha="center", fontsize=7.8, color=TEXT_COLOR)
    fig.subplots_adjust(top=0.84, bottom=0.42, left=0.11, right=0.98, wspace=0.38)

    png_path = f"{OUT_DIR}/fig3_code.png"
    fig.savefig(png_path, dpi=200)
    fig.savefig(f"{OUT_DIR}/fig3_code.pdf")
    plt.close(fig)
    print("wrote", png_path, "(+pdf)")
    return table_rows


def print_markdown_table(rows):
    print("| experiment | arm | seed | steps completed | final MBPP | final HumanEval | source directory |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        mbpp = f"{r['final_mbpp']:.3f}" if r["final_mbpp"] is not None else "-"
        he = f"{r['final_humaneval']:.3f}" if r["final_humaneval"] is not None else "-"
        print(f"| {r['experiment']} | {r['arm']} | {r['seed']} | {r['steps_completed']} | "
              f"{mbpp} | {he} | `{r['source']}` |")


if __name__ == "__main__":
    print("Building fig3_code (held-out MBPP pass@1 vs training step):")
    rows = make_figure()

    print("\nUntrained references:")
    for panel in PANELS:
        print(f"  {panel['key']}: Instruct {panel['instruct_ref']:.3f}, "
              f"base {panel['base_ref']:.3f} ({panel['ref_source']})")

    print("\nData table (paste into FIGURE3.md):")
    print_markdown_table(rows)
