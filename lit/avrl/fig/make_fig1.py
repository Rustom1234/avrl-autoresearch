"""
Headline figure for the AVRL thesis: cheap-feature structure statistics vs.
the 25% budget relative gain of greedy_online over static_matched.

Data is hardcoded from (and cross-checked against) the ledger entries and
run artifacts listed in FIGURE1.md. No new computation happens here beyond
unit conversion (fraction -> percent). Re-run this script if the ledger
numbers ever change; do not hand-edit the PNG/PDF outputs.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

OUT_DIR = "/homes/rdubash/active-audit/lit/avrl/fig"

# Colorblind-safe categorical pair validated with the dataviz skill's
# validate_palette.js (--pairs all, light mode): worst CVD delta-E 24.7,
# worst normal-vision delta-E 33.6, both well clear of the floors.
COLOR_EXACT = "#2a78d6"   # blue: exact-answer tasks (GSM8K, MBPP)
COLOR_JUDGE = "#eb6834"   # orange: judge-scored task (HelpSteer2)
GRID_COLOR = "#dddddd"    # light grey, lighter than default axis ink
TEXT_COLOR = "#222222"

# id -> (task, policy label, cheap_auc, proxy_error_rate [fraction],
#        confident_subset_neighbor_ratio_cell100, rel_gain_pct, ci_lo_pct, ci_hi_pct,
#        category, inconclusive)
#
# rel_gain/ci for all six points are greedy_online vs static_matched at 25%
# budget, read directly from avrl/runs/<point>/bootstrap.json (cell100, cheap
# features), cross-checked against RESEARCH_LOG.md's M01b/L10a/L9b/L9c/L9d tables.
#
POINTS = {
    "E01": dict(
        task="GSM8K",
        policy="Qwen2.5-1.5B-Instruct, majority vote",
        auc=0.9230886365179956,
        err_rate=0.14533333333333334,
        neighbor_ratio=1.1585434173669469,
        rel_gain=19.2,
        ci=(15.0, 26.3),
        category="exact",
        inconclusive=False,
        note="greedy_online vs static_matched, M01b / p0/bootstrap.json",
    ),
    "E06": dict(
        task="GSM8K (0.5B)",
        policy="Qwen2.5-0.5B-Instruct, majority vote",
        auc=0.8866092030800408,
        err_rate=0.384,
        neighbor_ratio=0.46374960076652827,
        rel_gain=20.9,
        ci=(18.7, 26.1),
        category="exact",
        inconclusive=False,
        note="greedy_online vs static_matched, L10a / E06/bootstrap.json",
    ),
    "E08": dict(
        task="MBPP",
        policy="Qwen2.5-1.5B-Instruct, output-signature vote / test execution",
        auc=0.8675833047184964,
        err_rate=0.48833333333333334,
        neighbor_ratio=1.4464285714285716,
        rel_gain=8.3,
        ci=(1.8, 13.1),
        category="exact",
        inconclusive=False,
        note="greedy_online vs static_matched, L10a / E08/bootstrap.json",
    ),
    "E09b": dict(
        task="HelpSteer2",
        policy="RM proxy / Qwen2.5-3B-Instruct judge (fallback)",
        auc=0.7460491123278581,
        err_rate=0.18066666666666667,
        neighbor_ratio=0.7042518028846154,
        rel_gain=25.7,
        ci=(5.3, 26.2),
        category="judge",
        inconclusive=False,
        note="greedy_online vs static_matched, L9b / E09b/bootstrap.json",
    ),
    "E09c": dict(
        task="HelpSteer2",
        policy="RM proxy / Qwen2.5-7B-Instruct-AWQ judge",
        auc=0.7988681278505654,
        err_rate=0.062,
        neighbor_ratio=1.8371096142069812,
        rel_gain=11.7,
        ci=(-5.7, 33.7),
        category="judge",
        inconclusive=True,
        note="greedy_online vs static_matched, L9c / E09c/bootstrap.json; CI includes zero",
    ),
    "E09d": dict(
        task="HelpSteer2 (4k)",
        policy="RM proxy / Qwen2.5-7B-Instruct-AWQ judge, 4,000 prompts",
        auc=0.7948275862068964,
        err_rate=0.0725,
        neighbor_ratio=2.024408583492394,
        rel_gain=18.8,
        ci=(7.8, 26.1),
        category="judge",
        inconclusive=False,
        note="greedy_online vs static_matched, L9d / E09d/bootstrap.json",
    ),
}

LABEL_OFFSETS = {
    # (dx, dy) in points, hand-tuned to avoid collisions
    "E01": (10, -16),
    "E06": (-105, 10),
    "E08": (8, -14),
    "E09b": (8, 6),
    "E09c": (-10, 12),
    "E09d": (8, -14),
}
LABEL_OFFSETS_B = {
    "E01": (8, 6),
    "E06": (-10, 10),
    "E08": (8, -4),
    "E09b": (8, 6),
    "E09c": (-10, 12),
    "E09d": (8, -16),
}


def size_from_rate(rate_frac):
    """Marker area in points^2, linear in proxy error rate (plain percent)."""
    pct = rate_frac * 100
    return 55 + pct * 9.0


def draw_point(ax, x, y, pid, d):
    color = COLOR_EXACT if d["category"] == "exact" else COLOR_JUDGE
    size = size_from_rate(d["err_rate"])
    ci_lo, ci_hi = d["ci"]
    yerr = [[y - ci_lo], [ci_hi - y]]
    ax.errorbar(
        [x], [y], yerr=yerr,
        fmt="none", ecolor=color, elinewidth=1.3, capsize=3, capthick=1.3,
        zorder=2,
    )
    if d["inconclusive"]:
        ax.scatter(
            [x], [y], s=size, facecolors="none", edgecolors=color,
            linewidths=1.8, zorder=3,
        )
    else:
        ax.scatter(
            [x], [y], s=size, facecolors=color, edgecolors=color,
            linewidths=0.8, alpha=0.9, zorder=3,
        )
    return color


def annotate(ax, x, y, pid, task, offsets):
    dx, dy = offsets[pid]
    ax.annotate(
        f"{pid}: {task}",
        xy=(x, y), xytext=(dx, dy), textcoords="offset points",
        fontsize=8.5, color=TEXT_COLOR, zorder=4,
    )


def style_axes(ax):
    ax.axhline(0, linestyle="--", linewidth=1.0, color="#999999", zorder=1)
    ax.grid(True, linewidth=0.6, color=GRID_COLOR, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#888888")
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)


def legend_handles():
    exact_h = plt.Line2D([0], [0], marker="o", linestyle="none",
                          markerfacecolor=COLOR_EXACT, markeredgecolor=COLOR_EXACT,
                          markersize=8, label="Exact-answer task (GSM8K, MBPP)")
    judge_h = plt.Line2D([0], [0], marker="o", linestyle="none",
                          markerfacecolor=COLOR_JUDGE, markeredgecolor=COLOR_JUDGE,
                          markersize=8, label="Judge-scored task (HelpSteer2)")
    hollow_h = plt.Line2D([0], [0], marker="o", linestyle="none",
                           markerfacecolor="none", markeredgecolor="#555555",
                           markersize=8, label="Inconclusive (95% CI includes zero)")
    return [exact_h, judge_h, hollow_h]


def make_figure(x_key, x_label, out_stem, label_offsets, write_pdf):
    plt.rcParams["font.size"] = 9.5
    fig, ax = plt.subplots(figsize=(5.6, 5.0), dpi=200)

    for pid, d in POINTS.items():
        x = d["auc"] if x_key == "auc" else d["err_rate"] * 100
        y = d["rel_gain"]
        draw_point(ax, x, y, pid, d)
        annotate(ax, x, y, pid, d["task"], label_offsets)

    style_axes(ax)
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel("Relative gain of greedy_online over\nstatic_matched at 25% budget (%)", fontsize=10)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin - 2, ymax + 2)

    # Legend lives above the axes so it never overlaps a whisker or point.
    fig.legend(
        handles=legend_handles(), loc="upper center", bbox_to_anchor=(0.56, 1.0),
        ncol=1, fontsize=8, frameon=False,
    )
    fig.subplots_adjust(top=0.78, left=0.16, right=0.78, bottom=0.11)

    png_path = f"{OUT_DIR}/{out_stem}.png"
    fig.savefig(png_path, dpi=200)
    if write_pdf:
        fig.savefig(f"{OUT_DIR}/{out_stem}.pdf")
    plt.close(fig)
    print("wrote", png_path, "(+pdf)" if write_pdf else "")


if __name__ == "__main__":
    make_figure(
        x_key="auc",
        x_label="Cheap-feature AUC for proxy error",
        out_stem="fig1_auc_vs_gain",
        label_offsets=LABEL_OFFSETS,
        write_pdf=True,
    )
    make_figure(
        x_key="err_rate",
        x_label="Proxy error rate (%)",
        out_stem="fig1b_error_rate_vs_gain",
        label_offsets=LABEL_OFFSETS_B,
        write_pdf=False,
    )
