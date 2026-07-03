"""
make_figures.py  -- publication figures for the AD small-molecule-miRNA benchmark.

Reads result CSVs produced by your Phase-5 pipeline and writes figures to FIGDIR.
Nothing here invents data: if an input file is missing, that figure is SKIPPED
with a printed message rather than fabricated.

INPUT FILES (see which figures each unlocks):
  phase5_aupr_per_seed.csv        [you already have this]   -> Fig 2 (AUPR panel), Fig 3
        wide format: columns RF, NBI, GNN_full, GNN_nocontext ; one row per seed ; values = AUPR
  phase5_metrics_per_seed.csv     [PATCH A required]        -> Fig 2 (AUPR + AUROC), bias values
        long format: method, seed, aupr, auroc, degree_bias_r
  predictions_seed{S}.csv         [PATCH B required]        -> Fig 4, Fig 5
        columns: m_id, d_id, te_deg, y_true, score_RF, score_NBI, score_GNN_full, score_GNN_nocontext
  ablation_metrics.csv            [PATCH C required]        -> Fig ABLATION (your headline integrity figure)
        long format: neg_mode, method, seed, aupr, auroc, degree_bias_r
  pos_edges.csv                   [you already have this]   -> Fig S1 (degree distribution)
        columns: mimat, drug_key

Run:  python make_figures.py
Deps: numpy, pandas, matplotlib, scipy, scikit-learn  (graphviz optional, Fig 1 only)
"""

import os
import os as _os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, spearmanr

# ----------------------------------------------------------------------------
G = _os.environ.get("MIRNA_DATA", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "data", "graph"))
FIGDIR = _os.environ.get("MIRNA_FIGS", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "figures"))
REPRESENTATIVE_SEED = 0                              # which predictions_seed{S}.csv to plot
RANDOM_AUPR_FLOOR   = 1.0 / 11.0                     # 1 / (1 + neg_ratio) = 0.0909
os.makedirs(FIGDIR, exist_ok=True)

METHODS = ["RF", "NBI", "GNN_full", "GNN_nocontext"]
LABELS  = ["RF\n(features)", "NBI\n(topology)", "GNN\n+context", "GNN\nno-context"]
COLORS  = {"RF": "#9e9e9e", "NBI": "#5b8db8", "GNN_full": "#d98c5f", "GNN_nocontext": "#3f7d56"}


def _ci95(a):
    a = np.asarray(a, dtype=float)
    return a.mean(), 1.96 * a.std(ddof=1) / np.sqrt(len(a))


def _box_strip(ax, data_by_method, ylabel, title):
    """Box plot + jittered points (no seaborn dependency)."""
    arrays = [np.asarray(data_by_method[m], dtype=float) for m in METHODS]
    bp = ax.boxplot(arrays, showfliers=False, widths=0.55, patch_artist=True,
                    showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white",
                                   markeredgecolor="black", markersize=6),
                    medianprops=dict(color="black", lw=1.4))
    for patch, m in zip(bp["boxes"], METHODS):
        patch.set_facecolor(COLORS[m]); patch.set_alpha(0.35)
    rng = np.random.default_rng(0)
    for i, (m, arr) in enumerate(zip(METHODS, arrays), start=1):
        jit = rng.uniform(-0.12, 0.12, size=len(arr))
        ax.scatter(np.full(len(arr), i) + jit, arr, s=14, color=COLORS[m],
                   edgecolor="white", linewidth=0.4, zorder=3, alpha=0.9)
    ax.set_xticks(range(1, len(METHODS) + 1)); ax.set_xticklabels(LABELS, fontsize=9)
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=11, loc="left")
    ax.grid(axis="y", alpha=0.25)


# ----------------------------------------------------------------------------
# Loader: prefer the rich per-seed file; fall back to the AUPR-only file.
# Returns {method: {"aupr": arr, "auroc": arr|None, "bias": arr|None}}
def load_per_seed():
    rich = os.path.join(G, "phase5_metrics_per_seed.csv")
    wide = os.path.join(G, "phase5_aupr_per_seed.csv")
    out = {m: {"aupr": None, "auroc": None, "bias": None} for m in METHODS}
    if os.path.exists(rich):
        d = pd.read_csv(rich)
        for m in METHODS:
            sub = d[d["method"] == m].sort_values("seed")
            out[m]["aupr"]  = sub["aupr"].to_numpy()
            out[m]["auroc"] = sub["auroc"].to_numpy()
            out[m]["bias"]  = sub["degree_bias_r"].to_numpy()
        print("[load] using phase5_metrics_per_seed.csv (AUPR + AUROC + bias)")
        return out, True
    if os.path.exists(wide):
        d = pd.read_csv(wide)
        for m in METHODS:
            if m in d.columns:
                out[m]["aupr"] = d[m].to_numpy()
        print("[load] using phase5_aupr_per_seed.csv (AUPR only -- AUROC/bias panels skipped)")
        return out, False
    print("[skip] no per-seed metrics file found")
    return None, False


# ----------------------------------------------------------------------------
def fig2_distributions(per_seed, has_rich):
    if per_seed is None or per_seed["RF"]["aupr"] is None:
        print("[skip] Fig 2: no AUPR data"); return
    ncols = 2 if has_rich else 1
    fig, axes = plt.subplots(1, ncols, figsize=(5.0 * ncols, 4.2), squeeze=False)
    _box_strip(axes[0][0], {m: per_seed[m]["aupr"] for m in METHODS},
               "AUPR", "a  AUPR across 30 seeds")
    axes[0][0].axhline(RANDOM_AUPR_FLOOR, ls="--", color="crimson", lw=1,
                       label=f"random floor ({RANDOM_AUPR_FLOOR:.3f})")
    axes[0][0].legend(fontsize=8, loc="upper left")
    if has_rich:
        _box_strip(axes[0][1], {m: per_seed[m]["auroc"] for m in METHODS},
                   "AUROC", "b  AUROC across 30 seeds")
    fig.tight_layout(); p = os.path.join(FIGDIR, "fig2_performance_distributions.png")
    fig.savefig(p, dpi=300); plt.close(fig); print("[ok]   Fig 2 ->", p)


# ----------------------------------------------------------------------------
def fig3_negative_result(per_seed):
    if per_seed is None or per_seed["GNN_nocontext"]["aupr"] is None or per_seed["GNN_full"]["aupr"] is None:
        print("[skip] Fig 3: need GNN_nocontext and GNN_full AUPR"); return
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.5, 4.3),
                                   gridspec_kw={"width_ratios": [1, 1.25]})

    # Left: paired per-seed delta (no-context minus full)
    nf = per_seed["GNN_nocontext"]["aupr"]; fu = per_seed["GNN_full"]["aupr"]
    diff = nf - fu
    rng = np.random.default_rng(1); jit = rng.uniform(-0.14, 0.14, size=len(diff))
    axL.axhline(0, color="black", lw=1)
    axL.scatter(np.ones(len(diff)) + jit, diff, s=26,
                color=["#3f7d56" if x > 0 else "#c0504d" for x in diff],
                edgecolor="white", linewidth=0.4, zorder=3)
    md, hd = _ci95(diff)
    axL.errorbar([1.45], [md], yerr=[[hd], [hd]], fmt="o", color="black", capsize=4, zorder=4)
    axL.set_xlim(0.5, 1.8); axL.set_xticks([1, 1.45])
    axL.set_xticklabels(["per-seed\nΔAUPR", "mean\n±95% CI"], fontsize=9)
    axL.set_ylabel("ΔAUPR  (GNN no-context − GNN +context)")
    axL.set_title(f"a  Paired difference  ({(diff > 0).sum()}/{len(diff)} seeds favour no-context)",
                  fontsize=10, loc="left")
    axL.grid(axis="y", alpha=0.25)

    # Right: forest plot of the four planned Wilcoxon comparisons
    comps = [("GNN_nocontext", "NBI"), ("GNN_nocontext", "GNN_full"),
             ("GNN_nocontext", "RF"), ("GNN_full", "NBI")]
    names, means, his, ps = [], [], [], []
    for a, b in comps:
        da, db = per_seed[a]["aupr"], per_seed[b]["aupr"]
        d = da - db
        try:
            _, p = wilcoxon(da, db)
        except ValueError:
            p = float("nan")
        m, h = _ci95(d)
        names.append(f"{a}\nvs {b}"); means.append(m); his.append(h); ps.append(p)
    ypos = np.arange(len(comps))[::-1]
    axR.axvline(0, color="black", lw=1)
    for y, m, h, p in zip(ypos, means, his, ps):
        sig = (not np.isnan(p)) and p < 0.0125
        col = "#3f7d56" if sig else "#9e9e9e"
        if np.isnan(p):
            ptxt = "p=NA"
        elif p < 0.0001:
            ptxt = "p<0.0001"
        else:
            ptxt = f"p={p:.4f}"
        axR.errorbar([m], [y], xerr=[[h], [h]], fmt="o", color=col, capsize=4, ms=7)
        axR.text(m, y + 0.15, ptxt + ("  (sig, Bonferroni)" if sig else "  (ns)"),
                 fontsize=8, ha="center", va="bottom", color=col)
    axR.set_ylim(-0.6, len(comps) - 1 + 0.85)
    axR.set_yticks(ypos); axR.set_yticklabels(names, fontsize=8)
    axR.set_xlabel("Mean ΔAUPR (±95% CI)")
    axR.set_title("b  Paired Wilcoxon comparisons (Bonferroni threshold p<0.0125)",
                  fontsize=10, loc="left", pad=10)
    axR.grid(axis="x", alpha=0.25)

    fig.tight_layout(); p = os.path.join(FIGDIR, "fig3_negative_result.png")
    fig.savefig(p, dpi=300); plt.close(fig); print("[ok]   Fig 3 ->", p)


# ----------------------------------------------------------------------------
def _load_predictions():
    f = os.path.join(G, f"predictions_seed{REPRESENTATIVE_SEED}.csv")
    if not os.path.exists(f):
        return None
    return pd.read_csv(f)


def fig4_degree_bias(pred):
    if pred is None:
        print("[skip] Fig 4: predictions_seed*.csv not found (apply PATCH B, re-run)"); return
    mean_bias = {}
    rich = os.path.join(G, "phase5_metrics_per_seed.csv")
    if os.path.exists(rich):
        dm = pd.read_csv(rich)
        for m in METHODS:
            mean_bias[m] = dm[dm.method == m]["degree_bias_r"].mean()
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for ax, m in zip(axes.ravel(), METHODS):
        s = pred[f"score_{m}"].to_numpy(); deg = pred["te_deg"].to_numpy()
        r = spearmanr(s, deg).correlation
        ax.scatter(deg, s, s=12, alpha=0.4, color=COLORS[m], edgecolor="none")
        title = f"{m}   seed-{REPRESENTATIVE_SEED} r = {r:+.3f}"
        if m in mean_bias:
            title += f"   (30-seed mean {mean_bias[m]:+.3f})"
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_xlabel("summed endpoint degree  k(m)+k(d)"); ax.set_ylabel("predicted score")
        ax.grid(alpha=0.2)
    fig.suptitle(f"Fig 4  Degree-bias on test pairs (single seed {REPRESENTATIVE_SEED} shown; "
                 f"30-seed mean is the Table 2 value; low r = not riding hubness)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); p = os.path.join(FIGDIR, "fig4_degree_bias.png")
    fig.savefig(p, dpi=300); plt.close(fig); print("[ok]   Fig 4 ->", p)


def fig5_pr_roc(pred):
    if pred is None:
        print("[skip] Fig 5: predictions_seed*.csv not found (apply PATCH B, re-run)"); return
    from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, roc_auc_score
    y = pred["y_true"].to_numpy()
    fig, (axP, axR) = plt.subplots(1, 2, figsize=(11, 4.6))
    for m in METHODS:
        s = pred[f"score_{m}"].to_numpy()
        pr, rc, _ = precision_recall_curve(y, s); ap = average_precision_score(y, s)
        axP.plot(rc, pr, color=COLORS[m], lw=1.8, label=f"{m} (AUPR={ap:.3f})")
        fpr, tpr, _ = roc_curve(y, s); au = roc_auc_score(y, s)
        axR.plot(fpr, tpr, color=COLORS[m], lw=1.8, label=f"{m} (AUROC={au:.3f})")
    axP.axhline(RANDOM_AUPR_FLOOR, ls="--", color="crimson", lw=1,
                label=f"random floor ({RANDOM_AUPR_FLOOR:.3f})")
    axP.set_xlabel("Recall"); axP.set_ylabel("Precision")
    axP.set_title("a  Precision-Recall", fontsize=11, loc="left"); axP.legend(fontsize=8)
    axR.plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
    axR.set_xlabel("False positive rate"); axR.set_ylabel("True positive rate")
    axR.set_title("b  ROC (ranking differs from PR)", fontsize=11, loc="left"); axR.legend(fontsize=8)
    fig.tight_layout(); p = os.path.join(FIGDIR, "fig5_pr_roc.png")
    fig.savefig(p, dpi=300); plt.close(fig); print("[ok]   Fig 5 ->", p)


# ----------------------------------------------------------------------------
def fig_ablation():
    f = os.path.join(G, "ablation_metrics.csv")
    if not os.path.exists(f):
        print("[skip] Fig ABLATION: ablation_metrics.csv not found (apply PATCH C, re-run)"); return
    d = pd.read_csv(f)
    modes = ["degree_matched", "uniform_random"]
    present = [mo for mo in modes if mo in set(d["neg_mode"])]
    if len(present) < 2:
        print("[skip] Fig ABLATION: need both 'degree_matched' and 'uniform_random' rows"); return
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.6))
    x = np.arange(len(METHODS)); w = 0.38
    for j, mode in enumerate(modes):
        au_m, au_h, bi_m, bi_h = [], [], [], []
        for m in METHODS:
            sub = d[(d["neg_mode"] == mode) & (d["method"] == m)]
            mm, hh = _ci95(sub["auroc"]); au_m.append(mm); au_h.append(hh)
            bm, bh = _ci95(sub["degree_bias_r"]); bi_m.append(bm); bi_h.append(bh)
        off = (-w / 2 if j == 0 else w / 2)
        hatch = "" if mode == "degree_matched" else "//"
        axA.bar(x + off, au_m, w, yerr=au_h, capsize=3, label=mode, hatch=hatch,
                color=["#3f7d56" if j == 0 else "#c0504d"][0], alpha=0.7)
        axB.bar(x + off, bi_m, w, yerr=bi_h, capsize=3, label=mode, hatch=hatch,
                color=["#3f7d56" if j == 0 else "#c0504d"][0], alpha=0.7)
    for ax, ttl, yl in [(axA, "a  AUROC: degree-matched vs random negatives", "AUROC (mean ±95% CI)"),
                        (axB, "b  Degree-bias r: the shortcut, made visible", "Spearman r (score vs degree)")]:
        ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=9)
        ax.set_title(ttl, fontsize=10, loc="left"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); p = os.path.join(FIGDIR, "fig_ablation_negatives.png")
    fig.savefig(p, dpi=300); plt.close(fig); print("[ok]   Fig ABLATION ->", p)


# ----------------------------------------------------------------------------
def figS1_degree_distribution(top_n=12):
    f = os.path.join(G, "pos_edges.csv")
    if not os.path.exists(f):
        print("[skip] Fig S1: pos_edges.csv not found"); return
    e = pd.read_csv(f)
    drug_deg = e["drug_key"].value_counts()
    mir_deg = e["mimat"].value_counts()
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.4))
    axA.bar(range(len(drug_deg)), drug_deg.values, color="#d98c5f")
    axA.set_xlabel("drug rank"); axA.set_ylabel("number of positive edges")
    axA.set_title("a  Drug degree distribution (hub-dominated)", fontsize=10, loc="left")
    top3 = drug_deg.head(3)
    box = "Top hubs:\n" + "\n".join(f"{name} ({int(v)})" for name, v in top3.items())
    axA.text(0.97, 0.95, box, transform=axA.transAxes, ha="right", va="top",
             fontsize=8, bbox=dict(boxstyle="round", fc="white", ec="#d98c5f", alpha=0.9))
    axB.bar(range(len(mir_deg)), mir_deg.values, color="#5b8db8")
    axB.set_xlabel("miRNA rank"); axB.set_ylabel("number of positive edges")
    axB.set_title("b  miRNA degree distribution", fontsize=10, loc="left")
    fig.tight_layout(); p = os.path.join(FIGDIR, "figS1_degree_distribution.png")
    fig.savefig(p, dpi=300); plt.close(fig); print("[ok]   Fig S1 ->", p)


# ----------------------------------------------------------------------------
def fig1_schematic():
    """Non-data schematic of the heterogeneous graph + benchmark ladder.
    Needs the `graphviz` python package AND the Graphviz system binary."""
    try:
        from graphviz import Digraph
    except Exception:
        print("[skip] Fig 1: graphviz not installed (pip install graphviz + system Graphviz). "
              "This figure is a schematic; drawing it by hand in Inkscape/PowerPoint is fine too.")
        return
    g = Digraph("schematic", format="png")
    g.attr(rankdir="LR", fontsize="11")
    g.node("mirna", "miRNA\n(25 nodes,\n64-d 3-mer)", shape="ellipse", style="filled", fillcolor="#cfe3f0")
    g.node("drug", "drug\n(257 nodes,\n512-bit Morgan)", shape="ellipse", style="filled", fillcolor="#f3d9c6")
    g.node("gene", "gene\n(324 nodes)", shape="ellipse", style="filled", fillcolor="#d8ead8")
    g.node("path", "pathway\nhsa05010", shape="box", style="filled", fillcolor="#eee2c0")
    g.edge("mirna", "drug", label="assoc (470)\nPREDICTION TARGET", color="#b5651d", penwidth="2")
    g.edge("mirna", "gene", label="targets (412)\ncontext only", style="dashed")
    g.edge("gene", "path", label="member (25)\ncontext only", style="dashed")
    g.render(os.path.join(FIGDIR, "fig1_schematic"), cleanup=True)
    print("[ok]   Fig 1 ->", os.path.join(FIGDIR, "fig1_schematic.png"))


def main():
    print("=== generating figures into", FIGDIR, "===")
    per_seed, has_rich = load_per_seed()
    fig2_distributions(per_seed, has_rich)
    fig3_negative_result(per_seed)
    pred = _load_predictions()
    fig4_degree_bias(pred)
    fig5_pr_roc(pred)
    fig_ablation()
    figS1_degree_distribution()
    fig1_schematic()
    print("=== done ===")


if __name__ == "__main__":
    main()
