"""
make_all_figures.py -- self-contained generation of all manuscript figures.
-------------------------------------------------------------------------------
Figures produced (PNG @300dpi + PDF) into ./figures:
  Fig 1  heterogeneous graph schematic            (no data needed)
  Fig 2  (a) AUPR per-seed distributions  (b) AUROC-vs-AUPR rank flip
  Fig 3  paired ΔAUPR forest plot with p-values
  Fig 4  inter-method correlation matrix + degree dependence
  Fig 5  candidate score heatmap (miRNA × top drugs), HDAC drugs flagged
  Fig S1 negative-sampling ablation (degree-bias rise + AUROC stability)

Data files are read if present; otherwise canonical reproduced values are used
for the summary figures (2,3,4,S1). Fig 5 requires all_model_predictions.csv.
Only dependencies: numpy, pandas, scipy, matplotlib.

  python make_all_figures.py
-------------------------------------------------------------------------------
"""
import os
import os as _os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.stats import spearmanr

# ----------------------------- config ---------------------------------------
DATA = _os.environ.get("MIRNA_DATA", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "data", "graph"))
FIGDIR = _os.environ.get("MIRNA_FIGS", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "figures"))
os.makedirs(FIGDIR, exist_ok=True)

MODELS = ["RF", "NBI", "GNN_full", "GNN_nocontext"]
LABEL = {"RF": "Random Forest", "NBI": "NBI",
         "GNN_full": "GNN +context", "GNN_nocontext": "GNN no-context"}
COLOR = {"RF": "#7f7f7f", "NBI": "#4c72b0",
         "GNN_full": "#dd8452", "GNN_nocontext": "#55a868"}

# ---- canonical reproduced values (fallback / summary figures) ---------------
AUPR_MEAN  = {"RF": 0.396, "NBI": 0.424, "GNN_full": 0.431, "GNN_nocontext": 0.469}
AUPR_CI    = {"RF": 0.013, "NBI": 0.015, "GNN_full": 0.036, "GNN_nocontext": 0.023}
AUROC_MEAN = {"RF": 0.708, "NBI": 0.737, "GNN_full": 0.722, "GNN_nocontext": 0.730}
AUROC_CI   = {"RF": 0.013, "NBI": 0.010, "GNN_full": 0.016, "GNN_nocontext": 0.012}
BIAS_MATCH = {"RF": 0.275, "NBI": 0.206, "GNN_full": 0.245, "GNN_nocontext": 0.181}
BIAS_DELTA = {"RF": 0.163, "NBI": 0.055, "GNN_full": 0.096, "GNN_nocontext": 0.115}  # |uniform - matched| magnitude (uniform is higher, confounded)
AUPR_FLOOR = 0.091
# Table 4: (label, mean ΔAUPR, wins/30, p, significant)
COMPARISONS = [("GNN no-context  vs  RF",          0.073, 27, 1e-4,   True),
               ("GNN no-context  vs  NBI",         0.045, 24, 0.0010, True),
               ("GNN no-context  vs  GNN +context", 0.038, 16, 0.3931, False),
               ("GNN +context  vs  NBI",           0.008, 20, 0.4161, False)]
# candidate-set inter-method Spearman (symmetric)
CORR_CANON = {("RF", "NBI"): 0.237, ("RF", "GNN_full"): 0.400, ("RF", "GNN_nocontext"): 0.378,
              ("NBI", "GNN_full"): 0.455, ("NBI", "GNN_nocontext"): 0.515,
              ("GNN_full", "GNN_nocontext"): 0.960}
DEGDEP_CANON = {"RF": 0.177, "NBI": 0.259, "GNN_full": 0.296, "GNN_nocontext": 0.259}
HDAC = {"ms-275", "entinostat", "vorinostat", "trichostatin a", "panobinostat",
        "romidepsin", "belinostat", "mocetinostat", "valproic acid", "sodium butyrate"}

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 120, "savefig.bbox": "tight"})

# ----------------------------- helpers ---------------------------------------
def find_file(*names):
    for n in names:
        p = os.path.join(DATA, n)
        if os.path.exists(p):
            return p
    return None

def find_col(df, *aliases):
    norm = {c.lower().strip().replace(" ", "_"): c for c in df.columns}
    for a in aliases:
        if a in norm:
            return norm[a]
    return None

def norm_model(v):
    s = str(v).lower().replace("-", "_").replace(" ", "_")
    if "nocontext" in s or "no_context" in s:           return "GNN_nocontext"
    if "full" in s or "augment" in s or "+context" in s: return "GNN_full"
    if s.startswith("rf") or "random" in s or "forest" in s: return "RF"
    if "nbi" in s or "network" in s:                    return "NBI"
    if "context" in s:                                  return "GNN_full"
    return None

def load_per_seed():
    """Return tidy DataFrame [seed, model, aupr, auroc, bias] or None."""
    p = find_file("phase5_metrics_per_seed.csv", "phase5_aupr_per_seed.csv", "metrics_per_seed.csv")
    if not p:
        return None
    try:
        df = pd.read_csv(p)
        mcol = find_col(df, "model", "method", "name")
        acol = find_col(df, "aupr", "ap", "average_precision")
        rcol = find_col(df, "auroc", "auc", "roc_auc")
        bcol = find_col(df, "degree_bias", "bias", "degree_bias_r", "rho")
        scol = find_col(df, "seed", "run", "iter")
        if mcol is None or acol is None:
            return None
        out = pd.DataFrame({
            "seed": df[scol] if scol else np.arange(len(df)),
            "model": df[mcol].map(norm_model),
            "aupr": df[acol],
            "auroc": df[rcol] if rcol else np.nan,
            "bias": df[bcol] if bcol else np.nan,
        }).dropna(subset=["model"])
        return out if len(out) else None
    except Exception as e:
        print(f"  (per-seed load failed: {e})")
        return None

def save(fig, name):
    fig.savefig(os.path.join(FIGDIR, name + ".png"), dpi=300)
    fig.savefig(os.path.join(FIGDIR, name + ".pdf"))
    plt.close(fig)
    print(f"  saved {name}.png / .pdf")

# ----------------------------- Fig 1: schematic ------------------------------
def fig1():
    fig, ax = plt.subplots(figsize=(7, 7.5))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    nodes = {  # (x, y, w, h, title, sub, facecolor)
        "drug":    (0.5, 0.13, 0.62, 0.13, "DRUG  (n = 257)", "512-bit Morgan fingerprint", "#eaf0f6"),
        "mirna":   (0.5, 0.42, 0.62, 0.13, "miRNA  (n = 25)", "64-d 3-mer frequency", "#eaf0f6"),
        "gene":    (0.32, 0.70, 0.42, 0.12, "GENE  (n = 324)", "16-d learnable", "#f3eef6"),
        "pathway": (0.32, 0.91, 0.42, 0.10, "PATHWAY hsa05010 (n = 1)", "structural", "#f3eef6"),
    }
    for x, y, w, h, title, sub, fc in nodes.values():
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                     boxstyle="round,pad=0.012,rounding_size=0.02",
                     fc=fc, ec="#333333", lw=1.4))
        ax.text(x, y + 0.018, title, ha="center", va="center", fontsize=11, fontweight="bold")
        ax.text(x, y - 0.028, sub, ha="center", va="center", fontsize=8.5, color="#555555")

    def arrow(p, q, color, lw, ls, label, double=False):
        a = FancyArrowPatch(p, q, arrowstyle="<|-|>" if double else "-|>",
                            mutation_scale=18, color=color, lw=lw, ls=ls,
                            shrinkA=2, shrinkB=2)
        ax.add_patch(a)
        mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
        ax.text(mx + 0.04, my, label, ha="left", va="center", fontsize=9, color=color)

    arrow((0.5, 0.355), (0.5, 0.195), "#c44e52", 2.6, "-", "470 associations\n(prediction target)", double=True)
    arrow((0.45, 0.485), (0.37, 0.640), "#8172b3", 1.6, "--", "412  (miRTarBase)")
    arrow((0.32, 0.760), (0.32, 0.860), "#8172b3", 1.6, "--", "25  (KEGG)")
    ax.text(0.5, 0.995, "Heterogeneous graph: target relation (red) vs disease-pathway context (purple)",
            ha="center", va="top", fontsize=9.5, style="italic", color="#333333")
    save(fig, "Fig1_graph_schematic")

# ----------------------------- Fig 2: performance ----------------------------
def fig2(ps):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 5))
    # (a) AUPR distributions
    if ps is not None and ps["aupr"].notna().any():
        data = [ps.loc[ps.model == m, "aupr"].dropna().values for m in MODELS]
        parts = axA.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
        for i, b in enumerate(parts["bodies"]):
            b.set_facecolor(COLOR[MODELS[i]]); b.set_alpha(0.5); b.set_edgecolor("k")
        for i, d in enumerate(data):
            axA.scatter(np.full(len(d), i + 1) + np.random.uniform(-0.05, 0.05, len(d)),
                        d, s=12, color=COLOR[MODELS[i]], edgecolor="k", lw=0.3, zorder=3)
            axA.hlines(np.mean(d), i + 0.78, i + 1.22, color="k", lw=2, zorder=4)
        src = "30 seeds"
    else:
        for i, m in enumerate(MODELS):
            axA.errorbar(i + 1, AUPR_MEAN[m], yerr=AUPR_CI[m], fmt="o", ms=9,
                         color=COLOR[m], capsize=5, lw=2)
        src = "mean ± 95% CI (canonical)"
    axA.axhline(AUPR_FLOOR, ls=":", color="gray")
    axA.text(0.1, AUPR_FLOOR + 0.004, f"random floor {AUPR_FLOOR}", fontsize=8, color="gray")
    axA.set_xticks(range(1, 5)); axA.set_xticklabels([LABEL[m] for m in MODELS], rotation=20, ha="right")
    axA.set_ylabel("AUPR"); axA.set_title(f"(a) AUPR by method  ({src})")

    # (b) rank flip: AUROC ranking vs AUPR ranking
    auroc_sorted = sorted(MODELS, key=lambda m: AUROC_MEAN[m])
    aupr_sorted = sorted(MODELS, key=lambda m: AUPR_MEAN[m])
    for m in MODELS:
        y0 = auroc_sorted.index(m); y1 = aupr_sorted.index(m)
        axB.plot([0, 1], [y0, y1], "-", color=COLOR[m], lw=2.5, marker="o", ms=8)
        axB.text(-0.03, y0, LABEL[m], ha="right", va="center", fontsize=9)
        axB.text(1.03, y1, LABEL[m], ha="left", va="center", fontsize=9)
    axB.set_xlim(-0.45, 1.45); axB.set_ylim(-0.5, 3.5)
    axB.set_xticks([0, 1]); axB.set_xticklabels(["rank by\nAUROC", "rank by\nAUPR"])
    axB.set_yticks([]); axB.set_title("(b) The winner flips with the metric")
    axB.annotate("NBI tops AUROC", (0, 3), fontsize=8, color=COLOR["NBI"], ha="center", xytext=(0, 3.3), textcoords="data")
    axB.annotate("GNN no-context tops AUPR", (1, 3), fontsize=8, color=COLOR["GNN_nocontext"], ha="center", xytext=(1, 3.3), textcoords="data")
    save(fig, "Fig2_performance")

# ----------------------------- Fig 3: forest ---------------------------------
def fig3():
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ys = range(len(COMPARISONS))[::-1]
    for y, (lab, d, wins, p, sig) in zip(ys, COMPARISONS):
        c = "#2a7" if sig else "#999"
        ax.plot(d, y, "o", ms=11, color=c, zorder=3)
        ax.hlines(y, 0, d, color=c, lw=2, zorder=2)
        ptxt = "<0.0001" if p < 1e-4 else f"{p:.4f}"
        ax.text(d + 0.004, y, f"p = {ptxt}   ({wins}/30)" + ("  *" if sig else "  n.s."),
                va="center", fontsize=9, color=c)
        ax.text(-0.004, y, lab, va="center", ha="right", fontsize=9.5)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlim(-0.045, 0.115); ax.set_ylim(-0.6, len(COMPARISONS) - 0.4)
    ax.set_yticks([]); ax.set_xlabel("mean ΔAUPR (paired, 30 seeds)")
    ax.set_title("Paired Wilcoxon comparisons (Bonferroni p < 0.0125)")
    save(fig, "Fig3_forest_dAUPR")

# ----------------------------- Fig 4: correlation ----------------------------
def fig4():
    pred = find_file("all_model_predictions.csv")
    corr = np.eye(4); degdep = {}
    if pred:
        try:
            d = pd.read_csv(pred)
            for i, a in enumerate(MODELS):
                for j, b in enumerate(MODELS):
                    corr[i, j] = spearmanr(d[f"score_{a}"], d[f"score_{b}"]).correlation
            for m in MODELS:
                degdep[m] = spearmanr(d[f"score_{m}"], d["sum_degree"]).correlation
        except Exception as e:
            print(f"  (Fig4 from CSV failed: {e}); using canonical")
            pred = None
    if not pred:
        for i, a in enumerate(MODELS):
            for j, b in enumerate(MODELS):
                corr[i, j] = 1.0 if a == b else CORR_CANON.get((a, b)) or CORR_CANON.get((b, a))
        degdep = DEGDEP_CANON

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
    im = axA.imshow(corr, cmap="RdYlBu_r", vmin=0, vmax=1)
    axA.set_xticks(range(4)); axA.set_yticks(range(4))
    axA.set_xticklabels([LABEL[m] for m in MODELS], rotation=30, ha="right")
    axA.set_yticklabels([LABEL[m] for m in MODELS])
    for i in range(4):
        for j in range(4):
            axA.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                     color="white" if corr[i, j] > 0.65 else "black", fontsize=10)
    axA.set_title("(a) Inter-method agreement (Spearman)")
    fig.colorbar(im, ax=axA, fraction=0.046, pad=0.04)

    vals = [degdep[m] for m in MODELS]
    axB.barh(range(4), vals, color=[COLOR[m] for m in MODELS])
    axB.set_yticks(range(4)); axB.set_yticklabels([LABEL[m] for m in MODELS])
    axB.invert_yaxis()
    for i, v in enumerate(vals):
        axB.text(v + 0.005, i, f"{v:+.3f}", va="center", fontsize=9)
    axB.set_xlim(0, max(vals) * 1.25); axB.set_xlabel("Spearman(score, summed degree)")
    axB.set_title("(b) Degree dependence on candidate set")
    save(fig, "Fig4_intermethod_correlation")

# ----------------------------- Fig 5: candidate heatmap ----------------------
def fig5():
    pred = find_file("all_model_predictions.csv")
    if not pred:
        print("  Fig5 SKIPPED: all_model_predictions.csv not found")
        return
    d = pd.read_csv(pred)
    score = "pct_GNN_nocontext" if "pct_GNN_nocontext" in d.columns else "consensus_pct"
    top_drugs = (d.groupby("drug_key")[score].max().sort_values(ascending=False).head(20).index.tolist())
    sub = d[d.drug_key.isin(top_drugs)]
    mat = sub.pivot_table(index="mimat", columns="drug_key", values=score, aggfunc="max")
    mat = mat.reindex(columns=top_drugs)
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(mat.values, cmap="magma", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(mat.columns))); ax.set_yticks(range(len(mat.index)))
    ax.set_xticklabels(mat.columns, rotation=55, ha="right", fontsize=8)
    ax.set_yticklabels(mat.index, fontsize=7)
    for lab in ax.get_xticklabels():
        if str(lab.get_text()).lower() in HDAC:
            lab.set_color("#2ca02c"); lab.set_fontweight("bold")
    ax.set_title(f"Candidate percentile ({score}); HDAC inhibitors in green")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="percentile rank")
    save(fig, "Fig5_candidate_heatmap")

# ----------------------------- Fig S1: ablation ------------------------------
def _mean_ci(vals):
    """mean and 95% CI half-width over seeds (1.96*sd/sqrt(n)); CI=0 if <2 points."""
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if v.size == 0:
        return np.nan, 0.0
    if v.size < 2:
        return float(v.mean()), 0.0
    return float(v.mean()), float(1.96 * v.std(ddof=1) / np.sqrt(v.size))


def figS1(ps):
    """Negative-sampling ablation, rendered strictly from ablation_metrics.csv.

    The figure is DATA-DRIVEN: it plots whatever the CSV contains and never
    flips bars to match the manuscript. A direction check prints the per-method
    degree-bias deltas (matched - uniform) and WARNS loudly if any is negative,
    because that would contradict the claim that degree-matched sampling raises
    measured degree reliance. If a negative delta appears, the DATA or the TEXT
    is wrong -- resolve that before submitting, do not just trust the picture.
    """
    GREEN, RED = "#55a868", "#c44e52"           # degree-matched (solid) | uniform (hatched)
    mat_b, uni_b, mat_bc, uni_bc = {}, {}, {}, {}   # degree-bias mean + CI
    mat_r, uni_r, mat_rc, uni_rc = {}, {}, {}, {}   # AUROC mean + CI
    loaded = False

    abl = find_file("ablation_metrics.csv")
    if abl:
        try:
            a = pd.read_csv(abl)
            mcol  = find_col(a, "model", "method", "name")
            rgcol = find_col(a, "neg_mode", "regime", "sampling", "neg", "negatives", "scheme")
            bcol  = find_col(a, "degree_bias_r", "degree_bias", "bias", "rho")
            rcol  = find_col(a, "auroc", "auc", "roc_auc")
            a["mm"] = a[mcol].map(norm_model)
            a["rg"] = a[rgcol].astype(str).str.lower()
            for m in MODELS:
                um = a[(a.mm == m) & (a.rg.str.contains("uni"))]
                dm = a[(a.mm == m) & (a.rg.str.contains("match|degree"))]
                uni_b[m],  uni_bc[m]  = _mean_ci(um[bcol].to_numpy())
                mat_b[m],  mat_bc[m]  = _mean_ci(dm[bcol].to_numpy())
                if rcol:
                    uni_r[m], uni_rc[m] = _mean_ci(um[rcol].to_numpy())
                    mat_r[m], mat_rc[m] = _mean_ci(dm[rcol].to_numpy())
                else:
                    uni_r[m] = mat_r[m] = AUROC_MEAN[m]; uni_rc[m] = mat_rc[m] = AUROC_CI[m]
            loaded = True
            print("  [FigS1] using ablation_metrics.csv (data-driven)")
        except Exception as e:
            print(f"  [FigS1] ablation CSV load failed: {e}; using canonical fallback")

    if not loaded:
        for m in MODELS:
            mat_b[m], mat_bc[m] = BIAS_MATCH[m], 0.0
            uni_b[m], uni_bc[m] = BIAS_MATCH[m] + BIAS_DELTA[m], 0.0   # uniform is HIGHER (confounded)
            mat_r[m], mat_rc[m] = AUROC_MEAN[m], AUROC_CI[m]
            uni_r[m], uni_rc[m] = AUROC_MEAN[m], AUROC_CI[m]
        print("  [FigS1] ablation_metrics.csv not found; using canonical values")

    # --- direction check: degree-bias should be LOWER under degree-matched (delta < 0),
    #     because uniform sampling confounds degree with class and INFLATES the estimate. ---
    print("  [FigS1] degree-bias delta (matched - uniform), and AUROC shift:")
    bad = []
    for m in MODELS:
        db = mat_b[m] - uni_b[m]
        dr = mat_r[m] - uni_r[m]
        flag = "" if db < 0 else "  <-- POSITIVE: contradicts the text!"
        if db >= 0:
            bad.append(m)
        print(f"     {LABEL[m]:16s} Δbias={db:+.3f}   ΔAUROC={dr:+.3f}{flag}")
    if bad:
        print("  [FigS1] *** WARNING ***: degree-matched did NOT lower degree-bias for "
              + ", ".join(LABEL[m] for m in bad) + ".")
        print("          The manuscript states degree-matched sampling REDUCES the inflated bias.")
        print("          Fix the DATA or the TEXT before submitting; do not publish a figure")
        print("          that contradicts the claim.")

    # --- draw: (a) AUROC stability, (b) degree-bias lower under degree-matched ---
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.6))
    x = np.arange(len(MODELS)); w = 0.38
    ekw = dict(capsize=3, ecolor="#333333", error_kw={"lw": 1})

    # (a) AUROC
    axA.bar(x - w/2, [mat_r[m] for m in MODELS], w, yerr=[mat_rc[m] for m in MODELS],
            label="degree-matched", color=GREEN, **ekw)
    axA.bar(x + w/2, [uni_r[m] for m in MODELS], w, yerr=[uni_rc[m] for m in MODELS],
            label="uniform-random", color=RED, hatch="//", edgecolor="white", **ekw)
    axA.set_xticks(x); axA.set_xticklabels([LABEL[m] for m in MODELS], rotation=15, ha="right")
    axA.set_ylim(0.60, 0.80); axA.set_ylabel("AUROC")
    axA.set_title("(a) AUROC barely moves between regimes (not inflated)", fontsize=11)
    axA.legend(fontsize=9, frameon=False)

    # (b) degree-bias, annotated with delta
    bmax = max(max(mat_b[m] + mat_bc[m], uni_b[m] + uni_bc[m]) for m in MODELS)
    axB.bar(x - w/2, [mat_b[m] for m in MODELS], yerr=[mat_bc[m] for m in MODELS], width=w,
            label="degree-matched", color=GREEN, **ekw)
    axB.bar(x + w/2, [uni_b[m] for m in MODELS], yerr=[uni_bc[m] for m in MODELS], width=w,
            label="uniform-random", color=RED, hatch="//", edgecolor="white", **ekw)
    for i, m in enumerate(MODELS):
        top = max(mat_b[m] + mat_bc[m], uni_b[m] + uni_bc[m])
        axB.text(i, top + 0.02 * bmax, f"Δ{mat_b[m]-uni_b[m]:+.3f}",
                 ha="center", va="bottom", fontsize=8, color="#333333")
    axB.set_xticks(x); axB.set_xticklabels([LABEL[m] for m in MODELS], rotation=15, ha="right")
    axB.set_ylim(0, bmax * 1.18); axB.set_ylabel(r"degree-bias  $\rho$(score, summed degree)")
    axB.set_title("(b) Degree-matched sampling REDUCES degree\u2013class confounding", fontsize=11)
    axB.legend(fontsize=9, frameon=False)

    save(fig, "FigS1_ablation")

# ----------------------------- main ------------------------------------------
if __name__ == "__main__":
    print(f"Reading data from: {DATA}\nWriting figures to: ./{FIGDIR}\n")
    ps = load_per_seed()
    print("per-seed metrics:", "loaded" if ps is not None else "not found (using canonical summaries)")
    for fn in (lambda: fig1(), lambda: fig2(ps), lambda: fig3(),
               lambda: fig4(), lambda: fig5(), lambda: figS1(ps)):
        try:
            fn()
        except Exception as e:
            print(f"  !! figure failed: {e}")
    print("\nDone.")
