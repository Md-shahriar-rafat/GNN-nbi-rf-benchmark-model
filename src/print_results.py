"""
print_results.py -- prints every TEXT result needed for the manuscript.
No figures, no model training. Reads CSVs you already have. Runs in seconds.

Reads:  phase5_metrics_per_seed.csv  and  ablation_metrics.csv   (both in G)
Prints: (1) Table 2  mean +/- 95% CI for AUPR, AUROC, degree-bias r
        (2) Table 3  paired Wilcoxon comparisons (the negative-result tests)
        (3) Ablation paired Wilcoxon: uniform_random vs degree_matched

Run:  python print_results.py     then paste the whole printout back.
"""
import pandas as pd
import os as _os
import numpy as np
from scipy.stats import wilcoxon

G = _os.environ.get("MIRNA_DATA", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "data", "graph"))
METHODS = ["RF", "NBI", "GNN_full", "GNN_nocontext"]


def ci(a):
    a = np.asarray(a, float)
    return a.mean(), 1.96 * a.std(ddof=1) / np.sqrt(len(a))


def sig_flag(p):
    if np.isnan(p):
        return "NA"
    return "SIG" if p < 0.0125 else ("." if p < 0.05 else "ns")


# ---------- (1) Table 2 + (2) Table 3 ----------
m = pd.read_csv(f"{G}/phase5_metrics_per_seed.csv")
print("=" * 70)
print("TABLE 2   mean +/- 95% CI over 30 seeds")
print(f"{'Method':16s}{'AUPR':>17s}{'AUROC':>17s}{'DegBias r':>13s}")
for k in METHODS:
    s = m[m.method == k]
    a, ah = ci(s.aupr); r, rh = ci(s.auroc); b, _ = ci(s.degree_bias_r)
    print(f"{k:16s}{a:.3f} ± {ah:.3f}   {r:.3f} ± {rh:.3f}   {b:+.3f}")

piv = m.pivot(index="seed", columns="method", values="aupr")
print("\nTABLE 3   paired Wilcoxon on AUPR (Bonferroni threshold p<0.0125)")
def paired(a, b):
    d = piv[a] - piv[b]
    try:
        _, p = wilcoxon(piv[a], piv[b])
    except ValueError:
        p = float("nan")
    print(f"{a:14s} vs {b:10s}  dAUPR {d.mean():+.4f}  wins {int((d > 0).sum())}/{len(d)}  p={p:.4f}  [{sig_flag(p)}]")
paired("GNN_nocontext", "NBI"); paired("GNN_nocontext", "GNN_full")
paired("GNN_nocontext", "RF");  paired("GNN_full", "NBI")

# ---------- (3) Ablation significance ----------
print("\n" + "=" * 70)
print("ABLATION   paired Wilcoxon: uniform_random vs degree_matched (per method)")
a = pd.read_csv(f"{G}/ablation_metrics.csv")
for metric in ["auroc", "degree_bias_r", "aupr"]:
    print(f"\n-- {metric} --")
    for k in METHODS:
        d = a[(a.method == k) & (a.neg_mode == "degree_matched")].sort_values("seed")[metric].to_numpy()
        u = a[(a.method == k) & (a.neg_mode == "uniform_random")].sort_values("seed")[metric].to_numpy()
        diff = u - d
        try:
            _, p = wilcoxon(u, d)
        except ValueError:
            p = float("nan")
        print(f"{k:16s} mean Δ(unif−deg) = {diff.mean():+.3f}   "
              f"wins {int((diff > 0).sum())}/{len(diff)}   p={p:.4f}   [{sig_flag(p)}]")

print("\n" + "=" * 70)
print("Done. Copy everything above and paste it back.")
