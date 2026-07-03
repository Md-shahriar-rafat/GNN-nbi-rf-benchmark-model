"""
ablation_significance.py -- which ablation deltas are real, not noise?
Paired Wilcoxon (per seed) of uniform-random vs degree-matched, per method,
for AUROC, degree-bias r, and AUPR. Run after run_patches_standalone.py.
"""
import pandas as pd
import os as _os
import numpy as np
from scipy.stats import wilcoxon

G = _os.environ.get("MIRNA_DATA", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "data", "graph"))
a = pd.read_csv(f"{G}/ablation_metrics.csv")
METHODS = ["RF", "NBI", "GNN_full", "GNN_nocontext"]

for metric in ["auroc", "degree_bias_r", "aupr"]:
    print(f"\n=== {metric}:  uniform_random  vs  degree_matched   (paired over seeds) ===")
    for k in METHODS:
        d = a[(a.method == k) & (a.neg_mode == "degree_matched")].sort_values("seed")[metric].to_numpy()
        u = a[(a.method == k) & (a.neg_mode == "uniform_random")].sort_values("seed")[metric].to_numpy()
        diff = u - d
        try:
            _, p = wilcoxon(u, d)
        except ValueError:
            p = float("nan")
        flag = "SIG" if (not np.isnan(p) and p < 0.0125) else ("." if (not np.isnan(p) and p < 0.05) else "ns")
        print(f"  {k:16s} mean Δ(unif−deg) = {diff.mean():+.3f}   "
              f"wins {int((diff > 0).sum())}/{len(diff)}   p={p:.4f}   [{flag}]")
print("\nBonferroni note: with 4 methods x 1 planned test per metric, treat p<0.0125 as significant.")
