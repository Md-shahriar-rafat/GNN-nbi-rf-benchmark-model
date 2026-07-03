"""
consensus_views.py -- turn the all-four predictions into views that escape the
hub trap. Reads all_model_predictions.csv (already saved); no retraining.

Prints:
  (1) inter-method Spearman matrix  -- expect RF to correlate weakly with the
      three graph methods, which should correlate strongly with each other
  (2) per-method degree dependence  -- does RF escape the hubs?
  (3) CROSS-PARADIGM consensus      -- pairs ranked high by BOTH the feature
      method (RF) and the graph methods; agreement despite different biases
  (4) DEGREE-CONTROLLED consensus   -- pairs all methods rank higher than their
      node degree predicts (the candidates that are NOT just hubs)
"""
import pandas as pd
import os as _os
import numpy as np
from scipy.stats import spearmanr

G = _os.environ.get("MIRNA_DATA", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "data", "graph"))
d = pd.read_csv(f"{G}/all_model_predictions.csv")
METHODS = ["RF", "NBI", "GNN_full", "GNN_nocontext"]

# (1) inter-method agreement
print("=" * 70)
print("(1) INTER-METHOD Spearman (raw scores over candidate pairs)")
print(f"{'':16s}" + "".join(f"{m:>15s}" for m in METHODS))
for a in METHODS:
    print(f"{a:16s}" + "".join(
        f"{spearmanr(d['score_'+a], d['score_'+b]).correlation:>15.3f}" for b in METHODS))

# (2) degree dependence per method
print("\n(2) DEGREE dependence (Spearman score vs k(m)+k(d))")
for a in METHODS:
    print(f"  {a:16s} r = {spearmanr(d['score_'+a], d['sum_degree']).correlation:+.3f}")

# (3) cross-paradigm consensus: high only if BOTH feature and graph rank it high
d["graph_pct"] = d[["pct_NBI", "pct_GNN_full", "pct_GNN_nocontext"]].mean(axis=1)
d["feat_pct"] = d["pct_RF"]
d["cross_paradigm"] = d[["graph_pct", "feat_pct"]].min(axis=1)
xp = d.sort_values("cross_paradigm", ascending=False).head(20)
print("\n(3) TOP 20 CROSS-PARADIGM consensus (high in BOTH features and graph)")
print(xp[["mimat", "drug_key", "sum_degree", "feat_pct", "graph_pct", "cross_paradigm"]]
      .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# (4) degree-controlled consensus: residual of consensus after regressing out degree
x = d["sum_degree"].to_numpy(dtype=float)
y = d["consensus_pct"].to_numpy(dtype=float)
b1, b0 = np.polyfit(x, y, 1)
d["consensus_resid"] = y - (b0 + b1 * x)
rc = d.sort_values("consensus_resid", ascending=False).head(20)
print(f"\n(4) TOP 20 DEGREE-CONTROLLED consensus  (consensus ~ {b0:.3f} + {b1:+.4f}*degree; "
      f"ranked by residual = scores higher than degree predicts)")
print(rc[["mimat", "drug_key", "sum_degree", "consensus_pct", "consensus_resid"]]
      .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# how much of the top-100 raw consensus is just the hub drugs?
top100 = d.sort_values("consensus_pct", ascending=False).head(100)
hub_share = top100["drug_key"].value_counts().head(8)
print("\nDrug composition of the top-100 raw consensus (hub check):")
print(hub_share.to_string())
print("\n" + "=" * 70)
print("Done. Paste (1), (2), (3), (4) back.")
