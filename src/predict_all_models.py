"""
predict_all_models.py
-------------------------------------------------------------------------------
Generate candidate predictions from ALL FOUR models (RF, NBI, context-free GNN,
context-augmented GNN) for every unobserved (miRNA, drug) pair, then compare
them on RANKS (scores are not comparable across methods).

Faithful to your pipeline: each model is retrained on ALL 470 positives (the
candidate-generation regime, no held-out test); RF and the GNNs are averaged
over 10 seeds with degree-matched negatives; NBI is deterministic.

WRITES (into G):
  all_model_predictions.csv   one row per unknown pair: ids + 4 scores + 4 ranks
  consensus_candidates.csv     top pairs by mean percentile rank across methods

PRINTS:
  - inter-method Spearman correlation matrix (do the methods agree?)
  - per-method degree dependence on the candidate set (hub concentration)
  - top-20 consensus candidates
  - top drug per miRNA under each method (agreement at the per-miRNA level)

Run in your benchmark environment:  python predict_all_models.py
Runtime ~ one arm of your benchmark (20 GNN trainings + 10 RF fits).
-------------------------------------------------------------------------------
"""
import os
import os as _os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import spearmanr

# ----------------------------- config ---------------------------------------
G = _os.environ.get("MIRNA_DATA", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "data", "graph"))
N_SEEDS      = 10        # averaging for RF and GNNs
RATIO        = 10        # negatives : positives for training
CAND_EPOCHS  = 120       # fixed epochs for candidate-mode GNN (no early stop, all positives)
HID, DROP    = 64, 0.5
TOP_N_CONSENSUS = 20

# ----------------------------- load ------------------------------------------
mf  = pd.read_csv(f"{G}/mirna_features.csv").set_index("mimat")
df_ = pd.read_csv(f"{G}/drug_features.csv").set_index("drug_key")
mf_arr, df_arr = mf.values, df_.values
n_mir, n_drug = len(mf), len(df_)
mimat_of = {i: m for i, m in enumerate(mf.index)}
drug_of  = {i: d for i, d in enumerate(df_.index)}

edges = pd.read_csv(f"{G}/pos_edges.csv")
mirna_idx = {m: i for i, m in enumerate(mf.index)}
drug_idx  = {d: i for i, d in enumerate(df_.index)}
edges = edges[edges["mimat"].isin(mirna_idx) & edges["drug_key"].isin(drug_idx)].copy()
edges["m_id"] = edges["mimat"].map(mirna_idx); edges["d_id"] = edges["drug_key"].map(drug_idx)
pos_pairs = edges[["m_id", "d_id"]].drop_duplicates().reset_index(drop=True)
all_pos = set(map(tuple, pos_pairs.values))

mg = pd.read_csv(f"{G}/mirna_gene_edges.csv")
gene_list = sorted(mg["gene"].unique()); gene_idx = {g: i for i, g in enumerate(gene_list)}
mg2 = mg[mg["mimat"].isin(mirna_idx)].copy()
mg2["m"] = mg2["mimat"].map(mirna_idx); mg2["g"] = mg2["gene"].map(gene_idx)
ei_mg_np = np.array([mg2["m"].values, mg2["g"].values])

mir_deg  = pos_pairs["m_id"].value_counts().reindex(range(n_mir), fill_value=0).values.astype(float)
drug_deg = pos_pairs["d_id"].value_counts().reindex(range(n_drug), fill_value=0).values.astype(float)
mir_p  = (mir_deg + 1) / (mir_deg + 1).sum()
drug_p = (drug_deg + 1) / (drug_deg + 1).sum()

# unknown (candidate) pairs
unknown = [(m, d) for m in range(n_mir) for d in range(n_drug) if (m, d) not in all_pos]
um = np.array([m for m, d in unknown]); ud = np.array([d for m, d in unknown])
print(f"{len(pos_pairs)} positives | {len(unknown)} unknown candidate pairs")

def sample_neg(n_needed, rng):
    negs = set()
    while len(negs) < n_needed:
        m = rng.choice(n_mir, size=n_needed * 2, p=mir_p)
        d = rng.choice(n_drug, size=n_needed * 2, p=drug_p)
        for mm, dd in zip(m, d):
            if (mm, dd) not in all_pos and (mm, dd) not in negs:
                negs.add((mm, dd))
                if len(negs) >= n_needed:
                    break
    return np.array(list(negs))

# ----------------------------- RF --------------------------------------------
def score_rf():
    feat = lambda mids, dids: np.hstack([mf_arr[mids], df_arr[dids]])
    Xcand = feat(um, ud)
    acc = np.zeros(len(unknown))
    for s in range(N_SEEDS):
        rng = np.random.default_rng(s)
        neg = sample_neg(len(pos_pairs) * RATIO, rng)
        Xtr = np.vstack([feat(pos_pairs["m_id"].values, pos_pairs["d_id"].values),
                         feat(neg[:, 0], neg[:, 1])])
        ytr = np.concatenate([np.ones(len(pos_pairs)), np.zeros(len(neg))])
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                     random_state=s, n_jobs=-1)
        clf.fit(Xtr, ytr)
        acc += clf.predict_proba(Xcand)[:, 1]
        print(f"  RF seed {s} done")
    return acc / N_SEEDS

# ----------------------------- NBI -------------------------------------------
def score_nbi():
    A = np.zeros((n_mir, n_drug)); A[pos_pairs["m_id"].values, pos_pairs["d_id"].values] = 1
    kd = A.sum(0); km = A.sum(1); kd[kd == 0] = 1; km[km == 0] = 1
    W = (A / km[:, None]).T @ (A / kd[None, :]); F = A @ W.T
    return F[um, ud]

# ----------------------------- GNN -------------------------------------------
def build_data(use_context):
    d = HeteroData()
    d["mirna"].x = torch.tensor(mf_arr, dtype=torch.float)
    d["drug"].x = torch.tensor(df_arr, dtype=torch.float)
    ei = torch.tensor([pos_pairs["m_id"].values, pos_pairs["d_id"].values], dtype=torch.long)
    d["mirna", "assoc", "drug"].edge_index = ei
    d["drug", "rev_assoc", "mirna"].edge_index = ei.flip(0)
    if use_context:
        d["gene"].x = torch.zeros((len(gene_list), 16))
        eg = torch.tensor(ei_mg_np, dtype=torch.long)
        d["mirna", "targets", "gene"].edge_index = eg
        d["gene", "rev_targets", "mirna"].edge_index = eg.flip(0)
    return d

class GNN(nn.Module):
    def __init__(self, use_context):
        super().__init__()
        lins = {"mirna": nn.Linear(64, HID), "drug": nn.Linear(512, HID)}
        if use_context:
            lins["gene"] = nn.Linear(16, HID)
        self.lin = nn.ModuleDict(lins)
        def conv():
            rels = {("mirna", "assoc", "drug"): SAGEConv((HID, HID), HID),
                    ("drug", "rev_assoc", "mirna"): SAGEConv((HID, HID), HID)}
            if use_context:
                rels[("mirna", "targets", "gene")] = SAGEConv((HID, HID), HID)
                rels[("gene", "rev_targets", "mirna")] = SAGEConv((HID, HID), HID)
            return HeteroConv(rels, aggr="sum")
        self.c1 = conv(); self.c2 = conv()
    def enc(self, d, tr):
        x = {k: Fn.relu(self.lin[k](d[k].x)) for k in self.lin}
        x = {k: Fn.dropout(v, p=DROP, training=tr) for k, v in x.items()}
        x = self.c1(x, d.edge_index_dict); x = {k: Fn.relu(v) for k, v in x.items()}
        x = {k: Fn.dropout(v, p=DROP, training=tr) for k, v in x.items()}
        return self.c2(x, d.edge_index_dict)
    def sc(self, x, m, dd):
        return (x["mirna"][m] * x["drug"][dd]).sum(-1)

def score_gnn(use_context):
    data = build_data(use_context)
    mt = torch.tensor(um); dt = torch.tensor(ud)
    acc = np.zeros(len(unknown))
    for s in range(N_SEEDS):
        torch.manual_seed(s)
        model = GNN(use_context)
        opt = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-3)
        rng = np.random.default_rng(1000 + s)
        neg = sample_neg(len(pos_pairs) * RATIO, rng)
        m_tr = torch.tensor(np.concatenate([pos_pairs["m_id"].values, neg[:, 0]]))
        d_tr = torch.tensor(np.concatenate([pos_pairs["d_id"].values, neg[:, 1]]))
        y_tr = torch.tensor(np.concatenate([np.ones(len(pos_pairs)), np.zeros(len(neg))]),
                            dtype=torch.float)
        for _ in range(CAND_EPOCHS):
            model.train(); opt.zero_grad()
            loss = Fn.binary_cross_entropy_with_logits(
                model.sc(model.enc(data, True), m_tr, d_tr), y_tr, pos_weight=torch.tensor(10.0))
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            acc += torch.sigmoid(model.sc(model.enc(data, False), mt, dt)).numpy()
        tag = "full" if use_context else "nocontext"
        print(f"  GNN_{tag} seed {s} done")
    return acc / N_SEEDS

# ----------------------------- run all ---------------------------------------
print("\nScoring all four models on the candidate set ...")
scores = {
    "RF": score_rf(),
    "NBI": score_nbi(),
    "GNN_full": score_gnn(True),
    "GNN_nocontext": score_gnn(False),
}
METHODS = ["RF", "NBI", "GNN_full", "GNN_nocontext"]

df = pd.DataFrame({
    "mimat": [mimat_of[m] for m in um],
    "drug_key": [drug_of[d] for d in ud],
    "m_id": um, "d_id": ud,
    "sum_degree": mir_deg[um] + drug_deg[ud],
})
for k in METHODS:
    df[f"score_{k}"] = scores[k]
    df[f"pct_{k}"] = pd.Series(scores[k]).rank(pct=True).values   # percentile rank
df["consensus_pct"] = df[[f"pct_{k}" for k in METHODS]].mean(axis=1)
df = df.sort_values("consensus_pct", ascending=False).reset_index(drop=True)
df.to_csv(f"{G}/all_model_predictions.csv", index=False)
print(f"\nwrote all_model_predictions.csv ({len(df)} rows)")

# ----------------------------- comparison ------------------------------------
print("\n" + "=" * 64)
print("INTER-METHOD agreement (Spearman rank correlation over candidate pairs)")
print(f"{'':16s}" + "".join(f"{k:>15s}" for k in METHODS))
for a in METHODS:
    row = "".join(f"{spearmanr(scores[a], scores[b]).correlation:>15.3f}" for b in METHODS)
    print(f"{a:16s}{row}")

print("\nPER-METHOD degree dependence on the candidate set (Spearman score vs k(m)+k(d))")
for k in METHODS:
    r = spearmanr(scores[k], df.set_index(["m_id", "d_id"]).loc[
        list(zip(um, ud)), "sum_degree"].values if False else (mir_deg[um] + drug_deg[ud])).correlation
    print(f"  {k:16s} r = {r:+.3f}")

print(f"\nTOP {TOP_N_CONSENSUS} CONSENSUS candidates (mean percentile rank across all 4 methods)")
cols = ["mimat", "drug_key", "consensus_pct"] + [f"pct_{k}" for k in METHODS]
top = df[cols].head(TOP_N_CONSENSUS)
top.to_csv(f"{G}/consensus_candidates.csv", index=False)
with pd.option_context("display.width", 200, "display.max_columns", 20):
    print(top.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

print("\nTOP DRUG PER miRNA under each method (do the methods pick the same drug?)")
print(f"{'miRNA':18s}" + "".join(f"{k:>16s}" for k in METHODS))
for m in range(n_mir):
    mask = um == m
    line = f"{mimat_of[m]:18s}"
    for k in METHODS:
        sub = scores[k][mask]
        best_d = ud[mask][int(np.argmax(sub))]
        line += f"{str(drug_of[best_d]):>16s}"
    print(line)

print("\n" + "=" * 64)
print("Done. Paste the printed comparison back.")
