"""
run_patches_standalone.py
-------------------------------------------------------------------------------
Self-contained equivalent of Patches A + B + C. Nothing to splice into your
Phase-5 script -- just run this. It reconstructs your pipeline exactly (same
data files, sampler, NBI, GNN, eval) and writes the three CSVs make_figures.py
needs:

  phase5_metrics_per_seed.csv   (Patch A)  method, seed, aupr, auroc, degree_bias_r
  predictions_seed0.csv         (Patch B)  m_id, d_id, te_deg, y_true, score_*
  ablation_metrics.csv          (Patch C)  neg_mode, method, seed, aupr, auroc, degree_bias_r

It is EFFICIENT: the degree-matched 30-seed run serves as both your main result
AND the degree-matched arm of the ablation, so the ablation only adds ONE extra
30-seed run (uniform-random), not two.

It also PRINTS your Table 2 (mean +/- 95% CI) and Table 3 (paired Wilcoxon).
Treat that printout as the SINGLE SOURCE OF TRUTH: the figures read the same
CSV this writes, so updating the manuscript tables to these numbers permanently
resolves the 0.3818-vs-0.3931 inconsistency.

INPUT (must already exist in G): mirna_features.csv, drug_features.csv,
pos_edges.csv, mirna_gene_edges.csv  -- the same files your benchmark used.

Run in the SAME environment as your benchmark:  python run_patches_standalone.py
Runtime: roughly 2x your original benchmark (set RUN_ABLATION=False for ~1x).
-------------------------------------------------------------------------------
"""
import os
import os as _os
import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.stats import spearmanr, wilcoxon

# ----------------------------- config ---------------------------------------
# BASE removed: data dir now resolved via MIRNA_DATA env var (see G below)
G = _os.environ.get("MIRNA_DATA", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "data", "graph"))
SEEDS        = range(30)
SAVE_SEED    = 0       # which seed's per-pair predictions to save (Patch B)
RATIO        = 10      # negatives : positives
RUN_ABLATION = True    # set False to skip the uniform-random arm (Patch C)
HID, DROP    = 64, 0.5

# ----------------------------- load static data -----------------------------
for fn in ["mirna_features.csv", "drug_features.csv", "pos_edges.csv", "mirna_gene_edges.csv"]:
    if not os.path.exists(f"{G}/{fn}"):
        raise FileNotFoundError(f"missing input: {G}/{fn} -- check the G path at the top of this script")

mf  = pd.read_csv(f"{G}/mirna_features.csv").set_index("mimat")
df_ = pd.read_csv(f"{G}/drug_features.csv").set_index("drug_key")
mf_arr, df_arr = mf.values, df_.values
n_mir, n_drug = len(mf), len(df_)
mirna_idx = {m: i for i, m in enumerate(mf.index)}
drug_idx  = {d: i for i, d in enumerate(df_.index)}

edges = pd.read_csv(f"{G}/pos_edges.csv")
edges = edges[edges["mimat"].isin(mirna_idx) & edges["drug_key"].isin(drug_idx)].copy()
edges["m_id"] = edges["mimat"].map(mirna_idx)
edges["d_id"] = edges["drug_key"].map(drug_idx)
pos_pairs = edges[["m_id", "d_id"]].drop_duplicates().reset_index(drop=True)
all_pos = set(map(tuple, pos_pairs.values))

mg = pd.read_csv(f"{G}/mirna_gene_edges.csv")
gene_list = sorted(mg["gene"].unique())
gene_idx = {g: i for i, g in enumerate(gene_list)}
mg2 = mg[mg["mimat"].isin(mirna_idx)].copy()
mg2["m"] = mg2["mimat"].map(mirna_idx)
mg2["g"] = mg2["gene"].map(gene_idx)
ei_mg_np = np.array([mg2["m"].values, mg2["g"].values])

mir_deg  = pos_pairs["m_id"].value_counts().reindex(range(n_mir), fill_value=0).values.astype(float)
drug_deg = pos_pairs["d_id"].value_counts().reindex(range(n_drug), fill_value=0).values.astype(float)
mir_p  = (mir_deg + 1) / (mir_deg + 1).sum()      # degree-matched weights
drug_p = (drug_deg + 1) / (drug_deg + 1).sum()

# ----------------------------- negative sampling -----------------------------
def sample_neg(n_needed, exclude, rng, p_m, p_d):
    negs = set()
    while len(negs) < n_needed:
        m = rng.choice(n_mir, size=n_needed * 2, p=p_m)
        d = rng.choice(n_drug, size=n_needed * 2, p=p_d)
        for mm, dd in zip(m, d):
            if (mm, dd) not in all_pos and (mm, dd) not in exclude and (mm, dd) not in negs:
                negs.add((mm, dd))
                if len(negs) >= n_needed:
                    break
    return list(negs)

# ----------------------------- transductive split ----------------------------
def make_split(seed, p_m, p_d):
    rng = np.random.default_rng(seed)
    deg = pos_pairs["d_id"].value_counts()
    pp = pos_pairs.copy(); pp["dd"] = pp["d_id"].map(deg)
    elig = pp[pp["dd"] >= 2].index.to_numpy()                 # leakage-safe: multi-edge drugs only
    n_test = min(int(round(0.2 * len(pp))), len(elig))
    test_idx = rng.choice(elig, size=n_test, replace=False)
    tmask = pp.index.isin(test_idx)
    for dlost in set(pp["d_id"]) - set(pp[~tmask]["d_id"]):    # return an edge if a drug got emptied
        e = pp[(pp["d_id"] == dlost) & tmask].index
        if len(e):
            test_idx = test_idx[test_idx != e[0]]
    tmask = pp.index.isin(test_idx)
    ptr = pp[~tmask][["m_id", "d_id"]].reset_index(drop=True)
    pte = pp[tmask][["m_id", "d_id"]].reset_index(drop=True)
    vperm = rng.permutation(len(ptr)); nv = int(0.1 * len(ptr))    # validation carved from train
    pval = ptr.iloc[vperm[:nv]].reset_index(drop=True)
    ptr2 = ptr.iloc[vperm[nv:]].reset_index(drop=True)
    ntr = pd.DataFrame(sample_neg(len(ptr2) * RATIO, set(), rng, p_m, p_d), columns=["m_id", "d_id"])
    nval = pd.DataFrame(sample_neg(len(pval) * RATIO, set(map(tuple, ntr.values)), rng, p_m, p_d),
                        columns=["m_id", "d_id"])
    nte = pd.DataFrame(sample_neg(len(pte) * RATIO,
                                  set(map(tuple, ntr.values)) | set(map(tuple, nval.values)),
                                  rng, p_m, p_d), columns=["m_id", "d_id"])
    lab = lambda p, n: pd.concat([p.assign(label=1), n.assign(label=0)], ignore_index=True)
    return lab(ptr2, ntr), lab(pval, nval), lab(pte, nte), ptr2

# ----------------------------- baselines -------------------------------------
def run_rf(tr, te):
    X = lambda d: np.hstack([mf_arr[d["m_id"].values], df_arr[d["d_id"].values]])
    clf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=0, n_jobs=-1)
    clf.fit(X(tr), tr["label"].values)
    return clf.predict_proba(X(te))[:, 1]

def run_nbi(train_pos, te):
    A = np.zeros((n_mir, n_drug)); A[train_pos["m_id"].values, train_pos["d_id"].values] = 1
    kd = A.sum(0); km = A.sum(1); kd[kd == 0] = 1; km[km == 0] = 1
    W = (A / km[:, None]).T @ (A / kd[None, :]); F = A @ W.T
    return F[te["m_id"].values, te["d_id"].values]

# ----------------------------- heterogeneous GNN -----------------------------
def build_data(train_pos, use_context):
    d = HeteroData()
    d["mirna"].x = torch.tensor(mf_arr, dtype=torch.float)
    d["drug"].x  = torch.tensor(df_arr, dtype=torch.float)
    ei = torch.tensor([train_pos["m_id"].values, train_pos["d_id"].values], dtype=torch.long)
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

def run_gnn(tr, val, train_pos, te, use_context, seed):
    torch.manual_seed(seed)
    data = build_data(train_pos, use_context); model = GNN(use_context)
    opt = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-3)
    mtr = torch.tensor(tr["m_id"].values); dtr = torch.tensor(tr["d_id"].values)
    ytr = torch.tensor(tr["label"].values, dtype=torch.float)
    mv = torch.tensor(val["m_id"].values); dv = torch.tensor(val["d_id"].values); yv = val["label"].values
    mte = torch.tensor(te["m_id"].values); dte = torch.tensor(te["d_id"].values)
    best, state, wait, pat = 0, None, 0, 30
    for _ in range(300):
        model.train(); opt.zero_grad()
        loss = Fn.binary_cross_entropy_with_logits(
            model.sc(model.enc(data, True), mtr, dtr), ytr, pos_weight=torch.tensor(10.0))
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model.sc(model.enc(data, False), mv, dv)).numpy()
        va = average_precision_score(yv, pv)                  # early stop on VALIDATION AUPR
        if va > best:
            best, state, wait = va, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= pat:
                break
    model.load_state_dict(state); model.eval()
    with torch.no_grad():
        return torch.sigmoid(model.sc(model.enc(data, False), mte, dte)).numpy()

# ----------------------------- 30-seed driver --------------------------------
def run_benchmark(seeds, p_m, p_d, save_pred_seed=None, label=""):
    records = []
    for s in seeds:
        tr, val, te, tp = make_split(s, p_m, p_d)
        te_deg = mir_deg[te["m_id"].values] + drug_deg[te["d_id"].values]
        y = te["label"].values
        preds = {"RF": run_rf(pd.concat([tr, val]), te),
                 "NBI": run_nbi(tp, te),
                 "GNN_full": run_gnn(tr, val, tp, te, True, s),
                 "GNN_nocontext": run_gnn(tr, val, tp, te, False, s)}
        for k, p in preds.items():
            records.append({"method": k, "seed": int(s),
                            "aupr": average_precision_score(y, p),
                            "auroc": roc_auc_score(y, p),
                            "degree_bias_r": spearmanr(p, te_deg).correlation})
        if save_pred_seed is not None and s == save_pred_seed:        # Patch B
            out = te[["m_id", "d_id"]].copy()
            out["te_deg"] = te_deg; out["y_true"] = y
            for k, p in preds.items():
                out[f"score_{k}"] = p
            out.to_csv(f"{G}/predictions_seed{s}.csv", index=False)
            print(f"  [B] wrote predictions_seed{s}.csv")
        print(f"  {label} seed {s} done")
    return pd.DataFrame(records)

def _ci(a):
    a = np.asarray(a, float)
    return a.mean(), 1.96 * a.std(ddof=1) / np.sqrt(len(a))

def summarize(dm):
    order = ["RF", "NBI", "GNN_full", "GNN_nocontext"]
    print(f"\n{'Method':16s} {'AUPR':>18s} {'AUROC':>18s} {'DegBias r':>12s}")
    for k in order:
        sub = dm[dm.method == k]
        ma, ha = _ci(sub.aupr); mr, hr = _ci(sub.auroc); mb, _ = _ci(sub.degree_bias_r)
        print(f"{k:16s} {ma:.3f} ± {ha:.3f}     {mr:.3f} ± {hr:.3f}    {mb:+.3f}")
    piv = dm.pivot(index="seed", columns="method", values="aupr")
    def paired(a, b):
        d = piv[a] - piv[b]
        try:
            _, p = wilcoxon(piv[a], piv[b])
        except ValueError:
            p = float("nan")
        sig = "**sig**" if p < 0.0125 else ("marginal" if p < 0.05 else "ns")
        print(f"{a} vs {b}: dAUPR {d.mean():+.4f} | wins {int((d > 0).sum())}/{len(d)} | p={p:.4f} {sig}")
    print("\n--- paired Wilcoxon (Bonferroni p<0.0125) -- USE THESE for Table 3 ---")
    paired("GNN_nocontext", "NBI"); paired("GNN_nocontext", "GNN_full")
    paired("GNN_nocontext", "RF");  paired("GNN_full", "NBI")

# ----------------------------- main ------------------------------------------
if __name__ == "__main__":
    print(f"env: scipy {scipy.__version__} | torch {torch.__version__}")
    print(f"graph: {n_mir} miRNAs, {n_drug} drugs, {len(pos_pairs)} positive edges")

    print("\n[main] degree-matched benchmark (30 seeds) = Patch A + B + degree arm of C ...")
    dm = run_benchmark(SEEDS, mir_p, drug_p, save_pred_seed=SAVE_SEED, label="degree")
    dm.to_csv(f"{G}/phase5_metrics_per_seed.csv", index=False)
    print("[A] wrote phase5_metrics_per_seed.csv")
    summarize(dm)

    if RUN_ABLATION:
        print("\n[ablation] uniform-random benchmark (30 seeds) = uniform arm of C ...")
        UNIF_M = np.ones(n_mir) / n_mir
        UNIF_D = np.ones(n_drug) / n_drug
        un = run_benchmark(SEEDS, UNIF_M, UNIF_D, label="uniform")
        abl = pd.concat([dm.assign(neg_mode="degree_matched"),
                         un.assign(neg_mode="uniform_random")], ignore_index=True)
        abl = abl[["neg_mode", "method", "seed", "aupr", "auroc", "degree_bias_r"]]
        abl.to_csv(f"{G}/ablation_metrics.csv", index=False)
        print("[C] wrote ablation_metrics.csv")
        # quick ablation readout: does random inflate AUROC and degree-bias?
        print("\n--- ablation summary (mean over 30 seeds) ---")
        for k in ["RF", "NBI", "GNN_full", "GNN_nocontext"]:
            d_au = abl[(abl.neg_mode == "degree_matched") & (abl.method == k)].auroc.mean()
            u_au = abl[(abl.neg_mode == "uniform_random") & (abl.method == k)].auroc.mean()
            d_b  = abl[(abl.neg_mode == "degree_matched") & (abl.method == k)].degree_bias_r.mean()
            u_b  = abl[(abl.neg_mode == "uniform_random") & (abl.method == k)].degree_bias_r.mean()
            print(f"{k:16s} AUROC deg={d_au:.3f} unif={u_au:.3f} (Δ{u_au-d_au:+.3f}) | "
                  f"bias r deg={d_b:+.3f} unif={u_b:+.3f} (Δ{u_b-d_b:+.3f})")

    print("\n=== done. now run: python make_figures.py ===")
