"""
pipeline_patches.py  -- three drop-in patches to your Phase-5 script (section 8.6).

These are INSERTIONS into your existing script; they reference your existing
names (res, bias, te, preds, n_mir, all_pos, mir_p, run_rf, run_gnn, ...).
Nothing here is standalone-runnable -- paste each block where indicated.

WHY: your current script writes ONLY AUPR per seed (line 640) and never saves
per-pair predictions, so Fig 2 (AUROC panel), Fig 4, Fig 5, and the ablation
cannot be drawn from what you have. These patches produce the missing CSVs.
"""

# ===========================================================================
# PATCH A  --  richer per-seed metrics  (unlocks Fig 2 AUROC panel + bias)
# WHERE: replace your line 640 (the phase5_aupr_per_seed.csv write) with this.
# Your `res[k]` already holds (aupr, auroc) tuples and `bias[k]` holds r,
# so this only changes what gets written to disk -- no recomputation.
# ---------------------------------------------------------------------------
def _write_rich_metrics(res, bias, G):
    import pandas as pd
    rows = []
    for k in res:
        for seed_i, (au, ro) in enumerate(res[k]):
            rows.append({"method": k, "seed": seed_i,
                         "aupr": au, "auroc": ro,
                         "degree_bias_r": bias[k][seed_i]})
    pd.DataFrame(rows).to_csv(f"{G}/phase5_metrics_per_seed.csv", index=False)
    # keep the old file too, for backward compatibility:
    pd.DataFrame({k: [x[0] for x in res[k]] for k in res}).to_csv(
        f"{G}/phase5_aupr_per_seed.csv", index=False)
    print("wrote phase5_metrics_per_seed.csv (long: method, seed, aupr, auroc, degree_bias_r)")
# call after the 30-seed loop:   _write_rich_metrics(res, bias, G)


# ===========================================================================
# PATCH B  --  persist per-pair predictions for ONE representative seed
#              (unlocks Fig 4 degree-bias scatter and Fig 5 PR/ROC curves)
# WHERE: inside your `for s in SEEDS:` loop (around line 617), right AFTER
#        `preds = {...}` and after `te_deg` and `y` are defined.
# ---------------------------------------------------------------------------
SAVE_SEED = 0   # which seed's predictions to dump; must be one of SEEDS

def _maybe_save_predictions(s, te, te_deg, y, preds, G, save_seed=0):
    import pandas as pd
    if s != save_seed:
        return
    out = te[["m_id", "d_id"]].copy()
    out["te_deg"] = te_deg
    out["y_true"] = y
    for k, p in preds.items():
        out[f"score_{k}"] = p
    out.to_csv(f"{G}/predictions_seed{s}.csv", index=False)
    print(f"wrote predictions_seed{s}.csv "
          f"(m_id, d_id, te_deg, y_true, score_RF, score_NBI, score_GNN_full, score_GNN_nocontext)")
# call inside loop:   _maybe_save_predictions(s, te, te_deg, y, preds, G, SAVE_SEED)


# ===========================================================================
# PATCH C  --  the random-vs-degree-matched ablation  (your headline figure)
# This is the experiment you have NOT run: it demonstrates, on YOUR graph,
# that uniform-random negatives inflate AUROC and degree-bias r relative to
# degree-matched negatives. Replace sample_neg and make_split with the
# parametrized versions below, then run run_ablation().
#
# The ONLY thing that changes between the two arms is the negative-sampling
# distribution (p_m, p_d). Positive test selection (leakage-safe multi-edge
# drugs) is held identical, so the comparison is clean.
# ---------------------------------------------------------------------------
# --- replace your sample_neg (lines 494-503) with this parametrized version:
def sample_neg(n_needed, exclude, rng, p_m, p_d):
    # n_mir, n_drug, all_pos are your existing module globals
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


# --- replace your make_split (lines 505-526) with this version that threads p_m, p_d:
def make_split(seed, p_m, p_d):
    import numpy as np, pandas as pd
    rng = np.random.default_rng(seed)
    deg = pos_pairs["d_id"].value_counts()
    pp = pos_pairs.copy(); pp["dd"] = pp["d_id"].map(deg)
    elig = pp[pp["dd"] >= 2].index.to_numpy()
    n_test = min(int(round(0.2 * len(pp))), len(elig))
    test_idx = rng.choice(elig, size=n_test, replace=False)
    tmask = pp.index.isin(test_idx)
    for dlost in set(pp["d_id"]) - set(pp[~tmask]["d_id"]):
        e = pp[(pp["d_id"] == dlost) & tmask].index
        if len(e):
            test_idx = test_idx[test_idx != e[0]]
    tmask = pp.index.isin(test_idx)
    ptr = pp[~tmask][["m_id", "d_id"]].reset_index(drop=True)
    pte = pp[tmask][["m_id", "d_id"]].reset_index(drop=True)
    vperm = rng.permutation(len(ptr)); nval_n = int(0.1 * len(ptr))
    pval = ptr.iloc[vperm[:nval_n]].reset_index(drop=True)
    ptr2 = ptr.iloc[vperm[nval_n:]].reset_index(drop=True)
    ntr = pd.DataFrame(sample_neg(len(ptr2) * RATIO, set(), rng, p_m, p_d), columns=["m_id", "d_id"])
    nval = pd.DataFrame(sample_neg(len(pval) * RATIO, set(map(tuple, ntr.values)), rng, p_m, p_d),
                        columns=["m_id", "d_id"])
    nte = pd.DataFrame(sample_neg(len(pte) * RATIO,
                                  set(map(tuple, ntr.values)) | set(map(tuple, nval.values)),
                                  rng, p_m, p_d), columns=["m_id", "d_id"])
    lab = lambda p, n: pd.concat([p.assign(label=1), n.assign(label=0)], ignore_index=True)
    return lab(ptr2, ntr), lab(pval, nval), lab(pte, nte), ptr2


# --- the ablation driver (add after your model/baseline definitions):
def run_ablation(seeds=range(30)):
    import numpy as np, pandas as pd
    from sklearn.metrics import average_precision_score, roc_auc_score
    from scipy.stats import spearmanr
    UNIF_M = np.ones(n_mir) / n_mir
    UNIF_D = np.ones(n_drug) / n_drug
    regimes = {"degree_matched": (mir_p, drug_p),      # your existing weights
               "uniform_random": (UNIF_M, UNIF_D)}     # the shortcut-exposing arm
    rows = []
    for mode, (pm, pd_) in regimes.items():
        for s in seeds:
            tr, val, te, tp = make_split(s, pm, pd_)
            te_deg = mir_deg[te["m_id"].values] + drug_deg[te["d_id"].values]
            y = te["label"].values
            preds = {"RF": run_rf(pd.concat([tr, val]), te),
                     "NBI": run_nbi(tp, te),
                     "GNN_full": run_gnn(tr, val, tp, te, True, s),
                     "GNN_nocontext": run_gnn(tr, val, tp, te, False, s)}
            for k, p in preds.items():
                rows.append({"neg_mode": mode, "method": k, "seed": s,
                             "aupr": average_precision_score(y, p),
                             "auroc": roc_auc_score(y, p),
                             "degree_bias_r": spearmanr(p, te_deg).correlation})
            print(f"  {mode} seed {s} done")
    pd.DataFrame(rows).to_csv(f"{G}/ablation_metrics.csv", index=False)
    print("wrote ablation_metrics.csv (neg_mode, method, seed, aupr, auroc, degree_bias_r)")
# run:   run_ablation()
#
# EXPECTED (the result that proves your thesis): under 'uniform_random',
# AUROC rises and degree_bias_r rises versus 'degree_matched'. If it does NOT,
# that is a real finding -- your degree-matched protocol matters less than
# claimed on this graph, and the manuscript must say so. Do not pre-judge it.
