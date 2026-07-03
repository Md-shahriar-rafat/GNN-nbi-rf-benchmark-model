# A controlled benchmark of graph neural networks for small-molecule–miRNA association prediction in Alzheimer's disease

Evaluation and analysis code for a **controlled, leakage-audited benchmark** of four method classes — a feature-only Random Forest, a topology-only Network-Based Inference (NBI) model, and heterogeneous GraphSAGE GNNs with and without disease-pathway context — for predicting Alzheimer's-related small-molecule–miRNA associations from the SmiRN-AD network.

The contribution is **evaluation integrity**, not a new architecture. The protocol neutralises the two failure modes that most often inflate reported performance on sparse biomedical graphs: degree (hub) bias from uniform-random negative sampling, and topological leakage from careless edge partitioning. The headline result is an honest negative one — adding disease-pathway context does not improve prediction — together with a cross-method consensus analysis that separates genuine signal from degree/promiscuity artefacts.

> **Provenance:** the positive associations derive from SmiRN-AD, a 2014 Connectivity-Map-based **computational** prediction set, not assayed binding events. Every model learns to reproduce and extend a prior computational prediction; results are bounded by that substrate.

---

## Quickstart for reviewers

```bash
git clone <repo-url> && cd gnn-mirna-ad-benchmark
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**(1) Reproduce the manuscript figures immediately — no data required.**
`make_all_figures.py` falls back to the reproduced canonical values when no CSVs are present, so it renders all six figures out of the box:

```bash
python figures/make_all_figures.py          # writes PNG+PDF into ./figures
```

**(2) Run the full pipeline end-to-end on synthetic data — no private data required.**
This proves the code executes. The generated data is **random** (correct shape, no biology), so the numbers are meaningless — it is a smoke test only:

```bash
python tools/make_synthetic_data.py          # writes toy CSVs into ./data/graph
python src/run_patches_standalone.py         # trains all models, writes the 3 result CSVs
python src/print_results.py                  # Table 2 + Table 3
python src/predict_all_models.py             # candidate generation
python src/consensus_views.py                # consensus / artefact analysis
python src/ablation_significance.py          # ablation Wilcoxon
python figures/make_all_figures.py           # figures now read the real CSVs
```

**(3) Reproduce the paper's numbers — requires the real data.**
Place the four SmiRN-AD-derived input files (below) in `./data/graph/` (or point `MIRNA_DATA` at their location), then run the same commands as (2). The dataset is **not redistributed here** (see *Data*).

---

## Paths are portable (no hard-coded directories)

Every script resolves its data directory in this order:
1. the `MIRNA_DATA` environment variable, if set;
2. otherwise `<repo>/data/graph` (relative to the script, so cwd does not matter).

```bash
# use data that lives elsewhere:
MIRNA_DATA=/path/to/graph python src/run_patches_standalone.py
# figures are written to <repo>/figures (override with MIRNA_FIGS)
```

There are **no absolute paths** anywhere in the code; nothing needs editing to run.

---

## Repository structure

```
gnn-mirna-ad-benchmark/
├── README.md · requirements.txt · .gitignore
├── data/graph/              # place the 4 input CSVs here (gitignored; .gitkeep only)
├── tools/
│   └── make_synthetic_data.py   # random data in the real schema (smoke test)
├── src/
│   ├── run_patches_standalone.py  # * core: sampler, NBI, GNNs, eval -> 3 result CSVs
│   ├── pipeline_patches.py        # the same additions as drop-in patches (reference)
│   ├── predict_all_models.py      # candidate generation across all 4 models
│   ├── consensus_views.py         # inter-method agreement + consensus views
│   ├── within_mirna_consensus.py  # within-miRNA ranking + HDAC-class enrichment
│   ├── ablation_significance.py   # paired Wilcoxon: uniform vs degree-matched
│   └── print_results.py           # Table 2 (mean +/- 95% CI) + Table 3 (Wilcoxon)
├── figures/
│   ├── make_all_figures.py        # * all 6 figures; canonical fallback if no CSVs
│   └── make_figures.py            # CSV-driven variant (skips, never fabricates)
└── docs/
    └── build_discussion.js        # generates the Discussion .docx (Node + docx)
```

### Required input files (schema)
| File | Key columns |
|------|-------------|
| `mirna_features.csv` | `mimat`, then 64 feature columns (3-mer frequency) |
| `drug_features.csv`  | `drug_key`, then 512 feature columns (Morgan fingerprint) |
| `pos_edges.csv`      | `mimat`, `drug_key` |
| `mirna_gene_edges.csv` | `mimat`, `gene` |

`run_patches_standalone.py` writes `phase5_metrics_per_seed.csv`, `predictions_seed0.csv`, and `ablation_metrics.csv` back into the same directory; the analysis and figure scripts read those.

---

## Canonical results (30 seeds, degree-matched 1:10, AUPR-primary)

| Method | AUPR (mean +/- 95% CI) | AUROC | degree-bias (degree-matched) |
|---|---|---|---|
| Random Forest (features) | 0.396 +/- 0.013 | 0.708 | 0.275 |
| NBI (topology)           | 0.424 +/- 0.015 | **0.737** | 0.206 |
| GNN +context             | 0.431 +/- 0.036 | 0.722 | 0.245 |
| **GNN no-context**       | **0.469 +/- 0.023** | 0.730 | **0.181** |

Random-classifier AUPR floor = 0.091. Key findings: the metric winner flips (NBI tops AUROC, context-free GNN tops AUPR); adding context does not help (dAUPR +0.038, p = 0.39; two GNNs correlate at Spearman rho = 0.96); **degree-matched sampling gives a lower, de-confounded estimate of degree reliance than uniform sampling** (which inflates it by confounding degree with the class label), while AUROC barely moves (<= 0.035); and the only candidate signal surviving every control is the class I HDAC-inhibitor axis (entinostat-miR-148b).

---

## Notes

- **Reproducibility:** the scripts are those used to produce the reported numbers; only the data location changes, handled by `MIRNA_DATA`. Avoid editing the analysis logic if you want to reproduce the canonical values.
- **GNN stack:** `run_patches_standalone.py` and `predict_all_models.py` require `torch` + `torch-geometric`; install `torch` first, then `torch-geometric` matching your torch/CUDA build. The other scripts need only the scientific-Python stack.
- **`pipeline_patches.py`** is not standalone — it documents the three additions as insertions into an original Phase-5 script. `run_patches_standalone.py` is the self-contained equivalent.
- The manuscript text is maintained separately from this code repository.
