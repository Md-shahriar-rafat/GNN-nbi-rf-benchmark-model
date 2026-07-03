"""
Generate SYNTHETIC data in the SmiRN-AD input schema so the full pipeline can be
run end-to-end without the real dataset.

    python tools/make_synthetic_data.py
    # then, e.g.:  python src/run_patches_standalone.py

*** WARNING ***  The data produced here is RANDOM. It reproduces only the SHAPE
of the real problem (25 miRNAs, 257 drugs, 470 hub-dominated positive edges, 412
miRNA-gene edges), NOT its biology. Any metrics computed on it are meaningless and
are provided ONLY to demonstrate that the code executes. Reproducing the numbers
in the paper requires the SmiRN-AD-derived CSVs (see README).

Writes to $MIRNA_DATA if set, else <repo>/data/graph.
"""
import os
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("MIRNA_DATA", os.path.join(REPO, "data", "graph"))
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(0)
N_MIR, N_DRUG, N_POS, N_GENE, N_MG = 25, 257, 470, 324, 412

mimats = [f"MIMAT{i:07d}" for i in range(1, N_MIR + 1)]
drugs = [f"DRUG{i:04d}" for i in range(N_DRUG)]
genes = [f"GENE{i:04d}" for i in range(N_GENE)]

# miRNA node features: 64-d 3-mer-frequency-like vectors (row-normalised)
mf = rng.random((N_MIR, 64))
mf = mf / mf.sum(axis=1, keepdims=True)
mf_df = pd.DataFrame(mf, columns=[f"f{i}" for i in range(64)])
mf_df.insert(0, "mimat", mimats)
mf_df.to_csv(f"{OUT}/mirna_features.csv", index=False)

# drug node features: sparse 512-bit Morgan-fingerprint-like binary vectors
dfp = (rng.random((N_DRUG, 512)) < 0.08).astype(int)
dfp_df = pd.DataFrame(dfp, columns=[f"b{i}" for i in range(512)])
dfp_df.insert(0, "drug_key", drugs)
dfp_df.to_csv(f"{OUT}/drug_features.csv", index=False)

# positive edges with a HUB-DOMINATED drug degree distribution (mimics reality:
# a few drugs carry many edges, most carry one). miRNAs are all fairly high-degree.
drug_w = 1.0 / (1.0 + np.arange(N_DRUG))        # ~power-law: low-index drugs are hubs
drug_w /= drug_w.sum()
pos = set()
while len(pos) < N_POS:
    m = int(rng.integers(0, N_MIR))
    d = int(rng.choice(N_DRUG, p=drug_w))
    pos.add((m, d))
pe = pd.DataFrame([(mimats[m], drugs[d]) for m, d in sorted(pos)],
                  columns=["mimat", "drug_key"])
pe.to_csv(f"{OUT}/pos_edges.csv", index=False)

# miRNA-gene edges (context arm). Only a subset of miRNAs connect onward, echoing
# the real graph's sparse pathway channel.
mg = set()
while len(mg) < N_MG:
    mg.add((int(rng.integers(0, N_MIR)), int(rng.integers(0, N_GENE))))
mge = pd.DataFrame([(mimats[m], genes[g]) for m, g in sorted(mg)],
                   columns=["mimat", "gene"])
mge.to_csv(f"{OUT}/mirna_gene_edges.csv", index=False)

deg = pe["drug_key"].value_counts()
print(f"Wrote SYNTHETIC data to {OUT}")
print(f"  mirna_features.csv   {N_MIR} x 64")
print(f"  drug_features.csv    {N_DRUG} x 512")
print(f"  pos_edges.csv        {N_POS} edges  (max drug degree = {int(deg.max())})")
print(f"  mirna_gene_edges.csv {N_MG} edges")
print("*** SYNTHETIC/RANDOM DATA -- metrics will be meaningless; proves the code runs only. ***")
