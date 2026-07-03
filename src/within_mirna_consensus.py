"""
within_mirna_consensus.py -- remove the promiscuity artifacts and test the HDAC thread.
Reads all_model_predictions.csv. Ranks drugs WITHIN each miRNA (removes miRNA-level
offset, so a promiscuous miRNA can't dominate), averages those within-miRNA ranks
across the four methods, then tests whether HDAC inhibitors are systematically
enriched versus cardiac glycosides and everything else.
"""
import pandas as pd
import os as _os
import numpy as np

G = _os.environ.get("MIRNA_DATA", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "data", "graph"))
d = pd.read_csv(f"{G}/all_model_predictions.csv")
METHODS = ["RF", "NBI", "GNN_full", "GNN_nocontext"]

# within-miRNA percentile rank per method, then average across methods
for m in METHODS:
    d[f"w_{m}"] = d.groupby("mimat")[f"score_{m}"].rank(pct=True)
d["within_consensus"] = d[[f"w_{m}" for m in METHODS]].mean(axis=1)

HDAC = {"ms-275", "entinostat", "vorinostat", "trichostatin a", "panobinostat",
        "romidepsin", "belinostat", "mocetinostat", "scriptaid", "apicidin",
        "valproic acid", "sodium butyrate", "depudecin"}
CG = {"digoxin", "digoxigenin", "lanatoside c", "ouabain", "helveticoside",
      "digitoxigenin", "digitoxin", "proscillaridin", "strophanthidin",
      "cymarin", "convallatoxin", "peruvoside"}

def cls(x):
    x = str(x).lower()
    if x in HDAC: return "HDAC_inhibitor"
    if x in CG:   return "cardiac_glycoside"
    return "other"
d["class"] = d["drug_key"].apply(cls)

print("=" * 64)
print("TOP DRUG PER miRNA by WITHIN-miRNA consensus (promiscuity removed)")
top = d.loc[d.groupby("mimat")["within_consensus"].idxmax()].sort_values(
    "within_consensus", ascending=False)
print(top[["mimat", "drug_key", "class", "within_consensus"]].to_string(
    index=False, float_format=lambda x: f"{x:.3f}"))
print("\nClass composition of those 25 top-per-miRNA picks:")
print(top["class"].value_counts().to_string())

print("\n" + "=" * 64)
print("CLASS-LEVEL enrichment: mean within-miRNA consensus percentile by class")
print("(if HDAC mean >> other, the HDAC thread is systematic, not coincidence)")
print(d.groupby("class")["within_consensus"].agg(["mean", "median", "count"]).to_string())

print("\nHDAC inhibitors present — top 3 miRNAs each (by within-miRNA consensus):")
hd = d[d["class"] == "HDAC_inhibitor"]
for drug in sorted(hd["drug_key"].unique()):
    sub = hd[hd.drug_key == drug].sort_values("within_consensus", ascending=False).head(3)
    print(f"  {drug:16s}: " + ", ".join(
        f"{r.mimat}({r.within_consensus:.2f})" for r in sub.itertuples()))

print("\n" + "=" * 64)
print("Done. Paste it all back.")
