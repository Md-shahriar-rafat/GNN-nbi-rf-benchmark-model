const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, BorderStyle, ShadingType
} = require("docx");

const FONT = "Calibri";
const border = { style: BorderStyle.SINGLE, size: 1, color: "BBBBBB" };
const borders = { top: border, bottom: border, left: border, right: border };
const cellMargins = { top: 80, bottom: 80, left: 120, right: 120 };

// helper: body paragraph from an array of runs (strings or {text,bold,italics})
function para(runs, opts = {}) {
  const children = (Array.isArray(runs) ? runs : [runs]).map(r =>
    typeof r === "string"
      ? new TextRun({ text: r, font: FONT, size: 22 })
      : new TextRun({ text: r.text, bold: !!r.bold, italics: !!r.italics, font: FONT, size: 22 })
  );
  return new Paragraph({
    children,
    spacing: { after: 160, line: 276 },
    alignment: AlignmentType.JUSTIFIED,
    ...opts
  });
}
function h1(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 200, after: 120 },
    children: [new TextRun({ text, bold: true, font: FONT, size: 30, color: "1F3864" })] });
}
function h2(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 200, after: 100 },
    children: [new TextRun({ text, bold: true, font: FONT, size: 25, color: "2E5496" })] });
}
function tcell(text, { head = false, w, bold = false, color } = {}) {
  return new TableCell({
    borders, width: { size: w, type: WidthType.DXA }, margins: cellMargins,
    shading: head ? { fill: "1F3864", type: ShadingType.CLEAR } : (color ? { fill: color, type: ShadingType.CLEAR } : undefined),
    children: [new Paragraph({ spacing: { after: 0 }, children: [
      new TextRun({ text, bold: head || bold, color: head ? "FFFFFF" : "000000", font: FONT, size: 18 })
    ] })]
  });
}

const COLS = [1700, 2700, 2300, 2660];
function trow(cells, head = false) {
  return new TableRow({ children: cells.map((c, i) =>
    tcell(c.text, { head, w: COLS[i], bold: c.bold, color: c.color })) });
}

const triageTable = new Table({
  width: { size: 9360, type: WidthType.DXA },
  columnWidths: COLS,
  rows: [
    trow([{ text: "Predicted drug(s)" }, { text: "Predicted miRNA partners" },
          { text: "Classification" }, { text: "Status" }], true),
    trow([{ text: "thioridazine" }, { text: "miR-134, miR-617, miR-15a, miR-572" },
          { text: "high-degree hub (deg 21)" }, { text: "degree artefact \u2014 not validated", color: "FCE4E4" }]),
    trow([{ text: "quinostatin" }, { text: "miR-29b/c, miR-382, miR-765, miR-575, miR-320" },
          { text: "high-degree hub (deg 13)" }, { text: "degree artefact \u2014 not validated", color: "FCE4E4" }]),
    trow([{ text: "perhexiline" }, { text: "miR-185, miR-30e-5p, miR-601" },
          { text: "high-degree hub (deg 12)" }, { text: "degree artefact \u2014 not validated", color: "FCE4E4" }]),
    trow([{ text: "melatonin" }, { text: "miR-188, miR-368, miR-598" },
          { text: "drug-side promiscuity (~12/25 miRNAs)" }, { text: "promiscuity artefact \u2014 not validated", color: "FCE4E4" }]),
    trow([{ text: "entinostat (MS-275),\nvorinostat, trichostatin A" }, { text: "miR-148b, miR-20b, miR-376a, miR-95" },
          { text: "class I HDAC; survives degree control" }, { text: "candidate hypothesis \u2014 testable", color: "E2F0E2" }]),
  ]
});

const children = [
  new Paragraph({ alignment: AlignmentType.LEFT, spacing: { after: 200 },
    children: [new TextRun({ text: "Discussion", bold: true, font: FONT, size: 36, color: "1F3864" })] }),

  // 4.1
  h2("Methodological validation of the controlled benchmark"),
  para("The evaluation of machine-learning models on sparse, heterogeneous biological graphs is frequently undermined by subtle, unrecognised evaluation biases that inflate apparent performance and mask a model\u2019s inability to generalise to unobserved interactions. This study therefore implemented a controlled, leakage-audited protocol addressing the two dominant failure modes of bipartite link prediction: degree (hub) bias introduced by uniform negative sampling, and topological data leakage introduced by careless edge partitioning."),
  para("The clearest evidence that metric choice is not a formality is the dissociation between the two ranking metrics. The topology-only Network-Based Inference (NBI) model achieved the highest AUROC (0.737 \u00B1 0.010), yet the context-free heterogeneous GraphSAGE (GNN no-context) model achieved the highest AUPR (0.469 \u00B1 0.023) and significantly outperformed every other method on it. The mechanism is class imbalance: in a bipartite space where unobserved pairs exceed ninety percent of all possible edges, the abundant true negatives suppress the false-positive rate (FPR = FP / (FP + TN)), so AUROC can remain high even when a model returns an unacceptable number of false positives. Precision (TP / (TP + FP)) is independent of the true-negative count and reflects top-of-list retrieval directly; AUPR was therefore adopted as the primary metric. A reader consulting AUROC alone would have selected NBI, whereas the metric that matches the prioritisation task selects the GNN."),
  para("A second and distinct mechanism concerns the negatives rather than the metric. Because the network is hub-dominated, uniform-random negatives fall predominantly on low-degree nodes, making the task artificially separable and allowing a model to rank pairs by node degree rather than genuine association. Critically, this popularity reliance does not inflate AUROC. Our negative-sampling ablation showed that switching from uniform to degree-matched negatives left raw AUROC essentially unchanged (absolute change \u2264 0.035) while decisively raising the measured degree-bias for every model (Random Forest +0.163, GNN no-context +0.115, GNN full +0.096, NBI +0.055; all p < 0.0001). Uniform sampling compresses the degree range over which the diagnostic is computed, hiding the reliance rather than removing it. Degree-matched negative sampling \u2014 endpoints drawn under a Laplace (+1) smoothed, degree-proportional distribution that forces the negative degree profile to match the positive one \u2014 is therefore essential not because it lowers headline scores (it does not), but because it is the only regime in which degree reliance becomes visible and controllable."),
  para("The protocol additionally used a leakage-safe transductive split: positive edges were partitioned 80/20, with a further ten percent of training positives reserved as a validation set for early stopping to prevent overfitting on this small graph. To prevent identity leakage while avoiding an unrealistic cold-start setting, positive test edges were held out only from drugs with at least two known edges, ensuring every drug retained at least one training edge and was represented during message passing."),

  // 4.2
  h2("The disease-pathway bottleneck: multi-omics destabilisation in sparse topologies"),
  para("A central finding is that incorporating multi-omics disease-pathway context \u2014 412 human miRNA\u2013gene interactions from miRTarBase v10 and 25 gene\u2013pathway associations from the KEGG Alzheimer\u2019s disease pathway (hsa05010) \u2014 failed to improve precision. The context-augmented model was statistically indistinguishable from the context-free GNN (\u0394AUPR = +0.038, 16/30 wins, p = 0.3931) and collapsed to the level of plain NBI (p = 0.42); the redundancy is visible in the predictions themselves, the two GNNs\u2019 candidate rankings correlating at Spearman \u03C1 = 0.96. The explanation is a topological bottleneck: although 412 miRNA\u2013gene edges exist, only 32 reach a gene that connects onward to the Alzheimer\u2019s pathway, and only 11 of the 25 miRNAs reach the pathway node at all. Passing messages through this sparse scaffold injects structural noise, destabilising training and widening the context model\u2019s confidence interval (0.431 \u00B1 0.036) relative to its context-free counterpart (0.469 \u00B1 0.023). Beyond sparseness, the scope is narrow: KEGG hsa05010 is amyloid- and tau-centric and excludes the epigenetic regulators that the surviving candidate signal subsequently implicates. The negative result is thus precise \u2014 it is the sparse, amyloid/tau-centric context that fails \u2014 and should not be generalised to every conceivable functional annotation."),

  // 4.3
  h2("Cross-method consensus separates signal from artefact"),
  para("Because the unobserved pairs have no experimental ground truth, candidate robustness was assessed across method paradigms, and the result is primarily cautionary. Naive consensus across all four methods is dominated by high-degree drugs (thioridazine, quinostatin, perhexiline, digoxin) on high-degree miRNAs, and requiring agreement between the feature method and the graph methods does not escape this, because a hub pair carries both good chemistry and good topology and is favoured by every paradigm. That four methodologically distinct approaches converge on the same hubs demonstrates the degree-concentration limitation is paradigm-independent, not a model quirk \u2014 and it is a direct caution against treating cross-method agreement as biological reality. Degree-controlled consensus proved confounded by the bipartite asymmetry of the graph, surfacing two promiscuous miRNAs that score highly with chemically unrelated drugs; melatonin and metrizamide display the analogous drug-side promiscuity, ranking as top partners for roughly half of all 25 miRNAs."),
  para("The discriminating power of this control is seen most clearly by contrasting two predictions of opposite character. Thioridazine, the highest-degree drug in the network (21 known associations), is returned as the top-ranked partner for four distinct miRNAs \u2014 miR-134, miR-617, miR-15a and miR-572 \u2014 and this breadth across unrelated miRNAs is itself the defining signature of a degree-driven artefact; the pairs do not survive degree control. Their standing is not rescued by thioridazine\u2019s documented history in dementia, both because that history is unfavourable \u2014 systematic reviews associate the drug with worsened behavioural scores, QTc prolongation, and anticholinergic cognitive impairment \u2014 and, more fundamentally, because disease relevance is a property of the drug rather than evidence for any specific miRNA pairing. Entinostat (MS-275) presents the opposite profile: it is nominated for miR-148b at moderate, non-hub degree (nine known associations), independently by all four methods, and the pair survives degree control. Here disease relevance and degree-independence coincide, and it is on this basis \u2014 not on the strength of a constructible mechanistic narrative, which can be assembled for artefacts and genuine signals alike \u2014 that entinostat\u2013miR-148b is carried forward as a testable hypothesis (Section 4.4), while the hub predictions are not. The remaining hub and promiscuity candidates summarised in Table 6 behave identically to thioridazine. This contrast establishes that the protocol separates specific association from node popularity; it does not establish that the surviving prediction is biologically validated."),
  para([{ text: "Table 6 triages the leading candidate predictions by their status under degree and promiscuity control.", italics: true }]),
  triageTable,
  new Paragraph({ spacing: { after: 120 }, children: [
    new TextRun({ text: "Table 6. Triage of leading candidate predictions. Degree is the number of known SmiRN-AD edges. Only the HDAC-inhibitor class survives degree and promiscuity control and is carried forward as a hypothesis.", italics: true, font: FONT, size: 18, color: "555555" }) ] }),

  // 4.4
  h2("The surviving signal: an epigenetic HDAC axis with a genuine Alzheimer\u2019s rationale"),
  para("The single thread that survives degree control, promiscuity control, and cross-method scrutiny is the histone-deacetylase (HDAC) inhibitor class. Entinostat (MS-275) is a four-method consensus prediction for miR-148b at moderate, non-hub drug degree, and two further structurally distinct class I HDAC inhibitors \u2014 vorinostat and trichostatin A \u2014 survive degree control. Three chemically diverse members of one mechanistic class recurring across paradigms is not the profile of a degree or transcriptional-signature artefact, and this signal alone merits mechanistic discussion."),
  para("Its biological standing must be stated in tiers. The supportive evidence is genuine and, importantly, independent of the cancer-derived SmiRN-AD substrate. Systemic class I HDAC inhibition \u2014 by vorinostat, sodium butyrate, or valproate \u2014 completely restores contextual memory in APP/PS1 Alzheimer\u2019s mice; HDAC2 is elevated in Alzheimer\u2019s patients and represses synaptic genes, and HDAC3 is elevated in the Alzheimer\u2019s hippocampus where its inhibition attenuates deficits; and entinostat is class-I-selective (HDAC1/2/3), targeting precisely the isoforms the disease literature implicates. The proposed link to the predicted miRNAs is also mechanistically coherent: miR-148b represses the 3\u2032-UTR of DNMT1, the maintenance DNA methyltransferase, and DNMT1 is itself dysregulated in the Alzheimer\u2019s brain, so a plausible loop connects HDAC inhibition to relief of methylation-mediated silencing of the miR-148/152 family and onward remodelling of an Alzheimer\u2019s-dysregulated methylome."),
  para("What remains unproven must be stated with equal force. The experimental evidence linking an HDAC inhibitor to the miR-148/152 family was generated in cancer cell lines and concerns the paralogues miR-148a and miR-152; the specific claim that an HDAC inhibitor modulates miR-148b in neurons, or that this loop operates in the Alzheimer\u2019s brain, is an extrapolation by seed-sequence homology and has not been demonstrated. Moreover, the model did not discover this biology \u2014 entinostat is favoured for miR-148b because of the structure of the training graph, and the concordance with a real pathway, however striking, could be coincidental given that the same model confidently nominates artefacts. The appropriate conclusion is intermediate: the HDAC\u2013miR-148/152\u2013DNMT1 axis is the most testable hypothesis this pipeline produced, resting on a drug class with genuine independent Alzheimer\u2019s relevance and an epigenetic target of established Alzheimer\u2019s relevance, but its miRNA-specific wiring requires direct experimental validation in a neuronal or disease model before any causal claim is warranted. That this strongest candidate is simultaneously the clearest instance of the cancer-to-Alzheimer\u2019s framing limitation is the honest shape of what a computational screen over a cancer-derived substrate yields: a well-motivated lead, not a result."),

  // 4.5
  h2("Limitations"),
  para("The limitations are integral to the interpretation. The positive associations derive from SmiRN-AD, a 2014 Connectivity-Map-based computational resource rather than a set of assayed binding events, so every model here learns, at best, to reproduce and extend a prior computational prediction. The framing is borrowed from oncology, and the mechanistic support for the leading candidate originates in cancer biology, a transfer that the strongest result simultaneously illustrates. The graph is small (25 miRNAs, 257 drugs, 470 edges), a pilot scale at which the modest GNN advantage may not generalise and at which a single pathway node makes the context experiment low-powered. High-confidence candidate ranks remain dominated by node degree despite degree-matched training. The negative context result is established for one annotation source \u2014 the amyloid/tau-centric KEGG pathway via miRTarBase targets \u2014 and should not be extrapolated to richer or differently scoped annotations. Finally, no experimental validation was performed; all candidates are hypotheses."),

  // 4.6
  h2("Conclusions"),
  para("A controlled, leakage-audited, degree-matched benchmark shows that a context-free heterogeneous GNN modestly but significantly outperforms feature-only and topology-only baselines for Alzheimer\u2019s small-molecule\u2013miRNA association prediction; that amyloid/tau-centric pathway context adds no usable signal on a sparse graph, a null established three independent ways; and that degree-matched negative sampling is a mandatory control because it exposes a degree reliance uniform sampling conceals. Applied honestly, cross-method consensus shows most candidate signal to be degree and promiscuity artefact, leaving the epigenetic HDAC-inhibitor axis \u2014 entinostat for miR-148b foremost \u2014 as the single defensible class-level hypothesis for experimental follow-up. We deliberately refrain from claiming clinically translatable associations or high precision: an AUPR of 0.47 on a computationally derived substrate is a modest, honest result whose value lies in the rigour of its evaluation and the candour of its candidate interpretation. The priority for future work is therefore experimental \u2014 direct testing of whether a class I HDAC inhibitor modulates the miR-148/152 family in neuronal models \u2014 and methodological: degree-matched negatives, leakage-safe splitting, and AUPR-primary reporting should become standard practice, since on present evidence evaluation integrity, not architectural complexity, is the field\u2019s binding constraint."),
];

const doc = new Document({
  styles: { default: { document: { run: { font: FONT, size: 22 } } } },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 },
      margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    children
  }]
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync("/mnt/user-data/outputs/Discussion_revised.docx", buf);
  console.log("written");
});
