# Review Feedback #1 on: June 08, 2026 23:30

# Journal-Style Review of Chapter 5

**Manuscript Title (inferred):** Explainability and Faithfulness Evaluation of GNN Methods for Drug–Drug Interaction Prediction

**Recommendation:** **Major Revision**

---

# Overall Assessment

This chapter is substantially stronger than most thesis-level explainability evaluations. The author demonstrates awareness of a critical problem in XAI research: **distinguishing predictive performance from explanation faithfulness**. The inclusion of multiple explainers, quantitative faithfulness metrics, topology-aware analysis, statistical testing, and an explicit discussion of graph construction decisions represents a mature experimental design.

The chapter's strongest contribution is not KEC itself, but the **systematic investigation of why explanation methods behave differently under a dense biomedical graph setting**. The discussion linking explanation behavior, attention dilution, oversmoothing, and graph topology is particularly valuable.

However, several methodological, statistical, and presentation issues remain that would likely be raised by reviewers of journals such as *Journal of Biomedical Informatics*, *Artificial Intelligence in Medicine*, *BMC Bioinformatics*, *IEEE JBHI*, or *Patterns*.

---

# Major Strengths

## 1. Clear Scope Definition

The chapter explicitly states that the goal is:

> evaluation of explanation faithfulness rather than SOTA DDI prediction

This is important because many reviewers would otherwise question the absence of comparisons with:

* MIRACLE
* SSI-DDI
* EmerGNN

The scope statement successfully preempts that criticism.

---

## 2. Strong Faithfulness Framework

The use of:

* Sufficiency
* Fidelity+
* Fidelity-
* Sparsity
* Connectivity

is appropriate and aligns with current GNN explainability literature.

The distinction between:

* necessity (Fidelity+)
* sufficiency (Fidelity-)

is correctly articulated.

This is one of the strongest sections of the chapter.

---

## 3. Honest Reporting of Negative Results

Many manuscripts attempt to justify proposed methods through selective reporting.

Here:

* KEC loses on Sufficiency
* KEC loses on Sparsity
* KEC loses on Strict Connectivity

yet wins on Fidelity+.

This improves credibility considerably.

---

## 4. Statistical Testing Included

Most explainability papers stop at mean comparisons.

Including:

* Wilcoxon signed-rank tests
* Holm–Bonferroni correction
* Rank-biserial effect size

is commendable.

---

## 5. Strong Discussion of PPI Effects

The observation that:

> NO_PPI slightly outperforms FULL

is scientifically interesting.

The chapter does not hide this result and instead investigates:

* attention dilution
* oversmoothing
* representational collapse

This is a reviewer-positive aspect.

---

# Major Concerns

## 1. The Clinical Usability Study Is Not Actually Performed

Section 5.7 is problematic.

The chapter is titled:

> Explanation Methods & Faithfulness Evaluation

but later introduces a clinical usability framework that has not been executed.

A reviewer will immediately ask:

> Why is a planned study presented as a results chapter?

Currently this section reads like a methodology chapter rather than a results chapter.

### Recommendation

Move Section 5.7 to:

* Future Work
* Study Design
* Proposed Validation Protocol

or explicitly label it:

> "Planned Future Clinical Evaluation"

otherwise readers may assume data exist.

---

## 2. KEC's "Novelty" Is Not Yet Fully Demonstrated

KEC appears to combine:

* local subgraph restriction
* biologically constrained candidate filtering
* greedy counterfactual search

The chapter repeatedly compares against:

* GNNExplainer
* PGExplainer
* Attention Rollout

but does not compare against:

CF-GNNExplainer

which is the natural baseline.

The text discusses CF-GNNExplainer theoretically but no empirical comparison is shown.

### Likely Reviewer Question

> How do we know KEC improves over existing counterfactual explainers rather than simply over descriptive explainers?

This is currently the biggest novelty threat.

---

## 3. Connectivity Metric Requires Stronger Justification

The chapter introduces:

* Strict Connectivity
* Lenient Connectivity

The strict metric is intuitive.

The lenient metric is novel but potentially controversial.

A reviewer may ask:

> Why does lying on a path imply explanatory value?

Many irrelevant edges can also lie on valid paths.

The metric is currently presented as if it automatically indicates biological plausibility.

That claim is not fully supported.

### Recommendation

Clarify:

* what lenient connectivity measures
* what it does not measure
* why it is only a proxy

---

## 4. Fidelity+ Values Are Extremely Small

KEC's best result:

0.0371 ± 0.1418

This is statistically significant against GNNExplainer.

However:

0.0371 is still a very small probability shift.

A reviewer may argue:

> Even the best explainer barely affects prediction confidence.

Indeed, your own discussion hints at this.

### Recommendation

Explicitly state:

* explanation necessity remains weak overall
* the model appears highly redundant
* current architecture may not support strong perturbation-based explanations

This is already partially discussed but should be emphasized more clearly.

---

## 5. Potential Circularity in KEC Candidate Selection

KEC only searches edges that:

> connect targets of the queried drugs

This greatly improves biological plausibility.

However it also biases explanations toward biologically meaningful pathways.

A reviewer may ask:

> Is the improved connectivity caused by better explanations or by hard-coded biological priors?

This distinction is important.

The manuscript should explicitly acknowledge:

> KEC's performance partly derives from imposed biological constraints.

That is not necessarily bad, but it should be stated.

---

# Statistical Concerns

## Multiple Comparisons

Only Fidelity+ receives significance testing.

Why not:

* Fidelity-
* Sufficiency
* Connectivity

A reviewer may question selective testing.

### Recommendation

Either:

1. Test all primary metrics

or

2. Justify Fidelity+ as the sole primary endpoint before results are presented.

Currently the justification appears later.

---

## Sample Size

Faithfulness evaluation uses:

100 sampled pairs

from

200 explained pairs.

This is acceptable computationally but requires stronger justification.

Questions reviewers may ask:

* Why not evaluate all 200?
* How stable are estimates across random samples?

A simple sensitivity analysis would strengthen the work.

---

# Interpretation Concerns

## Overstatement of Biological Plausibility

Several passages imply:

> KEC explanations are biologically plausible

Yet no biological validation is actually performed.

No:

* pathway enrichment
* literature validation
* pharmacist assessment

has yet occurred.

Therefore the manuscript should say:

> biologically motivated

rather than:

> biologically plausible

until external validation is completed.

---

## Overstatement of Oversmoothing

The chapter claims:

> representational collapse

based on cosine similarity:

0.0020 → 0.2465

This suggests increasing similarity but does not by itself prove oversmoothing.

A reviewer may ask for:

* layer-wise variance
* pairwise distance collapse
* spectral analysis

The wording should be softened.

Instead write:

> evidence consistent with oversmoothing

rather than asserting it conclusively.

---

# Writing and Presentation Issues

## Section 5.2.5 Title

Current:

> Graph Homogenization and backbone justification

Should be:

> Graph Homogenization and Backbone Justification

---

## Consistency Issue

You mention:

* 50 side effects

but later:

* Top 10 side effects selected

The relationship between:

* 50-class prediction
* top-10 evaluation

needs clearer explanation.

---

## Table 5.5.1

I would recommend adding:

| Method | Explanation Type |

Examples:

| Method            | Type                   |
| ----------------- | ---------------------- |
| Attention Rollout | Descriptive            |
| GNNExplainer      | Local perturbation     |
| PGExplainer       | Amortized perturbation |
| KEC               | Counterfactual         |

This improves interpretability.

---

# What a Strong Reviewer Would Probably Say

### Novelty

**Moderately strong**, but needs direct comparison with a counterfactual baseline.

### Technical Correctness

**Generally sound.**

No major methodological flaws were observed.

### Experimental Rigor

**Above average.**

Especially the inclusion of:

* significance testing
* ablation studies
* topology analysis

### Clinical Relevance

Currently **potential**, not demonstrated.

The usability study remains future work.

### Publication Potential

If this were submitted to a biomedical AI journal, I would likely recommend:

**Major Revision**

with the following mandatory items:

1. Add comparison against a counterfactual baseline (preferably CF-GNNExplainer).
2. Clarify the role and limitations of Lenient Connectivity.
3. Reduce claims regarding biological plausibility.
4. Move the planned usability study out of the results chapter.
5. Strengthen justification for evaluating only 100 pairs.
6. Temper claims about oversmoothing and representational collapse.

---

# Final Score (Reviewer Perspective)

| Criterion           | Score                       |
| ------------------- | --------------------------- |
| Novelty             | 7.5/10                      |
| Technical Soundness | 8.5/10                      |
| Experimental Design | 8.5/10                      |
| Statistical Rigor   | 8/10                        |
| Clarity             | 8.5/10                      |
| Clinical Relevance  | 6.5/10                      |
| Overall             | **8.0/10 (Major Revision)** |

This chapter is considerably stronger than a typical MSc thesis chapter and is approaching journal quality, but it still contains several claims that reviewers would challenge before publication.


# Reviewer Feedback #2 on: June 08, 2026 23:30

Peer Review — Journal Manuscript
Chapter 5: Explanation Methods & Faithfulness Evaluation
Explainable GNNs for DDI Prediction · Homogeneous GAT + Clinical KG · Four Explainer Methods
Major Revision
The work is methodologically sound and the writing is clear, but critical gaps in statistical validation, an incomplete clinical study, and a logical inconsistency in the connectivity metrics must be resolved before acceptance.
Originality
3.5/5
Technical rigor
3.5/5
Significance
3.0/5
Presentation
4.0/5
Statistical quality
3.0/5
Overall
6.5/10
Summary
This chapter presents an experimental evaluation of four GNN explanation methods — Attention Rollout, GNNExplainer, PGExplainer, and the proposed KEC — applied to a homogeneous GAT trained on a clinical knowledge graph for DDI side-effect prediction. The methodology is well-scoped and transparently framed as an explainability study rather than a predictive benchmark. KEC is the primary novel contribution: a greedy counterfactual search constrained to biologically plausible 2-hop neighborhoods, which achieves the best Fidelity+ and lenient connectivity among the evaluated methods. The faithfulness evaluation framework is comprehensive, covering sufficiency, fidelity+/−, sparsity, and dual (strict/lenient) topological connectivity. The conceptual taxonomy in §5.6.2 is a genuine contribution to the literature. Statistical testing follows appropriate practices (Wilcoxon signed-rank with Holm–Bonferroni correction). However, the ablation study is entirely unaccompanied by significance tests despite reporting near-identical scores, the planned clinical evaluation in §5.7 cannot constitute evidence for a journal submission, and a logical inconsistency in the connectivity metrics requires resolution. These issues are addressable in revision.
Strengths
KEC algorithm is well-motivated with precise pseudocode and a clear biological rationale for constraining the search to 2-hop neighborhoods and target-connecting paths.
Dual connectivity metrics (strict vs. lenient) are a novel, useful contribution; the random-baseline validation (0.03% lenient connectivity) credibly establishes KEC's 40.0% as non-trivial.
The internal diagnostic data — GAT attention dilution on PPI edges (Layer 1 mean 0.0069 vs. 0.1317 for target bindings) and cosine similarity collapse (0.0020 → 0.2465 across layers) — is illuminating and rarely reported at this level of detail.
The goal-mismatch taxonomy (§5.6.2) cleanly organizes why different explainers diverge across metrics, and will benefit readers unfamiliar with the explainability literature.
Design trade-offs (graph homogenization, backbone selection, exclusion of gradient/RL-based methods) are disclosed transparently with explicit justifications.
Rank-biserial effect sizes reported alongside p-values; correction method (Holm–Bonferroni) is appropriate for 6 pairwise comparisons.
Concerns
Critical — blocks acceptance
Major — requires revision
Minor — addressable in revision
C1
Ablation study lacks significance testing
The performance differences between Full GAT, No-PPI, and No-GNN are very small: the Test AUROC range is only 0.0037 (0.8507–0.8544). The authors state "graph message passing provides a modest but consistent improvement" and that No-PPI outperforms Full, but provide no confidence intervals or significance tests. The DeLong test for paired AUROC comparison, or bootstrap resampling, must be applied to determine whether these differences are statistically meaningful. Without this, no architectural conclusion can be drawn from Table 5.5.2.
C2
Section 5.7 — "Planned" study is not evidence
A journal article requires completed evidence. A planned protocol presented as a chapter section implies a level of rigor that has not yet been achieved. The clinical usability study should be either: (a) completed and reported, (b) removed and repositioned as a future-work paragraph in §5.8 or §6, or (c) submitted separately as a study protocol paper in a clinical informatics venue. As currently written, §5.7 occupies substantial space without delivering findings.
C3
Logical inconsistency in Attention Rollout connectivity metrics
Attention Rollout reports 9.0% strict connectivity but 0.0% lenient connectivity. This is paradoxical: lenient connectivity is defined as a weaker condition (explanation edges lying along some valid path in the original graph), whereas strict connectivity requires a complete path within the explanation subgraph. Any instance satisfying the strict criterion must also satisfy the lenient criterion, so strict% ≤ lenient% must always hold. A result of 9% strict / 0% lenient violates this and almost certainly indicates an implementation bug, a definitional mismatch between the two metrics, or an error in the evaluation code. This must be investigated and corrected before submission.
M1
KEC approximation quality is unvalidated
KEC is presented as an approximation of a minimal counterfactual edge set, but no analysis of approximation quality is given. The greedy heuristic could be arbitrarily far from the true minimum on dense subgraphs. Even a small-scale comparison against exact enumeration (e.g., exhaustive search on the subset of instances where the local subgraph has fewer than 20 candidate edges) would validate or qualify the "minimal" claim. Without this, the claim that KEC finds a minimal counterfactual set is unsupported.
M2
Lenient connectivity definition is under-specified
The metric is defined as "explanation edges are a subset of some valid path connecting d₁ and d₂ in the original graph." What constitutes a "valid path"? Any simple path? Shortest paths only? Biochemically annotated paths? The choice has a significant effect on the metric value. The definition must be made mathematically precise (e.g., any simple path of length ≤ k in the original undirected graph) and the choice of k justified.
M3
PGExplainer negative Fidelity+ — competing hypotheses unresolved
The two proposed mechanisms for PGExplainer's negative Fidelity+ (−0.0045) — mask-induced distribution shift vs. redundancy removal — are not distinguished. A simple diagnostic would help: compare the model's output distribution on unmasked subgraphs vs. fully random edge masks of the same sparsity. If random masking also increases prediction probability, the effect is structural (redundancy); if it does not, the effect is specific to PGExplainer's mask (distribution shift). The current discussion leaves this genuinely ambiguous in a way that weakens the claim about PGExplainer's behavior.
M4
The 100-pair evaluation subset representativeness is unverified
200 pairs were generated, but faithfulness metrics are evaluated on a random subsample of 100. The authors justify this by computational cost. However, no verification is provided that the 100-pair sample is representative of the full 200. A brief distributional comparison (e.g., drug degree distribution, mean local subgraph size, side-effect class frequency) between the 100 evaluated pairs and the full 200 should be included. Without this, potential selection bias cannot be ruled out.
m1
50-class evaluation claim is unverifiable
Table 5.5.3 shows only 10 of the 50 evaluated side-effect classes. The remaining 40 are described as showing "similar performance patterns," but no supporting data is provided. At minimum, a supplementary figure (box plot or CDF of per-class AUROC across all 50 classes) should be included so the claim can be evaluated.
m2
KEC max_removals=10 — binding frequency unreported
The procedure terminates and returns top-impact edges when no prediction flip occurs within 10 removals. How often does this fallback path activate in the 100-pair evaluation? If it fires frequently, the returned "counterfactual explanation" is not a true counterfactual, and the interpretation of KEC's Fidelity+ and lenient connectivity must be qualified accordingly.
m3
Seed variance is described but not quantified
The authors note "minor run-to-run seed variance" from non-deterministic PyG scatter/gather kernels. For reproducibility and to support the ablation claims in §5.5.2, report standard deviations across at least 3 seeds for the ablation models' test AUROC/AUPRC. A per-run variance of ±0.001 would still not challenge the conclusions; a variance of ±0.004 would call the No-PPI vs. Full comparison into question.
m4
Table 5.5.1 footnote is excessively long
The note under Table 5.5.1 is approximately 250 words — longer than many published paper paragraphs. Key content (bold-value convention, random baseline comparison, statistical significance results) should be distributed across the main text or a dedicated statistics subsection. The table note should be reduced to 2–3 sentences covering display conventions only.
m5
Fidelity− sign convention is non-standard
The paper states "a lower Fidelity− indicates that the explanation subgraph alone is sufficient," but then bolds the highest value in Table 5.5.1 (Attention Rollout, 0.0897) in the Fidelity− column. If lower is ideal, the bolded value should be the minimum, not the maximum. This appears to be a copy-paste inconsistency in the table bolding logic and will confuse readers. Confirm and correct.
m6
Section 5.1 scoping sentence placement
The sentence "Consequently, comparisons against specialized predictive models (such as MIRACLE, SSI-DDI, or EmerGNN) are out of scope" is placed mid-paragraph in §5.1. This is a paper-level scoping decision that is likely to be misread as a limitation admission rather than a design choice. Move it to the final sentence of the opening paragraph, or introduce it with "By design, this study does not…"
Statistical notes
The Wilcoxon signed-rank test with Holm–Bonferroni correction is appropriate for the six pairwise Fidelity+ comparisons on 100 samples. Reporting rank-biserial correlation as an effect size is commendable and adds interpretive value beyond p-values.
The AUPRC discussion in §5.5.3 is well-handled: noting that AUPRC is prevalence-inflated for 65–77% positive classes, and computing the random-classifier AUPRC baseline (56.91%), is exactly the right framing. No revision needed here.
The ablation study (§5.5.2) must be treated with the same statistical rigor as the explainability comparisons. AUROC differences of 0.0008–0.0037 are within typical run-to-run variance for PyG models and cannot be interpreted as directional without significance tests (DeLong or bootstrap). This is the most statistically consequential gap in the chapter.
Writing quality
Generally excellent. The mathematical notation is consistent throughout, and the prose explanations of each method (particularly PGExplainer's loss function and KEC's step-by-step heuristic) are unusually clear for this genre. The taxonomy in §5.6.2 is a highlight — it organizes disparate empirical results around a unifying conceptual framework in a way that will benefit both expert and non-expert readers.

The discussion in §5.6.1 around the localized representation learning findings (attention dilution + cosine similarity collapse) is technically rich and represents some of the most interesting content in the chapter. Consider surfacing this earlier — it currently appears at the end of Point 5 after several paragraphs of trade-off analysis, but it is arguably the deepest finding.

One structural issue: the chapter contains substantial methodological content in §5.2 (explainer descriptions, graph homogenization, exclusion rationale) that may overlap with a Methods chapter. If this is a standalone paper, the current structure is appropriate. If Chapter 5 is embedded in a larger thesis, some of §5.2 may belong in Chapter 3.

Recommendation to authors
The three critical concerns (C1–C3) must be fully resolved. C3 in particular suggests a possible implementation error that would undermine the connectivity results for Attention Rollout entirely. C1 and C2 are structural issues that require either new experiments (ablation significance tests) or editorial restructuring (§5.7 repositioning).

The major concerns (M1–M4) are each addressable within the existing experimental infrastructure: small validation experiments for M1, a tightened mathematical definition for M2, a diagnostic masking experiment for M3, and a distributional check for M4. The minor concerns are primarily presentation and reporting issues. Addressing all critical and major concerns would bring this work to a strong accept.


# Reviewer Feedback #3 on: June 08, 2026 23:30

Here is a comprehensive, rigorous peer review of your thesis chapter, styled exactly like a high-impact journal review (e.g., *IEEE Transactions on Pattern Analysis and Machine Intelligence*, *Bioinformatics*, or *ACM Transactions on Knowledge Discovery from Data*).

---

# 📝 Peer Review Report

**Manuscript Title:** Chapter 5: Explanation Methods & Faithfulness Evaluation

**Review Type:** Rigorous Journal Review

## ⚖️ General Evaluation

This manuscript presents a highly refreshing, methodology-oriented evaluation of GNN explainability frameworks within the critical domain of drug–drug interaction (DDI) prediction. The authors explicitly scope this work around **explanation faithfulness and usability** rather than chasing a raw predictive baseline arms race, which is a commendable and necessary direction for the field.

The standout strength of this work is its **intellectual honesty and diagnostic depth**. When faced with a classic GNN bottleneck—where adding a dense biological network (PPI) degrades performance and explanation models yield low structural connectivity—the author did not gloss over or hide the results. Instead, they ran excellent post-hoc profiling (attention dilution and representation collapse analyses) to explain *why* the model behaves this way. This elevates the work from a simple benchmarking exercise to a valuable structural critique of standard message-passing architectures on dense multi-relational topologies.

However, prior to publication or final thesis defense, there are several structural paradoxes, missing methodological details, and framing issues that must be addressed.

---

## 🔴 Major Revisions & Critical Concerns

### 1. The Predictive Paradox vs. Explainer Faithfulness

The ablation study (Table 5.5.2) reveals that the `No-PPI Ablation` model outperforms the `Full GAT Model` across almost all test metrics (Test AUROC 0.8544 vs 0.8536). Section 5.6.1 elegantly attributes this to GAT attention dilution over dense PPI edges (mean attention of 0.0069 vs 0.1317 for target bindings) and rapid representation over-smoothing.

* **The Reviewer’s Challenge:** If the underlying GAT model is actively discounting or being degraded by the PPI network, evaluating an explainer’s ability to find "biologically plausible pathways through the PPI interactome" becomes fundamentally paradoxical. The explainers are trying to map a decision-making path that the backbone GNN did not actually learn or rely upon.
* **Action Required:** Explicitly address this tension in the introduction of the Discussion. Frame the low strict connectivity of *all* explainers not as a failure of the explainability wrappers themselves, but as a faithful reflection of a model that operates primarily on localized drug-protein target attachments rather than deep graph routing.

### 2. The "Planned" Status of the Clinical Usability Study

Section 5.7 outlines a meticulous and robust protocol for clinical usability evaluation ($N=15$ pharmacists, triple-blinded, Krippendorff's $\alpha$, semi-structured interviews). While this is perfectly acceptable for an initial thesis draft, presenting a "planned" framework in a submitted journal manuscript is generally considered incomplete work.

* **Action Required:** If this is for a journal submission, you must either:
1. Complete the study and present the real-world data/Likert scores.
2. Execute a **pilot study** with a smaller sub-cohort ($N=3$ to $5$) to provide preliminary qualitative insights and thematic coding examples.
3. Re-frame the chapter title and abstract to strictly highlight "Quantitative Faithfulness and Topological Evaluation," marking the human-in-the-loop validation explicitly as future work.



### 3. Node Feature Initialization & Information Leakage Check

Section 5.3.2 does an excellent job explaining that target DDI edges are excluded from the message-passing graph `hom_edge` to prevent direct label leakage. However, the text is completely silent on how the initial node features ($X$) for drugs and proteins are constructed.

* **Action Required:** Clarify the nature of the input features. Are they learned structural embeddings (e.g., Node2Vec), molecular fingerprints (e.g., Morgan fingerprints for drugs), sequence embeddings (e.g., ESM/ProtBert for proteins), or simple one-hot/random initializations? If the initial features implicitly encapsulate relational density or side-effect correlations, it introduces confounding variables into your faithfulness metrics.

### 4. Mathematical Foundations of the KEC Heuristic

The Knowledge-Enhanced Counterfactual (KEC) algorithm operates as a greedy search heuristic (Section 5.2.4.1), sorting candidate edges by their single-edge prediction impact $I(e)$ and deleting them sequentially until the binary prediction label vector flips.

* **Action Required:** GNN outputs are notoriously non-linear and non-monotonic due to neighborhood aggregation. Deleting edge $A$ and edge $B$ sequentially may have a vastly different joint impact than the sum of their individual impacts $I(A) + I(B)$. Acknowledge this limitation. Briefly discuss whether your impact function exhibits submodular properties or if the greedy choice acts purely as a local first-order approximation of a combinatorial cut-set problem.

---

## 🟡 Minor Revisions & Clarifications

* **PGExplainer Loss Oscillations:** Section 5.2.3 notes that PGExplainer loss values exhibit significant oscillations, dropping into negative values due to regularization terms. Please provide the explicit mathematical formulations of $L_{sparsity}$ and $L_{entropy}$ or explain if the negative loss stems from the balance of the continuous relaxation optimizer.
* **Sparsity Metric Bounding:** In Table 5.5.1, GNNExplainer achieves a Local Sparsity of $1.0000 \pm 0.0000$. If sparsity is exactly 1, it implies the explainer selected *zero* edges, which conflicts with its ability to evaluate prediction changes. Clarify if GNNExplainer's continuous edge masks dropped entirely below your extraction threshold, effectively rendering the explanation empty.
* **Side-Effect Table Consistency:** Section 5.2.4.1 mentions $C=50$ side-effect classes, while Table 5.5.3 explicitly benchmarks the top 10. The explanatory note clarifies that the remaining 40 follow similar patterns, but the table title should be adjusted to "Table 5.5.3: Predictive Performance on Top 10 Most Prevalent Side-Effect Classes" to maintain absolute transparency.

---

## 🌟 Strengths to Retain

* The **Dual Connectivity Analysis (Strict vs. Lenient)** is brilliant. Introducing Lenient Connectivity to show that KEC hits a highly non-trivial 40% pathway alignment despite its design breaking strict connectivity is an excellent, original contribution to GNN explanation theory.
* The formatting, table structures, and statistical hygiene (using non-parametric paired Wilcoxon tests alongside Holm–Bonferroni corrections) are outstanding and ready for publication.

---

How would you prefer to address the trade-off between the model's predictive degradation on the full PPI graph and your evaluation of its biological pathway explanations?


# Reviewer Feedback #4 on: June 08, 2026 23:30


**Review of Chapter 5: Explanation Methods & Faithfulness Evaluation**

**Overall Assessment:**  
This chapter presents a solid, methodologically focused evaluation of GNN explainability techniques applied to a homogeneous GAT model for DDI prediction. The work is clearly scoped as an **explainability/faithfulness study** rather than a predictive modeling benchmark, which is appropriate and well-articulated. The introduction of **KEC (Knowledge-Enhanced Counterfactual)** as a domain-constrained counterfactual explainer is a reasonable contribution, and the multi-metric faithfulness evaluation (including connectivity analysis) is thoughtful. However, the results reveal substantial limitations in both the underlying model’s use of graph structure and the practical utility of the explanations, which the discussion acknowledges but does not fully resolve. With revisions, this could be suitable for publication in a specialized AI-for-health or graph ML venue (e.g., *Journal of Biomedical Informatics*, *Artificial Intelligence in Medicine*, or *NeurIPS/MLHC* workshop track). 

**Major Strengths:**
- **Clear scoping and intellectual honesty**: The authors repeatedly emphasize the methodological focus and avoid overclaiming predictive superiority. The discussion of model limitations (PPI dilution, oversmoothing, localized reliance on drug-target bindings) is candid and insightful.
- **Comprehensive faithfulness framework**: Combining discrete sufficiency, continuous fidelity metrics, sparsity, and both strict/lenient topological connectivity is a strong aspect. The random baseline comparison for lenient connectivity is particularly valuable.
- **Detailed method descriptions**: Especially strong for KEC, including implementation optimizations (vectorized extraction, candidate pruning) and the greedy heuristic. This supports reproducibility.
- **Ablation and diagnostic analysis**: The No-PPI and No-GNN ablations, attention weight profiling, and cosine similarity analysis provide mechanistic insight into why explanations are localized.
- **Planned clinical study**: The usability protocol is well-designed (blinding, Likert + qualitative, reasonable sample size, ethics considerations).

**Major Concerns (Revisions Required):**

1. **Limited Practical Utility of Explanations**  
   The quantitative results are underwhelming: very low strict connectivity (0–9%), near-zero or negative Fidelity+ for most methods, and high sparsity with minimal impact on predictions for non-KEC methods. This suggests the explanations often fail to capture coherent biological pathways. While discussed, the chapter should more strongly address whether these explanations are clinically meaningful *even if faithful*. The low lenient connectivity for baseline methods (0%) versus KEC (40%) is one bright spot, but needs clearer visualization or examples (e.g., a figure showing a representative KEC vs. PGExplainer subgraph).

2. **Model Limitations Undermine Explainer Evaluation**  
   The GAT appears to learn mostly local drug-target signals rather than pathway-based reasoning, as evidenced by:
   - Negligible PPI benefit (No-PPI slightly outperforms Full).
   - Attention dilution on PPI edges.
   - Oversmoothing (cosine similarity rising to 0.2465).
   
   Evaluating explainers on a model that does not strongly rely on multi-hop structure raises questions about generalizability. The authors should discuss this more explicitly as a threat to validity: “Explanations are faithful to a model that itself may not capture the intended biological relational inductive bias.”

3. **Statistical and Evaluation Scale Issues**  
   - Only 100 pairs evaluated despite generating for 200 — justified by compute, but still modest.  
   - Many metrics have large standard deviations (e.g., Sufficiency SD ~0.31–0.38), indicating high instance-level variability.  
   - Holm–Bonferroni correction leaves only one pairwise comparison significant. Effect sizes are reported but should be interpreted more cautiously.  
   - Consider reporting additional metrics (e.g., explanation stability across seeds for stochastic methods, or per-class fidelity).

4. **Methodological Details Needing Clarification**  
   - **KEC stopping criterion**: Clarify what “binary prediction flips for any class” means in a multi-label setting (any of the 50 side effects?). Threshold?  
   - **PGExplainer negative Fidelity+**: The discussion of possible causes (distribution shift vs. redundancy removal) is good but speculative. More ablation on this artifact would strengthen the paper.  
   - **Homogenization trade-offs**: Well-described, but quantify information loss more (e.g., compare to a heterogeneous model if feasible, or at least discuss expected impact).  
   - **GNNExplainer details**: Confirm training details (epochs, hyperparameters, regularization strengths) for reproducibility.  
   - **Threshold τ=0.5**: Justified for clinical decisions but should note sensitivity analysis or calibration status of the model.

5. **Presentation and Structure**  
   - Some tables are dense; consider moving detailed side-effect per-class results to supplementary material.  
   - No figures in the provided draft — strongly recommend adding: (a) example explanation subgraphs, (b) attention/PPI dilution plots, (c) distribution of explanation sizes/hop distances.  
   - Minor inconsistencies (e.g., overall Test AUROC cited as 0.8536/0.8537 across sections).

**Minor Comments:**
- Expand acronyms on first use where needed (some are clear from context, but best practice).  
- “Knowledge-Enhanced Counterfactual” — consider a more distinctive name or clearer acronym justification.  
- In KEC candidate filtering: “paths of length ≤3” — confirm this aligns with 2-hop GNN receptive field.  
- Discussion of “conceptual explanation goal mismatch taxonomy” is useful; could be expanded into a small table for clarity.  
- Future work: Suggest architectural mitigations (e.g., residual connections, GATv2, relational GNNs, contrastive pathway objectives) more concretely.  
- References: Ensure all cited works (Ying et al., Luo et al., Lucic et al., Shen et al. 2025) are properly formatted.

**Recommendation: Major Revision**  
The chapter has a strong methodological core and honest analysis but needs strengthening in (1) addressing the gap between faithfulness and clinical utility, (2) better visualization of results and examples, (3) deeper discussion of model limitations as they pertain to explainability claims, and (4) polishing of statistical reporting and reproducibility details. After addressing these, it would make a worthwhile contribution to the growing literature on trustworthy GNNs in biomedicine. The planned human evaluation is a crucial next step that should be highlighted as differentiating this work.

I encourage the authors to proceed with revisions — this direction (domain-constrained counterfactuals + multi-faceted faithfulness + clinical usability) is promising.