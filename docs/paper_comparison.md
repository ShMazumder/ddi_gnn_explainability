# Comparative Analysis: DDI GNN Explainability Thesis vs. DDI-Ben (Bioinformatics 2025)

This document compares our master's thesis project (**"Explainable Graph Neural Networks for Drug–Drug Interaction Prediction: A Faithfulness-Evaluated Study over Clinical Knowledge Graphs"**) with the benchmarking paper:
> **"Benchmarking drug-drug interaction prediction methods: a perspective of distribution changes"** (Z. Shen et al., *Bioinformatics* 2025, arXiv:2410.18583).

---

## 1. Quick Comparison Matrix

| Dimension | DDI-Ben Paper (Shen et al., 2025) | Our GNN Explainability Thesis |
| :--- | :--- | :--- |
| **Primary Objective** | Benchmarking prediction model **generalization/robustness** under distribution shifts (known vs. new drugs). | Evaluating the **faithfulness and usability of explanations** for GNN DDI predictions. |
| **Methods Evaluated** | 10 DDI Predictors (MLP, Decagon, EmerGNN, DDI-GPT, TIGER, etc.). | 4 Explainers (Attention Rollout, GNNExplainer, PGExplainer, and KEC). |
| **Graph Modeling** | Independent molecular graphs + static biomedical networks (for GNNs/Transformers). | Heterogeneous Clinical Knowledge Graph (CKG) with bidirectional drug-protein bindings and protein-protein interactions (PPI). |
| **Key Metrics** | Prediction performance: F1-Macro, ROC-AUC, PR-AUC, Accuracy. | Explanation quality: Sufficiency, Necessity, Fidelity+, Fidelity-, and Sparsity. |
| **Key Insight** | Existing predictors degrade significantly under distribution shifts; LLMs/text-augmented methods are most robust. | Directed message flows block drug updates; bidirectional flow is mathematically required for faithful explanation extraction. |

---

## 2. What We Do Differently (Our Contributions)

### A. Explainability & Faithfulness focus vs. Raw Prediction
While DDI-Ben benchmarks how well models *predict* interactions, they treat the model predictions as black boxes. Our work addresses the **explainability gap** in clinical settings. We evaluate whether the topological subgraphs highlighted by explainers are **faithful** (i.e. if the model actually relies on them to make the prediction).

### B. Biologically Constrained Explanations (KEC)
Generic explainability tools (like standard GNNExplainer or PGExplainer) optimize masks based purely on mathematical mutual information, often selecting disconnected or biologically meaningless edges. We introduce **Knowledge-Enhanced Counterfactuals (KEC)**, which greedily perturb edges restricted to the query drug pairs' biological $k$-hop neighborhood. KEC generates highly targeted, path-connected, and clinically relevant explanations (e.g., showing drug $A \rightarrow$ binds Target $T_A \rightarrow$ interacts with Target $T_B \leftarrow$ bound by drug $B$).

### C. Clinical Knowledge Graph Integration
Instead of relying only on drug molecular graphs (structures), we construct a heterogeneous Clinical Knowledge Graph (CKG) by integrating:
- **DrugBank** (clinical target bindings)
- **TWOSIDES** (known drug-drug interactions and multi-class side effects)
- **STRING** (systems-biology protein-protein interactions)

By mapping this to a homogeneous, bidirectional GAT setup, our model models systems-biology pathways directly in its layers.

---

## 3. Potential Improvements & Synergies

### A. Assessing Explanation Faithfulness under Distribution Shifts
A highly novel contribution we can add to Chapter 5 is evaluating **how explanation faithfulness behaves under the distribution shifts** described in the DDI-Ben paper. 
* *Hypothesis*: When models make predictions on "unseen" or "emerging" drug classes (S1/S2 splits), the GNN explainers might produce less faithful subgraphs (lower sufficiency/higher sparsity) compared to known drug splits, indicating the model's reasoning is less coherent under covariate shifts.
* *Implementation*: We can implement their **cluster-based drug split** (based on Tanimoto coefficient similarity threshold $\gamma$) or **chronological approval split** in our pipeline and run faithfulness checks on both splits.

### B. Integrating Textual Side Information
The DDI-Ben paper shows that Large Language Models (LLM) and textual side information (like the drug text summaries in `TextDDI` or `DDI-GPT`) are exceptionally robust against distribution shifts. We could explore integrating text embeddings of drug target descriptions as initial GAT node features, enhancing both the predictor's robustness and the semantic quality of the generated subgraphs.
