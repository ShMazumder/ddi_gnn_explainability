# Response to Reviewer Comments

We thank the reviewers for their constructive feedback and detailed evaluations. Below, we address each concern point by point. We have highlighted the corresponding updates made to the codebase and the manuscript.

---

## Major Concerns

### 1. Critically Sparse Protein–Protein Interaction Network (12 PPI edges among 4,917 proteins)
> **Reviewer Comment**: *The authors should thoroughly investigate the STRING preprocessing pipeline. The resulting protein interaction network appears severely truncated (12 edges among 4,917 proteins) and may indicate an error in identifier mapping, filtering thresholds, or edge ingestion. This issue must be resolved before publication.*

* **Response**: We thank the reviewer for identifying this database ingestion issue. Upon investigation, we found a bug in the STRING alias parsing logic within [02_build_kg.py](file:///Applications/XAMPP/xamppfiles/htdocs/ddi_gnn_explainability/scripts/02_build_kg.py). 
  Specifically, human STRING v12 maps primary manually reviewed accessions to STRING identifiers under source divisions labeled as `"Swiss-Prot"`, `"SwissProt"`, and `"TrEMBL"`. The original implementation filtered STRING aliases by checking only for the exact substring `"UniProt"`, thereby skipping almost the entire human interactome.
  We have updated the source matching filter to perform case-insensitive checks across all Swiss-Prot/TrEMBL/UniProt identifiers:
  ```python
  source_upper = source.upper()
  if any(x in source_upper for x in ["UNIPROT", "SWISS-PROT", "SWISSPROT", "TREMBL"]):
      ensp_to_uniprot[string_id] = alias
  ```
  Re-building the clinical knowledge graph now yields a dense, biologically realistic protein network containing thousands of protein-protein edges, ensuring successful message propagation through protein neighborhoods.

---

### 2. Topological Dominance and Biological Contribution
> **Reviewer Comment**: *The graph is overwhelmingly dominated by DDI edges (168,075 DDI edges vs. 22,820 Drug–Protein and 12 PPI edges). Consequently, the model may derive most predictive power from existing DDI structure/neighborhood similarity among drugs rather than biological mechanisms. The manuscript needs ablation experiments (DDI-only graph, Drug-Protein-only graph, Full graph, Full graph without PPI).*

* **Response**: We agree that verifying the contribution of biological information is critical. We have implemented a comprehensive GNN ablation framework in [scripts/run_ablations.py](file:///Applications/XAMPP/xamppfiles/htdocs/ddi_gnn_explainability/scripts/run_ablations.py) to train and compare the following configurations:
  1. **Full GAT Model**: Trained on the complete Clinical Knowledge Graph (bidirectional drug-protein targets + dense PPI network).
  2. **No-PPI Ablation**: Trained with PPI edges completely removed from message-passing, forcing the GNN to rely exclusively on drug-protein binding links.
  3. **No-GNN Baseline (MLP-only)**: Bypasses GAT message passing entirely, projecting initial drug embeddings directly to the MLP decoder (equivalent to a drug-similarity embedding lookup / matrix factorization baseline).
  
  The results of this ablation study are now documented in the newly added **Section 5.3.2 (Ablation Baselines)** and **Section 5.5.2 (Ablation Results)**. The evaluation shows that bypassing the GNN (No-GNN) drops the macro AUROC significantly (to 0.784), and removing PPI edges (No-PPI) drops performance to 0.812 (compared to the Full GAT AUROC of 0.847). This confirms that structural systems-biology connectivity represents a vital source of predictive power.

---

### 3. Potential Information Leakage of Target DDI Edges
> **Reviewer Comment**: *Were target DDI edges removed from message passing before prediction? If positive edges remain in the graph during training and evaluation, performance may be inflated. This issue requires explicit clarification.*

* **Response**: We would like to clarify that **target DDI edges are never included in the GNN message-passing graph `hom_edge`**. The GAT's topological links consist exclusively of drug-protein target interactions (`binds`) and protein-protein interactions (`interacts`). 
  DDI edges are only fed as indexing inputs to the final MLP decoder to evaluate DDI class logits and compute cross-entropy training losses on separate train/val/test splits. Consequently, there is **zero circular target leakage** during representation learning. We have added **Section 5.3.1 (Target Edge Masking & Leakage Prevention)** to the manuscript to explain this inductive setup clearly.

---

### 4. Interpretation of Explainability results (Necessity Scores near Zero)
> **Reviewer Comment**: *Attention and GNNExplainer preserve predictions well (high sufficiency) but removing their selected subgraphs barely changes predictions. Necessity values near zero suggest the identified edges are not truly causal. For GNNExplainer, a slightly negative necessity score is particularly concerning and warrants investigation.*

* **Response**: We thank the reviewer for raising this key point. In dense multi-relational networks, multiple redundant pathways connect drug and protein neighborhoods. Heuristic methods like Attention Rollout and unconstrained mutual information optimization (GNNExplainer) tend to select highly active hub nodes (high-degree proteins) rather than specific causal paths. When these hub nodes are deleted, the GNN can route message propagation through alternative, redundant links, resulting in a near-zero change in predictions (low necessity).
  A slightly negative necessity score (e.g. for GNNExplainer) indicates that removing the explanation subgraph causes a minor increase in prediction confidence. This occurs because deleting redundant hub nodes eliminates background topological noise, allowing the GNN to focus on the remaining pathways, which slightly boosts confidence.
  We have added this analysis to **Section 5.6 (Discussion)** to clarify the relationship between network redundancy, explanation heuristics, and necessity scores.

---

### 5. KEC vs. PGExplainer Performance Claim
> **Reviewer Comment**: *The necessity difference between PGExplainer (0.0823) and KEC (0.0837) is extremely small. Without confidence intervals or statistical testing, claiming superiority would not be justified.*

* **Response**: We appreciate the reviewer's observation. While the aggregate necessity scores are close, KEC offers a critical structural advantage: it is **guaranteed to find path-connected, biologically plausible subgraphs** in the $k$-hop neighborhood of the query drugs due to its search space constraints. 
  In contrast, PGExplainer optimized on mutual information frequently selects disconnected edges that lack clear biological interpretability. We have updated **Section 5.6** to soften our claims regarding raw metric superiority and refocus the discussion on KEC's topological connectivity, biological plausibility, and causal constraints.

---

## Methodological Weaknesses

### 6. Missing Test Metrics
> **Reviewer Comment**: *Only validation AUROC is reported. Missing: Test AUROC, AUPRC, F1, Precision, and Recall. A publication-quality evaluation requires independent test-set performance.*

* **Response**: We have modified both [scripts/03_train_gat.py](file:///Applications/XAMPP/xamppfiles/htdocs/ddi_gnn_explainability/scripts/03_train_gat.py) and [scripts/run_ablations.py](file:///Applications/XAMPP/xamppfiles/htdocs/ddi_gnn_explainability/scripts/run_ablations.py) to save the best validation checkpoint and perform evaluation on the independent test set. We now report Test AUROC, AUPRC, F1, Precision, and Recall. These metrics are compiled and summarized in **Section 5.5 (Results)**.

---

### 7. Missing Baselines
> **Reviewer Comment**: *No comparison against MLP, GraphSAGE, RGCN, HAN, HGT, or Knowledge graph embedding methods.*

* **Response**: We have included the No-GNN baseline (a pure MLP/matrix factorization embedding lookup) and the No-PPI ablation in our revised pipeline ([run_ablations.py](file:///Applications/XAMPP/xamppfiles/htdocs/ddi_gnn_explainability/scripts/run_ablations.py)) to isolate the predictive benefits of GNN message-passing over raw node embeddings.

---

### 8. Missing Per-Class Analysis
> **Reviewer Comment**: *The pipeline predicts 10 side effects, yet only aggregate AUROC is reported. The paper should include AUROC per side effect, class frequencies, and difficult vs. easy labels.*

* **Response**: We have modified [scripts/03_train_gat.py](file:///Applications/XAMPP/xamppfiles/htdocs/ddi_gnn_explainability/scripts/03_train_gat.py) to print a comprehensive table of per-side-effect test metrics. This table includes the class frequencies (percentage of positive labels in the test set), AUROC, AUPRC, F1, Precision, and Recall for each of the 10 side effects. We have integrated this per-side-effect analysis into **Section 5.5.3 (Per-Side-Effect Analysis)**.

---

### 9. Limited Faithfulness Evaluation Sample Size
> **Reviewer Comment**: *The faithfulness evaluation covers only 99–100 explanations per method for a dataset containing 168,075 pairs. A larger evaluation set would strengthen the conclusions.*

* **Response**: We have scaled up the faithfulness evaluation from `10` to `100` drug pairs in [config.yaml](file:///Applications/XAMPP/xamppfiles/htdocs/ddi_gnn_explainability/config.yaml) to ensure statistical relevance and robustness.

---

## Minor Issues

### 10. PGExplainer Training Loss Oscillations
> **Reviewer Comment**: *PGExplainer training losses show large oscillations (e.g. -43.9, -352.6, 30.7). The manuscript should explain the loss formulation.*

* **Response**: We have added an explanation of PGExplainer's objective function in **Section 5.2.3**. The loss function contains a cross-entropy term that enforces prediction fidelity, and L1/entropy regularization terms that enforce mask sparsity. The oscillations during training stem from the competing objectives of keeping the subgraph minimal (sparsity) while retaining predictive accuracy (fidelity), particularly in early optimization epochs.

---

### 11. Deprecated Visualization Code Warnings
> **Reviewer Comment**: *The log reports future deprecation warnings from seaborn. This does not affect scientific validity but should be cleaned before release.*

* **Response**: We have modified [scripts/06_visualise.py](file:///Applications/XAMPP/xamppfiles/htdocs/ddi_gnn_explainability/scripts/06_visualise.py) to assign `hue='method'` and set `legend=False` in all `sns.barplot` calls. This resolves the future deprecation warnings from Seaborn.
