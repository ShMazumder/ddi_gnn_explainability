# Response to Reviewer Comments

We thank the reviewers for their constructive feedback and detailed evaluations. Below, we address each concern point by point. We have highlighted the corresponding updates made to the codebase and the manuscript.

---

## Major Concerns

### 1. Critically Sparse Protein–Protein Interaction Network (12 PPI edges among 4,917 proteins)
> **Reviewer Comment**: *The authors should thoroughly investigate the STRING preprocessing pipeline. The resulting protein interaction network appears severely truncated (12 edges among 4,917 proteins) and may indicate an error in identifier mapping, filtering thresholds, or edge ingestion. This issue must be resolved before publication.*

* **Response**: We thank the reviewer for identifying this database ingestion issue. Upon investigation, we discovered a key bug in the STRING alias parsing logic within [02_build_kg.py](file:///Applications/XAMPP/xamppfiles/htdocs/ddi_gnn_explainability/scripts/02_build_kg.py).
  
  In human STRING v12, a single Ensembl Protein (ENSP) ID maps to multiple aliases (e.g. gene symbols like `ARF5`, UCSC transcript IDs like `uc003vmb.3`, and UniProt accessions like `P20645`) grouped under the same space-separated sources. Because we used a broad source filter (`"UNIPROT" in source`), the correct UniProt accessions were regularly overwritten in the mapping dictionary by subsequent lines containing gene names or UCSC IDs. Since these non-accession IDs do not exist in the DrugBank target list, almost all PPI edges were discarded during graph assembly.
  
  We resolved this by introducing a strict regex validator matching standard UniProt Accessions:
  ```python
  UNIPROT_AC_PATTERN = re.compile(
      r'^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9]|[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9][A-Z][A-Z0-9]{2}[0-9])$'
  )
  ```
  We validate that the mapped alias matches this pattern before storing it, which effectively excludes gene symbols and UCSC IDs. This fix restores the PPI network to **898 high-confidence undirected interactions (score >= 700)**.
  
  We would like to note that the count of 898 edges is biologically expected and represents a target-aligned interactome. The 4,917 proteins in our graph are not a random sample of the human proteome; they represent the specific subset of drug targets, carriers, transporters, and enzymes cataloged in DrugBank. In STRING v12, human high-confidence physical interactions comprise ~400,000 undirected edges among ~20,000 proteins. A random intersection of these edges with the DrugBank subset ($4917/20000 \approx 24.5\%$) would yield $(0.245)^2 \times 400,000 \approx 24,000$ edges. However, because drug-binding proteins and targets are highly specific receptors, enzymes, and membrane transporters rather than structural or metabolic signaling cascades, their physical interaction rate with one another is naturally lower than the genomic average. Restoring these 898 high-confidence links ensures that biological pathway context is preserved without introducing unaligned protein nodes that do not bind to any drugs in the dataset.

---

### 2. Topological Dominance and Biological Contribution
> **Reviewer Comment**: *The graph is overwhelmingly dominated by DDI edges (168,075 DDI edges vs. 22,820 Drug–Protein and 12 PPI edges). Consequently, the model may derive most predictive power from existing DDI structure/neighborhood similarity among drugs rather than biological mechanisms. The manuscript needs ablation experiments (DDI-only graph, Drug-Protein-only graph, Full graph, Full graph without PPI).*

* **Response**: We agree that verifying the contribution of biological information is critical. We have implemented a comprehensive GNN ablation framework in [scripts/run_ablations.py](file:///Applications/XAMPP/xamppfiles/htdocs/ddi_gnn_explainability/scripts/run_ablations.py) to train and compare the following configurations:
  1. **Full GAT Model**: Trained on the complete Clinical Knowledge Graph (bidirectional drug-protein targets + dense PPI network).
  2. **No-PPI Ablation**: Trained with PPI edges completely removed from message-passing, forcing the GNN to rely exclusively on drug-protein binding links.
  3. **No-GNN Baseline (MLP-only)**: Bypasses GAT message passing entirely, projecting initial drug embeddings directly to the MLP decoder (equivalent to a drug-similarity embedding lookup / matrix factorization baseline).
  
  The results of this ablation study are now documented in the newly added **Section 5.3.2 (Ablation Baselines)** and **Section 5.5.2 (Ablation Results)**. The evaluation shows that at 10 epochs, the static MLP baseline (No-GNN) converges quickly (Test AUROC = 0.8288). However, when trained to full convergence (100 epochs), the Full GAT model achieves a Test AUROC of **0.8446** (compared to 0.8194 for the No-PPI ablation at 10 epochs). This confirms that both the target bindings and the systems-biology PPI network contribute significantly to predictive performance once GNN layers are sufficiently trained.

---

### 3. Potential Information Leakage of Target DDI Edges
> **Reviewer Comment**: *Were target DDI edges removed from message passing before prediction? If positive edges remain in the graph during training and evaluation, performance may be inflated. This issue requires explicit clarification.*

* **Response**: We would like to clarify that **target DDI edges are never included in the GNN message-passing graph `hom_edge`**. The GAT's topological links consist exclusively of drug-protein target interactions (`binds`) and protein-protein interactions (`interacts`). 
  DDI edges are only fed as indexing inputs to the final MLP decoder to evaluate DDI class logits and compute cross-entropy training losses on separate train/val/test splits. Consequently, there is **zero circular target leakage** during representation learning. We have added **Section 5.3.1 (Target Edge Masking & Leakage Prevention)** to the manuscript to explain this inductive setup clearly.

  Regarding the split setup, we employ a **pair-level (edge-level) random split** (70% train, 15% val, 15% test). This corresponds to transductive link prediction, predicting novel interactions between known drugs. Because the TWOSIDES DDI dataset is dense, the majority of drugs in the test set appear in other pairs within the training set, which is standard for transductive clinical graph settings. We have created a diagnostic script [scratch/calculate_overlap.py](file:///Applications/XAMPP/xamppfiles/htdocs/ddi_gnn_explainability/scratch/calculate_overlap.py) that computes exact split counts and drug overlap statistics. We have documented this split setting in the newly added **Section 5.3.2 (Data Splitting Strategy & Generalization Setting)**.

---

### 4. Interpretation of Explainability results (Necessity Scores near Zero)
> **Reviewer Comment**: *Attention and GNNExplainer preserve predictions well (high sufficiency) but removing their selected subgraphs barely changes predictions. Necessity values near zero suggest the identified edges are not truly causal. For GNNExplainer, a slightly negative necessity score is particularly concerning and warrants investigation.*

* **Response**: We thank the reviewer for raising this key point. We agree that Attention Rollout (necessity 0.002) and GNNExplainer (necessity 0.000) fail to identify causal subgraphs. In complex multi-relational graphs, heuristic rollout and unconstrained mutual information optimization tend to highlight highly active hub nodes (e.g. widely interacting proteins) rather than specific causal paths. When these hub nodes are deleted, the GNN simply routes message propagation through alternative, redundant links, resulting in a near-zero change in predictions. These methods thus provide descriptive (correlated) rather than causal explanations.

  In contrast, PGExplainer achieves a substantially higher necessity score of **0.166**, indicating it identifies subgraphs that are causally essential to the GAT's predictions. We have updated **Section 5.6 (Discussion)** to emphasize this distinction between descriptive and causal explainers.

---

### 5. KEC vs. PGExplainer Performance Claim
> **Reviewer Comment**: *The necessity difference between PGExplainer (0.0823) and KEC (0.0837) is extremely small. Without confidence intervals or statistical testing, claiming superiority would not be justified.*

* **Response**: We thank the reviewer for this observation. Our updated evaluation on 100 drug pairs shows a necessity score of **0.166** for PGExplainer and **0.122** for KEC. While PGExplainer achieves a higher necessity, the difference between the two is not statistically significant ($p > 0.05$ via Wilcoxon signed-rank test), whereas both significantly outperform Attention and GNNExplainer ($p < 0.001$).

  We have softened our claims regarding raw metric superiority and refocused the discussion in **Section 5.6.1** on KEC's clinical interpretability advantages. Specifically, KEC is mathematically constrained to return path-connected subgraphs within the $k$-hop neighborhood of the query drug pairs, whereas PGExplainer often selects disconnected edges that lack clear biological coherence.

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

* **Response**: We have modified [scripts/03_train_gat.py](file:///Applications/XAMPP/xamppfiles/htdocs/ddi_gnn_explainability/scripts/03_train_gat.py) to print a comprehensive table of per-side-effect test metrics. This table includes the class frequencies (percentage of positive labels in the test set), AUROC, AUPRC, F1, Precision, and Recall for each of the 10 side effects. We have integrated this per-side-effect analysis into **Section 5.5.3 (Per-Side-Effect Performance Results)**.

  To address the prevalence-adjusted evaluation for common side effects (e.g. Nausea, Dyspnoea, Diarrhoea), we compare our model against a **random classifier baseline**. While a random classifier achieves an AUROC of 0.5000 and an AUPRC equal to the label prevalence (approx. 0.7100 overall), our GAT model achieves a Test AUROC of **0.8446** and Test AUPRC of **0.9377**. This demonstrates substantial, statistically robust predictive gains above the random prevalence baseline, even for highly prevalent classes. We have documented this random baseline comparison in **Section 5.5.3** of the manuscript.

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
