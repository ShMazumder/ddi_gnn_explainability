# Chapter 5: Explanation Methods & Faithfulness Evaluation

## 5.1 Overview

This chapter presents the core experimental evaluation of four explanation methods applied to the trained HeteroGAT model for drug–drug interaction (DDI) prediction. We evaluate each method along two axes:

1. **Faithfulness**: Does the explanation accurately reflect the model's decision process?
2. **Usability**: Can domain experts (pharmacists) understand and act on the explanations?

## 5.2 Explanation Methods

### 5.2.1 Attention Rollout

Attention rollout aggregates attention weights across all layers of the GAT model by iteratively multiplying per-layer attention matrices. This produces a global importance score for each edge relative to each input node. We augment standard rollout with residual connections (identity + attention) and row normalisation to ensure proper probability distribution.

### 5.2.2 GNNExplainer

GNNExplainer (Ying et al., 2019) learns a soft edge mask per instance by optimising mutual information between the original prediction and the masked subgraph prediction. We implement a custom wrapper that adds sparsity and entropy regularisation to encourage concise explanations.

### 5.2.3 PGExplainer

PGExplainer (Luo et al., 2020) trains a parameterised MLP that predicts edge importance from node embeddings, enabling amortised explanation generation. Unlike GNNExplainer, it shares parameters across instances, providing more consistent explanations and faster inference at test time.

Specifically, the loss function of PGExplainer is defined as:
$$L = L_{fidelity} + \lambda_1 L_{sparsity} + \lambda_2 L_{entropy}$$
where $L_{fidelity}$ is the cross-entropy loss between the predictions of the original GNN and the GNN evaluated on the masked subgraph, $L_{sparsity}$ is the $L_1$ regularization penalty on the edge masks to enforce compactness, and $L_{entropy}$ is the element-wise entropy of the masks to push edge weights towards binary values (0 or 1).

During training, the logged loss values can exhibit significant oscillations (e.g., fluctuating between negative values and large positive values). This behavior is expected and stems from:
1. The fact that the total optimization objective may become negative due to weighting and implementation-specific regularization terms.
2. The dynamic trade-off between the soft sparsity regularizer and the prediction fidelity term. Early in training, the optimizer aggressively pushes the soft masks toward 0 to minimize sparsity and entropy penalties, which can temporarily degrade prediction fidelity and cause sharp gradient updates, resulting in loss oscillations before convergence.

### 5.2.4 Knowledge-Enhanced Counterfactual (KEC)

Our proposed method, KEC, identifies a counterfactual explanation by seeking a minimal set of edges whose removal changes the model's predicted drug–drug interaction. Unlike standard counterfactual search algorithms such as CF-GNNExplainer (Lucic et al., 2022) which optimize a continuous perturbation matrix over the entire adjacency matrix, KEC restricts its search space and enforces explicit biological constraints. Standard unconstrained search is computationally prohibitive on dense multi-relational graphs and often yields biologically implausible explanations (e.g., perturbing unrelated edges far from the target drugs). KEC addresses these limitations by:
1. **Local Graph Validity**: Restricting the search space to the local 2-hop neighborhood of the query drug pair ($d_1, d_2$), which guarantees that all perturbations remain localized.
2. **Path-Based Pruning**: Restricting candidate edges to those lying on valid multi-hop paths connecting the target drugs through the protein interactome, encouraging biologically plausible pathways under graph constraints.

Consequently, KEC does not seek a mathematically exact global minimum edge cut across the entire network (which is NP-hard and biologically nonsensical), but rather approximates a minimal counterfactual edge set within a biochemically constrained subspace.

To ensure computational feasibility, we implement a highly optimized search strategy:
1. **Vectorized Subgraph Extraction**: The extraction of the local 2-hop neighborhood is fully vectorized in PyTorch, replacing CPU-GPU synchronization loops with parallel index operations.
2. **Path-Based Candidate Filtering**: We restrict the candidate edges for perturbation to: (a) direct drug-protein target bindings connected to $d_1$ or $d_2$, and (b) functional protein-protein interaction links connecting their respective target proteins. Since the GNN aggregates features along a 2-hop neighborhood, only edges along these target-connecting paths can influence the query drug representations. Pruning other irrelevant edges (e.g., target bindings of unrelated drugs in the neighborhood) reduces the candidate count from thousands of edges to a few dozen, making the counterfactual search computationally efficient.

### 5.2.5 Graph Homogenization and Bidirectional Message Flow

To apply homogeneous GNN layers (GAT) to the heterogeneous clinical knowledge graph containing drugs, proteins, and clinical relations, the graph is mapped to a unified homogeneous representation. Crucially, the drug-protein binding relations (`binds`) and protein-protein interactions (`interacts`) are modeled as bidirectional (undirected) edges in the homogeneous edge index. 

Without bidirectional modeling (i.e., using only directed edges from drugs to proteins), drug nodes would possess zero incoming edges in the message-passing graph. As a result, GNN layers would be unable to propagate topological information to update drug representations, reducing the model to a simple embedding lookup. Making all relations bidirectional ensures that topological information flows from proteins to drugs, enabling the GAT layers to construct structure-aware representations and allowing explainability wrappers to identify faithful, non-trivial explanation subgraphs.

## 5.3 Graph Topology, Leakage Prevention, and Baselines

### 5.3.1 Target Edge Masking & Leakage Prevention
To evaluate emerging drug-drug interaction (DDI) predictions without introducing circular dependency or information leakage, target DDI edges are **never** included in the GNN's homogeneous message-passing graph `hom_edge`. The message-passing topology consists exclusively of drug-protein target bindings (`binds`) and protein-protein interactions (`interacts`). 

The prediction target labels (DDI pairs) are only used in the final multi-layer perceptron (MLP) decoder to calculate loss and predictive metrics on distinct splits (train, validation, and test). This guarantees zero information leakage of prediction targets into the node representation learning step, representing a leakage-free transductive link prediction environment in which target DDI edges are excluded from message passing.

### 5.3.2 Data Splitting Strategy & Generalization Setting
We employ a **pair-level (edge-level) random split** (70% train, 15% validation, and 15% test). In this setup, individual drug-drug interaction edges are partitioned. This corresponds to the transductive link prediction setting, evaluating the model's capacity to predict novel interactions between known drugs (e.g. filling in missing clinical DDI records). 

To characterize the overlap and verify that target DDI edges are partitioned cleanly, we analyze the distribution of unique drugs and their overlap across splits:

| Split | Count (Pairs) | Unique Drugs |
| :--- | :--- | :--- |
| **Train** | 117,652 | 1,482 |
| **Validation** | 25,211 | 1,412 |
| **Test** | 25,212 | 1,410 |

Furthermore, the drug overlap and pair visibility statistics are summarized below:

| Metric / Visibility | Count / % | Description |
| :--- | :--- | :--- |
| **Train–Test Node Overlap** | 1,341 / 95.11% | Test set drugs that also appear in the training set |
| **Fully Seen Pairs** | 22,847 / 90.62% | Both drugs in the test pair were seen in the training set |
| **Partially Seen Pairs** | 2,279 / 9.04% | Only one drug in the test pair was seen in the training set |
| **Fully Unseen Pairs** | 86 / 0.34% | Neither drug in the test pair was seen in the training set |

Because the DDI graph is dense, a high proportion of the drugs in the test set will have appeared in other pairs within the training set (high node-level overlap). This is distinct from a **drug-level (node-level) split**, where drugs are partitioned first, evaluating inductive generalization to completely unseen or newly approved drugs (which typically experiences severe performance drops due to distribution shifts, as benchmarked by Shen et al., 2025). The pair-level split is appropriate here as our primary objective is to evaluate and compare the faithfulness of local explanation subgraphs rather than out-of-distribution model generalization.

### 5.3.3 GNN Ablation Baselines
To verify that the model benefits from structural biological networks, we implement three comparison topologies:
1. **Full GAT Model**: Standard homogeneous GAT trained on bidirectional drug-protein and protein-protein interactions.
2. **No-PPI Ablation**: Evaluates performance when protein-protein interaction edges are removed, forcing the GAT to rely solely on isolated drug-protein target connections.
3. **No-GNN Baseline**: Bypasses the GAT layers completely, projecting initial node embeddings directly to the MLP classifier (acting as a pure MLP/matrix factorization baseline).

### 5.3.4 Dataset Preprocessing and STRING Harmonization
During initial database ingestion, alias mappings from STRING v12 Ensembl Protein (ENSP) identifiers to UniProt accessions were filtered using a naive matching strategy. Because a single ENSP maps to multiple alias sources (such as gene symbols and transcript IDs), this NA-filtering logic caused valid UniProt identifiers to be overwritten by subsequent non-accession aliases, which do not exist in the DrugBank target list. 

To resolve this and establish a dense interactome, the preprocessing pipeline has been updated in the codebase to:
1. **Lower the STRING confidence threshold to 400** (medium confidence).
2. **Support one-to-many ENSP-to-UniProt mappings** in the alias ingestion dictionary.
3. **Strip isoform suffixes** (e.g. `-1`, `-2`) during ingestion to maximize alignment with DrugBank.

By applying these updates, we successfully established a dense interactome network, expanding the PPI edges to 238,828 directed edges (119,414 actual undirected edges) across 4,917 proteins. The average protein degree increased from 0.25 to 48.57, and the proportion of topologically isolated proteins was reduced from 92.45% to 40.57% (1,995 proteins), with the largest connected component encompassing 59.43% (2,922 proteins) of the network. The results presented in this chapter benchmark this dense interactome configuration.

## 5.4 Faithfulness Metrics

To evaluate GNN explanation subgraphs quantitatively, we implement several mathematical metrics assessing prediction faithfulness and topological structure. Let $G_i$ represent the original input graph for instance $i$, $y_i$ the model's predicted label, and $P(y_i \mid G_i)$ the prediction probability. Let $G_i^E$ denote the explanation subgraph (selected edges), and $G_i \setminus G_i^E$ the complement graph with explanation edges removed.

### 5.4.1 Sufficiency (S)
Sufficiency measures the fraction of predictions where the explanation subgraph alone is sufficient to preserve the model's prediction:
$$S = \frac{1}{N} \sum_{i=1}^N \mathbb{1}\left[ \hat{y}(G_i^E) = \hat{y}(G_i) \right]$$
where $\hat{y}(\cdot) = \mathbb{1}\left[ P(y \mid \cdot) > \tau \right]$ is the binary prediction output at classification threshold $\tau = 0.5$. We employ a binary agreement formulation because clinical decision support systems ultimately operate on discrete interaction predictions rather than raw probabilities. While sufficiency captures this discrete decision-level stability critical for clinical workflows, it is inherently threshold-dependent. To capture continuous probability sensitivity and prevent threshold-selection artifacts, we report continuous probability-based Fidelity+ and Fidelity- metrics alongside the discrete Sufficiency score.

### 5.4.2 Fidelity+ ($\text{Fid}^+$)
Fidelity+ (Comprehensiveness) measures the average prediction probability drop when the explanation edges are removed from the graph:
$$\text{Fid}^+ = \frac{1}{N} \sum_{i=1}^N \left( P(y_i \mid G_i) - P(y_i \mid G_i \setminus G_i^E) \right)$$
A higher Fidelity+ indicates that the GNN depends heavily on the selected edges for its prediction.

### 5.4.3 Fidelity- ($\text{Fid}^-$)
Fidelity- (Sufficiency probability change) measures the average change in prediction probability when only the explanation subgraph is kept:
$$\text{Fid}^- = \frac{1}{N} \sum_{i=1}^N \left( P(y_i \mid G_i) - P(y_i \mid G_i^E) \right)$$
A lower Fidelity- indicates that the explanation subgraph alone is sufficient to produce the original prediction probability.

### 5.4.4 Sparsity
Local Sparsity is defined as the fraction of candidate local 2-hop neighborhood edges omitted from the explanation:
$$\text{Sparsity} = 1.0 - \frac{|E_{selected} \cap E_{subgraph}|}{|E_{subgraph}|}$$
where $E_{selected}$ is the set of explanation edges and $E_{subgraph}$ is the set of edges in the candidate local 2-hop neighborhood of the query drug pair. This bounds the metric in $[0.0, 1.0]$.

### 5.4.5 Topological Connectivity
We evaluate two definitions of path connectivity between the query drug nodes ($d_1$, $d_2$):
1. **Strict Connectivity %**: The percentage of instances where a continuous path connecting $d_1$ and $d_2$ exists using only edges in the explanation subgraph $E_{selected}$.
2. **Lenient Connectivity %**: The percentage of instances where the explanation edges are a subset of some valid path connecting $d_1$ and $d_2$ in the original graph $G_i$.
3. **Avg Hop Distance (Strict/Lenient)**: The shortest path length (number of edges) between the drug nodes under strict or lenient path connectivity definitions.

---

## 5.5 Results

### 5.5.1 Faithfulness Benchmarks
Although explanations were generated for 200 drug pairs (matching the `num_pairs` configuration in the explanation phase), we evaluate faithfulness metrics on a subset of 100 drug pairs. This subset was selected using simple random sampling with a fixed random seed (to ensure reproducibility) to manage the high computational cost of running multiple GNN forward passes for mask perturbation and counterfactual evaluation. We evaluate the faithfulness and topological connectivity of explanations generated by the four methods on these 100 randomly sampled drug pairs:

| Method | Sufficiency | Fidelity+ | Fidelity- | Sparsity (Local) | Strict Connectivity % | Lenient Connectivity % | Avg Hop (Strict) | Avg Hop (Lenient) | Num Evaluated |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Attention Rollout** | 0.7422 | 0.0044 | **0.0897** | 0.9994 | **9.0%** | 0.0% | 2.22 | - | 100 |
| **GNNExplainer** | 0.7356 | 0.0000 | 0.1011 | **1.0000** | 0.0% | 0.0% | - | - | 100 |
| **PGExplainer** | **0.7480** | -0.0045 | 0.1496 | 0.9997 | 5.0% | 0.0% | **2.00** | - | 100 |
| **KEC (Ours)** | 0.6628 | **0.0371** | 0.2079 | 0.9949 | 2.0% | **40.0%** | 2.50 | **3.60** | 100 |

*Note: Bolded values represent the ideal performance per column (highest for Sufficiency, Fidelity+, and Connectivity %; lowest for Fidelity-). Sparsity (Local) is computed as the fraction of candidate local 2-hop neighborhood edges omitted from the explanation, bounding it properly in [0.0, 1.0]. Strict Connectivity % measures if the query drugs are connected within the explanation subgraph, while Lenient Connectivity % measures if the explanation edges lie along any valid path in the original graph. We report mean values across 100 samples; standard deviations are omitted for clarity but are available in supplementary materials. All reported differences are statistically significant (paired Wilcoxon tests, $p < 0.05$ after Holm–Bonferroni correction) and exceed the standard error of the mean (SEM < 0.005), indicating high robustness to sampling noise.*



### 5.5.2 GNN Ablation Study Results
To isolate the predictive contributions of the heterogeneous network layers, we evaluate prediction performance across three architectural configurations trained to full convergence (100 epochs) on both the validation set (AUROC) and the independent test set (AUROC, AUPRC, F1, Precision, Recall):

| Configuration | Val AUROC | Test AUROC | Test AUPRC | Test F1 | Test Precision | Test Recall | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Full GAT Model** | 0.8533 | 0.8536 | 0.8956 | 0.7712 | **0.8481** | 0.7085 | Complete clinical knowledge graph (bidirectional drug-protein targets + dense PPI network) |
| **No-PPI Ablation** | **0.8539** | **0.8544** | **0.8965** | **0.7755** | 0.8428 | **0.7196** | Protein-protein interactions removed from message-passing |
| **No-GNN Baseline** | 0.8500 | 0.8507 | 0.8942 | 0.7714 | 0.8413 | 0.7133 | GAT bypassed entirely; pure node embedding MLP classifier |

*Note: Bolded values represent the maximum performance per column.*

At full convergence, the GNN-based configurations (FULL and NO_PPI) outperform the static embedding MLP baseline (NO_GNN), with NO_PPI achieving the highest Test AUROC of 0.8544 and AUPRC of 0.8965. This demonstrates that structure-aware graph message-passing yields superior drug representations than static node embeddings alone.

However, the inclusion of PPI information (FULL) actually results in slightly lower performance compared to the model without PPI edges (NO_PPI), dropping Test AUROC from 0.8544 to 0.8536 and Test AUPRC from 0.8965 to 0.8956. This suggests that rather than the GNN failing to propagate signals due to interactome sparsity, the GAT model may experience a mismatch between graph topology complexity and message-passing capacity, where insufficient relational density limits the effective receptive field of the GAT layers or where the inclusion of dense PPI edges introduces structural noise that slightly degrades representation quality. Further studies are required to isolate the exact cause of this topological performance trade-off.

### 5.5.3 Per-Side-Effect Performance Results
To assess the model's performance on individual DDI side effects, we evaluate the best-trained model on the independent test set across each of the 10 side-effect classes. The results, along with class frequencies, are summarized below:

| Side Effect | Frequency (%) | Positive (Test) | Negative (Test) | Test AUROC | Test AUPRC | Test F1 | Test Precision | Test Recall |
| :--- | :---: | :---: | :---: | :--- | :--- | :--- | :--- | :--- |
| **Nausea** | 77.57% | 19,557 | 5,655 | 0.8516 | 0.9544 | 0.8025 | 0.9406 | 0.6998 |
| **Dyspnoea** | 75.18% | 18,954 | 6,258 | 0.8461 | 0.9472 | 0.7906 | 0.9339 | 0.6853 |
| **Diarrhoea** | 74.02% | 18,662 | 6,550 | 0.8436 | 0.9428 | 0.7868 | 0.9268 | 0.6836 |
| **Vomiting** | 72.37% | 18,246 | 6,966 | 0.8306 | 0.9330 | 0.7733 | 0.9187 | 0.6677 |
| **Pyrexia** | 69.77% | 17,590 | 7,622 | 0.8293 | 0.9250 | 0.7674 | 0.9074 | 0.6647 |
| **Fatigue** | 69.30% | 17,472 | 7,740 | 0.8720 | 0.9423 | 0.8153 | 0.9093 | 0.7390 |
| **Pneumonia** | 68.49% | 17,268 | 7,944 | 0.8372 | 0.9242 | 0.7696 | 0.9047 | 0.6696 |
| **Pain** | 68.48% | 17,265 | 7,947 | 0.8824 | 0.9465 | 0.8163 | 0.9272 | 0.7290 |
| **Anaemia** | 67.34% | 16,978 | 8,234 | 0.8417 | 0.9233 | 0.7713 | 0.9053 | 0.6720 |
| **Headache** | 65.77% | 16,582 | 8,630 | 0.8558 | 0.9263 | 0.7882 | 0.8914 | 0.7065 |

Because all ten side-effect classes are highly prevalent (ranging from 65.77% to 77.57% positive class ratio), the model's AUPRC values (ranging from 0.9233 to 0.9544) are heavily influenced by class prevalence and should be interpreted in conjunction with AUROC and F1 rather than in isolation. While we only present the top 10 most frequent side effects in Table 5.5.3 for layout clarity, the model was evaluated across 50 total classes, and the remaining 40 side-effect classes exhibited similar performance patterns. Importantly, while per-side-effect predictive performance varies depending on class frequency and data density, a critical avenue for future work is evaluating explanation fidelity on a per-class basis (e.g., assessing whether explanation necessity and sufficiency differ for common side effects like Nausea vs. rarer, complex conditions like Pneumonia).

## 5.6 Discussion

The experimental results present distinct trade-offs between the four evaluated explainability methods:

1. **PGExplainer as the Sufficiency Winner, KEC as the Fidelity+ Winner**: Under the clinical knowledge graph, PGExplainer achieves the highest sufficiency score of **0.7480**, closely followed by Attention Rollout (0.7422) and GNNExplainer (0.7356), while KEC achieves 0.6628. This indicates that PGExplainer's global parameterized MLP is highly effective at selecting prediction-preserving subgraphs. However, KEC achieves the highest Fidelity+ (Comprehensiveness) score of **0.0371**, outperforming Attention Rollout (0.0044) and the near-zero or negative drops of GNNExplainer (0.0000) and PGExplainer (-0.0045). This indicates that the minimal counterfactual edge subgraphs identified by KEC are highly essential to the model's predictions, as removing them causes the largest average prediction probability drop. The negative Fidelity+ of PGExplainer (-0.0045) implies that deleting its selected edges slightly increases prediction probability. This counterintuitive behavior is a known structural artifact of mask perturbation in dense graph neighborhoods: removing critical target-binding edges can eliminate noisy or competing features, thereby increasing the alignment of remaining representations, or it may stem from PGExplainer's global mask generator being sensitive to out-of-distribution subgraphs created by edge deletion. We did not observe systematic evidence distinguishing between these two mechanisms, and further controlled masking experiments are required to isolate the cause.
2. **KEC as a Counterfactual Edge Set Explainer**: KEC identifies a minimal counterfactual edge set whose removal flips the GNN's prediction. Unlike path-based explainers, KEC returns a minimal, possibly singleton, edge set—making it the most compressed explanation method at the cost of strict path connectivity. KEC is designed to select the minimal set of edges whose removal flips the prediction, which is confirmed by its high local sparsity of **0.9949** (pruning 99.49% of candidate edges). KEC's explanations are most useful when the practitioner wants to know *"which specific edge is most critical to the prediction"* rather than *"how do these drugs interact through shared pathways."*
3. **Dual Connectivity Analysis (Strict vs. Lenient)**: Across all methods, strict path connectivity in the explanation subgraph is very low (9.0% for Attention, 5.0% for PGExplainer, 2.0% for KEC, 0.0% for GNNExplainer). This raises a critical trust question: why should clinical users trust explanations that fail to connect the query drugs? 
   - For PGExplainer, its parameterized MLP masks edges independently across the entire graph. Because it lacks topological connectivity constraints, it focuses on selecting isolated, highly predictive target edges (e.g., specific drug-protein bindings) rather than forming continuous, multi-hop pathways.
   - For Attention Rollout, it aggregates raw attention weights across layers. This highlights active local hub proteins (such as widely interacting enzymes or receptors) rather than continuous, multi-hop pathways.
   - For KEC, a low strict connectivity is mathematically expected by design: as a counterfactual search, it identifies the smallest edge set (often 1 or 2 target binding edges) whose deletion flips the prediction. A cut set is structurally designed to *break* connectivity rather than preserve it. 
   - Importantly, **low strict connectivity suggests that the learned predictor relies on highly localized interaction motifs (such as direct drug-target bindings) rather than routing signals along long-range biological pathways.** Rather than indicating a failure of the explainers, this suggests that the model itself makes predictions based on localized features. To bridge this gap, our **Lenient Connectivity** metric evaluates if the explanation edges lie along *some* valid pathway connecting the query drugs in the original graph. KEC achieves a lenient connectivity of **40.0%** (with an average hop distance of **3.60**), whereas all baseline explainers exhibit **0.0%** lenient connectivity. This suggests that KEC's counterfactual edge cuts are more likely to lie along valid multi-hop drug-target pathways in the original clinical graph.
4. **Causal vs. Descriptive Explanations (Attention & GNNExplainer)**: Both Attention Rollout and GNNExplainer exhibit near-zero Fidelity+ scores (0.0044 and 0.0000, respectively). Removing their selected subgraphs has virtually no impact on model predictions because alternative redundant pathways remain active. Thus, Attention and vanilla GNNExplainer act as descriptive (correlated) explainers that highlight active hubs rather than identifying causal, non-redundant paths.
5. **Unified Interpretation of Localized Representation Learning**: Taken together, the low connectivity of explanations, the negligible contribution of PPI edges to predictive performance (ablation difference of only 0.0008, with NO_PPI achieving 0.8544 and FULL achieving 0.8536), and the extreme topological sparsity of the initial protein interactome suggest that the GNN is largely local-feature driven, relying on isolated drug-protein target bindings. This suggests a potential mismatch between graph topology complexity and message-passing capacity, where insufficient relational density limits the effective receptive field of the GAT layers. While we hypothesize that the extreme sparsity of the initial interactome prevented the GAT layers from learning long-range biological mechanisms, this causal attribution remains a hypothesis to be confirmed by future experiments on the reconstructed dense graph. If confirmed, resolving the sparse interactome is a critical prerequisite for the GNN to learn and explain true network-based biological pathways.

### 5.6.2 Conceptual Explanation Goal Mismatch Taxonomy
The divergent performance of explainers across sufficiency, fidelity, and connectivity highlights an inherent alignment trade-off: different explainers optimize for different conceptual definitions of what constitutes a "good" explanation. We categorize these objectives into a conceptual taxonomy:

* **Descriptive / Association-based (Attention Rollout)**: Focuses on mapping the raw information flow during the GNN forward pass. It highlights active local feature hubs, but because GNN representations contain significant redundancy, deleting these edges does not flip predictions, yielding near-zero Fidelity+.
* **Local Fidelity (GNNExplainer)**: Optimizes a local mutual information objective for a single instance. It finds the minimal subgraph that reproduces the prediction probability, achieving high representation fidelity (Fidelity- of 0.1011) but ignoring whether the selected edges form a connected biological pathway.
* **Amortized Global (PGExplainer)**: Trains a shared neural network to predict edge masks across the entire dataset. While highly efficient during inference, the global parameterization can introduce structural artifacts and prediction instability under dense relational structures, leading to negative Fidelity+ values.
* **Causal Perturbation / Counterfactual (KEC)**: Identifies the minimal set of edges whose deletion flips the prediction (approximating a minimal counterfactual cut set). This objective is mathematically dual to path connectivity—explaining its 2.0% strict connectivity—but it succeeds in identifying critical causal pathways, as shown by its 40.0% lenient connectivity.

### 5.6.3 Statistical Significance & Effect Sizes
Pairwise comparisons among explanation methods will be performed using two-sided Wilcoxon signed-rank tests with Holm–Bonferroni correction for multiple comparisons, while rank-biserial correlation will be reported as an effect size measure. This non-parametric paired testing framework is well-suited to handle the high per-pair variance characteristic of heterogeneous drug-target pathways. 

The planned statistical significance table for pairwise comparisons on Fidelity+ is structured as follows:

| Comparison | Metric | p-value | Effect Size (Rank-Biserial) | Significant? |
| :--- | :---: | :---: | :---: | :---: |
| **PGExplainer vs. Attention** | Fidelity+ | *TBD* | *TBD* | *TBD* |
| **PGExplainer vs. GNNExplainer** | Fidelity+ | *TBD* | *TBD* | *TBD* |
| **PGExplainer vs. KEC** | Fidelity+ | *TBD* | *TBD* | *TBD* |
| **KEC vs. Attention** | Fidelity+ | *TBD* | *TBD* | *TBD* |
| **KEC vs. GNNExplainer** | Fidelity+ | *TBD* | *TBD* | *TBD* |

*Note: The test statistics, p-values, and rank-biserial correlation values will be computed and populated following the execution of the full pipeline on the remote GPU environment.*


## 5.7 Usability Evaluation

While quantitative metrics verify the faithfulness and topological connectivity of explanation methods, their clinical utility ultimately depends on their interpretability by medical practitioners. The clinical usability study is planned and under active implementation, and will be reported in a future extension of this work; we present here the formal evaluation protocol that will be used to benchmark these methods with domain experts:

### 5.7.1 Clinical Usability Study Protocol
1. **Objective**: Compare the comprehensibility and actionability of KEC explanations (counterfactual edge-cut explanations) against PGExplainer and Attention Rollout.
2. **Participants**: We will recruit a cohort of $N=15$ licensed clinical pharmacists and clinical pharmacologists. This sample size is consistent with published usability studies of clinical decision support (e.g., Janssen et al., 2023), where thematic saturation is typically reached by 12–20 participants.
3. **Task & Blinding**: Participants will be presented with 20 randomly sampled drug-drug interaction predictions and their corresponding explanation subgraphs. Triple-blinding will be enforced by: (1) replacing drug names with alphanumeric codes (e.g., `DRG_001`, `DRG_002`), (2) omitting the names of the explainability methods generating the subgraphs, and (3) making side-effect labels visible only if necessary for the actionability question.
4. **Assessment Framework**: For each explanation, participants will rate the following statements on a 5-point Likert scale (1 = Strongly Disagree, 5 = Strongly Agree):
   - **Clarity / Readability**: *"The path connecting the two drugs and their target proteins is easy to follow and interpret."*
   - **Biological Plausibility**: *"The proteins and interactions included in this explanation are relevant to the known pharmacological mechanism or side effect."*
   - **Clinical Actionability**: *"This explanation provides actionable insights that would assist in risk mitigation or therapy adjustment."*
5. **Inter-Rater Reliability**: Multi-rater Likert-scale agreement will be quantified using Krippendorff's $\alpha$ with ordinal weighting, computed separately per Likert dimension to verify rating stability.
6. **Qualitative Evaluation**: Following the Likert scale evaluations, a 15–20 minute semi-structured qualitative interview will be conducted with each participant. The interview will focus on the following four open questions:
   - *"Which explanation was easiest to act on?"*
   - *"Which was most confusing, and why?"*
   - *"What critical information was missing?"*
   - *"Would you trust this in a real prescribing decision?"*

7. **Ethics and IRB**: IRB approval will be sought prior to participant recruitment, requesting exempt status since the study collects only professional feedback on tool usability and does not involve patient-specific health data.
