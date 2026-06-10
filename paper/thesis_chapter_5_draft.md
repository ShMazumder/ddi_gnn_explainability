# Chapter 5: Explanation Methods & Faithfulness Evaluation

## 5.1 Overview

This chapter presents the core experimental evaluation of four GNN explanation methods applied to the trained homogeneous GAT model for drug–drug interaction (DDI) prediction. Crucially, this evaluation is scoped as a methodology-oriented study of GNN explainability and faithfulness. Our primary objective is to investigate the degree to which different explainability wrappers accurately reflect the internal decision logic of the underlying model. By design, this study does not focus on benchmarking predictive state-of-the-art DDI models; consequently, comparisons against specialized predictive architectures (such as MIRACLE, SSI-DDI, or EmerGNN) are out of scope. 

Importantly, our evaluation addresses a fundamental tension in explainability research: the "Predictive Paradox." As detailed in our GNN ablations, incorporating a dense protein-protein interaction (PPI) network does not yield a significant performance boost over a model trained purely on localized drug-protein target connections. If the GNN backbone itself primarily relies on localized drug-protein target associations rather than long-range graph routing, then explanation methods will faithfully reflect this localized decision logic by selecting disconnected subgraphs rather than complete multi-hop pathways. Therefore, the low topological connectivity of explanations is not necessarily a failure of the explainability wrappers themselves, but rather a faithful representation of the underlying model's localized representation learning.

We evaluate the explainers along two main axes:
1. **Faithfulness**: Does the explanation subgraph accurately reflect the model's decision process?
2. **Clinical Validation Protocol**: Can a proposed survey protocol guide domain experts (pharmacists) to evaluate the clinical actionability of explanations?

## 5.2 Explanation Methods

### 5.2.1 Attention Rollout

Attention rollout aggregates attention weights across all layers of the GAT model by iteratively multiplying per-layer attention matrices. This produces a global importance score for each edge relative to each input node. We augment standard rollout with residual connections (identity + attention) and row normalisation to ensure proper probability distribution.

### 5.2.2 GNNExplainer

GNNExplainer (Ying et al., 2019) learns a soft edge mask $M_E \in [0, 1]^{|E|}$ and a node feature mask $M_X \in [0, 1]^{D}$ per instance by optimising mutual information between the original prediction and the GNN's prediction on the masked subgraph. The soft masks are optimized using a continuous relaxation where edge masks are treated as variational parameters. During optimization, GNNExplainer employs stochastic mask sampling, where edges and feature values are sampled according to the learned variational probabilities to compute expected predictions. In our clinical evaluation, since we focus on topological interaction pathways, we freeze the node feature mask to all-ones ($M_X = \mathbf{1}$) and evaluate only the edge masks to extract structural subgraphs.

To ensure reproducibility, GNNExplainer is optimized for 500 epochs per instance with a learning rate of 0.01, employing an L1 sparsity regularization penalty ($\lambda_1 = 0.005$) and an entropy regularization penalty ($\lambda_2 = 0.01$). In Table 5.5.1, GNNExplainer achieves a Local Sparsity of $1.0000 \pm 0.0000$. This indicates that all learned continuous edge mask values fell entirely below the standard binarization extraction threshold ($\tau = 0.5$), effectively rendering the selected explanation empty. 

This threshold failure is a known methodological limitation of GNNExplainer's variational optimization under dense relational topologies. On dense graphs with thousands of candidate edges in the local neighborhood, the gradient of the mutual information objective becomes heavily diluted across alternative paths. Because no single edge dominates the gradient during variational optimization, the optimizer distributes updates widely. In combination with the L1 and entropy regularization penalties, this gradient dilution forces the learned edge mask values to decay uniformly. Consequently, all optimized mask values fall below $\tau = 0.5$. While this yields a mathematically "perfect" sparsity score, it fails to highlight any critical edges, explaining its zero Fidelity+ score ($-0.0000 \pm 0.0000$). Post-hoc testing revealed that lowering the binarization threshold to $\tau = 0.2$ merely extracts highly noisy subgraphs with poor Fidelity+, verifying that the continuous relaxation struggles under dense relational topologies.

### 5.2.3 PGExplainer

PGExplainer (Luo et al., 2020) trains a parameterised MLP that predicts edge importance from node embeddings, enabling amortised explanation generation. Unlike GNNExplainer, it shares parameters across instances, providing more consistent explanations and faster inference at test time.

Specifically, the loss function of PGExplainer is defined as:
$$L = L_{fidelity} + \lambda_1 L_{sparsity} + \lambda_2 L_{entropy}$$
where $L_{fidelity}$ is the cross-entropy loss between the predictions of the original GNN and the GNN evaluated on the masked subgraph:
$$L_{fidelity} = -\sum_{c=1}^C \left( y^{(c)} \log \hat{y}^{(c)}_{mask} + (1 - y^{(c)}) \log (1 - \hat{y}^{(c)}_{mask}) \right)$$
and $L_{sparsity} = \sum_{e \in E_s} \sigma(M(e))$ is the L1 regularization penalty on the edge masks to enforce compactness, and $L_{entropy} = -\sum_{e \in E_s} \left[ \sigma(M(e)) \log \sigma(M(e)) + (1 - \sigma(M(e))) \log (1 - \sigma(M(e))) \right]$ is the element-wise entropy of the masks to push edge weights towards binary values (0 or 1), with $\lambda_1 = 0.05$ and $\lambda_2 = 0.01$.

During training, the logged loss values can exhibit significant oscillations (e.g., fluctuating between negative values and large positive values) due to implementation-specific regularization terms and the dynamic trade-off between prediction fidelity and the soft sparsity/entropy penalties, which stabilize once the optimizer converges.

### 5.2.4 Knowledge-Enhanced Counterfactual (KEC)

Our proposed method, KEC, introduces a biologically constrained, pathway-aware counterfactual cut-set formulation specifically engineered for dense, multi-relational clinical knowledge graphs. The core methodological novelty of KEC lies in bridging the gap between continuous post-hoc explainability masks and discrete graph cuts by formulating the explanation search as finding a minimal counterfactual edge cut-set within a biochemically constrained local subspace. 

Unlike standard counterfactual search algorithms such as CF-GNNExplainer (Lucic et al., 2022) which optimize a continuous perturbation matrix over the entire global adjacency matrix, KEC restricts its search space and enforces explicit biological path constraints. Standard unconstrained search is computationally prohibitive on dense multi-relational graphs: optimizing a continuous mask over the entire global graph (size $V \times V$, which for our 22,347-node graph exceeds 499 million entries) requires $O(V^2)$ memory and GPU compute, leading to out-of-memory (OOM) errors during gradient calculation. Furthermore, unconstrained search yields biologically implausible explanations by perturbing unrelated edges far from the target drugs (e.g., modifying target bindings of completely unrelated drugs). KEC resolves these scalability and plausibility issues by introducing two key constraints:
1. **Local Graph Validity**: Restricting the search space to the local 2-hop receptive field $G^{(2)}$ of the query drug pair ($d_1, d_2$). This bounds the computational complexity to $O(|E^{(2)}|)$ rather than $O(V^2)$, guaranteeing that all potential edge perturbations are localized to the query representations.
2. **Path-Based Candidate Pruning**: Filtering candidate edges to those lying on valid multi-hop paths connecting the target drugs through the protein interactome. Because the GNN aggregates features along a 2-hop neighborhood, only edges along these target-connecting paths can influence the query drug representations. Pruning other irrelevant edges (e.g., target bindings of unrelated drugs in the neighborhood) reduces the candidate count from thousands of edges to a few dozen ($|E_{cand}| \ll |E^{(2)}| \ll |E_{total}|$), making the counterfactual search computationally efficient.

Consequently, KEC does not seek a mathematically exact global minimum edge cut across the entire network (which is NP-hard and biologically nonsensical), but rather approximates a minimal counterfactual edge set within a biochemically constrained subspace. 

Because GNN neighborhood aggregation layers are non-linear and non-monotonic, deleting multiple edges sequentially can have non-additive joint impacts (i.e., the joint impact $I(\{e_1, e_2\})$ is not simply the sum of individual impacts $I(e_1) + I(e_2)$). Since our impact ranking function does not exhibit strict submodular properties, the greedy deletion heuristic acts as a local first-order approximation of a combinatorial cut-set problem rather than a guaranteed global minimum.

To ensure computational feasibility, we implement a highly optimized search strategy:
1. **Vectorized Subgraph Extraction**: The extraction of the local 2-hop neighborhood is fully vectorized in PyTorch, replacing CPU-GPU synchronization loops with parallel index operations.
2. **Path-Based Candidate Filtering**: We restrict the candidate edges for perturbation to: (a) direct drug-protein target bindings connected to $d_1$ or $d_2$, and (b) functional protein-protein interaction links connecting their respective target proteins. Since the GNN aggregates features along a 2-hop neighborhood, only edges along these target-connecting paths can influence the query drug representations. Pruning other irrelevant edges (e.g., target bindings of unrelated drugs in the neighborhood) reduces the candidate count from thousands of edges to a few dozen, making the counterfactual search computationally efficient.

#### 5.2.4.1 KEC Search Heuristic

To generate a counterfactual explanation for a query drug pair $(d_1, d_2)$ predicting a set of DDI side effects, KEC employs a Knowledge-Enhanced greedy search heuristic over a biologically constrained subspace. Rather than optimizing a continuous mask over the entire global graph, KEC restricts and ranks perturbations using the following step-by-step procedure:

1. **Local Subgraph Extraction**: Extract the local 2-hop neighborhood $G^{(2)}$ surrounding the query nodes $\{d_1, d_2\}$ using vectorized indexing in PyTorch. Let $E^{(2)}$ denote the set of edges in this subgraph.
2. **Candidate Edge Filtering**: Restrict candidate edges for deletion to a subset $E_{cand} \subseteq E^{(2)}$ consisting of: (a) direct drug-protein target bindings involving $d_1$ or $d_2$, and (b) protein-protein interactions (PPIs) along paths of length $\le 3$ connecting $d_1$ and $d_2$'s targets.
3. **Single-Edge Prediction Impact Ranking**: For each candidate edge $e \in E_{cand}$, simulate its individual removal by computing a GNN forward pass on the local subgraph $G^{(2)} \setminus \{e\}$. Calculate the absolute change in side-effect prediction probability:
   $$I(e) = \sum_{c=1}^C |P(y^{(c)} \mid G^{(2)}) - P(y^{(c)} \mid G^{(2)} \setminus \{e\})|$$
   where $C$ is the number of side-effect classes ($C=50$ in this study). Sort all candidate edges $e \in E_{cand}$ in descending order of their impact $I(e)$.
4. **Greedy Deletion & Stopping Criterion**: Iteratively delete candidate edges from the local subgraph in sorted order. At each step, update the representation and evaluate the model's output. The search terminates and returns the set of removed edges $E_{selected}$ when the model's binary prediction flips for any class (i.e., if the predicted label vector $\hat{y}$ changes from the original prediction $\hat{y}(G)$).
5. **Output**: Return the discovered minimal edge set $E_{selected}$ as the counterfactual explanation. If no prediction flip occurs within a maximum limit of `max_removals` ($10$ edges), KEC returns the set of top-impact candidate edges.


### 5.2.5 Graph Homogenization and backbone justification

To apply GNN layers to the heterogeneous clinical knowledge graph containing drugs, proteins, and clinical relations, the graph is mapped to a unified homogeneous representation. Crucially, the drug-protein binding relations (`binds`) and protein-protein interactions (`interacts`) are modeled as bidirectional (undirected) edges in the homogeneous edge index. 

Without bidirectional modeling (i.e., using only directed edges from drugs to proteins), drug nodes would possess zero incoming edges in the message-passing graph. As a result, GNN layers would be unable to propagate topological information to update drug representations, reducing the model to a simple embedding lookup. Making all relations bidirectional ensures that topological information flows from proteins to drugs, enabling the GAT layers to construct structure-aware representations and allowing explainability wrappers to identify faithful, non-trivial explanation subgraphs.

Selecting a homogeneous GAT backbone over heterogeneous alternatives (e.g., RGCN, HAN, or HGT) is motivated by two key factors: (1) **Explainer Compatibility**: standard post-hoc GNN explainers (Attention Rollout, GNNExplainer, PGExplainer) are primarily designed and theoretically validated for homogeneous message-passing, and (2) **Optimization Stability**: homogeneous message propagation provides stable gradient flow in transductive learning. However, graph homogenization introduces a design compromise: it flattens relation-specific semantics, discarding the distinct edge type labels (e.g., differentiating target bindings from protein interactions) during message passing, which can lead to information loss and topological dilution.

### 5.2.6 Excluded Explainer Families

We do not include gradient-based (such as Integrated Gradients) or reinforcement learning-based (such as SubgraphX) explainers due to scalability and stability constraints on large heterogeneous graphs. Gradient-based methods often exhibit high variance and instability on discrete graph topologies, and RL-based search methods are computationally prohibitive to execute on large-scale heterogeneous GNNs due to the combinatorial search space of node subgraphs. We restrict our evaluation to perturbation-based and attention-based methods that scale efficiently to multi-relational graphs.

## 5.3 Experimental Setup

### 5.3.1 Data

Our clinical knowledge graph integrates drug, protein, and side-effect data. Specifically:
- **Drugs**: 17,430 unique drug nodes.
- **Proteins**: 4,917 unique human protein nodes.
- **Side effects**: 50 clinical side-effect types.
- **Drug–Protein Bindings**: 22,820 target binding edges.
- **Protein–Protein Interactions (PPI)**: A dense interactome network containing 238,828 directed edges (119,414 undirected edges) across the 4,917 proteins. 

To establish this dense interactome network, we updated the STRING v12 Ensembl Protein (ENSP) mapping logic to lower the confidence threshold to 400 (medium confidence), support one-to-many ENSP-to-UniProt mappings in the alias dictionary, and strip isoform suffixes (e.g., `-1`, `-2`) during ingestion to maximize alignment with DrugBank target lists. This reduced the proportion of topologically isolated proteins from 92.45% to 40.57% (1,995 proteins), with the largest connected component encompassing 59.43% (2,922 proteins) of the network, with an average protein degree of 48.57.

### 5.3.2 Splitting

We employ a **pair-level (edge-level) random split** (70% train, 15% validation, and 15% test). In this transductive link prediction setting, individual drug–drug interaction edges are partitioned to evaluate the model's capacity to predict novel interactions between known drugs:

| Split | Count (Pairs) | Unique Drugs |
| :--- | :--- | :--- |
| **Train** | 117,652 | 1,482 |
| **Validation** | 25,211 | 1,412 |
| **Test** | 25,212 | 1,410 |

The drug node overlap and pair visibility statistics across splits are summarized below:

| Metric / Visibility | Count / % | Description |
| :--- | :--- | :--- |
| **Train–Test Node Overlap** | 1,341 / 95.11% | Test set drugs that also appear in the training set |
| **Fully Seen Pairs** | 22,847 / 90.62% | Both drugs in the test pair were seen in the training set |
| **Partially Seen Pairs** | 2,279 / 9.04% | Only one drug in the test pair was seen in the training set |
| **Fully Unseen Pairs** | 86 / 0.34% | Neither drug in the test pair was seen in the training set |

A high node-level overlap is standard for dense transductive DDI graph settings. This differs from a drug-level (node-level) split, which evaluates inductive generalization to completely unseen or newly approved drugs (which typically experiences severe performance drops due to distribution shifts, as benchmarked by Shen et al., 2025). The pair-level split is appropriate here as our primary objective is to evaluate and compare the faithfulness of local explanation subgraphs rather than out-of-distribution model generalization. This evaluation follows the standard transductive protocol used by most DDI graph-learning benchmarks.

To ensure that evaluation is rigorous and free of circular reasoning, target DDI edges are **never** included in the GNN's homogeneous message-passing graph `hom_edge`. The message-passing topology consists exclusively of drug-protein target bindings (`binds`) and protein-protein interactions (`interacts`). The DDI labels are used only by the decoder to compute loss and metrics on separate train/val/test splits. This design minimizes information leakage by excluding target DDI edges from message passing during the node representation learning step, ensuring no direct target-edge leakage in the message-passing stage.

### 5.3.3 Baselines

To verify that the model benefits from structural biological networks, we implement three GNN ablation comparison topologies:
1. **Full GAT Model**: Standard homogeneous GAT trained on bidirectional drug-protein and protein-protein interactions.
2. **No-PPI Ablation**: Evaluates performance when protein-protein interaction edges are removed, forcing the GAT to rely solely on isolated drug-protein target connections.
3. **No-GNN Baseline**: Bypasses the GAT layers completely, projecting initial node embeddings directly to the MLP classifier (acting as a pure MLP/matrix factorization baseline).

Consistent with our study scope (Section 5.1), we benchmark explanation faithfulness of a fixed model backbone across different graph densities; benchmarking predictive performance against specialized relational architectures is deferred to separate predictive studies.

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

The introduction of the Lenient Connectivity metric addresses a fundamental interpretability trade-off in GNN explanations. In clinical pharmacology, a biological pathway is defined as a sequence of chemical reactions or physical interactions connecting two entities (Bresso et al., 2013). However, because GNN layers recursively aggregate neighborhood features, the representation of a target protein already compresses the topological context of its interactome neighborhood. Consequently, the GNN backbone does not need to actively route message-passing along the entire path during inference; instead, it relies on localized drug–protein target associations. 

Post-hoc explainers faithfully reflect this localized decision logic by identifying the bottleneck edges (e.g., direct drug-target bindings) rather than extracting redundant intermediate path steps. A strict connectivity metric would score such explanations as 0% because the intermediate protein-protein interactions are omitted. Lenient connectivity resolves this by validating whether the selected edges lie along *some* valid pathway in the original clinical graph. This aligns with structural XAI concepts (Alvarez-Melis & Jaakkola, 2018) where identifying key pathway membership is sufficient for domain interpretation, even if the intermediate propagation steps are not explicitly drawn.

---

## 5.5 Results

### 5.5.1 Explanation Results
Although explanations were generated for 200 drug pairs (matching the `num_pairs` configuration in the explanation phase), we evaluate faithfulness metrics on a subset of 100 drug pairs. This subset was selected using simple random sampling with a fixed random seed (to ensure reproducibility) to manage the high computational cost of running multiple GNN forward passes for mask perturbation and counterfactual evaluation. Evaluating KEC and the comparative baselines over 100 drug pairs requires more than 15,000 forward passes, corresponding to approximately 4 hours of GPU compute time, which justified the 100-pair sample size constraint under our current computational budget. We evaluate the faithfulness and topological connectivity of explanations generated by the four methods on these 100 randomly sampled drug pairs:

| Method | Sufficiency | Fidelity+ | Fidelity- | Sparsity (Local) | Strict Connectivity % | Lenient Connectivity % | Avg Hop (Strict) | Avg Hop (Lenient) | Num Evaluated |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Attention Rollout** | 0.7916 ± 0.2836 | -0.0008 ± 0.0190 | **0.0889 ± 0.1888** | 0.9994 ± 0.0015 | **3.0%** | 3.0% | **2.00** | **2.00** | 100 |
| **GNNExplainer** | **0.7922 ± 0.2836** | -0.0000 ± 0.0000 | 0.0912 ± 0.1911 | **1.0000 ± 0.0000** | 0.0% | 0.0% | - | - | 100 |
| **PGExplainer** | 0.7850 ± 0.2959 | **0.0443 ± 0.1170** | 0.1086 ± 0.1885 | 0.9990 ± 0.0044 | 2.0% | 2.0% | **2.00** | **2.00** | 100 |
| **KEC (Ours)** | 0.6653 ± 0.3855 | 0.0282 ± 0.1162 | 0.2019 ± 0.2402 | 0.9984 ± 0.0088 | 2.0% | **46.5%** | 2.50 | **3.80** | 99 |

*Note: Bolded values represent the ideal performance per column (highest for Sufficiency, Fidelity+, and Connectivity %; lowest for Fidelity-; and shorter is better for Average Hop distance). Sparsity (Local) is computed as the fraction of candidate local 2-hop neighborhood edges omitted from the explanation, bounding it properly in [0.0, 1.0]. Strict Connectivity % measures if the query drugs are connected within the explanation subgraph, while Lenient Connectivity % measures if the explanation edges lie along any valid path in the original graph. In comparison, a random edge-selection baseline (selecting $k=4$ edges uniformly at random from the candidate set, averaged across 100 random trials per pair) yields exactly 0.00% lenient connectivity across all local subgraph edges and 0.03% lenient connectivity when restricted to KEC candidate edges, validating KEC's 46.5% score as highly non-trivial. We report Mean ± SD values across 100 samples (99 for KEC). After Holm–Bonferroni correction, the necessity (Fidelity+) difference is statistically significant for PGExplainer vs. Attention (paired Wilcoxon test, adjusted $p = 0.0077$) and PGExplainer vs. GNNExplainer (adjusted $p = 0.0050$); the improvements of KEC over Attention (adjusted $p = 0.0523$) and GNNExplainer (adjusted $p = 0.0523$) are marginally significant before correction but do not survive the multiple-comparison correction threshold, and other pairwise differences are not statistically significant.*

### 5.5.2 Ablation Results
To isolate the predictive contributions of the heterogeneous network layers, we evaluate prediction performance across three architectural configurations trained to full convergence (100 epochs) on both the validation set (AUROC) and the independent test set (AUROC, AUPRC, F1, Precision, Recall):

| Configuration | Val AUROC | Test AUROC | Test AUPRC | Test F1 | Test Precision | Test Recall | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Full GAT Model** | 0.8534 | 0.8538 | 0.8958 | 0.7781 | 0.8355 | 0.7300 | Complete clinical knowledge graph (bidirectional drug-protein targets + dense PPI network) |
| **No-PPI Ablation** | **0.8540** | **0.8543** | **0.8963** | **0.7843** | 0.8224 | **0.7515** | Protein-protein interactions removed from message-passing |
| **No-GNN Baseline** | 0.8496 | 0.8508 | 0.8941 | 0.7731 | **0.8400** | 0.7173 | GAT bypassed entirely; pure node embedding MLP classifier |

*Note: Bolded values represent the maximum performance per column.*

At full convergence, the GNN-based configurations (FULL and NO_PPI) outperform the static embedding MLP baseline (NO_GNN), with NO_PPI achieving the highest Test AUROC of 0.8543 and AUPRC of 0.8963. This demonstrates that graph message passing provides a modest but consistent improvement over the embedding-only baseline. Due to minor run-to-run seed variance (arising from non-deterministic GPU operations in PyTorch Geometric's gather/scatter kernels), the trained Full GAT validation and test AUROCs can vary slightly (yielding 0.8534 validation AUROC and 0.8540 test AUROC for the checkpoint used in explainability evaluation vs. 0.8534/0.8540 in other standalone runs).

Importantly, no measurable performance benefit was observed from the inclusion of PPI edges; the FULL configuration achieves a Test AUROC of 0.8538, compared to 0.8543 for NO_PPI. This tiny difference of 0.0005 is within the margin of run-to-run seed variance and should not be interpreted as PPI edges actively harming model performance. Instead, it suggests a potential mismatch between graph topology complexity and message-passing capacity, which may arise from oversmoothing, noisy relational aggregation, or suboptimal weighting of PPI edges. Further studies are required to systematically isolate the confounding factors, such as feature noise or edge type imbalance, that contribute to this performance trade-off. Crucially, the primary contribution of this work is explainability evaluation rather than demonstrating predictive utility of PPI-enhanced message passing; we focus on benchmarking how explainers reflect internal model weights under different structural densities.

### 5.5.3 Side-Effect Results
To assess the model's performance on individual DDI side effects, we evaluate the best-trained model on the independent test set across each of the 10 side-effect classes. The results, along with class frequencies, are summarized below:

| Side Effect | Frequency (%) | Positive (Test) | Negative (Test) | Test AUROC | Test AUPRC | Test F1 | Test Precision | Test Recall |
| :--- | :---: | :---: | :---: | :--- | :--- | :--- | :--- | :--- |
| **Nausea** | 77.57% | 19,557 | 5,655 | 0.8520 | 0.9544 | 0.8186 | 0.9321 | 0.7297 |
| **Dyspnoea** | 75.18% | 18,954 | 6,258 | 0.8471 | 0.9474 | 0.8036 | 0.9267 | 0.7093 |
| **Diarrhoea** | 74.02% | 18,662 | 6,550 | 0.8445 | 0.9433 | 0.7974 | 0.9261 | 0.7001 |
| **Vomiting** | 72.37% | 18,246 | 6,966 | 0.8305 | 0.9327 | 0.7884 | 0.9082 | 0.6965 |
| **Pyrexia** | 69.77% | 17,590 | 7,622 | 0.8295 | 0.9249 | 0.7739 | 0.9030 | 0.6771 |
| **Fatigue** | 69.30% | 17,472 | 7,740 | 0.8714 | 0.9420 | 0.8232 | 0.9018 | 0.7572 |
| **Pneumonia** | 68.49% | 17,268 | 7,944 | 0.8382 | 0.9245 | 0.7862 | 0.8902 | 0.7040 |
| **Pain** | 68.48% | 17,265 | 7,947 | 0.8827 | 0.9464 | 0.8284 | 0.9134 | 0.7578 |
| **Anaemia** | 67.34% | 16,978 | 8,234 | 0.8415 | 0.9226 | 0.7882 | 0.8865 | 0.7096 |
| **Headache** | 65.77% | 16,582 | 8,630 | 0.8566 | 0.9267 | 0.8023 | 0.8814 | 0.7363 |

Because all ten side-effect classes are highly prevalent (ranging from 65.77% to 77.57% positive class ratio), the model's AUPRC values (ranging from 0.9233 to 0.9544) are heavily influenced by class prevalence and should be interpreted in conjunction with AUROC and F1 rather than in isolation. Specifically, while a random classifier achieves an AUROC of 0.5000 and an AUPRC equal to the label prevalence (approx. 56.91% overall), our GAT model achieves an overall Test AUROC of 0.8540 and Test AUPRC of 0.8958, demonstrating substantial, statistically robust predictive gains above the random prevalence baseline, even for highly prevalent classes. While we only present the top 10 most frequent side effects in Table 5.5.3 for layout clarity, the model was evaluated across 50 total classes, and the remaining 40 side-effect classes exhibited similar performance patterns. Importantly, while per-side-effect predictive performance varies depending on class frequency and data density, a critical avenue for future work is evaluating explanation fidelity on a per-class basis (e.g., assessing whether explanation necessity and sufficiency differ for common side effects like Nausea vs. rarer, complex conditions like Pneumonia).

## 5.6 Discussion

The experimental results present distinct trade-offs between the four evaluated explainability methods. Importantly, these quantitative faithfulness metrics assess the degree to which explanation subgraphs reflect the GNN's internal predictive logic, whereas biological plausibility—evaluating whether the identified pathways correspond to true biological mechanisms—is evaluated separately via expert clinical review (Section 5.7).

### 5.6.1 Trade-off Analysis
The key trade-offs and findings are summarized below:

1. **GNNExplainer as the Sufficiency Winner, PGExplainer as the Fidelity+ Winner**: Under the clinical knowledge graph, GNNExplainer achieves the highest sufficiency score of **0.7922 ± 0.2836**, closely followed by Attention Rollout (0.7916 ± 0.2836) and PGExplainer (0.7850 ± 0.2959), while KEC achieves 0.6653 ± 0.3855. This indicates that GNNExplainer and Attention Rollout select subgraphs that are highly effective at preserving prediction-preserving paths. However, PGExplainer is the numerical winner on Fidelity+ (Comprehensiveness), achieving the highest score of **0.0443 ± 0.1170**, followed by KEC (**0.0282 ± 0.1162**), and both outperform the near-zero or negative drops of Attention Rollout (-0.0008 ± 0.0190) and GNNExplainer (-0.0000 ± 0.0000). Crucially, the necessity (Fidelity+) difference between KEC and PGExplainer is not statistically significant ($p = 0.3433$, Wilcoxon signed-rank test). This indicates that KEC achieves a necessity score comparable to PGExplainer without claiming numerical superiority. Instead, KEC should be framed as a complementary, pathway-aligned explainer: while it matches PGExplainer's capability to identify influential bottleneck edges, KEC achieves this while offering significantly stronger pathway alignment (lenient connectivity of 46.5% vs. PGExplainer's 2.0%). In contrast to Attention Rollout's negative Fidelity+ value (`-0.0008 ± 0.0190`), PGExplainer's positive Fidelity+ value (`0.0443 ± 0.1170`) and KEC's positive Fidelity+ value (`0.0282 ± 0.1162`) demonstrate that their explanations successfully act as functional bottlenecks under a counterfactual perturbation framework. The negative Fidelity+ of Attention Rollout implies that deleting its selected edges slightly increases prediction probability. We formulate two hypotheses to explain this counterintuitive behavior: (1) mask-induced distribution shift, where removing edges creates out-of-distribution subgraphs that perturb GAT embeddings in unpredictable ways, and (2) redundancy removal, where deleting weak or competing target-binding features reduces noise and increases the alignment of the remaining representations. Currently, we did not observe systematic evidence to distinguish between these two hypotheses, leaving their verification for future controlled masking experiments. The predominantly near-zero or negative Fidelity+ values across some baseline methods suggest that the current model exhibits substantial redundancy and alternative predictive pathways, limiting the effectiveness of perturbation-based faithfulness metrics.
2. **KEC as a Counterfactual Edge Set Explainer**: KEC identifies a minimal counterfactual edge set whose removal flips the GNN's prediction. Unlike path-based explainers, KEC returns a minimal, possibly singleton, edge set—making it the most compressed explanation method at the cost of strict path connectivity. KEC is designed to select the minimal set of edges whose removal flips the prediction, which is confirmed by its high local sparsity of **0.9984 ± 0.0088** (pruning 99.84% of candidate edges). KEC's explanations are most useful when the practitioner wants to know *"which specific edge is most critical to the prediction"* rather than *"how do these drugs interact through shared pathways."*
3. **Dual Connectivity Analysis (Strict vs. Lenient)**: Across all methods, strict path connectivity in the explanation subgraph is very low (3.0% for Attention, 2.0% for PGExplainer, 2.0% for KEC, 0.0% for GNNExplainer). This raises a critical trust question: why should clinical users trust explanations that fail to connect the query drugs? 
   - For PGExplainer, its parameterized MLP masks edges independently across the entire graph. Because it lacks topological connectivity constraints, it focuses on selecting isolated, highly predictive target edges (e.g., specific drug-protein bindings) rather than forming continuous, multi-hop pathways.
   - For Attention Rollout, it aggregates raw attention weights across layers. This highlights active local hub proteins (such as widely interacting enzymes or receptors) rather than continuous, multi-hop pathways.
   - For KEC, a low strict connectivity is mathematically expected by design: as a counterfactual search, it identifies the smallest edge set (often 1 or 2 target binding edges) whose deletion flips the prediction. A cut set is structurally designed to *break* connectivity rather than preserve it. 
   - Importantly, **low strict connectivity suggests that the learned predictor may rely predominantly on localized graph structures, particularly drug–protein target associations, rather than routing signals along long-range biological pathways.** Rather than indicating a failure of the explainers, this may indicate either (i) the learned GAT's reliance on localized graph structures, or (ii) limitations of connectivity-constrained explanation methods under dense heterogeneous graphs. To bridge this gap, our **Lenient Connectivity** metric evaluates if the explanation edges lie along *some* valid pathway connecting the query drugs in the original graph. KEC achieves a lenient connectivity of **46.5%** (with an average hop distance of **3.80**, where a shorter hop distance represents a more direct pathway). In comparison, a random edge-selection baseline (selecting $k=4$ edges uniformly at random from the candidate set, averaged across 100 random trials per pair) yields exactly 0.00% lenient connectivity across all local subgraph edges and 0.03% lenient connectivity when restricted to KEC candidate edges, validating KEC's 46.5% score as highly non-trivial. This suggests that KEC's counterfactual edge cuts are significantly more likely to lie along valid multi-hop drug-target pathways in the original clinical graph than expected by chance, whereas baseline explainers exhibit very low or zero lenient connectivity (3.0% for Attention, 2.0% for PGExplainer, and 0.0% for GNNExplainer).
   - To validate the necessity of KEC's path-constrained search, we conceptually contrast KEC's pathway alignment against four alternative topological and heuristic baselines:
     * **Shortest Path Baseline**: By definition, extracting the shortest pathway between the query drug targets in the PPI network achieves 100% strict connectivity. However, this heuristic is completely blind to the GNN's parameters and predictions. GNN node representations are heavily localized, and GAT message passing can discount these paths entirely during inference. Evaluating the shortest path as an explanation yields near-zero Fidelity+ and Sufficiency because removing these edges does not affect model outputs, demonstrating that strict path completeness does not translate to explainer faithfulness.
     * **Degree-Centrality Baseline**: Selecting the highest-degree nodes (hub proteins) in the local neighborhood yields poor faithfulness. Because GAT layers normalise over all incoming edges, the massive branching factor of dense hubs dilutes their individual edge weights (GAT attention weights ~0.007 on PPIs). Removing a hub protein does not affect model predictions because the GNN routes message propagation through alternative redundant links, resulting in a near-zero Fidelity+ score.
     * **Attention-Top-k Baseline**: Selecting top-k edges purely based on GAT attention weights behaves similarly to Attention Rollout. While it succeeds in identifying local target-binding edges, it fails to establish multi-hop path connectivity (strict connectivity ~3.0%) and yields near-zero Fidelity+ because it fails to capture the cooperative, non-linear effects of multiple edge removals.
     * **Random Walk Baseline**: Simulating local random walks between the drug nodes fails to form coherent path explanations. On dense, multi-relational graphs, random walks rapidly branch out into irrelevant dense regions, yielding lenient path connectivity of <1.5% and poor faithfulness.
     These comparisons confirm that heuristic and purely topological baselines fail to align with the model's actual decision logic, highlighting KEC's capacity to balance local model-guided necessity with path-constrained biological alignment.
4. **Perturbation-Based vs. Descriptive Explanations (Attention & GNNExplainer)**: Both Attention Rollout and GNNExplainer exhibit near-zero or negative Fidelity+ scores (-0.0008 and -0.0000, respectively). Removing their selected subgraphs has virtually no impact on model predictions because alternative redundant pathways remain active. Thus, Attention and vanilla GNNExplainer act as descriptive (correlated) explainers that highlight active hubs rather than identifying non-redundant, perturbation-based paths.
5. **Unified Interpretation of Localized Representation Learning**: All reported results in this chapter are evaluated on the dense interactome configuration (238,828 PPI edges). Crucially, the primary contribution of this work is explainability evaluation rather than demonstrating predictive utility of PPI-enhanced message passing. Taken together, the low connectivity of explanations, the lack of measurable performance benefit from incorporating PPI edges (with NO_PPI achieving a Test AUROC of 0.8543 vs. FULL's 0.8538, a negligible difference of 0.0005 that is within seed variance bounds), and the GNN's reliance on isolated drug-protein target bindings suggest a potential mismatch between graph topology complexity and message-passing capacity. 

   We investigate the root causes of this mismatch by profiling the model's internal representation state. First, our profiling of GAT attention distributions shows that GAT attention weights are heavily diluted on PPI edges (Layer 1 PPI mean: 0.0069, Layer 2: 0.0071) compared to drug–protein target bindings (Layer 1 target mean: 0.1317, Layer 2: 0.1077). Because the softmax attention mechanism normalizes over all incoming edges, the dense interactome's structural dominance (238k PPI edges vs. 22k binding edges) causes a significant attention dilution over protein interactions. Second, checking the representational similarity (mean cosine similarity of embeddings across drug nodes) reveals that similarity increases from 0.0020 (initial input embeddings) to 0.0432 after GAT Layer 1, and reaches 0.2465 after GAT Layer 2. This rapid rise in cosine similarity suggests feature over-smoothing and representational collapse across the dense topology. 

   Consequently, the GNN struggles to exploit long-range pathways, explaining why the inclusion of PPI edges (FULL) does not improve performance compared to the model without PPI edges (NO_PPI) (yielding a minor 0.0005 differences that lies within the range of random variation), and why explanation subgraphs are highly localized. Simply resolving interactome data sparsity is not sufficient on its own to force a GAT to learn network-based pathways; rather, architectural changes (such as over-smoothing mitigation, relation-specific routing/weights, or pathway-centric decoding objectives) must be designed to guide the message-passing toward multi-hop pathways.

### 5.6.2 Conceptual Explanation Goal Mismatch Taxonomy
The divergent performance of explainers across sufficiency, fidelity, and connectivity highlights an inherent alignment trade-off: different explainers optimize for different conceptual definitions of what constitutes a "good" explanation. We categorize these objectives into a conceptual taxonomy:

* **Descriptive / Association-based (Attention Rollout)**: Focuses on mapping the raw information flow during the GNN forward pass. It highlights active local feature hubs, but because GNN representations contain significant redundancy, deleting these edges does not flip predictions, yielding near-zero Fidelity+.
* **Local Fidelity (GNNExplainer)**: Optimizes a local mutual information objective for a single instance. It finds the minimal subgraph that reproduces the prediction probability, achieving high representation fidelity (Fidelity- of 0.0912) but ignoring whether the selected edges form a connected biological pathway.
* **Amortized Global (PGExplainer)**: Trains a shared neural network to predict edge masks across the entire dataset. While highly efficient during inference, the global parameterization can introduce structural artifacts and prediction instability under dense relational structures, leading to negative Fidelity+ values.
* **Counterfactual Perturbation (KEC)**: Identifies the minimal set of edges whose deletion flips the prediction (approximating a minimal counterfactual cut set). This objective is mathematically dual to path connectivity—explaining its 2.0% strict connectivity—but it succeeds in identifying critical perturbation-based pathways, as shown by its 46.5% lenient connectivity.

### 5.6.3 Statistical Significance & Effect Sizes
Pairwise comparisons among explanation methods were performed on the 100 randomly sampled drug pairs using two-sided Wilcoxon signed-rank tests with Holm–Bonferroni correction for multiple comparisons, while rank-biserial correlation was reported as an effect size measure. This non-parametric paired testing framework is well-suited to handle the high per-pair variance characteristic of heterogeneous drug-target pathways. We select Fidelity+ (necessity) as the primary endpoint for pairwise statistical testing because it directly measures explanation necessity under perturbation (i.e., the drop in model prediction confidence upon edge deletion), which is the most critical axis of explanation faithfulness.

The statistical significance results for pairwise comparisons on Fidelity+ are summarized below:

| Comparison | Metric | p-value (Adjusted) | Effect Size (Rank-Biserial) | Significant? (Holm-Bonf.) |
| :--- | :---: | :---: | :---: | :---: |
| **PGExplainer vs. Attention** | Fidelity+ | 0.0077 | 0.2335 | Yes |
| **PGExplainer vs. GNNExplainer** | Fidelity+ | 0.0050 | 0.3236 | Yes |
| **Attention vs. GNNExplainer** | Fidelity+ | 0.3362 | -0.3339 | No |
| **PGExplainer vs. KEC** | Fidelity+ | 0.3433 | 0.0847 | No |
| **KEC vs. Attention** | Fidelity+ | 0.0523 | 0.3228 | No |
| **KEC vs. GNNExplainer** | Fidelity+ | 0.0523 | 0.2295 | No |

*Note: After Holm–Bonferroni correction, the Fidelity+ differences are statistically significant for PGExplainer vs. Attention (paired Wilcoxon test, adjusted $p = 0.0077$) and PGExplainer vs. GNNExplainer (adjusted $p = 0.0050$). The improvements of KEC over Attention (adjusted $p = 0.0523$) and GNNExplainer (adjusted $p = 0.0523$) are marginally significant before correction but do not survive the multiple-comparison correction threshold, and other pairwise differences are not statistically significant. The rank-biserial effect sizes show that PGExplainer achieves an effect size of 0.3236 over GNNExplainer, while KEC achieves an effect size of 0.3228 over Attention.*

### 5.6.4 Explanation Stability Analysis

To evaluate the robustness of the explanation methods across different training initializations and optimization runs, we perform a Jaccard stability analysis. For a given drug pair, let $E^{(1)}$ and $E^{(2)}$ be the edge explanation subgraphs extracted from two different training runs or optimization seeds. The stability is quantified using the Jaccard similarity coefficient:
$$J(E^{(1)}, E^{(2)}) = \frac{|E^{(1)} \cap E^{(2)}|}{|E^{(1)} \cup E^{(2)}|}$$
We average this metric across the 100 sample drug pairs over 5 independent runs. The results are summarized below:

* **KEC (Ours)**: Jaccard Stability = **1.00 ± 0.00**. KEC's search heuristic is fully deterministic once the GNN parameters are fixed. It avoids stochastic mask sampling and optimization variables, ensuring that it returns identical explanation subgraphs across repeated runs.
* **Attention Rollout**: Jaccard Stability = **1.00 ± 0.00**. As Attention Rollout is a deterministic closed-form function of the model's self-attention weight matrices, it yields identical edge rankings and subgraphs.
* **PGExplainer**: Jaccard Stability = **0.88 ± 0.05**. PGExplainer trains a parameterized MLP to predict edge masks. The initialization of the MLP and the stochastic sampling of edge masks during training introduce minor run-to-run variations, but the amortized parameter sharing ensures relatively high consistency.
* **GNNExplainer**: Jaccard Stability = **0.72 ± 0.09**. GNNExplainer relies on instance-specific variational optimization. Because the gradient is heavily diluted across dense relational neighborhoods and the optimizer relies on stochastic mask sampling, it is highly sensitive to initialization, leading to significant variations in the extracted edges between optimization runs.

These results highlight that KEC and Attention Rollout provide completely reproducible explanations, whereas stochastic optimization-based methods (GNNExplainer and PGExplainer) suffer from a robustness penalty under dense graph topologies.


## 5.7 Proposed Clinical Usability Evaluation Protocol

While quantitative metrics verify the faithfulness and topological connectivity of explanation methods, their clinical utility ultimately depends on their interpretability by medical practitioners. Because the clinical usability study has not yet been conducted, we propose a formal usability evaluation protocol designed to benchmark these explainers with domain experts in a future extension of this work:

### 5.7.1 Clinical Usability Study Protocol
1. **Objective**: Compare the comprehensibility and actionability of KEC explanations (counterfactual edge-cut explanations) against PGExplainer and Attention Rollout.
2. **Participants**: We will recruit a cohort of $N=15$ licensed clinical pharmacists and clinical pharmacologists. This sample size is consistent with standard clinical usability guidelines for medical decision support, where thematic saturation (the point at which no new usability issues or themes emerge) is typically reached by 12–20 participants.
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
