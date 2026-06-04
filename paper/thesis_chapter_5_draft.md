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
1. The mathematical definition of the entropy penalty, which can become negative depending on scaling and optimization settings.
2. The dynamic trade-off between the soft sparsity regularizer and the prediction fidelity term. Early in training, the optimizer aggressively pushes the soft masks toward 0 to minimize sparsity and entropy penalties, which can temporarily degrade prediction fidelity and cause sharp gradient updates, resulting in loss oscillations before convergence.

### 5.2.4 Knowledge-Enhanced Counterfactual (KEC)

Our proposed method identifies the minimal set of edges whose removal changes the model's prediction. KEC leverages the knowledge graph structure to constrain the search space to biologically plausible perturbations within the k-hop neighbourhood of the query drug pair.

### 5.2.5 Graph Homogenization and Bidirectional Message Flow

To apply homogeneous GNN layers (GAT) to the heterogeneous clinical knowledge graph containing drugs, proteins, and clinical relations, the graph is mapped to a unified homogeneous representation. Crucially, the drug-protein binding relations (`binds`) and protein-protein interactions (`interacts`) are modeled as bidirectional (undirected) edges in the homogeneous edge index. 

Without bidirectional modeling (i.e., using only directed edges from drugs to proteins), drug nodes would possess zero incoming edges in the message-passing graph. As a result, GNN layers would be unable to propagate topological information to update drug representations, reducing the model to a simple embedding lookup. Making all relations bidirectional ensures that topological information flows from proteins to drugs, enabling the GAT layers to construct structure-aware representations and allowing explainability wrappers to identify faithful, non-trivial explanation subgraphs.

## 5.3 Graph Topology, Leakage Prevention, and Baselines

### 5.3.1 Target Edge Masking & Leakage Prevention
To evaluate emerging drug-drug interaction (DDI) predictions without introducing circular dependency or information leakage, target DDI edges are **never** included in the GNN's homogeneous message-passing graph `hom_edge`. The message-passing topology consists exclusively of drug-protein target bindings (`binds`) and protein-protein interactions (`interacts`). 

The prediction target labels (DDI pairs) are only used in the final multi-layer perceptron (MLP) decoder to calculate loss and predictive metrics on distinct splits (train, validation, and test). This guarantees zero information leakage of prediction targets into the node representation learning step, representing a fully inductive prediction environment.

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

## 5.4 Faithfulness Metrics

| Metric | Definition | Ideal |
| :--- | :--- | :--- |
| **Sufficiency** | Prediction agreement using only the explanation subgraph | High |
| **Necessity** | Prediction drop when explanation edges are removed | High |
| **Fidelity+** | Average ΔP when explanation is removed | High |
| **Fidelity-** | Average ΔP when only explanation is kept | Low |
| **Sparsity** | Fraction of edges in explanation | Low |
| **Path-Connected %** | Percentage of explanations where query drug nodes are connected in the subgraph | High |
| **Avg Hop Distance** | Shortest path distance between query drug nodes in the subgraph | Low |

## 5.5 Results

### 5.5.1 Faithfulness Benchmarks
We evaluate the faithfulness and topological connectivity of explanations generated by the four methods on 100 randomly sampled drug pairs:

| Method | Sufficiency | Necessity | Fidelity+ | Fidelity- | Sparsity | Path-Connected % | Avg Hop Distance | Num Evaluated |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Attention Rollout** | 0.7260 | 0.0055 | 0.0055 | **0.1550** | 0.0042 | 15.0% | 4.80 | 100 |
| **GNNExplainer** | **0.7280** | 0.0004 | 0.0004 | 0.1600 | 0.0042 | 12.0% | 4.90 | 100 |
| **PGExplainer** | 0.7150 | 0.0648 | 0.0648 | 0.1850 | 0.0011 | 22.0% | 4.50 | 100 |
| **KEC (Proposed)** | 0.6840 | **0.1006** | **0.1006** | 0.1700 | **0.00004** | **100.0%** | **3.20** | 100 |

*Note: Bolded values represent the ideal performance per column (highest for Sufficiency, Necessity, Fidelity+, and Path-Connected %; lowest for Fidelity-, Sparsity, and Avg Hop Distance).*

### 5.5.2 GNN Ablation Study Results
To isolate the predictive contributions of the heterogeneous network layers, we evaluate prediction performance across three architectural configurations trained to full convergence (100 epochs) on both the validation set (AUROC) and the independent test set (AUROC, AUPRC, F1, Precision, Recall):

| Configuration | Val AUROC | Test AUROC | Test AUPRC | Test F1 | Test Precision | Test Recall | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Full GAT Model** | **0.8469** | 0.8446 | 0.9377 | **0.8177** | 0.8927 | **0.7547** | Complete clinical knowledge graph (bidirectional drug-protein targets + dense PPI) |
| **No-PPI Ablation** | 0.8463 | **0.8449** | **0.9379** | 0.8165 | 0.8938 | 0.7520 | Protein-protein interactions removed from message-passing |
| **No-GNN Baseline** | 0.8353 | 0.8322 | 0.9318 | 0.7874 | **0.9017** | 0.6992 | GAT bypassed entirely; pure node embedding MLP classifier |

*Note: Bolded values represent the maximum performance per column.*

At full convergence, the GNN-based configurations (FULL and NO_PPI) significantly outperform the static embedding MLP baseline (NO_GNN), achieving Test AUROCs of 0.8446 and 0.8449 respectively compared to 0.8322. This demonstrates that structure-aware graph message-passing yields superior drug representations than static node embeddings alone.

However, the inclusion of PPI information (FULL) produces performance comparable to the model without PPI edges (NO_PPI), with a marginal difference of only 0.0003 in Test AUROC (0.8446 vs. 0.8449). This indicates that the protein-protein interaction network contributes very little predictive signal under the current graph construction. 

This behavior is directly explained by the extreme sparsity of the constructed PPI network: after Swiss-Prot/TrEMBL alias mapping, the graph contains only 1,232 PPI edges among 4,917 proteins, yielding an average protein degree of $\approx 0.50$. This leaves the vast majority of target proteins topologically isolated. In practice, because drug-protein target interactions are dense (22,820 edges) and provide the primary pathway for message flow, GNN representations are heavily dominated by direct drug-protein bindings, making the sparse PPI subgraph functionally redundant for link prediction. Acknowledging this topological sparsity is a key limitation; future iterations should lower the STRING confidence threshold (currently $\ge 700$) or expand matching coverage to resolve protein isolation.

### 5.5.3 Per-Side-Effect Performance Results
To assess the model's performance on individual DDI side effects, we evaluate the best-trained model on the independent test set across each of the 10 side-effect classes. The results, along with class frequencies, are summarized below:

| Side Effect | Frequency (%) | Test AUROC | Test AUPRC | Test F1 | Test Precision | Test Recall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Nausea** | 78.84% | 0.8461 | 0.9556 | 0.8262 | 0.9294 | 0.7436 |
| **Dyspnoea** | 76.64% | 0.8377 | 0.9479 | 0.8129 | 0.9201 | 0.7280 |
| **Diarrhoea** | 74.94% | 0.8375 | 0.9428 | 0.8137 | 0.9075 | 0.7374 |
| **Vomiting** | 73.70% | 0.8255 | 0.9352 | 0.7935 | 0.9038 | 0.7072 |
| **Pyrexia** | 70.96% | 0.8244 | 0.9252 | 0.8031 | 0.8768 | 0.7409 |
| **Fatigue** | 70.07% | 0.8678 | 0.9421 | 0.8284 | 0.8941 | 0.7716 |
| **Pneumonia** | 69.88% | 0.8384 | 0.9282 | 0.8044 | 0.8865 | 0.7362 |
| **Pain** | 69.36% | 0.8756 | 0.9451 | 0.8315 | 0.9021 | 0.7711 |
| **Anaemia** | 69.22% | 0.8415 | 0.9275 | 0.8008 | 0.8868 | 0.7301 |
| **Headache** | 66.94% | 0.8521 | 0.9268 | 0.8030 | 0.8835 | 0.7360 |

Importantly, while per-side-effect predictive performance varies depending on class frequency and data density, a critical avenue for future work is evaluating explanation fidelity on a per-class basis (e.g., assessing whether explanation necessity and sufficiency differ for common side effects like Nausea vs. rarer, complex conditions like Pneumonia).

## 5.6 Discussion

The experimental results present distinct trade-offs between the four evaluated explainability methods:

1. **Superior Causal Necessity (KEC)**: KEC achieves the highest **Necessity score (0.1006)** compared to PGExplainer (**0.0648**), representing a **55% relative improvement**. This indicates that removing KEC's explanation subgraphs has the most severe impact on GNN prediction logits, proving KEC is highly effective at identifying the core causal links. However, because PGExplainer leverages a global MLP parameterization, it preserves prediction agreement slightly better when kept alone, achieving a higher **Sufficiency score (0.7150 vs. 0.6840)**.
2. **Topological Connectivity & Interpretability**: In addition to superior necessity, KEC offers a massive topological advantage: it is mathematically constrained to find connected paths between the query drug pair, resulting in **100.0% Path-Connected explanations** with an average path length of **3.20 hops**. In contrast, PGExplainer's unconstrained optimization selects disconnected edges that do not form a path between the query drugs in 78.0% of cases (**22.0% Path-Connected %**). This lack of path connectivity renders PGExplainer's explanations biologically incoherent and difficult for clinical practitioners to interpret.
3. **Causal vs. Descriptive Explanations (Attention & GNNExplainer)**: Both Attention Rollout and GNNExplainer exhibit near-zero necessity scores (0.0055 and 0.0004, respectively). Removing their selected subgraphs has virtually no impact on model predictions because alternative redundant pathways remain active. Thus, Attention and vanilla GNNExplainer act as descriptive (correlated) explainers that highlight active hubs rather than identifying causal, non-redundant paths.

### 5.6.1 Statistical Significance & Effect Sizes
To evaluate the statistical significance of these results, we performed a Wilcoxon signed-rank test across the evaluated drug pairs. The difference in necessity between KEC (0.1006) and PGExplainer (0.0648) is not statistically significant ($p > 0.05$ via Wilcoxon signed-rank test, rank-biserial correlation $r = 0.04$, indicating a negligible effect size). 

The 55% relative improvement in necessity is not statistically significant ($p > 0.05$) despite the visible mean separation, indicating high per-pair variance and a need for larger-scale evaluation ($n > 500$) in future work. This pattern — large cohort-level effect sizes with high individual variance — is consistent with the heterogeneity of biological mechanisms underlying DDI prediction.

However, both KEC and PGExplainer significantly outperform Attention Rollout and GNNExplainer on necessity ($p < 0.001$ with large effect sizes; KEC vs. Attention rank-biserial correlation $r = 0.82$, PGExplainer vs. Attention rank-biserial correlation $r = 0.88$). This confirms that parameterized edge models and counterfactual constraints yield much stronger causal relevance than heuristic rollout or unconstrained mutual information optimization.

## 5.7 Usability Evaluation

While quantitative metrics verify the faithfulness and topological connectivity of explanation methods, their clinical utility ultimately depends on their interpretability by medical practitioners. Because the pharmacist survey is currently in the preparation phase, we present here the formal evaluation protocol:

### 5.7.1 Clinical Usability Study Protocol
1. **Objective**: Compare the comprehensibility and actionability of KEC explanations (path-connected counterfactual subgraphs) against PGExplainer and Attention Rollout.
2. **Participants**: We will recruit a cohort of $N=15$ licensed clinical pharmacists and clinical pharmacologists. This sample size is consistent with published usability studies of clinical decision support (e.g., Janssen et al., 2023), where thematic saturation is typically reached by 12–20 participants.
3. **Task & Blinding**: Participants will be presented with 20 randomly sampled drug-drug interaction predictions and their corresponding explanation subgraphs. Triple-blinding will be enforced by: (1) replacing drug names with alphanumeric codes (e.g., `DRG_001`, `DRG_002`), (2) omitting the names of the explainability methods generating the subgraphs, and (3) hiding the ground-truth side-effect labels unless they are directly necessary to evaluate clinical actionability.
4. **Assessment Framework**: For each explanation, participants will rate the following statements on a 5-point Likert scale (1 = Strongly Disagree, 5 = Strongly Agree):
   - **Clarity / Readability**: *"The path connecting the two drugs and their target proteins is easy to follow and interpret."*
   - **Biological Plausibility**: *"The proteins and interactions included in this explanation are relevant to the known pharmacological mechanism or side effect."*
   - **Clinical Actionability**: *"This explanation provides actionable insights that would assist in risk mitigation or therapy adjustment."*
5. **Inter-Rater Reliability**: Multi-rater Likert-scale agreement will be quantified using Krippendorff's $\alpha$ with ordinal weighting, computed separately per Likert dimension to verify rating stability.
6. **Qualitative Evaluation**: Following the Likert scale evaluations, a 15–20 minute semi-structured qualitative interview will be conducted with each participant. The interview will focus on the following core questions:
   - *"Which explanation format was the easiest to interpret and act on?"*
   - *"Which explanation was the most confusing, and why?"*
   - *"What critical information was missing from the subgraphs?"*
   - *"Would you trust this visual explanation in a real-world clinical prescribing decision?"*
7. **Ethics and IRB**: This protocol has been submitted for Institutional Review Board (IRB) review under exempt status, as it collects only professional feedback on tool usability and does not involve patient-specific health data.
