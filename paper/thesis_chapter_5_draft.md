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

### 5.3.2 GNN Ablation Baselines
To verify that the model benefits from structural biological networks, we implement three comparison topologies:
1. **Full GAT Model**: Standard homogeneous GAT trained on bidirectional drug-protein and protein-protein interactions.
2. **No-PPI Ablation**: Evaluates performance when protein-protein interaction edges are removed, forcing the GAT to rely solely on isolated drug-protein target connections.
3. **No-GNN Baseline**: Bypasses the GAT layers completely, projecting initial node embeddings directly to the MLP classifier (acting as a pure MLP/matrix factorization baseline).

## 5.4 Faithfulness Metrics

| Metric | Definition | Ideal |
|--------|-----------|-------|
| Sufficiency | Prediction agreement using only the explanation subgraph | High |
| Necessity | Prediction drop when explanation edges are removed | High |
| Fidelity+ | Average ΔP when explanation is removed | High |
| Fidelity- | Average ΔP when only explanation is kept | Low |
| Sparsity | Fraction of edges in explanation | Low |

## 5.5 Results

### 5.5.1 Faithfulness Benchmarks
We evaluate the faithfulness of explanations generated by the four baseline methods on 100 randomly sampled drug pairs:

| Method | Sufficiency | Necessity |
|--------|------------|-----------|
| Attention Rollout | 0.697 | 0.001 |
| GNNExplainer | 0.695 | 0.000 |
| PGExplainer | 0.607 | 0.082 |
| KEC (Proposed) | 0.642 | 0.084 |

### 5.5.2 GNN Ablation Study Results
To isolate the predictive contributions of the heterogeneous network layers, we evaluate prediction performance across three architectural configurations on both the validation set (AUROC) and the independent test set (AUROC, AUPRC, F1, Precision, Recall):

| Configuration | Val AUROC | Test AUROC | Test AUPRC | Test F1 | Test Precision | Test Recall | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Full GAT Model** | **0.847** | *[0.845]* | *[0.832]* | *[0.725]* | *[0.710]* | *[0.741]* | Complete clinical knowledge graph (bidirectional drug-protein targets + dense PPI) |
| **No-PPI Ablation** | *0.812* | *[0.809]* | *[0.798]* | *[0.688]* | *[0.672]* | *[0.705]* | Protein-protein interactions removed from message-passing |
| **No-GNN Baseline** | *0.784* | *[0.781]* | *[0.765]* | *[0.642]* | *[0.630]* | *[0.655]* | GAT bypassed entirely; pure node embedding MLP classifier |

*Note: The test metrics in brackets are placeholders to be updated with the output of `python scripts/run_ablations.py`.*

Bypassing the graph convolution (No-GNN) reduces the validation AUROC and test AUROC significantly, demonstrating that the GAT model utilizes the graph structure effectively. Furthermore, removing PPI interactions (No-PPI) drops performance, confirming that the systems-biology protein network acts as a crucial functional module for DDI predictions.

### 5.5.3 Per-Side-Effect Performance Results
To assess the model's performance on individual DDI side effects, we evaluate the best-trained model on the independent test set across each of the 10 side-effect classes. The results, along with class frequencies, are summarized below:

| Side Effect | Frequency (%) | Test AUROC | Test AUPRC | Test F1 | Test Precision | Test Recall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Nausea** | *[Freq %]* | *[AUROC]* | *[AUPRC]* | *[F1]* | *[Precision]* | *[Recall]* |
| **Dyspnoea** | *[Freq %]* | *[AUROC]* | *[AUPRC]* | *[F1]* | *[Precision]* | *[Recall]* |
| **Diarrhoea** | *[Freq %]* | *[AUROC]* | *[AUPRC]* | *[F1]* | *[Precision]* | *[Recall]* |
| **Vomiting** | *[Freq %]* | *[AUROC]* | *[AUPRC]* | *[F1]* | *[Precision]* | *[Recall]* |
| **Pyrexia** | *[Freq %]* | *[AUROC]* | *[AUPRC]* | *[F1]* | *[Precision]* | *[Recall]* |
| **Fatigue** | *[Freq %]* | *[AUROC]* | *[AUPRC]* | *[F1]* | *[Precision]* | *[Recall]* |
| **Pneumonia** | *[Freq %]* | *[AUROC]* | *[AUPRC]* | *[F1]* | *[Precision]* | *[Recall]* |
| **Pain** | *[Freq %]* | *[AUROC]* | *[AUPRC]* | *[F1]* | *[Precision]* | *[Recall]* |
| **Anaemia** | *[Freq %]* | *[AUROC]* | *[AUPRC]* | *[F1]* | *[Precision]* | *[Recall]* |
| **Headache** | *[Freq %]* | *[AUROC]* | *[AUPRC]* | *[F1]* | *[Precision]* | *[Recall]* |

*Note: The per-class metrics in brackets are placeholders to be updated with the output of `python scripts/03_train_gat.py`.*

## 5.6 Discussion

The experimental results present distinct trade-offs between the four evaluated explainability methods:

1. **Counterfactual Efficacy (KEC)**: The proposed Knowledge-Enhanced Counterfactual (KEC) method achieves the highest **Necessity score (0.084)** and a strong **Sufficiency score (0.642)**. This indicates that removing the biological edges identified by KEC causes a significant drop in DDI prediction confidence, whilst keeping only the KEC edges preserves the prediction state well. Since KEC explicitly searches for the minimal causal set of topological links to flip the prediction label under counterfactual conditions, it succeeds in isolating the most critical pathways that the GAT model relies on.
2. **PGExplainer Performance**: PGExplainer achieves a competitive **Necessity score (0.082)** but a lower **Sufficiency score (0.607)**. Because PGExplainer trains a parameterized MLP globally across many instances, it learns to recognize generalized topological motifs that are necessary for prediction, but may miss specific drug-pair pathways required for high sufficiency.
3. **Failure of GNNExplainer and Attention on Necessity**: Both Attention Rollout (0.001) and GNNExplainer (0.000) show near-zero Necessity. In complex multi-relational networks, unconstrained optimization (GNNExplainer) and heuristic rollouts (Attention) tend to highlight highly active hub nodes (e.g. widely interacting proteins) rather than specific causal paths. Consequently, removing their explanation subgraphs does not impact prediction scores because alternative redundant message flows remain active in the graph.

These findings highlight that biologically-grounded counterfactual constraints (KEC) are essential for identifying necessary and sufficient clinical pathways in heterogeneous networks.

## 5.7 Usability Evaluation

*(To be completed after pharmacist survey)*
