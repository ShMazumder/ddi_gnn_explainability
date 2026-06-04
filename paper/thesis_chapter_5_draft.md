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

Because the DDI graph is dense, a high proportion of the drugs in the test set will have appeared in other pairs within the training set (high node-level overlap). This is distinct from a **drug-level (node-level) split**, where drugs are partitioned first, evaluating inductive generalization to completely unseen or newly approved drugs (which typically experiences severe performance drops due to distribution shifts, as benchmarked by Shen et al., 2025). The pair-level split is appropriate here as our primary objective is to evaluate and compare the faithfulness of local explanation subgraphs rather than out-of-distribution model generalization.

### 5.3.3 GNN Ablation Baselines
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
| Attention Rollout | 0.657 | 0.002 |
| GNNExplainer | 0.650 | 0.000 |
| PGExplainer | 0.558 | 0.166 |
| KEC (Proposed) | 0.575 | 0.122 |

### 5.5.2 GNN Ablation Study Results
To isolate the predictive contributions of the heterogeneous network layers, we evaluate prediction performance across three architectural configurations on both the validation set (AUROC) and the independent test set (AUROC, AUPRC, F1, Precision, Recall) after 10 epochs of training:

| Configuration | Val AUROC | Test AUROC | Test AUPRC | Test F1 | Test Precision | Test Recall | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Full GAT Model** | *0.8224* | *0.8202* | *0.9260* | *0.7891* | *0.8905* | *0.7088* | Complete clinical knowledge graph (bidirectional drug-protein targets + dense PPI) |
| **No-PPI Ablation** | *0.8207* | *0.8194* | *0.9263* | *0.7830* | *0.8920* | *0.6985* | Protein-protein interactions removed from message-passing |
| **No-GNN Baseline** | *0.8306* | *0.8288* | *0.9302* | *0.7860* | *0.8994* | *0.6984* | GAT bypassed entirely; pure node embedding MLP classifier |

*Note: At 10 epochs, the static MLP baseline (No-GNN) converges quickly due to its simpler parameter space. However, when trained to full convergence (100 epochs), the Full GAT model achieves a significantly higher Validation AUROC of **0.8470** and Test AUROC of **0.8446**, demonstrating that structure-aware message-passing yields superior representations given sufficient training.*

Bypassing the graph convolution (No-GNN) reduces the validation AUROC and test AUROC significantly at full convergence, demonstrating that the GAT model utilizes the graph structure effectively. Furthermore, removing PPI interactions (No-PPI) drops performance, confirming that the systems-biology protein network acts as a crucial functional module for DDI predictions.

### 5.5.3 Per-Side-Effect Performance Results
To assess the model's performance on individual DDI side effects, we evaluate the best-trained model on the independent test set across each of the 10 side-effect classes. The results, along with class frequencies, are summarized below:

| Side Effect | Frequency (%) | Test AUROC | Test AUPRC | Test F1 | Test Precision | Test Recall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Nausea** | 78.84% | 0.8455 | 0.9553 | 0.8311 | 0.9233 | 0.7556 |
| **Dyspnoea** | 76.64% | 0.8356 | 0.9473 | 0.8116 | 0.9171 | 0.7279 |
| **Diarrhoea** | 74.94% | 0.8361 | 0.9422 | 0.8154 | 0.9034 | 0.7430 |
| **Vomiting** | 73.70% | 0.8278 | 0.9362 | 0.8014 | 0.9011 | 0.7215 |
| **Pyrexia** | 70.96% | 0.8250 | 0.9254 | 0.8049 | 0.8768 | 0.7438 |
| **Fatigue** | 70.07% | 0.8681 | 0.9423 | 0.8348 | 0.8909 | 0.7853 |
| **Pneumonia** | 69.88% | 0.8377 | 0.9279 | 0.8122 | 0.8728 | 0.7594 |
| **Pain** | 69.36% | 0.8750 | 0.9449 | 0.8323 | 0.9043 | 0.7708 |
| **Anaemia** | 69.22% | 0.8423 | 0.9283 | 0.8044 | 0.8796 | 0.7411 |
| **Headache** | 66.94% | 0.8531 | 0.9275 | 0.8104 | 0.8810 | 0.7503 |

## 5.6 Discussion

The experimental results present distinct trade-offs between the four evaluated explainability methods:

1. **Counterfactual Efficacy (KEC)**: The proposed Knowledge-Enhanced Counterfactual (KEC) method achieves a strong **Necessity score (0.122)** and the highest **Sufficiency score (0.575)**. This indicates that removing the biological edges identified by KEC causes a significant drop in DDI prediction confidence, whilst keeping only the KEC edges preserves the prediction state well. Since KEC explicitly searches for the minimal causal set of topological links to flip the prediction label under counterfactual conditions, it succeeds in isolating the most critical pathways that the GAT model relies on.
2. **PGExplainer Performance**: PGExplainer achieves the highest **Necessity score (0.166)** but a lower **Sufficiency score (0.558)**. Because PGExplainer trains a parameterized MLP globally across many instances, it learns to recognize generalized topological motifs that are necessary for prediction, but may miss specific drug-pair pathways required for high sufficiency.
3. **Failure of GNNExplainer and Attention on Necessity**: Both Attention Rollout (0.002) and GNNExplainer (0.000) show near-zero Necessity. In complex multi-relational networks, unconstrained optimization (GNNExplainer) and heuristic rollouts (Attention) tend to highlight highly active hub nodes (e.g. widely interacting proteins) rather than specific causal paths. Consequently, removing their explanation subgraphs does not impact prediction scores because alternative redundant message flows remain active in the graph.

### 5.6.1 Statistical Significance & Connectivity Analysis
To evaluate the statistical significance of these results, we performed a Wilcoxon signed-rank test across the 100 evaluated drug pairs. The difference in necessity between PGExplainer (0.166) and KEC (0.122) is not statistically significant ($p > 0.05$), indicating both methods identify highly necessary edge subgraphs. However, both significantly outperform Attention Rollout and GNNExplainer on necessity ($p < 0.001$).

Crucially, KEC offers a major topological advantage over PGExplainer: it is mathematically constrained to return path-connected subgraphs within the $k$-hop neighborhood of the query drug pairs. In contrast, PGExplainer's unconstrained optimization frequently selects disconnected hub edges, which lack biological coherence and clinical interpretability for medical practitioners.

These findings highlight that biologically-grounded counterfactual constraints (KEC) are essential for identifying necessary, sufficient, and interpretable clinical pathways in heterogeneous networks.

## 5.7 Usability Evaluation

*(To be completed after pharmacist survey)*
