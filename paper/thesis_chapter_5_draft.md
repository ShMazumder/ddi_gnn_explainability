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

### 5.2.4 Knowledge-Enhanced Counterfactual (KEC)

Our proposed method identifies the minimal set of edges whose removal changes the model's prediction. KEC leverages the knowledge graph structure to constrain the search space to biologically plausible perturbations within the k-hop neighbourhood of the query drug pair.

### 5.2.5 Graph Homogenization and Bidirectional Message Flow

To apply homogeneous GNN layers (GAT) to the heterogeneous clinical knowledge graph containing drugs, proteins, and clinical relations, the graph is mapped to a unified homogeneous representation. Crucially, the drug-protein binding relations (`binds`) and protein-protein interactions (`interacts`) are modeled as bidirectional (undirected) edges in the homogeneous edge index. 

Without bidirectional modeling (i.e., using only directed edges from drugs to proteins), drug nodes would possess zero incoming edges in the message-passing graph. As a result, GNN layers would be unable to propagate topological information to update drug representations, reducing the model to a simple embedding lookup. Making all relations bidirectional ensures that topological information flows from proteins to drugs, enabling the GAT layers to construct structure-aware representations and allowing explainability wrappers to identify faithful, non-trivial explanation subgraphs.

## 5.3 Faithfulness Metrics

| Metric | Definition | Ideal |
|--------|-----------|-------|
| Sufficiency | Prediction agreement using only the explanation subgraph | High |
| Necessity | Prediction drop when explanation edges are removed | High |
| Fidelity+ | Average ΔP when explanation is removed | High |
| Fidelity- | Average ΔP when only explanation is kept | Low |
| Sparsity | Fraction of edges in explanation | Low |

## 5.4 Results

| Method | Sufficiency | Necessity |
|--------|------------|-----------|
| Attention Rollout | 0.620 | 0.001 |
| GNNExplainer | 0.620 | 0.000 |
| PGExplainer | 0.650 | 0.081 |
| KEC | 0.580 | 0.139 |

## 5.5 Discussion

The experimental results present distinct trade-offs between the four evaluated explainability methods:

1. **Counterfactual Efficacy (KEC)**: The proposed Knowledge-Enhanced Counterfactual (KEC) method achieves the highest **Necessity score (0.139)**. This indicates that removing the edges identified by KEC causes a significant drop (13.9% on average) in DDI prediction confidence. Since KEC explicitly searches for the minimal causal set of topological links to flip the prediction label under counterfactual conditions, it succeeds in isolating the most critical pathways that the GAT model relies on. KEC sacrifices a small amount of Sufficiency (0.580) to maximize this causal necessity.
2. **PGExplainer Generalization**: PGExplainer achieves the highest **Sufficiency score (0.650)** and a solid **Necessity score (0.081)**. Because PGExplainer trains a parameterized MLP globally across many instances, it learns to recognize generalized topological motifs that are highly sufficient on their own to preserve prediction states, while avoiding instance-specific noise.
3. **Failure of GNNExplainer and Attention on Necessity**: Both Attention Rollout (0.001) and GNNExplainer (0.000) show near-zero Necessity. In complex multi-relational networks, unconstrained optimization (GNNExplainer) and heuristic rollouts (Attention) tend to highlight highly active hub nodes (e.g. widely interacting proteins) rather than specific causal paths. Consequently, removing their explanation subgraphs does not impact prediction scores because alternative redundant message flows remain active in the graph.

These findings highlight that biologically-grounded counterfactual constraints (KEC) are essential for identifying necessary clinical pathways in heterogeneous networks.

## 5.6 Usability Evaluation

*(To be completed after pharmacist survey)*
