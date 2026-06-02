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

## 5.3 Faithfulness Metrics

| Metric | Definition | Ideal |
|--------|-----------|-------|
| Sufficiency | Prediction agreement using only the explanation subgraph | High |
| Necessity | Prediction drop when explanation edges are removed | High |
| Fidelity+ | Average ΔP when explanation is removed | High |
| Fidelity- | Average ΔP when only explanation is kept | Low |
| Sparsity | Fraction of edges in explanation | Low |

## 5.4 Results

*Table 5.1: Faithfulness comparison across explanation methods (to be filled after running experiments)*

| Method | Sufficiency | Necessity | Fidelity+ | Fidelity- | Sparsity |
|--------|------------|-----------|-----------|-----------|----------|
| Attention Rollout | — | — | — | — | — |
| GNNExplainer | — | — | — | — | — |
| PGExplainer | — | — | — | — | — |
| KEC | — | — | — | — | — |

## 5.5 Discussion

*(To be completed after experiments)*

## 5.6 Usability Evaluation

*(To be completed after pharmacist survey)*
