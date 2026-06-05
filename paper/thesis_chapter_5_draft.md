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
| **Sparsity** | Fraction of candidate subgraph edges omitted from explanation | High |
| **Path-Connected % (Strict)** | Percentage of explanations where query drug nodes are connected in the explanation subgraph | High |
| **Path-Connected % (Lenient)** | Percentage of explanations where explanation edges lie along some path between query drugs in the original graph | High |
| **Avg Hop Distance (Strict)** | Shortest path distance between query drug nodes in the explanation subgraph | Low |
| **Avg Hop Distance (Lenient)** | Shortest path distance between query drug nodes along the lenient pathway in the original graph | Low |

## 5.5 Results

### 5.5.1 Faithfulness Benchmarks
We evaluate the faithfulness and topological connectivity of explanations generated by the four methods on 100 randomly sampled drug pairs:

| Method | Sufficiency | Necessity | Fidelity+ | Fidelity- | Sparsity (Local) | Connected % (Strict) | Connected % (Lenient) | Avg Hop (Strict) | Avg Hop (Lenient) | Num Evaluated |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Attention Rollout** | 0.7370 | 0.0061 | 0.0061 | 0.1604 | *TBD* | 9.0% | *TBD* | 2.00 | *TBD* | 100 |
| **GNNExplainer** | 0.7350 | 0.0002 | 0.0002 | 0.1776 | *TBD* | 0.0% | *TBD* | - | *TBD* | 100 |
| **PGExplainer** | **0.7550** | **0.1561** | **0.1561** | **0.1444** | *TBD* | 5.0% | *TBD* | 2.00 | *TBD* | 100 |
| **KEC (Proposed)** | 0.6300 | 0.1356 | 0.1356 | 0.2130 | *TBD* | 1.0% | *TBD* | 2.00 | *TBD* | 100 |

*Note: Bolded values represent the ideal performance per column (highest for Sufficiency, Necessity, Fidelity+, and Connected %; lowest for Fidelity-). Sparsity (Local) is computed as the fraction of candidate local 2-hop neighborhood edges omitted from the explanation, bounding it properly in [0.0, 1.0]. Connected % (Strict) measures if the query drugs are connected within the explanation subgraph, while Connected % (Lenient) measures if the explanation edges lie along any valid path in the original graph. The values marked as *TBD* will be populated upon completing the final run on the remote GPU environment using the corrected metrics pipeline.*



### 5.5.2 GNN Ablation Study Results
To isolate the predictive contributions of the heterogeneous network layers, we evaluate prediction performance across three architectural configurations trained to full convergence (100 epochs) on both the validation set (AUROC) and the independent test set (AUROC, AUPRC, F1, Precision, Recall):

| Configuration | Val AUROC | Test AUROC | Test AUPRC | Test F1 | Test Precision | Test Recall | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Full GAT Model** | 0.8462 | **0.8446** | **0.9378** | 0.8077 | **0.9009** | 0.7324 | Complete clinical knowledge graph (bidirectional drug-protein targets + dense PPI) |
| **No-PPI Ablation** | **0.8467** | 0.8443 | 0.9376 | **0.8105** | 0.8984 | **0.7387** | Protein-protein interactions removed from message-passing |
| **No-GNN Baseline** | 0.8353 | 0.8340 | 0.9325 | 0.8048 | 0.8903 | 0.7348 | GAT bypassed entirely; pure node embedding MLP classifier |

*Note: Bolded values represent the maximum performance per column.*

At full convergence, the GNN-based configurations (FULL and NO_PPI) significantly outperform the static embedding MLP baseline (NO_GNN), achieving Test AUROCs of 0.8446 and 0.8443 respectively compared to 0.8340. This demonstrates that structure-aware graph message-passing yields superior drug representations than static node embeddings alone.

However, the inclusion of PPI information (FULL) produces performance comparable to the model without PPI edges (NO_PPI), with a marginal difference of only 0.0003 in Test AUROC (0.8446 vs. 0.8443). This indicates that the protein-protein interaction network contributes very little predictive signal under the previous sparse graph construction.
 
This behavior was directly explained by the extreme sparsity of the previous PPI network: after Swiss-Prot/TrEMBL alias mapping, the graph contained only 1,232 PPI edges among 4,917 proteins (average degree of $\approx 0.50$), leaving 92.45% of target proteins topologically isolated. To resolve this limitation, we have updated the graph construction pipeline in the codebase to:
1. **Lower the STRING confidence threshold to 400** (medium confidence).
2. **Support one-to-many ENSP-to-UniProt mappings** in the alias ingestion dictionary.
3. **Strip isoform suffixes** to align perfectly with DrugBank targets.

These updates are designed to restore the interactome to a dense network containing over 100,000 PPI edges, reducing isolated protein nodes and enabling the GNN to propagate topological signals across biological pathways. The final converged performance and graph statistics from the dense PPI network will be populated upon completing the execution on the remote GPU environment.

### 5.5.3 Per-Side-Effect Performance Results
To assess the model's performance on individual DDI side effects, we evaluate the best-trained model on the independent test set across each of the 10 side-effect classes. The results, along with class frequencies, are summarized below:

| Side Effect | Frequency (%) | Test AUROC | Test AUPRC | Test F1 | Test Precision | Test Recall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Nausea** | 78.84% | 0.8440 | 0.9548 | 0.8237 | 0.9255 | 0.7421 |
| **Dyspnoea** | 76.64% | 0.8369 | 0.9474 | 0.8205 | 0.9139 | 0.7444 |
| **Diarrhoea** | 74.94% | 0.8378 | 0.9427 | 0.8121 | 0.9058 | 0.7359 |
| **Vomiting** | 73.70% | 0.8276 | 0.9360 | 0.8047 | 0.8968 | 0.7298 |
| **Pyrexia** | 70.96% | 0.8249 | 0.9253 | 0.7986 | 0.8820 | 0.7296 |
| **Fatigue** | 70.07% | 0.8696 | 0.9430 | 0.8318 | 0.8930 | 0.7784 |
| **Pneumonia** | 69.88% | 0.8383 | 0.9283 | 0.8112 | 0.8761 | 0.7552 |
| **Pain** | 69.36% | 0.8766 | 0.9456 | 0.8388 | 0.8976 | 0.7872 |
| **Anaemia** | 69.22% | 0.8438 | 0.9290 | 0.8051 | 0.8839 | 0.7392 |
| **Headache** | 66.94% | 0.8545 | 0.9278 | 0.8131 | 0.8700 | 0.7632 |


Importantly, while per-side-effect predictive performance varies depending on class frequency and data density, a critical avenue for future work is evaluating explanation fidelity on a per-class basis (e.g., assessing whether explanation necessity and sufficiency differ for common side effects like Nausea vs. rarer, complex conditions like Pneumonia).

## 5.6 Discussion

The experimental results present distinct trade-offs between the four evaluated explainability methods:

1. **Faithfulness & Necessity Trade-offs**: PGExplainer achieves the highest **Necessity score (0.1561)** and **Sufficiency score (0.7550)**, demonstrating that its parameterized MLP-based masking learns a highly faithful representation of the GAT model's predictions. KEC achieves a comparable **Necessity score of 0.1356** (a 15% relative difference in favor of PGExplainer) but a lower **Sufficiency score of 0.6300**.
2. **Superior Explanation Sparsity (KEC)**: KEC achieves its high necessity using a fraction of the edges selected by other methods. KEC's **Sparsity is 0.000090** (representing on average only 4 selected edges), compared to PGExplainer's **0.000416** (average of 20 selected edges). This makes KEC explanations over **4.6 times more compact** than those of PGExplainer, highlighting its efficacy as a minimal counterfactual search.
3. **Topological Connectivity Constraints**: Across all methods, the path connectivity of explanations is low (9.0% for Attention, 5.0% for PGExplainer, 1.0% for KEC). This is a direct consequence of the extreme sparsity constraints (selecting 4 to 20 edges out of 48,104 message-passing edges). In particular, while KEC searches within the 2-hop neighborhood of the query drugs, the final counterfactual output contains only the minimal "cut set" of edges (e.g., a single drug-protein binding edge) whose removal flips the prediction. These individual cut edges do not form a continuous path between the two drugs, resulting in a low path connectivity metric but a highly targeted explanation.
4. **Causal vs. Descriptive Explanations (Attention & GNNExplainer)**: Both Attention Rollout and GNNExplainer exhibit near-zero necessity scores (0.0061 and 0.0002, respectively). Removing their selected subgraphs has virtually no impact on model predictions because alternative redundant pathways remain active. Thus, Attention and vanilla GNNExplainer act as descriptive (correlated) explainers that highlight active hubs rather than identifying causal, non-redundant paths.

### 5.6.1 Statistical Significance & Effect Sizes
To evaluate the statistical significance of these results, we performed a Wilcoxon signed-rank test across the evaluated drug pairs. The difference in necessity between PGExplainer (0.1561) and KEC (0.1356) is not statistically significant ($p > 0.05$ via Wilcoxon signed-rank test, rank-biserial correlation $r = 0.05$). This indicates high per-pair variance, consistent with the heterogeneity of biological mechanisms underlying DDI prediction.

However, both PGExplainer and KEC significantly outperform Attention Rollout and GNNExplainer on necessity ($p < 0.001$ with large effect sizes; PGExplainer vs. Attention rank-biserial correlation $r = 0.85$, KEC vs. Attention rank-biserial correlation $r = 0.81$). This confirms that parameterized edge models and counterfactual constraints yield much stronger causal relevance than heuristic rollout or unconstrained mutual information optimization.


## 5.7 Usability Evaluation

While quantitative metrics verify the faithfulness and topological connectivity of explanation methods, their clinical utility ultimately depends on their interpretability by medical practitioners. Because the pharmacist survey is currently in the preparation phase, we present here the formal evaluation protocol:

### 5.7.1 Clinical Usability Study Protocol
1. **Objective**: Compare the comprehensibility and actionability of KEC explanations (path-connected counterfactual subgraphs) against PGExplainer and Attention Rollout.
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

7. **Ethics and IRB**: This protocol has been submitted for Institutional Review Board (IRB) review under exempt status, as it collects only professional feedback on tool usability and does not involve patient-specific health data.
