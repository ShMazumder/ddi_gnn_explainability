# Explainable GNN for Drug–Drug Interaction Prediction

This repository contains the complete implementation for the master's thesis:
**"Explainable Graph Neural Networks for Drug–Drug Interaction Prediction: A Faithfulness-Evaluated Study over Clinical Knowledge Graphs"**

## Quick start

```bash
# Clone repository
git clone https://github.com/yourname/ddi_gnn_explainability.git
cd ddi_gnn_explainability

# Install dependencies
pip install -r requirements.txt

# Download datasets (requires internet, ~2 GB)
python scripts/01_download_data.py

# Build knowledge graph
python scripts/02_build_kg.py

# Train GAT model
python scripts/03_train_gat.py

# Run explanation methods and faithfulness evaluation
python scripts/04_explain_methods.py
python scripts/05_faithfulness.py

# Generate visualisations
python scripts/06_visualise.py

# Run GNN ablation studies
python scripts/run_ablations.py

# Run graph diagnostics
python scripts/diagnose_graph.py
```

## Testing & Model Dependency Verification

To verify that GAT node representations depend on the graph topology, you can run the unit test on synthetic data:

```bash
python3 tests/test_model_dependence.py
```

This verifies that the GNN is executing bidirectional message-passing correctly and that the explanation subgraphs are faithful.

## STRING PPI Graph Diagnostics

To analyze the topology, connectivity components, and sparsity of the assembled STRING PPI network, run the diagnostic script:

```bash
python scripts/diagnose_graph.py
```

This calculates the average protein degree, edge density, the size of the largest connected component (LCC), and the ratio of connected vs. isolated target proteins.

## Homogeneous Graph Representation & Bidirectional Flow

The clinical knowledge graph aggregates multi-source datasets (DrugBank, TWOSIDES, STRING) into a heterogeneous structure, which is then mapped to a unified homogeneous representation for GAT Conv processing. 

Crucially, **edges are modeled as bidirectional (undirected)**:
1. Drug-to-protein and protein-to-drug relations (`binds`, `bound_by`).
2. Protein-protein interactions (`interacts`).

By ensuring edges flow in both directions, drug nodes can successfully aggregate topological features from their binding proteins and neighbors. This prevents the GNN from degrading into a trivial MLP embedding lookup, enabling explainability wrappers (Attention Rollout, GNNExplainer, PGExplainer, and KEC) to compute faithful, structure-dependent explanation subgraphs.

## Datasets

The script `01_download_data.py` automatically downloads:
- DrugBank (requires academic license – you need to place the XML file manually)
- TWOSIDES (public)
- SIDER (public)
- STRING (public)

See `config.yaml` for file paths.

## Citation

If you use this code, please cite the thesis (to be added).
