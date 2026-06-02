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
```

## Datasets

The script `01_download_data.py` automatically downloads:
- DrugBank (requires academic license – you need to place the XML file manually)
- TWOSIDES (public)
- SIDER (public)
- STRING (public)

See `config.yaml` for file paths.

## Citation

If you use this code, please cite the thesis (to be added).
