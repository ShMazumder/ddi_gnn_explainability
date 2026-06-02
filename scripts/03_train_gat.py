"""
Train the HeteroGATDDI model on the constructed knowledge graph.
Uses weighted BCE loss and evaluates with macro AUROC.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import yaml
from pathlib import Path
import sys
sys.path.append(".")
from models.gat_ddi import HeteroGATDDI


def train():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    graph_path = Path(config["data"]["graph_dir"]) / "hetero_data.pt"
    data = torch.load(graph_path)
    num_drugs = data['drug'].num_nodes
    num_proteins = data['protein'].num_nodes
    edge_index = data['drug', 'ddi', 'drug'].edge_index
    labels = data['drug', 'ddi', 'drug'].side_effect_label  # (num_pairs, num_side_effects)

    # Create homogeneous edge index (drugs + proteins)
    # For simplicity, combine drug-protein and protein-protein edges
    drug_protein_edges = data['drug', 'binds', 'protein'].edge_index
    # Offset protein indices by num_drugs
    drug_protein_edges[1] += num_drugs
    ppi_edges = data['protein', 'interacts', 'protein'].edge_index
    ppi_edges[0] += num_drugs
    ppi_edges[1] += num_drugs
    hom_edge = torch.cat([drug_protein_edges, ppi_edges], dim=1)

    # Train/val/test split on drug pairs
    num_pairs = edge_index.size(1)
    indices = np.arange(num_pairs)
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

    train_pairs = edge_index[:, train_idx].t()
    train_labels = labels[train_idx]
    val_pairs = edge_index[:, val_idx].t()
    val_labels = labels[val_idx]
    test_pairs = edge_index[:, test_idx].t()
    test_labels = labels[test_idx]

    # Data loaders
    train_dataset = TensorDataset(train_pairs, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=config["model"]["batch_size"], shuffle=True)
    val_loader = DataLoader(TensorDataset(val_pairs, val_labels), batch_size=config["model"]["batch_size"])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HeteroGATDDI(
        num_drugs=num_drugs,
        num_proteins=num_proteins,
        drug_hidden=config["model"]["drug_hidden"],
        protein_hidden=config["model"]["protein_hidden"],
        gat_hidden=128,
        gat_heads=config["model"]["gat_heads"],
        out_dim=labels.shape[1],
        dropout=config["model"]["dropout"]
    ).to(device)

    # Weighted BCE
    pos_count = labels.sum(dim=0)
    neg_count = labels.shape[0] - pos_count
    pos_weight = (neg_count / (pos_count + 1e-8)).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=config["model"]["learning_rate"])
    hom_edge = hom_edge.to(device)

    # Ensure results directory exists
    Path("results").mkdir(parents=True, exist_ok=True)

    best_auroc = 0.0
    for epoch in range(config["model"]["epochs"]):
        model.train()
        total_loss = 0
        for batch_pairs, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            batch_pairs = batch_pairs.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            drug_emb = model.forward({'drug': None, 'protein': None}, hom_edge, num_drugs)
            preds = model.predict_side_effects(drug_emb, batch_pairs)
            loss = criterion(preds, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            drug_emb = model.forward({'drug': None, 'protein': None}, hom_edge, num_drugs)
            for batch_pairs, batch_labels in val_loader:
                batch_pairs = batch_pairs.to(device)
                preds = model.predict_side_effects(drug_emb, batch_pairs)
                all_preds.append(preds.cpu())
                all_labels.append(batch_labels.cpu())
        preds_cat = torch.cat(all_preds).numpy()
        labels_cat = torch.cat(all_labels).numpy()
        auroc = roc_auc_score(labels_cat, preds_cat, average='macro')
        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | Val AUROC: {auroc:.4f}")

        if auroc > best_auroc:
            best_auroc = auroc
            torch.save(model.state_dict(), "results/model_checkpoint.pt")

    print(f"Training complete. Best AUROC: {best_auroc:.4f}")


if __name__ == "__main__":
    train()
