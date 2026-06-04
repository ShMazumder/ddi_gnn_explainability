"""
Train the HeteroGATDDI model on the constructed knowledge graph.
Uses weighted BCE loss and evaluates with macro AUROC.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
import json
import pandas as pd
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
    data = torch.load(graph_path, weights_only=False)
    num_drugs = data['drug'].num_nodes
    num_proteins = data['protein'].num_nodes
    edge_index = data['drug', 'ddi', 'drug'].edge_index
    labels = data['drug', 'ddi', 'drug'].side_effect_label  # (num_pairs, num_side_effects)

    # Create bidirectional drug-protein edges
    dp_edges = data['drug', 'binds', 'protein'].edge_index.clone()
    dp_edges[1] += num_drugs
    pd_edges = torch.stack([dp_edges[1], dp_edges[0]], dim=0)
    
    # Create bidirectional protein-protein interactions
    p_edges = data['protein', 'interacts', 'protein'].edge_index.clone()
    p_edges[0] += num_drugs
    p_edges[1] += num_drugs
    p_edges_rev = torch.stack([p_edges[1], p_edges[0]], dim=0)
    
    hom_edge = torch.cat([dp_edges, pd_edges, p_edges, p_edges_rev], dim=1)

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
            preds_logits = model.predict_side_effects_logits(drug_emb, batch_pairs)
            loss = criterion(preds_logits, batch_labels)
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

    # Test evaluation
    print("\n--- Evaluating Best Model on Test Set ---")
    model.load_state_dict(torch.load("results/model_checkpoint.pt", map_location=device))
    model.eval()
    
    test_loader = DataLoader(TensorDataset(test_pairs, test_labels), batch_size=config["model"]["batch_size"])
    all_test_preds = []
    all_test_labels = []
    with torch.no_grad():
        drug_emb = model.forward({'drug': None, 'protein': None}, hom_edge, num_drugs)
        for batch_pairs, batch_labels in test_loader:
            batch_pairs = batch_pairs.to(device)
            preds = model.predict_side_effects(drug_emb, batch_pairs)
            all_test_preds.append(preds.cpu())
            all_test_labels.append(batch_labels.cpu())
            
    test_preds_cat = torch.cat(all_test_preds).numpy()
    test_labels_cat = torch.cat(all_test_labels).numpy()
    
    # Load side effect names
    side_effects_path = Path(config["data"]["graph_dir"]) / "side_effects.txt"
    if side_effects_path.exists():
        with open(side_effects_path, 'r', encoding='utf-8') as f:
            side_effect_names = [line.strip() for line in f if line.strip()]
    else:
        side_effect_names = [f"SideEffect_{i}" for i in range(test_labels_cat.shape[1])]
        
    # Overall metrics
    overall_auroc = roc_auc_score(test_labels_cat, test_preds_cat, average='macro')
    overall_auprc = average_precision_score(test_labels_cat, test_preds_cat, average='macro')
    test_preds_binary = (test_preds_cat >= 0.5).astype(int)
    overall_f1 = f1_score(test_labels_cat, test_preds_binary, average='macro', zero_division=0)
    overall_precision = precision_score(test_labels_cat, test_preds_binary, average='macro', zero_division=0)
    overall_recall = recall_score(test_labels_cat, test_preds_binary, average='macro', zero_division=0)
    
    print("\n==========================================")
    print("Overall Test Set Metrics")
    print("==========================================")
    print(f"Test AUROC (Macro): {overall_auroc:.4f}")
    print(f"Test AUPRC (Macro): {overall_auprc:.4f}")
    print(f"Test F1 (Macro):    {overall_f1:.4f}")
    print(f"Test Precision:     {overall_precision:.4f}")
    print(f"Test Recall:        {overall_recall:.4f}")
    print("==========================================\n")
    
    # Save overall metrics
    overall_metrics = {
        'test_auroc': float(overall_auroc),
        'test_auprc': float(overall_auprc),
        'test_f1': float(overall_f1),
        'test_precision': float(overall_precision),
        'test_recall': float(overall_recall)
    }
    with open("results/test_metrics.json", "w") as f:
        json.dump(overall_metrics, f, indent=4)
        
    # Per-class analysis
    print("Per-Side-Effect Metrics Table:")
    print("| Side Effect | Frequency (%) | AUROC | AUPRC | F1 | Precision | Recall |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    
    per_class_rows = []
    for c_idx, se_name in enumerate(side_effect_names):
        freq = float(test_labels_cat[:, c_idx].mean()) * 100
        if len(np.unique(test_labels_cat[:, c_idx])) < 2:
            se_auroc = 0.5
            se_auprc = 0.5
        else:
            se_auroc = roc_auc_score(test_labels_cat[:, c_idx], test_preds_cat[:, c_idx])
            se_auprc = average_precision_score(test_labels_cat[:, c_idx], test_preds_cat[:, c_idx])
            
        se_bin_pred = test_preds_binary[:, c_idx]
        se_f1 = f1_score(test_labels_cat[:, c_idx], se_bin_pred, zero_division=0)
        se_precision = precision_score(test_labels_cat[:, c_idx], se_bin_pred, zero_division=0)
        se_recall = recall_score(test_labels_cat[:, c_idx], se_bin_pred, zero_division=0)
        
        print(f"| {se_name:<11} | {freq:>13.2f}% | {se_auroc:>5.4f} | {se_auprc:>5.4f} | {se_f1:>5.4f} | {se_precision:>9.4f} | {se_recall:>6.4f} |")
        
        per_class_rows.append({
            'side_effect': se_name,
            'frequency_pct': freq,
            'auroc': se_auroc,
            'auprc': se_auprc,
            'f1': se_f1,
            'precision': se_precision,
            'recall': se_recall
        })
        
    pd.DataFrame(per_class_rows).to_csv("results/per_class_metrics.csv", index=False)


if __name__ == "__main__":
    train()
