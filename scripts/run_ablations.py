import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import yaml
from pathlib import Path
import sys
sys.path.append(".")
from models.gat_ddi import HeteroGATDDI

def train_and_evaluate_mode(mode, config, data, train_idx, val_idx, test_idx, device, epochs=10):
    num_drugs = data['drug'].num_nodes
    num_proteins = data['protein'].num_nodes
    edge_index = data['drug', 'ddi', 'drug'].edge_index
    labels = data['drug', 'ddi', 'drug'].side_effect_label

    # Build GNN homogeneous edge index based on mode
    if mode == "no_gnn":
        hom_edge = torch.empty((2, 0), dtype=torch.long, device=device)
    else:
        # Create bidirectional drug-protein edges
        dp_edges = data['drug', 'binds', 'protein'].edge_index.clone()
        dp_edges[1] += num_drugs
        pd_edges = torch.stack([dp_edges[1], dp_edges[0]], dim=0)
        
        if mode == "no_ppi":
            # Remove PPI edges
            hom_edge = torch.cat([dp_edges, pd_edges], dim=1).to(device)
        else:  # "full"
            # Create bidirectional PPI edges
            p_edges = data['protein', 'interacts', 'protein'].edge_index.clone()
            p_edges[0] += num_drugs
            p_edges[1] += num_drugs
            p_edges_rev = torch.stack([p_edges[1], p_edges[0]], dim=0)
            hom_edge = torch.cat([dp_edges, pd_edges, p_edges, p_edges_rev], dim=1).to(device)

    # Train/val/test splits
    train_pairs = edge_index[:, train_idx].t()
    train_labels = labels[train_idx]
    val_pairs = edge_index[:, val_idx].t()
    val_labels = labels[val_idx]
    test_pairs = edge_index[:, test_idx].t()
    test_labels = labels[test_idx]

    train_dataset = TensorDataset(train_pairs, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=config["model"]["batch_size"], shuffle=True)
    val_loader = DataLoader(TensorDataset(val_pairs, val_labels), batch_size=config["model"]["batch_size"])
    test_loader = DataLoader(TensorDataset(test_pairs, test_labels), batch_size=config["model"]["batch_size"])

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

    pos_count = labels.sum(dim=0)
    neg_count = labels.shape[0] - pos_count
    pos_weight = (neg_count / (pos_count + 1e-8)).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=config["model"]["learning_rate"])

    print(f"\n--- Training in '{mode}' mode for {epochs} epochs ---")
    
    best_val_auroc = 0.0
    best_model_state = None
    import copy

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_pairs, batch_labels in train_loader:
            batch_pairs = batch_pairs.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            
            # Use bypass_gat flag if mode is no_gnn
            drug_emb = model.forward(
                {'drug': None, 'protein': None}, 
                hom_edge, 
                num_drugs, 
                bypass_gat=(mode == "no_gnn")
            )
            
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
            drug_emb = model.forward(
                {'drug': None, 'protein': None}, 
                hom_edge, 
                num_drugs, 
                bypass_gat=(mode == "no_gnn")
            )
            for batch_pairs, batch_labels in val_loader:
                batch_pairs = batch_pairs.to(device)
                preds = model.predict_side_effects(drug_emb, batch_pairs)
                all_preds.append(preds.cpu())
                all_labels.append(batch_labels.cpu())
                
        preds_cat = torch.cat(all_preds).numpy()
        labels_cat = torch.cat(all_labels).numpy()
        auroc = roc_auc_score(labels_cat, preds_cat, average='macro')
        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | Val AUROC: {auroc:.4f}")
        
        if auroc > best_val_auroc:
            best_val_auroc = auroc
            best_model_state = copy.deepcopy(model.state_dict())

    # Evaluate best model on test set
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.eval()

    all_test_preds = []
    all_test_labels = []
    with torch.no_grad():
        drug_emb = model.forward(
            {'drug': None, 'protein': None}, 
            hom_edge, 
            num_drugs, 
            bypass_gat=(mode == "no_gnn")
        )
        for batch_pairs, batch_labels in test_loader:
            batch_pairs = batch_pairs.to(device)
            preds = model.predict_side_effects(drug_emb, batch_pairs)
            all_test_preds.append(preds.cpu())
            all_test_labels.append(batch_labels.cpu())

    test_preds_cat = torch.cat(all_test_preds).numpy()
    test_labels_cat = torch.cat(all_test_labels).numpy()

    from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
    test_auroc = roc_auc_score(test_labels_cat, test_preds_cat, average='macro')
    test_auprc = average_precision_score(test_labels_cat, test_preds_cat, average='macro')
    test_preds_binary = (test_preds_cat >= 0.5).astype(int)
    test_f1 = f1_score(test_labels_cat, test_preds_binary, average='macro', zero_division=0)
    test_precision = precision_score(test_labels_cat, test_preds_binary, average='macro', zero_division=0)
    test_recall = recall_score(test_labels_cat, test_preds_binary, average='macro', zero_division=0)

    return {
        'val_auroc': best_val_auroc,
        'test_auroc': test_auroc,
        'test_auprc': test_auprc,
        'test_f1': test_f1,
        'test_precision': test_precision,
        'test_recall': test_recall
    }

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    graph_path = Path(config["data"]["graph_dir"]) / "hetero_data.pt"
    if not graph_path.exists():
        print(f"Graph file not found at {graph_path}. Please run build_kg script first.")
        return

    data = torch.load(graph_path, weights_only=False)
    
    num_pairs = data['drug', 'ddi', 'drug'].edge_index.size(1)
    indices = np.arange(num_pairs)
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # We run for epochs defined in config (default 100) to ensure full convergence
    epochs = config["model"].get("epochs", 100)
    results = {}
    for mode in ["no_gnn", "no_ppi", "full"]:
        results[mode] = train_and_evaluate_mode(mode, config, data, train_idx, val_idx, test_idx, device, epochs=epochs)

    print("\n==========================================================================================")
    print("Ablation Study Results Table")
    print("==========================================================================================")
    print("| Configuration | Val AUROC | Test AUROC | Test AUPRC | Test F1 | Test Precision | Test Recall |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for mode, metrics in results.items():
        print(f"| {mode.upper():<13} | {metrics['val_auroc']:>9.4f} | {metrics['test_auroc']:>10.4f} | {metrics['test_auprc']:>10.4f} | {metrics['test_f1']:>7.4f} | {metrics['test_precision']:>14.4f} | {metrics['test_recall']:>11.4f} |")
    print("==========================================================================================\n")

    # Save to CSV
    import pandas as pd
    rows = []
    for mode, metrics in results.items():
        r = {'mode': mode}
        r.update(metrics)
        rows.append(r)
    pd.DataFrame(rows).to_csv("results/ablation_results.csv", index=False)

if __name__ == "__main__":
    main()
