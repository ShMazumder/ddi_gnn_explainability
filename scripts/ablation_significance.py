import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import yaml
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.append(".")

from models.gat_ddi import HeteroGATDDI

def train_ablation_model(mode, config, data, train_idx, val_idx, test_idx, device, epochs=15):
    num_drugs = data['drug'].num_nodes
    num_proteins = data['protein'].num_nodes
    edge_index = data['drug', 'ddi', 'drug'].edge_index
    labels = data['drug', 'ddi', 'drug'].side_effect_label

    # Build GNN homogeneous edge index based on mode
    if mode == "no_gnn":
        hom_edge = torch.empty((2, 0), dtype=torch.long, device=device)
    else:
        dp_edges = data['drug', 'binds', 'protein'].edge_index.clone()
        dp_edges[1] += num_drugs
        pd_edges = torch.stack([dp_edges[1], dp_edges[0]], dim=0)
        
        if mode == "no_ppi":
            hom_edge = torch.cat([dp_edges, pd_edges], dim=1).to(device)
        else:  # "full"
            p_edges = data['protein', 'interacts', 'protein'].edge_index.clone()
            p_edges[0] += num_drugs
            p_edges[1] += num_drugs
            p_edges_rev = torch.stack([p_edges[1], p_edges[0]], dim=0)
            hom_edge = torch.cat([dp_edges, pd_edges, p_edges, p_edges_rev], dim=1).to(device)

    # Train/val/test splits
    train_idx_tensor = torch.tensor(train_idx, dtype=torch.long) if isinstance(train_idx, np.ndarray) else train_idx
    test_idx_tensor = torch.tensor(test_idx, dtype=torch.long) if isinstance(test_idx, np.ndarray) else test_idx
    
    train_pairs = edge_index[:, train_idx_tensor].t()
    train_labels = labels[train_idx_tensor]
    test_pairs = edge_index[:, test_idx_tensor].t()
    test_labels = labels[test_idx_tensor]

    train_dataset = TensorDataset(train_pairs, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=config["model"]["batch_size"], shuffle=True)
    test_loader = DataLoader(TensorDataset(test_pairs, test_labels), batch_size=config["model"]["batch_size"])

    model = HeteroGATDDI(
        num_drugs=num_drugs,
        num_proteins=num_proteins,
        drug_hidden=config["model"]["drug_hidden"],
        protein_hidden=config["model"]["protein_hidden"],
        gat_hidden=config["model"].get("gat_hidden", 128),
        gat_heads=config["model"]["gat_heads"],
        out_dim=labels.shape[1],
        dropout=config["model"]["dropout"]
    ).to(device)

    pos_count = labels.sum(dim=0)
    neg_count = labels.shape[0] - pos_count
    pos_weight = (neg_count / (pos_count + 1e-8)).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=config["model"]["learning_rate"])

    print(f"Training '{mode}' model for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        for batch_pairs, batch_labels in train_loader:
            batch_pairs = batch_pairs.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
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

    # Predict on test set
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
        for batch_pairs, batch_labels in test_loader:
            batch_pairs = batch_pairs.to(device)
            preds = model.predict_side_effects(drug_emb, batch_pairs)
            all_preds.append(preds.cpu())
            all_labels.append(batch_labels.cpu())

    test_preds = torch.cat(all_preds).numpy()
    test_labels = torch.cat(all_labels).numpy()
    return test_preds, test_labels

def run_bootstrap(preds_a, preds_b, labels, num_resamples=1000):
    """Perform bootstrap comparison between two models."""
    n = len(labels)
    diffs_auroc = []
    diffs_auprc = []
    
    # Pre-calculate macro stats for bootstrap efficiency
    for _ in range(num_resamples):
        indices = np.random.choice(n, size=n, replace=True)
        lbl_b = labels[indices]
        pr_a_b = preds_a[indices]
        pr_b_b = preds_b[indices]
        
        # Calculate macro AUROC
        try:
            auroc_a = roc_auc_score(lbl_b, pr_a_b, average='macro')
            auroc_b = roc_auc_score(lbl_b, pr_b_b, average='macro')
            diffs_auroc.append(auroc_a - auroc_b)
        except ValueError:
            pass # Skip if bootstrap sample has no positive/negative instances for some class
            
        try:
            auprc_a = average_precision_score(lbl_b, pr_a_b, average='macro')
            auprc_b = average_precision_score(lbl_b, pr_b_b, average='macro')
            diffs_auprc.append(auprc_a - auprc_b)
        except ValueError:
            pass
            
    diffs_auroc = np.array(diffs_auroc)
    diffs_auprc = np.array(diffs_auprc)
    
    # p-value = fraction of differences that are <= 0 (if we assume a > b)
    p_auroc = np.mean(diffs_auroc <= 0) if np.mean(diffs_auroc) > 0 else np.mean(diffs_auroc >= 0)
    p_auprc = np.mean(diffs_auprc <= 0) if np.mean(diffs_auprc) > 0 else np.mean(diffs_auprc >= 0)
    
    return p_auroc, p_auprc, np.std(diffs_auroc), np.std(diffs_auprc)

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    graph_path = Path(config["data"]["graph_dir"]) / "hetero_data.pt"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.load(graph_path, map_location=device, weights_only=False)

    num_pairs = data['drug', 'ddi', 'drug'].edge_index.size(1)
    indices = np.arange(num_pairs)
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

    # Train and get predictions
    preds_full, test_labels = train_ablation_model("full", config, data, train_idx, val_idx, test_idx, device, epochs=15)
    preds_no_ppi, _ = train_ablation_model("no_ppi", config, data, train_idx, val_idx, test_idx, device, epochs=15)
    preds_no_gnn, _ = train_ablation_model("no_gnn", config, data, train_idx, val_idx, test_idx, device, epochs=15)

    print("\nCalculating bootstrap statistics (1000 resamples)...")
    # 1. Estimate standard deviations for each model
    n = len(test_labels)
    boot_auroc = {"full": [], "no_ppi": [], "no_gnn": []}
    boot_auprc = {"full": [], "no_ppi": [], "no_gnn": []}
    
    for _ in tqdm(range(1000), desc="Bootstrap Single"):
        idx = np.random.choice(n, size=n, replace=True)
        lbl = test_labels[idx]
        try:
            boot_auroc["full"].append(roc_auc_score(lbl, preds_full[idx], average='macro'))
            boot_auroc["no_ppi"].append(roc_auc_score(lbl, preds_no_ppi[idx], average='macro'))
            boot_auroc["no_gnn"].append(roc_auc_score(lbl, preds_no_gnn[idx], average='macro'))
        except ValueError:
            pass
            
        try:
            boot_auprc["full"].append(average_precision_score(lbl, preds_full[idx], average='macro'))
            boot_auprc["no_ppi"].append(average_precision_score(lbl, preds_no_ppi[idx], average='macro'))
            boot_auprc["no_gnn"].append(average_precision_score(lbl, preds_no_gnn[idx], average='macro'))
        except ValueError:
            pass

    print("\n--- Model Standard Deviations ---")
    sd_results = {}
    for m in ["full", "no_ppi", "no_gnn"]:
        sd_auc = np.std(boot_auroc[m])
        sd_prc = np.std(boot_auprc[m])
        print(f"{m.upper()}: AUROC SD = {sd_auc:.6f} | AUPRC SD = {sd_prc:.6f}")
        sd_results[m] = (sd_auc, sd_prc)

    # 2. Paired comparisons
    print("\n--- Paired Bootstrap Comparisons ---")
    p_no_ppi_full_auc, p_no_ppi_full_prc, _, _ = run_bootstrap(preds_no_ppi, preds_full, test_labels, num_resamples=1000)
    p_full_no_gnn_auc, p_full_no_gnn_prc, _, _ = run_bootstrap(preds_full, preds_no_gnn, test_labels, num_resamples=1000)
    p_no_ppi_no_gnn_auc, p_no_ppi_no_gnn_prc, _, _ = run_bootstrap(preds_no_ppi, preds_no_gnn, test_labels, num_resamples=1000)

    print(f"NO_PPI vs. FULL:  AUROC p = {p_no_ppi_full_auc:.4f} | AUPRC p = {p_no_ppi_full_prc:.4f}")
    print(f"FULL vs. NO_GNN:  AUROC p = {p_full_no_gnn_auc:.4f} | AUPRC p = {p_full_no_gnn_prc:.4f}")
    print(f"NO_PPI vs. NO_GNN: AUROC p = {p_no_ppi_no_gnn_auc:.4f} | AUPRC p = {p_no_ppi_no_gnn_prc:.4f}")

    # Save results to file
    with open("results/ablation_significance.txt", "w") as f:
        f.write("=== Ablation Study Significance Testing ===\n")
        for m in ["full", "no_ppi", "no_gnn"]:
            f.write(f"{m.upper()} SD: AUROC = {sd_results[m][0]:.6f}, AUPRC = {sd_results[m][1]:.6f}\n")
        f.write(f"\nPaired p-values:\n")
        f.write(f"NO_PPI vs. FULL:  AUROC p = {p_no_ppi_full_auc:.6f}, AUPRC p = {p_no_ppi_full_prc:.6f}\n")
        f.write(f"FULL vs. NO_GNN:  AUROC p = {p_full_no_gnn_auc:.6f}, AUPRC p = {p_full_no_gnn_prc:.6f}\n")
        f.write(f"NO_PPI vs. NO_GNN: AUROC p = {p_no_ppi_no_gnn_auc:.6f}, AUPRC p = {p_no_ppi_no_gnn_prc:.6f}\n")

if __name__ == "__main__":
    main()
