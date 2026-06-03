import torch
import yaml
from pathlib import Path
import sys
sys.path.append(".")
from models.gat_ddi import HeteroGATDDI

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    graph_path = Path(config["data"]["graph_dir"]) / "hetero_data.pt"
    data = torch.load(graph_path, weights_only=False)
    num_drugs = data['drug'].num_nodes
    num_proteins = data['protein'].num_nodes
    edge_index = data['drug', 'ddi', 'drug'].edge_index
    labels = data['drug', 'ddi', 'drug'].side_effect_label

    drug_protein_edges = data['drug', 'binds', 'protein'].edge_index.clone()
    drug_protein_edges[1] += num_drugs
    ppi_edges = data['protein', 'interacts', 'protein'].edge_index.clone()
    ppi_edges[0] += num_drugs
    ppi_edges[1] += num_drugs
    hom_edge = torch.cat([drug_protein_edges, ppi_edges], dim=1)

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
    
    # Try loading the checkpoint if it exists
    checkpoint_path = Path("results/model_checkpoint.pt")
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Loaded checkpoint successfully.")
    else:
        print("Checkpoint not found, using random weights.")
    
    model.eval()

    # Sample a few pairs
    sample_pairs = edge_index[:, :5].t().to(device)

    # 1. Prediction with full hom_edge
    with torch.no_grad():
        drug_emb_full = model.forward({'drug': None, 'protein': None}, hom_edge.to(device), num_drugs)
        preds_full = model.predict_side_effects(drug_emb_full, sample_pairs)

    # 2. Prediction with empty edge index (only self-loops)
    empty_edge = torch.empty((2, 0), dtype=torch.long, device=device)
    with torch.no_grad():
        drug_emb_empty = model.forward({'drug': None, 'protein': None}, empty_edge, num_drugs)
        preds_empty = model.predict_side_effects(drug_emb_empty, sample_pairs)

    # 3. Check difference
    diff = torch.abs(preds_full - preds_empty).max().item()
    print(f"Max absolute difference in predictions between full graph and empty graph: {diff}")
    
    # Check if hom_edge contains any edges pointing to drug nodes (< num_drugs)
    target_nodes = hom_edge[1]
    num_edges_to_drugs = (target_nodes < num_drugs).sum().item()
    print(f"Number of edges in hom_edge: {hom_edge.size(1)}")
    print(f"Number of edges pointing to drug nodes: {num_edges_to_drugs}")

if __name__ == "__main__":
    main()
