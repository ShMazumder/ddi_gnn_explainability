import torch
import torch.nn.functional as F
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.append(".")

from models.gat_ddi import HeteroGATDDI
from explanations.kec import get_k_hop_subgraph
from faithfulness.metrics import fidelity_plus

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    graph_path = Path(config["data"]["graph_dir"]) / "hetero_data.pt"
    data = torch.load(graph_path, map_location=device, weights_only=False)
    num_drugs = data['drug'].num_nodes
    num_proteins = data['protein'].num_nodes
    labels = data['drug', 'ddi', 'drug'].side_effect_label

    # Build homogeneous edge index
    dp_edges = data['drug', 'binds', 'protein'].edge_index.clone()
    dp_edges[1] += num_drugs
    pd_edges = torch.stack([dp_edges[1], dp_edges[0]], dim=0)
    
    p_edges = data['protein', 'interacts', 'protein'].edge_index.clone()
    p_edges[0] += num_drugs
    p_edges[1] += num_drugs
    p_edges_rev = torch.stack([p_edges[1], p_edges[0]], dim=0)
    
    hom_edge = torch.cat([dp_edges, pd_edges, p_edges, p_edges_rev], dim=1)

    # Load model
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
    model.load_state_dict(torch.load("results/model_checkpoint.pt", map_location=device), strict=False)
    model.eval()

    hom_edge = hom_edge.to(device)
    results_dir = Path("results")
    all_explanations = torch.load(results_dir / "explanations.pt", map_location=device)
    sample_pairs = torch.load(results_dir / "sample_pairs.pt", map_location=device)
    
    pg_explanations = all_explanations.get('pgexplainer', None)
    if pg_explanations is None:
        print("Error: PGExplainer explanations not found.")
        return

    num_samples = config["explanation"]["faithfulness_samples"]

    # Get full predictions
    with torch.no_grad():
        drug_emb = model.forward({'drug': None, 'protein': None}, hom_edge, num_drugs)
        full_preds = model.predict_side_effects(drug_emb, sample_pairs[:num_samples].to(device))

    pg_drops = []
    random_drops = []
    
    np.random.seed(42)

    print("Running PGExplainer negative Fidelity+ diagnostic...")
    for i in tqdm(range(num_samples)):
        expl = pg_explanations[i]
        if expl is None:
            continue
            
        expl_edges = expl.get('edge_mask', None)
        if expl_edges is None:
            continue

        d1, d2 = sample_pairs[i].tolist()
        
        # Get PGExplainer prediction drop
        complement_mask = ~(expl_edges > 0.5) if isinstance(expl_edges, torch.Tensor) else torch.ones(hom_edge.size(1), dtype=torch.bool, device=device)
        compl_edge_index = hom_edge[:, complement_mask]
        with torch.no_grad():
            drug_emb_compl = model.forward({'drug': None, 'protein': None}, compl_edge_index, num_drugs)
            compl_pred = model.predict_side_effects(drug_emb_compl, sample_pairs[i:i+1])
        pg_drop = (full_preds[i:i+1] - compl_pred).mean().item()
        pg_drops.append(pg_drop)

        # Get local subgraph
        _, subgraph_mask = get_k_hop_subgraph(hom_edge.cpu(), [d1, d2], k=2)
        subgraph_indices = torch.where(subgraph_mask)[0].tolist()
        
        # Count selected edges in the subgraph
        if isinstance(expl_edges, torch.Tensor):
            num_selected = (expl_edges[subgraph_mask] > 0.5).sum().item()
        else:
            num_selected = sum(1 for idx in subgraph_indices if expl_edges[idx] > 0.5)

        if num_selected == 0:
            random_drops.append(0.0)
            continue

        # Generate random masks of same size within local subgraph and average drops
        trial_drops = []
        for _ in range(5):
            random_subgraph_idx = np.random.choice(subgraph_indices, size=min(num_selected, len(subgraph_indices)), replace=False)
            random_subgraph_idx_tensor = torch.tensor(random_subgraph_idx, dtype=torch.long, device=device)
            random_mask = torch.ones(hom_edge.size(1), dtype=torch.bool, device=device)
            random_mask[random_subgraph_idx_tensor] = False
            
            rand_edge_index = hom_edge[:, random_mask]
            with torch.no_grad():
                drug_emb_rand = model.forward({'drug': None, 'protein': None}, rand_edge_index, num_drugs)
                rand_pred = model.predict_side_effects(drug_emb_rand, sample_pairs[i:i+1])
            rand_drop = (full_preds[i:i+1] - rand_pred).mean().item()
            trial_drops.append(rand_drop)
            
        random_drops.append(np.mean(trial_drops))

    mean_pg_drop = np.mean(pg_drops)
    mean_random_drop = np.mean(random_drops)

    print("\n=== PGExplainer Diagnostic Results ===")
    print(f"PGExplainer Mean Fidelity+ (Fidelity drop): {mean_pg_drop:.6f}")
    print(f"Random Mask Mean Fidelity+ (Fidelity drop): {mean_random_drop:.6f}")
    
    diff = mean_pg_drop - mean_random_drop
    print(f"Difference (PGExplainer - Random):           {diff:.6f}")
    
    if mean_pg_drop < 0 and mean_random_drop >= 0:
        conclusion = "The negative Fidelity+ is SPECIFIC to PGExplainer's selected mask, supporting the mask-induced distribution shift hypothesis."
    elif mean_pg_drop < 0 and mean_random_drop < 0:
        if mean_pg_drop < mean_random_drop:
            conclusion = "Both PGExplainer and random masking yield negative Fidelity+, but PGExplainer is more negative, suggesting a combination of redundancy and mask-specific distribution shift."
        else:
            conclusion = "Both yield negative Fidelity+ of comparable magnitude, supporting the redundancy removal hypothesis where removing edges from dense regions slightly increases model confidence."
    else:
        conclusion = "Fidelity+ is positive, indicating removing edges drops confidence as expected."

    print(f"Conclusion: {conclusion}")

    # Save to file
    with open("results/pgexplainer_diagnostic.txt", "w") as f:
        f.write(f"PGExplainer Mean Fidelity+: {mean_pg_drop:.6f}\n")
        f.write(f"Random Mask Mean Fidelity+: {mean_random_drop:.6f}\n")
        f.write(f"Difference: {diff:.6f}\n")
        f.write(f"Conclusion: {conclusion}\n")

if __name__ == "__main__":
    main()
