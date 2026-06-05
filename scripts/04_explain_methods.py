"""
Run explanation methods on the trained GAT model.

Loads the trained model checkpoint and runs the following explanation methods
on a sample of drug pairs:
  1. Attention Rollout
  2. GNNExplainer
  3. PGExplainer
  4. Knowledge-Enhanced Counterfactual (KEC)

Results are saved to results/ for downstream faithfulness evaluation.
"""

import torch
import numpy as np
import yaml
from pathlib import Path
import sys
sys.path.append(".")

from models.gat_ddi import HeteroGATDDI
from explanations.attention_rollout import attention_rollout
from explanations.gnnexplainer_wrapper import run_gnnexplainer
from explanations.pgexplainer_wrapper import run_pgexplainer
from explanations.kec import run_kec


def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    graph_path = Path(config["data"]["graph_dir"]) / "hetero_data.pt"
    data = torch.load(graph_path, weights_only=False)
    num_drugs = data['drug'].num_nodes
    num_proteins = data['protein'].num_nodes
    edge_index = data['drug', 'ddi', 'drug'].edge_index
    labels = data['drug', 'ddi', 'drug'].side_effect_label

    # Build homogeneous edge index (bidirectional)
    dp_edges = data['drug', 'binds', 'protein'].edge_index.clone()
    dp_edges[1] += num_drugs
    pd_edges = torch.stack([dp_edges[1], dp_edges[0]], dim=0)
    
    p_edges = data['protein', 'interacts', 'protein'].edge_index.clone()
    p_edges[0] += num_drugs
    p_edges[1] += num_drugs
    p_edges_rev = torch.stack([p_edges[1], p_edges[0]], dim=0)
    
    hom_edge = torch.cat([dp_edges, pd_edges, p_edges, p_edges_rev], dim=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load trained model
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

    # Sample drug pairs for explanation
    num_pairs = config["explanation"]["num_pairs"]
    total_pairs = edge_index.size(1)
    sample_idx = np.random.choice(total_pairs, size=min(num_pairs, total_pairs), replace=False)
    sample_pairs = edge_index[:, sample_idx].t()

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    methods = config["explanation"]["methods"]
    all_explanations = {}

    with torch.no_grad():
        drug_emb = model.forward({'drug': None, 'protein': None}, hom_edge, num_drugs)

    for method_name in methods:
        print(f"\n=== Running {method_name} ===")

        if method_name == "attention":
            explanations = attention_rollout(model, hom_edge, num_drugs, sample_pairs, device)
        elif method_name == "gnnexplainer":
            explanations = run_gnnexplainer(model, hom_edge, num_drugs, sample_pairs, device)
        elif method_name == "pgexplainer":
            explanations = run_pgexplainer(model, hom_edge, num_drugs, sample_pairs, device)
        elif method_name == "kec":
            explanations = run_kec(model, hom_edge, num_drugs, sample_pairs, device, data)
        else:
            print(f"Unknown method: {method_name}")
            continue

        all_explanations[method_name] = explanations
        print(f"  Generated {len(explanations)} explanations")

    # Save explanations
    torch.save(all_explanations, results_dir / "explanations.pt")
    torch.save(sample_pairs, results_dir / "sample_pairs.pt")
    print(f"\nExplanations saved to {results_dir}")


if __name__ == "__main__":
    main()
