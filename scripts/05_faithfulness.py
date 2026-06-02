"""
Evaluate faithfulness of explanation methods using sufficiency and necessity metrics.

Loads the saved explanations from 04_explain_methods.py and computes:
  - Sufficiency: does the explanation subgraph alone preserve the prediction?
  - Necessity: does removing the explanation change the prediction?

Results are saved to results/faithfulness_results.csv.
"""

import torch
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import sys
sys.path.append(".")

from models.gat_ddi import HeteroGATDDI
from faithfulness.metrics import sufficiency, necessity


def evaluate_faithfulness():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    graph_path = Path(config["data"]["graph_dir"]) / "hetero_data.pt"
    data = torch.load(graph_path, weights_only=False)
    num_drugs = data['drug'].num_nodes
    num_proteins = data['protein'].num_nodes
    labels = data['drug', 'ddi', 'drug'].side_effect_label

    # Build homogeneous edge index
    drug_protein_edges = data['drug', 'binds', 'protein'].edge_index.clone()
    drug_protein_edges[1] += num_drugs
    ppi_edges = data['protein', 'interacts', 'protein'].edge_index.clone()
    ppi_edges[0] += num_drugs
    ppi_edges[1] += num_drugs
    hom_edge = torch.cat([drug_protein_edges, ppi_edges], dim=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
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
    model.load_state_dict(torch.load("results/model_checkpoint.pt", map_location=device))
    model.eval()

    hom_edge = hom_edge.to(device)

    # Load explanations
    results_dir = Path("results")
    all_explanations = torch.load(results_dir / "explanations.pt")
    sample_pairs = torch.load(results_dir / "sample_pairs.pt")

    # Get full predictions
    with torch.no_grad():
        drug_emb = model.forward({'drug': None, 'protein': None}, hom_edge, num_drugs)
        full_preds = model.predict_side_effects(drug_emb, sample_pairs.to(device))

    # Evaluate each method
    results_rows = []
    num_samples = config["explanation"]["faithfulness_samples"]

    for method_name, explanations in all_explanations.items():
        print(f"\nEvaluating {method_name}...")

        suff_scores = []
        nec_scores = []

        for i, expl in enumerate(explanations[:num_samples]):
            if expl is None:
                continue

            # Explanation is a set of edge indices (subgraph)
            expl_edges = expl.get('edge_mask', None)
            if expl_edges is None:
                continue

            # Sufficiency: predict with only explanation edges
            expl_edge_index = hom_edge[:, expl_edges > 0.5] if isinstance(expl_edges, torch.Tensor) else hom_edge
            with torch.no_grad():
                drug_emb_sub = model.forward({'drug': None, 'protein': None}, expl_edge_index, num_drugs)
                sub_pred = model.predict_side_effects(drug_emb_sub, sample_pairs[i:i+1].to(device))
            suff = sufficiency(full_preds[i:i+1], sub_pred)
            suff_scores.append(suff)

            # Necessity: predict without explanation edges
            complement_mask = ~(expl_edges > 0.5) if isinstance(expl_edges, torch.Tensor) else torch.ones(hom_edge.size(1), dtype=torch.bool)
            compl_edge_index = hom_edge[:, complement_mask]
            with torch.no_grad():
                drug_emb_compl = model.forward({'drug': None, 'protein': None}, compl_edge_index, num_drugs)
                compl_pred = model.predict_side_effects(drug_emb_compl, sample_pairs[i:i+1].to(device))
            nec = necessity(full_preds[i:i+1], compl_pred)
            nec_scores.append(nec)

        avg_suff = np.mean(suff_scores) if suff_scores else 0.0
        avg_nec = np.mean(nec_scores) if nec_scores else 0.0
        results_rows.append({
            'method': method_name,
            'sufficiency': avg_suff,
            'necessity': avg_nec,
            'num_evaluated': len(suff_scores)
        })
        print(f"  Sufficiency: {avg_suff:.4f}, Necessity: {avg_nec:.4f}")

    # Save results
    df = pd.DataFrame(results_rows)
    df.to_csv(results_dir / "faithfulness_results.csv", index=False)
    print(f"\nResults saved to {results_dir / 'faithfulness_results.csv'}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    evaluate_faithfulness()
