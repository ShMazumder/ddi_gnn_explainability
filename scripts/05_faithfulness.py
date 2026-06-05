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
from faithfulness.metrics import (
    sufficiency, necessity, fidelity_plus, fidelity_minus, sparsity,
    check_connectivity_and_dist, check_lenient_connectivity_and_dist
)
from explanations.kec import get_k_hop_subgraph


def evaluate_faithfulness():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    graph_path = Path(config["data"]["graph_dir"]) / "hetero_data.pt"
    data = torch.load(graph_path, weights_only=False)
    num_drugs = data['drug'].num_nodes
    num_proteins = data['protein'].num_nodes
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
        fid_plus_scores = []
        fid_minus_scores = []
        sparsity_scores = []
        conn_strict_scores = []
        hop_strict_scores = []
        conn_lenient_scores = []
        hop_lenient_scores = []

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

            # Fidelity+
            fid_p = fidelity_plus(full_preds[i:i+1], compl_pred)
            fid_plus_scores.append(fid_p)

            # Fidelity-
            fid_m = fidelity_minus(full_preds[i:i+1], sub_pred)
            fid_minus_scores.append(fid_m)

            # Sparsity relative to local 2-hop subgraph edges count
            d1, d2 = sample_pairs[i].tolist()
            _, subgraph_mask = get_k_hop_subgraph(hom_edge.cpu(), [d1, d2], k=2)
            expl_sparsity = sparsity(expl_edges, subgraph_mask)
            sparsity_scores.append(expl_sparsity)

            # Strict connectedness and hop distance
            is_strict_conn, strict_hop_dist = check_connectivity_and_dist(expl_edges, d1, d2, hom_edge)
            conn_strict_scores.append(float(is_strict_conn))
            if is_strict_conn:
                hop_strict_scores.append(float(strict_hop_dist))

            # Lenient connectedness and hop distance
            is_lenient_conn, lenient_hop_dist = check_lenient_connectivity_and_dist(expl_edges, d1, d2, hom_edge)
            conn_lenient_scores.append(float(is_lenient_conn))
            if is_lenient_conn:
                hop_lenient_scores.append(float(lenient_hop_dist))

        avg_suff = np.mean(suff_scores) if suff_scores else 0.0
        avg_nec = np.mean(nec_scores) if nec_scores else 0.0
        avg_fid_plus = np.mean(fid_plus_scores) if fid_plus_scores else 0.0
        avg_fid_minus = np.mean(fid_minus_scores) if fid_minus_scores else 0.0
        avg_sparsity = np.mean(sparsity_scores) if sparsity_scores else 0.0
        pct_strict = np.mean(conn_strict_scores) * 100.0 if conn_strict_scores else 0.0
        avg_hop_strict = np.mean(hop_strict_scores) if hop_strict_scores else -1.0
        pct_lenient = np.mean(conn_lenient_scores) * 100.0 if conn_lenient_scores else 0.0
        avg_hop_lenient = np.mean(hop_lenient_scores) if hop_lenient_scores else -1.0

        results_rows.append({
            'method': method_name,
            'sufficiency': avg_suff,
            'necessity': avg_nec,
            'fidelity_plus': avg_fid_plus,
            'fidelity_minus': avg_fid_minus,
            'sparsity': avg_sparsity,
            'path_connected_strict_pct': pct_strict,
            'avg_hop_dist_strict': avg_hop_strict,
            'path_connected_lenient_pct': pct_lenient,
            'avg_hop_dist_lenient': avg_hop_lenient,
            'num_evaluated': len(suff_scores)
        })
        print(f"  Sufficiency: {avg_suff:.4f}, Necessity: {avg_nec:.4f}, Fidelity+: {avg_fid_plus:.4f}, Fidelity-: {avg_fid_minus:.4f}, Sparsity: {avg_sparsity:.4f}")
        print(f"  Strict Conn: {pct_strict:.1f}%, Hop: {avg_hop_strict:.2f} | Lenient Conn: {pct_lenient:.1f}%, Hop: {avg_hop_lenient:.2f}")

    # Save results
    df = pd.DataFrame(results_rows)
    df.to_csv(results_dir / "faithfulness_results.csv", index=False)
    print(f"\nResults saved to {results_dir / 'faithfulness_results.csv'}")
    print(df.to_string(index=False))



if __name__ == "__main__":
    evaluate_faithfulness()
