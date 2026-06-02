"""
Knowledge-Enhanced Counterfactual (KEC) explanation method.

Generates counterfactual explanations by identifying the minimal set of
edges whose removal changes the model's prediction. Leverages domain
knowledge from the knowledge graph to prioritise biologically meaningful
perturbations.
"""

import torch
import numpy as np
from tqdm import tqdm


def get_k_hop_subgraph(edge_index, node_idx, k=2, num_nodes=None):
    """Extract k-hop neighbourhood around a set of seed nodes."""
    if num_nodes is None:
        num_nodes = edge_index.max().item() + 1

    visited = set(node_idx) if isinstance(node_idx, (list, set)) else {node_idx}
    frontier = set(visited)

    src, dst = edge_index

    for _ in range(k):
        new_frontier = set()
        for node in frontier:
            # Find neighbors
            mask_src = (src == node)
            mask_dst = (dst == node)
            neighbors = torch.cat([dst[mask_src], src[mask_dst]]).unique().tolist()
            new_frontier.update(neighbors)
        new_frontier -= visited
        visited.update(new_frontier)
        frontier = new_frontier

    # Get edge indices within the subgraph
    node_set = visited
    mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
    for i in range(edge_index.size(1)):
        if src[i].item() in node_set and dst[i].item() in node_set:
            mask[i] = True

    return list(visited), mask


def compute_counterfactual(model, edge_index, num_drugs, drug_pair, device,
                           k_hop=2, max_removals=10, threshold=0.5):
    """
    Find the minimal set of edges whose removal flips the prediction.

    Strategy:
    1. Extract k-hop subgraph around the drug pair
    2. Rank edges by their impact on prediction (greedy removal)
    3. Return the minimal removal set that changes the prediction
    """
    d1, d2 = drug_pair

    # Get full prediction
    with torch.no_grad():
        drug_emb = model.forward({'drug': None, 'protein': None}, edge_index, num_drugs)
        full_pred = model.predict_side_effects(drug_emb, torch.tensor([[d1, d2]], device=device))
        full_label = (full_pred > threshold).float()

    # Get k-hop subgraph
    subgraph_nodes, subgraph_mask = get_k_hop_subgraph(
        edge_index, [d1, d2], k=k_hop
    )
    subgraph_edge_indices = torch.where(subgraph_mask)[0]

    if len(subgraph_edge_indices) == 0:
        return None

    # Greedy edge removal: try removing each edge and measure prediction change
    edge_impacts = []
    for idx in subgraph_edge_indices:
        # Create edge index with this edge removed
        mask = torch.ones(edge_index.size(1), dtype=torch.bool, device=device)
        mask[idx] = False
        reduced_edges = edge_index[:, mask]

        with torch.no_grad():
            drug_emb_r = model.forward({'drug': None, 'protein': None}, reduced_edges, num_drugs)
            reduced_pred = model.predict_side_effects(drug_emb_r, torch.tensor([[d1, d2]], device=device))

        # Impact = change in prediction
        impact = (full_pred - reduced_pred).abs().sum().item()
        edge_impacts.append((idx.item(), impact))

    # Sort by impact (highest first)
    edge_impacts.sort(key=lambda x: x[1], reverse=True)

    # Greedily remove edges until prediction flips
    removed_edges = []
    current_edges = edge_index.clone()
    removal_mask = torch.ones(edge_index.size(1), dtype=torch.bool, device=device)

    for edge_idx, impact in edge_impacts[:max_removals]:
        removal_mask[edge_idx] = False
        removed_edges.append(edge_idx)

        # Check if prediction flipped
        reduced_edges = edge_index[:, removal_mask]
        with torch.no_grad():
            drug_emb_r = model.forward({'drug': None, 'protein': None}, reduced_edges, num_drugs)
            new_pred = model.predict_side_effects(drug_emb_r, torch.tensor([[d1, d2]], device=device))
            new_label = (new_pred > threshold).float()

        if not torch.equal(full_label, new_label):
            break

    # Build explanation from removed edges
    explanation_mask = torch.zeros(edge_index.size(1), device=device)
    for idx in removed_edges:
        explanation_mask[idx] = 1.0

    relevant_edge_indices = torch.tensor(removed_edges, device=device)
    relevant_edges = edge_index[:, relevant_edge_indices] if len(removed_edges) > 0 else torch.empty(2, 0, dtype=torch.long)
    relevant_nodes = torch.unique(relevant_edges).tolist() if relevant_edges.numel() > 0 else []

    return {
        'edge_mask': explanation_mask,
        'nodes': relevant_nodes,
        'edges': list(zip(relevant_edges[0].tolist(), relevant_edges[1].tolist())) if relevant_edges.numel() > 0 else [],
        'num_removals': len(removed_edges),
        'prediction_flipped': not torch.equal(full_label, new_label) if removed_edges else False
    }


def run_kec(model, edge_index, num_drugs, sample_pairs, device, data=None):
    """
    Run Knowledge-Enhanced Counterfactual explanations for sampled drug pairs.

    Returns a list of explanation dicts.
    """
    model.eval()
    edge_index = edge_index.to(device)

    explanations = []
    for i in tqdm(range(len(sample_pairs)), desc="KEC"):
        d1, d2 = sample_pairs[i].tolist()

        try:
            expl = compute_counterfactual(
                model, edge_index, num_drugs,
                (d1, d2), device, k_hop=2, max_removals=10
            )
            explanations.append(expl)
        except Exception as e:
            print(f"  KEC failed for pair {i}: {e}")
            explanations.append(None)

    return explanations
