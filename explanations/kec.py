"""
Knowledge-Enhanced Counterfactual (KEC) explanation method.

Generates counterfactual explanations by identifying the minimal set of
edges whose removal changes the model's prediction. Leverages domain
knowledge from the knowledge graph to prioritise biologically meaningful
perturbations.
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


def get_k_hop_subgraph(edge_index, node_idx, k=2, num_nodes=None):
    """Extract k-hop neighbourhood around a set of seed nodes using vectorized PyTorch operations."""
    if num_nodes is None:
        num_nodes = edge_index.max().item() + 1

    device = edge_index.device
    src, dst = edge_index

    # Initialize visited and frontier using boolean tensors for vectorized operations
    visited_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    node_idx_tensor = torch.tensor(list(node_idx), dtype=torch.long, device=device) if not isinstance(node_idx, torch.Tensor) else node_idx.to(torch.long)
    visited_mask[node_idx_tensor] = True
    
    frontier_mask = visited_mask.clone()

    for _ in range(k):
        # Find neighbors of frontier nodes
        src_in_frontier = frontier_mask[src]
        dst_in_frontier = frontier_mask[dst]

        neighbors_from_src = dst[src_in_frontier]
        neighbors_from_dst = src[dst_in_frontier]

        # Combine and unique
        new_nodes = torch.cat([neighbors_from_src, neighbors_from_dst]).unique()

        # Filter out already visited nodes
        new_nodes = new_nodes[~visited_mask[new_nodes]]

        # Update frontier and visited masks
        frontier_mask.zero_()
        frontier_mask[new_nodes] = True
        visited_mask[new_nodes] = True

    # Get edge indices within the subgraph (both src and dst must be in the visited set)
    mask = visited_mask[src] & visited_mask[dst]
    
    # Return as list and boolean mask as required by callers
    visited_nodes = torch.where(visited_mask)[0].tolist()
    return visited_nodes, mask


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
    src, dst = edge_index

    # Get k-hop subgraph
    subgraph_nodes, subgraph_mask = get_k_hop_subgraph(
        edge_index, [d1, d2], k=k_hop
    )
    
    # Global indices of edges in the subgraph
    subgraph_edge_indices = torch.where(subgraph_mask)[0]

    if len(subgraph_edge_indices) == 0:
        return None

    # Pre-compute mapped node structures (once per drug pair!)
    subgraph_nodes_tensor = torch.tensor(subgraph_nodes, dtype=torch.long, device=device)
    node_to_local = {node: i for i, node in enumerate(subgraph_nodes)}
    d1_local_idx = node_to_local[d1]
    d2_local_idx = node_to_local[d2]
    
    is_drug = subgraph_nodes_tensor < num_drugs
    drug_ids = subgraph_nodes_tensor[is_drug]
    protein_ids = subgraph_nodes_tensor[~is_drug] - num_drugs
    
    # Slice initial node embeddings once
    x_local = torch.zeros(len(subgraph_nodes), model.drug_embed.embedding_dim, device=device)
    if is_drug.sum() > 0:
        x_local[is_drug] = model.drug_embed(drug_ids)
    if (~is_drug).sum() > 0:
        x_local[~is_drug] = model.protein_embed(protein_ids)

    # Build node mapping tensor for fast edge mapping
    max_node_idx = max(subgraph_nodes)
    mapping_tensor = torch.zeros(max_node_idx + 1, dtype=torch.long, device=device)
    mapping_tensor[subgraph_nodes_tensor] = torch.arange(len(subgraph_nodes), device=device)

    # Map the entire subgraph edges to local indices
    subgraph_src_global = src[subgraph_edge_indices]
    subgraph_dst_global = dst[subgraph_edge_indices]
    
    src_local = mapping_tensor[subgraph_src_global]
    dst_local = mapping_tensor[subgraph_dst_global]
    edge_index_local = torch.stack([src_local, dst_local], dim=0)

    # Map global edge index to its local index in subgraph_edge_indices
    global_to_local_edge = torch.zeros(edge_index.size(1), dtype=torch.long, device=device)
    global_to_local_edge[subgraph_edge_indices] = torch.arange(len(subgraph_edge_indices), device=device)

    # Helper function to get prediction on a subset of local edges
    def get_prediction_on_local_edges(local_edge_mask):
        # Run GAT Conv layers on the local subgraph
        local_edges = edge_index_local[:, local_edge_mask]
        h = F.elu(model.gat1(x_local, local_edges))
        h = model.gat2(h, local_edges)
        
        d1_emb = h[d1_local_idx].unsqueeze(0)
        d2_emb = h[d2_local_idx].unsqueeze(0)
        
        # Virtual embeddings lookup
        drug_emb_virtual = torch.zeros(num_drugs, model.drug_embed.embedding_dim, device=device)
        drug_emb_virtual[d1] = d1_emb
        drug_emb_virtual[d2] = d2_emb
        return model.predict_side_effects(drug_emb_virtual, torch.tensor([[d1, d2]], device=device))

    # Get full prediction (using all local edges)
    with torch.no_grad():
        full_local_mask = torch.ones(edge_index_local.size(1), dtype=torch.bool, device=device)
        full_pred = get_prediction_on_local_edges(full_local_mask)
        full_label = (full_pred > threshold).float()

    # Find neighbors of d1 and d2 in PyTorch to compute path candidates
    d1_neighbors = torch.cat([dst[src == d1], src[dst == d1]])
    d2_neighbors = torch.cat([dst[src == d2], src[dst == d2]])
    
    num_nodes_total = edge_index.max().item() + 1
    d1_target_mask = torch.zeros(num_nodes_total, dtype=torch.bool, device=device)
    d1_target_mask[d1_neighbors] = True
    
    d2_target_mask = torch.zeros(num_nodes_total, dtype=torch.bool, device=device)
    d2_target_mask[d2_neighbors] = True
    
    is_ppi_path = (d1_target_mask[src] & d2_target_mask[dst]) | (d2_target_mask[src] & d1_target_mask[dst])
    is_direct = (src == d1) | (dst == d1) | (src == d2) | (dst == d2)
    
    candidate_mask = is_direct | is_ppi_path
    final_candidate_mask = subgraph_mask & candidate_mask
    candidate_indices = torch.where(final_candidate_mask)[0]

    # Fallback to all subgraph edges if path candidate filter returns empty
    if len(candidate_indices) == 0:
        candidate_indices = subgraph_edge_indices

    # Greedy edge removal: try removing each candidate edge and measure prediction change
    edge_impacts = []
    for idx in candidate_indices:
        # Find local index of this global edge
        local_edge_idx = global_to_local_edge[idx].item()
        
        # Create mask with this edge removed
        local_mask = torch.ones(edge_index_local.size(1), dtype=torch.bool, device=device)
        local_mask[local_edge_idx] = False

        with torch.no_grad():
            reduced_pred = get_prediction_on_local_edges(local_mask)

        # Impact = change in prediction
        impact = (full_pred - reduced_pred).abs().sum().item()
        edge_impacts.append((idx.item(), impact))

    # Sort by impact (highest first)
    edge_impacts.sort(key=lambda x: x[1], reverse=True)

    # Greedily remove edges until prediction flips
    removed_edges = []
    removal_mask = torch.ones(edge_index_local.size(1), dtype=torch.bool, device=device)

    for edge_idx, impact in edge_impacts[:max_removals]:
        local_edge_idx = global_to_local_edge[edge_idx].item()
        removal_mask[local_edge_idx] = False
        removed_edges.append(edge_idx)

        # Check if prediction flipped
        with torch.no_grad():
            new_pred = get_prediction_on_local_edges(removal_mask)
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
