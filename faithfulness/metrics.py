"""
Faithfulness metrics for evaluating GNN explanations.

Implements:
  - Sufficiency: does the explanation subgraph alone preserve the prediction?
  - Necessity: does removing the explanation change the prediction?
  - Fidelity+: prediction drop when explanation is removed
  - Fidelity-: prediction drop when only explanation is kept
  - Sparsity: fraction of edges in the explanation
"""

import torch
import numpy as np


def sufficiency(full_pred, subgraph_pred, threshold=0.5):
    """
    Fraction of predictions that remain consistent using the explanation subgraph alone.

    High sufficiency means the explanation captures the essential structure
    needed for the prediction.

    Args:
        full_pred: predictions using the full graph (Tensor)
        subgraph_pred: predictions using only the explanation subgraph (Tensor)
        threshold: classification threshold

    Returns:
        float: sufficiency score in [0, 1]
    """
    full_bin = (full_pred > threshold).float()
    sub_bin = (subgraph_pred > threshold).float()
    return (full_bin == sub_bin).float().mean().item()


def necessity(full_pred, pred_without_expl, threshold=0.5):
    """
    Average drop in prediction probability when explanation edges are removed.

    High necessity means removing the explanation significantly impacts
    the prediction, indicating the explanation is important.

    Args:
        full_pred: predictions using the full graph (Tensor)
        pred_without_expl: predictions with explanation edges removed (Tensor)
        threshold: classification threshold

    Returns:
        float: average prediction drop
    """
    drop = full_pred - pred_without_expl
    return drop.mean().item()


def fidelity_plus(full_pred, pred_without_expl):
    """
    Fidelity+ (comprehensiveness): average prediction change when
    explanation edges are removed from the graph.

    Higher is better — means the explanation captures important structure.

    Args:
        full_pred: predictions using the full graph (Tensor)
        pred_without_expl: predictions with explanation edges removed (Tensor)

    Returns:
        float: fidelity+ score
    """
    return (full_pred - pred_without_expl).mean().item()


def fidelity_minus(full_pred, subgraph_pred):
    """
    Fidelity- (sufficiency): average prediction change when only
    explanation edges are kept.

    Lower is better — means the explanation alone is sufficient.

    Args:
        full_pred: predictions using the full graph (Tensor)
        subgraph_pred: predictions using only the explanation subgraph (Tensor)

    Returns:
        float: fidelity- score
    """
    return (full_pred - subgraph_pred).mean().item()


def sparsity(edge_mask, total_edges):
    """
    Fraction of edges included in the explanation.

    Lower sparsity means a more concise explanation.

    Args:
        edge_mask: binary mask or importance scores (Tensor)
        total_edges: total number of edges in the graph

    Returns:
        float: sparsity ratio in [0, 1]
    """
    if isinstance(edge_mask, torch.Tensor):
        num_selected = (edge_mask > 0.5).sum().item()
    else:
        num_selected = sum(1 for x in edge_mask if x > 0.5)
    return num_selected / max(total_edges, 1)


def check_connectivity_and_dist(expl_edges_mask, d1, d2, hom_edge):
    """
    Check if the two query drugs (d1, d2) are connected via selected explanation edges
    in hom_edge, and if so, return the shortest path length (hop distance).
    """
    from collections import defaultdict
    adj = defaultdict(list)
    
    # Get indices of selected edges (weight > 0.5)
    if isinstance(expl_edges_mask, torch.Tensor):
        selected_indices = torch.where(expl_edges_mask > 0.5)[0].tolist()
    else:
        selected_indices = [idx for idx, val in enumerate(expl_edges_mask) if val > 0.5]
        
    selected_edges = hom_edge[:, selected_indices].tolist()
    for u, v in zip(selected_edges[0], selected_edges[1]):
        adj[u].append(v)
        adj[v].append(u)
        
    # BFS from d1 to find shortest path to d2
    visited = {d1: 0}
    queue = [d1]
    while queue:
        curr = queue.pop(0)
        dist = visited[curr]
        if curr == d2:
            return True, dist
        for neighbor in adj[curr]:
            if neighbor not in visited:
                visited[neighbor] = dist + 1
                queue.append(neighbor)
    return False, -1
