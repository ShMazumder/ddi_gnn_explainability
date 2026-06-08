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


def sparsity(edge_mask, subgraph_mask):
    """
    Fraction of candidate subgraph edges omitted from the explanation.

    Args:
        edge_mask: binary mask or importance scores (Tensor)
        subgraph_mask: boolean/binary mask of the candidate local subgraph (Tensor)

    Returns:
        float: local sparsity in [0, 1]
    """
    if isinstance(edge_mask, torch.Tensor):
        edge_bool = edge_mask > 0.5
        sub_bool = subgraph_mask.to(edge_bool.device) > 0.5
        selected_in_subgraph = edge_bool & sub_bool
        num_selected = selected_in_subgraph.sum().item()
        subgraph_edges_count = sub_bool.sum().item()
    else:
        edge_bool = np.array(edge_mask) > 0.5
        sub_bool = np.array(subgraph_mask) > 0.5
        selected_in_subgraph = edge_bool & sub_bool
        num_selected = int(selected_in_subgraph.sum())
        subgraph_edges_count = int(sub_bool.sum())
        
    return 1.0 - (num_selected / max(subgraph_edges_count, 1))


def check_connectivity_and_dist(expl_edges_mask, d1, d2, hom_edge):
    """
    Check if the two query drugs (d1, d2) are connected via selected explanation edges
    in hom_edge (Strict Connectedness), and if so, return the shortest path length.
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


def check_lenient_connectivity_and_dist(expl_edges_mask, d1, d2, hom_edge):
    """
    Check if the selected explanation edges are a subset of some d1-to-d2 path
    in the original graph (within 4 hops).
    """
    # If the explanation is strictly connected, it is leniently connected by definition
    is_strict_conn, strict_hop_dist = check_connectivity_and_dist(expl_edges_mask, d1, d2, hom_edge)
    if is_strict_conn:
        return True, strict_hop_dist

    if isinstance(expl_edges_mask, torch.Tensor):
        selected_indices = torch.where(expl_edges_mask > 0.5)[0].tolist()
    else:
        selected_indices = [idx for idx, val in enumerate(expl_edges_mask) if val > 0.5]
    
    if not selected_indices:
        return False, -1
        
    selected_edges_set = set()
    for idx in selected_indices:
        u, v = hom_edge[0, idx].item(), hom_edge[1, idx].item()
        selected_edges_set.add(tuple(sorted((u, v))))
        
    # Since any path of length <= 4 has at most 4 edges, if the explanation has more than 4 edges,
    # it can never be a subset of a single path. This prevents DFS combinatorial explosion on large masks.
    if len(selected_edges_set) > 4:
        return False, -1
        
    src = hom_edge[0].tolist()
    dst = hom_edge[1].tolist()
    
    # Extract local 2-hop neighborhood of query drugs to restrict path search space
    from explanations.kec import get_k_hop_subgraph
    _, local_subgraph_mask = get_k_hop_subgraph(hom_edge.cpu(), [d1, d2], k=2)
    local_edge_indices = torch.where(local_subgraph_mask)[0].tolist()
    
    from collections import defaultdict
    adj = defaultdict(list)
    for idx in local_edge_indices:
        u, v = src[idx], dst[idx]
        adj[u].append(v)
        adj[v].append(u)
        
    # Run BFS from d1 and d2 to find shortest distances in local graph
    def bfs_dist(start):
        dist = {start: 0}
        queue = [start]
        while queue:
            curr = queue.pop(0)
            d = dist[curr]
            if d >= 4:
                continue
            for neighbor in adj[curr]:
                if neighbor not in dist:
                    dist[neighbor] = d + 1
                    queue.append(neighbor)
        return dist

    dist_d1 = bfs_dist(d1)
    dist_d2 = bfs_dist(d2)

    # If d2 is not reachable from d1 in <= 4 steps, then no path exists
    if d2 not in dist_d1 or dist_d1[d2] > 4:
        return False, -1

    # Filter local adjacency list to prune nodes and edges that cannot be on any path of length <= 4
    pruned_adj = defaultdict(list)
    for idx in local_edge_indices:
        u, v = src[idx], dst[idx]
        # Check if u and v are both within reach of d1 and d2, and the edge lies on some path of length <= 4
        if u in dist_d1 and v in dist_d2 and dist_d1[u] + 1 + dist_d2[v] <= 4:
            pruned_adj[u].append((v, idx))
        if v in dist_d1 and u in dist_d2 and dist_d1[v] + 1 + dist_d2[u] <= 4:
            pruned_adj[v].append((u, idx))
            
    paths_found = []
    
    def dfs(curr, path_nodes, path_edges):
        if curr == d2:
            paths_found.append((list(path_nodes), list(path_edges)))
            return
        if len(path_nodes) > 4: # path length <= 4 edges (means <= 5 nodes)
            return
            
        for neighbor, edge_idx in pruned_adj[curr]:
            if neighbor not in path_nodes:
                # Check if this step is promising (can reach d2 within remaining hops)
                if neighbor in dist_d2 and len(path_nodes) + dist_d2[neighbor] <= 5:
                    path_nodes.append(neighbor)
                    path_edges.append(edge_idx)
                    dfs(neighbor, path_nodes, path_edges)
                    path_edges.pop()
                    path_nodes.pop()
                
    dfs(d1, [d1], [])
    
    best_dist = -1
    for p_nodes, p_edges in paths_found:
        path_edges_set = set()
        for idx in p_edges:
            u, v = src[idx], dst[idx]
            path_edges_set.add(tuple(sorted((u, v))))
            
        if selected_edges_set.issubset(path_edges_set):
            dist = len(p_edges)
            if best_dist == -1 or dist < best_dist:
                best_dist = dist
                
    if best_dist != -1:
        return True, best_dist
    return False, -1

