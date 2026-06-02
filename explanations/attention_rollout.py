"""
Attention Rollout explanation method for multi-head GAT.

Computes layer-wise attention rollout by multiplying attention weight matrices
across GAT layers, producing a per-edge importance score.
"""

import torch
import numpy as np
from tqdm import tqdm


def get_attention_weights(model, edge_index, num_drugs, device):
    """Extract attention weights from each GAT layer via forward hooks."""
    attention_weights = []

    def hook_fn(module, input, output):
        # GATConv returns (out, (edge_index, alpha)) when return_attention_weights is set
        if isinstance(output, tuple) and len(output) == 2:
            _, (_, alpha) = output
            attention_weights.append(alpha.detach())

    # Register hooks on GAT layers
    hooks = []
    for name, module in model.named_modules():
        if hasattr(module, '_alpha'):  # GATConv stores attention in _alpha
            hooks.append(module.register_forward_hook(hook_fn))

    # Forward pass to capture attention
    with torch.no_grad():
        model.forward({'drug': None, 'protein': None}, edge_index, num_drugs)

    # Remove hooks
    for h in hooks:
        h.remove()

    return attention_weights


def compute_rollout(attention_weights, edge_index, num_nodes):
    """
    Compute attention rollout across layers.

    For each layer, the attention matrix is averaged across heads,
    added with an identity (residual connection), and renormalised.
    The final rollout is the product of all layer matrices.
    """
    # Start with identity
    rollout = torch.eye(num_nodes, device=attention_weights[0].device)

    for alpha in attention_weights:
        # alpha shape: (num_edges, num_heads) or (num_edges,)
        if alpha.dim() == 2:
            alpha = alpha.mean(dim=1)  # average across heads

        # Build sparse attention matrix
        attn_matrix = torch.zeros(num_nodes, num_nodes, device=alpha.device)
        src, dst = edge_index
        attn_matrix[dst, src] = alpha

        # Add identity (residual connection)
        attn_matrix = attn_matrix + torch.eye(num_nodes, device=alpha.device)

        # Row-normalise
        row_sum = attn_matrix.sum(dim=1, keepdim=True)
        attn_matrix = attn_matrix / (row_sum + 1e-8)

        # Multiply into rollout
        rollout = rollout @ attn_matrix

    return rollout


def attention_rollout(model, edge_index, num_drugs, sample_pairs, device):
    """
    Run attention rollout for each sampled drug pair.

    Returns a list of explanation dicts, each containing:
      - 'edge_mask': per-edge importance scores
      - 'nodes': relevant node indices
      - 'edges': list of (src, dst) tuples in the explanation
    """
    model.eval()
    edge_index = edge_index.to(device)

    # Get attention weights
    attn_weights = get_attention_weights(model, edge_index, num_drugs, device)

    if not attn_weights:
        print("Warning: No attention weights captured. Ensure model uses GATConv with attention.")
        # Fallback: uniform edge importance
        explanations = []
        for i in range(len(sample_pairs)):
            explanations.append({
                'edge_mask': torch.ones(edge_index.size(1), device=device) / edge_index.size(1),
                'nodes': sample_pairs[i].tolist(),
                'edges': []
            })
        return explanations

    num_nodes = num_drugs + (edge_index.max().item() + 1 - num_drugs)
    rollout = compute_rollout(attn_weights, edge_index, num_nodes)

    explanations = []
    for i in tqdm(range(len(sample_pairs)), desc="Attention Rollout"):
        d1, d2 = sample_pairs[i].tolist()

        # Importance of each edge to this drug pair
        src, dst = edge_index
        edge_importance = rollout[d1, src] * rollout[d2, dst] + rollout[d2, src] * rollout[d1, dst]

        # Normalise
        if edge_importance.max() > 0:
            edge_importance = edge_importance / edge_importance.max()

        # Top-k edges as explanation
        k = min(20, edge_index.size(1))
        topk_idx = edge_importance.topk(k).indices
        mask = torch.zeros(edge_index.size(1), device=device)
        mask[topk_idx] = 1.0

        # Collect relevant nodes
        relevant_edges = edge_index[:, topk_idx]
        relevant_nodes = torch.unique(relevant_edges).tolist()

        explanations.append({
            'edge_mask': mask,
            'nodes': relevant_nodes,
            'edges': list(zip(relevant_edges[0].tolist(), relevant_edges[1].tolist()))
        })

    return explanations
