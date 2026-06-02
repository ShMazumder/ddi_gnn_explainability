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
