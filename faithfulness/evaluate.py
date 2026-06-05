"""
Faithfulness evaluation harness.

Loads explanations and evaluates them using multiple faithfulness metrics.
Can be used standalone or imported by scripts/05_faithfulness.py.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path

from .metrics import sufficiency, necessity, fidelity_plus, fidelity_minus, sparsity
from explanations.kec import get_k_hop_subgraph


def evaluate_single_explanation(model, edge_index, num_drugs, drug_pair,
                                 explanation, device, threshold=0.5):
    """
    Evaluate a single explanation using all faithfulness metrics.

    Args:
        model: trained HeteroGATDDI model
        edge_index: full graph edge index
        num_drugs: number of drug nodes
        drug_pair: tensor of shape (2,) with drug indices
        explanation: dict with 'edge_mask' key
        device: torch device
        threshold: classification threshold

    Returns:
        dict with metric scores
    """
    if explanation is None:
        return None

    edge_mask = explanation.get('edge_mask', None)
    if edge_mask is None:
        return None

    pair_tensor = drug_pair.unsqueeze(0).to(device) if drug_pair.dim() == 1 else drug_pair.to(device)

    # Full prediction
    with torch.no_grad():
        drug_emb_full = model.forward({'drug': None, 'protein': None}, edge_index, num_drugs)
        full_pred = model.predict_side_effects(drug_emb_full, pair_tensor)

    # Prediction with only explanation edges (sufficiency)
    expl_mask = edge_mask > 0.5
    if expl_mask.sum() > 0:
        expl_edges = edge_index[:, expl_mask]
        with torch.no_grad():
            drug_emb_expl = model.forward({'drug': None, 'protein': None}, expl_edges, num_drugs)
            expl_pred = model.predict_side_effects(drug_emb_expl, pair_tensor)
    else:
        expl_pred = torch.zeros_like(full_pred)

    # Prediction without explanation edges (necessity)
    compl_mask = ~expl_mask
    if compl_mask.sum() > 0:
        compl_edges = edge_index[:, compl_mask]
        with torch.no_grad():
            drug_emb_compl = model.forward({'drug': None, 'protein': None}, compl_edges, num_drugs)
            compl_pred = model.predict_side_effects(drug_emb_compl, pair_tensor)
    else:
        compl_pred = torch.zeros_like(full_pred)

    d1, d2 = drug_pair.tolist()
    _, subgraph_mask = get_k_hop_subgraph(edge_index.cpu(), [d1, d2], k=2)

    return {
        'sufficiency': sufficiency(full_pred, expl_pred, threshold),
        'necessity': necessity(full_pred, compl_pred, threshold),
        'fidelity_plus': fidelity_plus(full_pred, compl_pred),
        'fidelity_minus': fidelity_minus(full_pred, expl_pred),
        'sparsity': sparsity(edge_mask, subgraph_mask)
    }


def evaluate_method(model, edge_index, num_drugs, sample_pairs,
                    explanations, device, method_name="unknown"):
    """
    Evaluate all explanations from a single method.

    Returns a summary dict with averaged metrics.
    """
    all_scores = []

    for i, expl in enumerate(explanations):
        scores = evaluate_single_explanation(
            model, edge_index, num_drugs,
            sample_pairs[i], expl, device
        )
        if scores is not None:
            all_scores.append(scores)

    if not all_scores:
        return {
            'method': method_name,
            'sufficiency': 0.0,
            'necessity': 0.0,
            'fidelity_plus': 0.0,
            'fidelity_minus': 0.0,
            'sparsity': 0.0,
            'num_evaluated': 0
        }

    df = pd.DataFrame(all_scores)
    return {
        'method': method_name,
        'sufficiency': df['sufficiency'].mean(),
        'necessity': df['necessity'].mean(),
        'fidelity_plus': df['fidelity_plus'].mean(),
        'fidelity_minus': df['fidelity_minus'].mean(),
        'sparsity': df['sparsity'].mean(),
        'num_evaluated': len(all_scores)
    }
