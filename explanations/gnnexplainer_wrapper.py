"""
GNNExplainer wrapper for the HeteroGATDDI model.

Uses PyTorch Geometric's GNNExplainer to generate per-edge importance masks
that explain individual drug-drug interaction predictions.
"""

import torch
import numpy as np
from tqdm import tqdm

try:
    from torch_geometric.explain import Explainer, GNNExplainer
    HAS_EXPLAINER = True
except ImportError:
    HAS_EXPLAINER = False
    print("Warning: torch_geometric.explain not available. Install torch-geometric>=2.3.0")


def run_gnnexplainer(model, edge_index, num_drugs, sample_pairs, device):
    """
    Run GNNExplainer on each sampled drug pair.

    Returns a list of explanation dicts with edge_mask, nodes, and edges.
    """
    model.eval()
    edge_index = edge_index.to(device)

    if not HAS_EXPLAINER:
        print("GNNExplainer not available – returning empty explanations")
        return [None] * len(sample_pairs)

    explanations = []

    for i in tqdm(range(len(sample_pairs)), desc="GNNExplainer"):
        d1, d2 = sample_pairs[i].tolist()

        try:
            # Create a simplified forward function for GNNExplainer
            num_nodes = edge_index.max().item() + 1
            x = torch.zeros(num_nodes, 1, device=device)  # dummy features

            # GNNExplainer optimises an edge mask
            edge_mask = torch.nn.Parameter(torch.randn(edge_index.size(1), device=device))
            optimizer = torch.optim.Adam([edge_mask], lr=0.01)

            # Get target prediction
            with torch.no_grad():
                drug_emb = model.forward({'drug': None, 'protein': None}, edge_index, num_drugs)
                target_pred = model.predict_side_effects(drug_emb, sample_pairs[i:i+1].to(device))

            # Optimise edge mask using REINFORCE policy gradient
            baseline = 0.0
            for epoch in range(100):
                optimizer.zero_grad()
                sigmoid_mask = torch.sigmoid(edge_mask)
                
                # Sample binary mask stochastically
                dist = torch.distributions.Bernoulli(sigmoid_mask)
                sampled_mask = dist.sample()
                log_probs = dist.log_prob(sampled_mask).sum()
                
                # Ensure we keep at least a few edges (stochastic fallback)
                if sampled_mask.sum() == 0:
                    sampled_mask[torch.randint(0, len(sampled_mask), (5,))] = 1.0
                
                # Run forward pass with sampled edges
                masked_edge_index = edge_index[:, sampled_mask > 0.5]
                
                # Forward with masked edges
                drug_emb = model.forward({'drug': None, 'protein': None}, masked_edge_index, num_drugs)
                pred = model.predict_side_effects(drug_emb, sample_pairs[i:i+1].to(device))
                
                # Loss: prediction fidelity
                pred_loss = torch.nn.functional.mse_loss(pred, target_pred.detach())
                
                # Update running baseline
                if epoch == 0:
                    baseline = pred_loss.item()
                else:
                    baseline = 0.95 * baseline + 0.05 * pred_loss.item()
                
                # Policy gradient loss (REINFORCE)
                advantage = pred_loss.item() - baseline
                policy_loss = advantage * log_probs
                
                # Size and entropy regularisation
                size_loss = sigmoid_mask.sum() * 0.01
                entropy_loss = -sigmoid_mask * torch.log(sigmoid_mask + 1e-8) - (1 - sigmoid_mask) * torch.log(1 - sigmoid_mask + 1e-8)
                
                loss = policy_loss + size_loss + entropy_loss.mean() * 0.1
                loss.backward()
                optimizer.step()

            # Extract explanation
            final_mask = torch.sigmoid(edge_mask).detach()
            k = min(20, edge_index.size(1))
            topk_idx = final_mask.topk(k).indices

            mask_binary = torch.zeros(edge_index.size(1), device=device)
            mask_binary[topk_idx] = 1.0

            relevant_edges = edge_index[:, topk_idx]
            relevant_nodes = torch.unique(relevant_edges).tolist()

            explanations.append({
                'edge_mask': mask_binary,
                'nodes': relevant_nodes,
                'edges': list(zip(relevant_edges[0].tolist(), relevant_edges[1].tolist()))
            })

        except Exception as e:
            print(f"  GNNExplainer failed for pair {i}: {e}")
            explanations.append(None)

    return explanations
