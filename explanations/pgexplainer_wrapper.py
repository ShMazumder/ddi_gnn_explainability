"""
PGExplainer wrapper for the HeteroGATDDI model.

PGExplainer (Parameterized GNN Explainer) trains a separate MLP to predict
edge importance masks from node embeddings, enabling amortised explanation
generation across multiple instances.
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


class PGExplainerMLP(nn.Module):
    """MLP that predicts edge importance from concatenated node embeddings."""

    def __init__(self, emb_dim, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, emb_src, emb_dst):
        """Predict edge importance from source and destination embeddings."""
        pair = torch.cat([emb_src, emb_dst], dim=1)
        return self.mlp(pair).squeeze(-1)


def train_pgexplainer(model, edge_index, num_drugs, train_pairs, device, epochs=30, lr=0.003):
    """
    Train the PGExplainer MLP on a set of drug pairs.

    The MLP learns to predict which edges are important for preserving
    the model's predictions across all training instances.
    """
    model.eval()
    edge_index = edge_index.to(device)

    # Get node embeddings from the trained model
    with torch.no_grad():
        drug_emb = model.forward({'drug': None, 'protein': None}, edge_index, num_drugs)

    # Full node embeddings (drug + protein)
    num_nodes = edge_index.max().item() + 1
    # Use drug embeddings for drug nodes, zeros for proteins (simplified)
    node_emb = torch.zeros(num_nodes, drug_emb.size(1), device=device)
    node_emb[:num_drugs] = drug_emb

    emb_dim = node_emb.size(1)
    pg_model = PGExplainerMLP(emb_dim).to(device)
    optimizer = torch.optim.Adam(pg_model.parameters(), lr=lr)

    src, dst = edge_index

    # Vectorised PGExplainer training using REINFORCE
    baseline = 0.0
    for epoch in range(epochs):
        pg_model.train()
        optimizer.zero_grad()

        # Predict edge importance probabilities
        edge_logits = pg_model(node_emb[src], node_emb[dst])
        edge_probs = torch.sigmoid(edge_logits)

        # Sample binary mask stochastically
        dist = torch.distributions.Bernoulli(edge_probs)
        sampled_mask = dist.sample()
        log_probs = dist.log_prob(sampled_mask).sum()

        # Ensure we keep at least a few edges
        if sampled_mask.sum() == 0:
            sampled_mask[torch.randint(0, len(sampled_mask), (5,))] = 1.0

        # Masked forward pass
        masked_edge_index = edge_index[:, sampled_mask > 0.5]
        drug_emb_masked = model.forward({'drug': None, 'protein': None}, masked_edge_index, num_drugs)

        # Predict DDI on full and masked graphs
        train_pairs_tensor = train_pairs.to(device)
        with torch.no_grad():
            target_pred = model.predict_side_effects(drug_emb, train_pairs_tensor)
        masked_pred = model.predict_side_effects(drug_emb_masked, train_pairs_tensor)

        # Loss: prediction fidelity
        pred_loss = nn.functional.mse_loss(masked_pred, target_pred.detach())

        # Update baseline
        if epoch == 0:
            baseline = pred_loss.item()
        else:
            baseline = 0.95 * baseline + 0.05 * pred_loss.item()

        # REINFORCE loss + sparsity regularisation
        advantage = pred_loss.item() - baseline
        policy_loss = advantage * log_probs
        sparsity_loss = edge_probs.mean() * 0.1

        total_loss = policy_loss + sparsity_loss
        total_loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"  PGExplainer epoch {epoch+1}/{epochs} | Loss: {total_loss.item():.4f} | Fidelity Loss: {pred_loss.item():.4f}")

    return pg_model, node_emb


def run_pgexplainer(model, edge_index, num_drugs, sample_pairs, device):
    """
    Train PGExplainer and generate explanations for sampled drug pairs.

    Returns a list of explanation dicts with edge_mask, nodes, and edges.
    """
    print("Training PGExplainer...")
    pg_model, node_emb = train_pgexplainer(
        model, edge_index, num_drugs, sample_pairs, device
    )

    pg_model.eval()
    edge_index = edge_index.to(device)
    src, dst = edge_index

    explanations = []
    with torch.no_grad():
        edge_logits = pg_model(node_emb[src], node_emb[dst])
        edge_probs = torch.sigmoid(edge_logits)

    for i in tqdm(range(len(sample_pairs)), desc="PGExplainer"):
        d1, d2 = sample_pairs[i].tolist()

        # Weight edges by relevance to this pair (proximity-based)
        pair_importance = edge_probs.clone()

        # Boost edges connected to the query drugs
        drug_mask = (src == d1) | (src == d2) | (dst == d1) | (dst == d2)
        pair_importance[drug_mask] *= 2.0

        # Normalise
        if pair_importance.max() > 0:
            pair_importance = pair_importance / pair_importance.max()

        # Top-k edges
        k = min(20, edge_index.size(1))
        topk_idx = pair_importance.topk(k).indices

        mask_binary = torch.zeros(edge_index.size(1), device=device)
        mask_binary[topk_idx] = 1.0

        relevant_edges = edge_index[:, topk_idx]
        relevant_nodes = torch.unique(relevant_edges).tolist()

        explanations.append({
            'edge_mask': mask_binary,
            'nodes': relevant_nodes,
            'edges': list(zip(relevant_edges[0].tolist(), relevant_edges[1].tolist()))
        })

    return explanations
