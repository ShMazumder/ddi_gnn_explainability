import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, Linear


class HeteroGATDDI(nn.Module):
    def __init__(self, num_drugs, num_proteins, drug_hidden=128, protein_hidden=128,
                 gat_hidden=128, gat_heads=4, out_dim=964, dropout=0.2):
        super().__init__()
        self.drug_embed = nn.Embedding(num_drugs, drug_hidden)
        self.protein_embed = nn.Embedding(num_proteins, protein_hidden)
        self.gat1 = GATConv(drug_hidden + protein_hidden, gat_hidden, heads=gat_heads, concat=True)
        self.gat2 = GATConv(gat_hidden * gat_heads, gat_hidden, heads=1, concat=False)
        self.mlp = nn.Sequential(
            nn.Linear(gat_hidden * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, out_dim)
        )
        self.dropout = dropout

    def forward(self, x_dict, edge_index_homogeneous, num_drugs, return_attention_weights=False):
        # x_dict: {'drug': None, 'protein': None} – we use embeddings
        drug_emb = self.drug_embed(torch.arange(num_drugs, device=next(self.parameters()).device))
        num_proteins = self.protein_embed.num_embeddings
        protein_emb = self.protein_embed(torch.arange(num_proteins, device=next(self.parameters()).device))
        x = torch.cat([drug_emb, protein_emb], dim=0)

        x = F.dropout(x, p=self.dropout, training=self.training)
        
        if return_attention_weights:
            x, (_, alpha1) = self.gat1(x, edge_index_homogeneous, return_attention_weights=True)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x, (_, alpha2) = self.gat2(x, edge_index_homogeneous, return_attention_weights=True)
            drug_out = x[:num_drugs]
            return drug_out, [alpha1, alpha2]
        else:
            x = F.elu(self.gat1(x, edge_index_homogeneous))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.gat2(x, edge_index_homogeneous)
            drug_out = x[:num_drugs]
            return drug_out

    def predict_side_effects(self, drug_embeddings, drug_pairs):
        emb_i = drug_embeddings[drug_pairs[:, 0]]
        emb_j = drug_embeddings[drug_pairs[:, 1]]
        pair_emb = torch.cat([emb_i, emb_j], dim=1)
        logits = self.mlp(pair_emb)
        return torch.sigmoid(logits)
