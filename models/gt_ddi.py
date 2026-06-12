import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, SAGEConv
from torch_geometric.utils import softmax


class KernelLinearAttention(nn.Module):
    """
    Linear attention (Performer/Katharopoulos style) that runs in O(N) memory and time.
    Softmax(QK^T)V is approximated via kernel feature maps phi(Q) (phi(K)^T V).
    We use the positive feature map phi(x) = elu(x) + 1.
    """
    def __init__(self, in_dim, out_dim=None, heads=4):
        super().__init__()
        self.heads = heads
        if out_dim is None:
            out_dim = in_dim
        self.head_dim = out_dim // heads
        assert out_dim % heads == 0, f"out_dim ({out_dim}) must be divisible by heads ({heads})"
        
        self.q_proj = nn.Linear(in_dim, out_dim)
        self.k_proj = nn.Linear(in_dim, out_dim)
        self.v_proj = nn.Linear(in_dim, out_dim)
        self.out_proj = nn.Linear(out_dim, out_dim)

    def forward(self, x, edge_index=None, return_attention_weights=False):
        # x: (N, in_dim)
        N, _ = x.shape
        H = self.heads
        d = self.head_dim
        D = H * d
        
        q = self.q_proj(x).view(N, H, d)
        k = self.k_proj(x).view(N, H, d)
        v = self.v_proj(x).view(N, H, d)
        
        # Apply kernel function: elu(x) + 1
        phi_q = F.elu(q) + 1.0
        phi_k = F.elu(k) + 1.0
        
        # Permute for batched matrix multiplication over heads
        phi_q_perm = phi_q.permute(1, 0, 2) # (H, N, d)
        phi_k_perm = phi_k.permute(1, 0, 2) # (H, N, d)
        v_perm = v.permute(1, 0, 2)         # (H, N, d)
        
        # Sum of keys: (H, d)
        sum_k = phi_k_perm.sum(dim=1) # (H, d)
        
        # Denominator normalizer for each head and query: (H, N, 1)
        den = torch.einsum('hnd,hd->hn', phi_q_perm, sum_k).unsqueeze(-1) + 1e-9
        
        # Key-Value dot product: (H, d, d)
        phi_k_t = phi_k_perm.permute(0, 2, 1) # (H, d, N)
        k_v = torch.bmm(phi_k_t, v_perm)     # (H, d, d)
        
        # Numerator: (H, N, d)
        num = torch.bmm(phi_q_perm, k_v)     # (H, N, d)
        
        # Normalised output: (H, N, d) -> (N, H, d) -> (N, D)
        out = num / den
        out = out.permute(1, 0, 2).reshape(N, D)
        out = self.out_proj(out)
        
        if return_attention_weights and edge_index is not None:
            # Compute query-key dot product attention scores only for the edges in edge_index
            src, dst = edge_index
            q_dst = q[dst] # (E, H, d)
            k_src = k[src] # (E, H, d)
            
            # Dot product normalized by sqrt(d)
            scores = (q_dst * k_src).sum(dim=-1) / math.sqrt(d) # (E, H)
            alpha = softmax(scores, dst, num_nodes=N)
            return out, alpha
            
        return out


class LinearGraphTransformerConv(nn.Module):
    """
    Graph Transformer layer utilizing kernel-based global linear attention
    combined with local message passing (SAGEConv) to encode structural bias.
    """
    def __init__(self, in_channels, out_channels, heads=4, dropout=0.2):
        super().__init__()
        self.heads = heads
        self.out_channels = out_channels
        self.attn = KernelLinearAttention(in_channels, out_dim=out_channels, heads=heads)
        self.conv = SAGEConv(in_channels, out_channels)
        self.proj = nn.Linear(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, return_attention_weights=False):
        # Global linear attention
        if return_attention_weights:
            h_attn, alpha = self.attn(x, edge_index, return_attention_weights=True)
        else:
            h_attn = self.attn(x)
            alpha = None
            
        # Local structural feature propagation
        h_conv = self.conv(x, edge_index)
        
        # Combine global and local features
        h = h_attn + h_conv
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Skip connection & LayerNorm
        h = self.norm(h + self.proj(x))
        
        if return_attention_weights:
            return h, (edge_index, alpha)
        return h


class HeteroGTDDI(nn.Module):
    def __init__(self, num_drugs, num_proteins, drug_hidden=128, protein_hidden=128,
                 gat_hidden=128, gat_heads=4, out_dim=964, dropout=0.2, attention_type="sparse"):
        super().__init__()
        self.drug_embed = nn.Embedding(num_drugs, drug_hidden)
        self.protein_embed = nn.Embedding(num_proteins, protein_hidden)
        assert drug_hidden == protein_hidden, "drug_hidden and protein_hidden must be equal to concatenate along the node axis"
        
        self.attention_type = attention_type
        self.dropout = dropout
        
        if attention_type == "sparse":
            # Uses TransformerConv (O(E) sparse message-passing graph attention)
            self.gt1 = TransformerConv(drug_hidden, gat_hidden, heads=gat_heads, concat=True, dropout=dropout)
            self.gt2 = TransformerConv(gat_hidden * gat_heads, gat_hidden, heads=1, concat=False, dropout=dropout)
        elif attention_type == "linear":
            # Uses custom LinearGraphTransformerConv (O(N) global linear attention + local structural bias)
            self.gt1 = LinearGraphTransformerConv(drug_hidden, gat_hidden * gat_heads, heads=gat_heads, dropout=dropout)
            self.gt2 = LinearGraphTransformerConv(gat_hidden * gat_heads, gat_hidden, heads=1, dropout=dropout)
        else:
            raise ValueError(f"Unknown attention_type: {attention_type}")
            
        # Compatibility aliases for downstream explainers (e.g. KEC) and diagnostic scripts
        self.gat1 = self.gt1
        self.gat2 = self.gt2

        self.mlp = nn.Sequential(
            nn.Linear(gat_hidden * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, out_dim)
        )
        self.bypass_proj = nn.Linear(drug_hidden, gat_hidden)

    def forward(self, x_dict, edge_index_homogeneous, num_drugs, return_attention_weights=False, bypass_gat=False):
        drug_emb = self.drug_embed(torch.arange(num_drugs, device=next(self.parameters()).device))
        
        if bypass_gat:
            drug_out = F.relu(self.bypass_proj(drug_emb))
            if return_attention_weights:
                return drug_out, []
            return drug_out

        num_proteins = self.protein_embed.num_embeddings
        protein_emb = self.protein_embed(torch.arange(num_proteins, device=next(self.parameters()).device))
        x = torch.cat([drug_emb, protein_emb], dim=0)

        x = F.dropout(x, p=self.dropout, training=self.training)
        
        if return_attention_weights:
            x, (edge_idx1, alpha1) = self.gt1(x, edge_index_homogeneous, return_attention_weights=True)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x, (edge_idx2, alpha2) = self.gt2(x, edge_index_homogeneous, return_attention_weights=True)
            drug_out = x[:num_drugs]
            return drug_out, [(edge_idx1, alpha1), (edge_idx2, alpha2)]
        else:
            x = F.elu(self.gt1(x, edge_index_homogeneous))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.gt2(x, edge_index_homogeneous)
            drug_out = x[:num_drugs]
            return drug_out

    def predict_side_effects_logits(self, drug_embeddings, drug_pairs):
        emb_i = drug_embeddings[drug_pairs[:, 0]]
        emb_j = drug_embeddings[drug_pairs[:, 1]]
        pair_emb = torch.cat([emb_i, emb_j], dim=1)
        logits = self.mlp(pair_emb)
        return logits

    def predict_side_effects(self, drug_embeddings, drug_pairs):
        logits = self.predict_side_effects_logits(drug_embeddings, drug_pairs)
        return torch.sigmoid(logits)
