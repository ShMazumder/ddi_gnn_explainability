import sys
sys.path.append(".")
import torch
import numpy as np
from torch_geometric.data import HeteroData
from models.gat_ddi import HeteroGATDDI
from explanations.attention_rollout import attention_rollout
from explanations.gnnexplainer_wrapper import run_gnnexplainer
from explanations.pgexplainer_wrapper import run_pgexplainer
from explanations.kec import run_kec
from faithfulness.metrics import (
    sufficiency, necessity, fidelity_plus, fidelity_minus, sparsity,
    check_connectivity_and_dist, check_lenient_connectivity_and_dist
)
from explanations.kec import get_k_hop_subgraph

def build_synthetic_data():
    data = HeteroData()
    
    # 5 drugs, 5 proteins, 3 side effects
    num_drugs = 5
    num_proteins = 5
    num_side_effects = 3
    
    data['drug'].num_nodes = num_drugs
    data['protein'].num_nodes = num_proteins
    data['side_effect'].num_nodes = num_side_effects
    
    # Edges: drug-protein (binds)
    # drug 0 binds protein 0, drug 1 binds protein 1, etc.
    binds_edges = torch.tensor([
        [0, 1, 2, 3, 0],
        [0, 1, 2, 3, 4]
    ], dtype=torch.long)
    data['drug', 'binds', 'protein'].edge_index = binds_edges
    
    # Edges: protein-protein (interacts)
    ppi_edges = torch.tensor([
        [0, 1, 2],
        [1, 2, 3]
    ], dtype=torch.long)
    data['protein', 'interacts', 'protein'].edge_index = ppi_edges
    
    # Edges: drug-drug (ddi)
    ddi_edges = torch.tensor([
        [0, 1],
        [1, 2]
    ], dtype=torch.long)
    data['drug', 'ddi', 'drug'].edge_index = ddi_edges
    data['drug', 'ddi', 'drug'].side_effect_label = torch.ones((2, num_side_effects), dtype=torch.float)
    
    return data

def test_pipeline():
    data = build_synthetic_data()
    num_drugs = data['drug'].num_nodes
    num_proteins = data['protein'].num_nodes
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize GAT model
    model = HeteroGATDDI(
        num_drugs=num_drugs,
        num_proteins=num_proteins,
        drug_hidden=16,
        protein_hidden=16,
        gat_hidden=16,
        gat_heads=2,
        out_dim=3,
        dropout=0.0
    ).to(device)
    model.eval()
    
    # Create bidirectional homogeneous edge index
    dp_edges = data['drug', 'binds', 'protein'].edge_index.clone()
    dp_edges[1] += num_drugs
    pd_edges = torch.stack([dp_edges[1], dp_edges[0]], dim=0)
    
    p_edges = data['protein', 'interacts', 'protein'].edge_index.clone()
    p_edges[0] += num_drugs
    p_edges[1] += num_drugs
    p_edges_rev = torch.stack([p_edges[1], p_edges[0]], dim=0)
    
    hom_edge = torch.cat([dp_edges, pd_edges, p_edges, p_edges_rev], dim=1).to(device)
    
    sample_pairs = data['drug', 'ddi', 'drug'].edge_index.t().to(device)
    
    print("\n--- Running Explanation Methods on Bidirectional Graph ---")
    
    # Run each explainer
    explanations_dict = {}
    
    # 1. Attention Rollout
    print("Running Attention Rollout...")
    explanations_dict['attention'] = attention_rollout(model, hom_edge, num_drugs, sample_pairs, device)
    
    # 2. GNNExplainer
    print("Running GNNExplainer...")
    explanations_dict['gnnexplainer'] = run_gnnexplainer(model, hom_edge, num_drugs, sample_pairs, device)
    
    # 3. PGExplainer
    print("Running PGExplainer...")
    explanations_dict['pgexplainer'] = run_pgexplainer(model, hom_edge, num_drugs, sample_pairs, device)
    
    # 4. KEC
    print("Running KEC...")
    explanations_dict['kec'] = run_kec(model, hom_edge, num_drugs, sample_pairs, device, data)
    
    # Evaluate faithfulness metrics
    # Get full predictions
    with torch.no_grad():
        drug_emb = model.forward({'drug': None, 'protein': None}, hom_edge, num_drugs)
        full_preds = model.predict_side_effects(drug_emb, sample_pairs)
        
    print("\n--- Evaluating Faithfulness Metrics ---")
    for method_name, explanations in explanations_dict.items():
        print(f"Method: {method_name}")
        suff_scores = []
        nec_scores = []
        fid_plus_scores = []
        fid_minus_scores = []
        sparsity_scores = []
        conn_strict_scores = []
        hop_strict_scores = []
        conn_lenient_scores = []
        hop_lenient_scores = []
        
        for i, expl in enumerate(explanations):
            if expl is None:
                continue
            expl_edges = expl.get('edge_mask', None)
            if expl_edges is None:
                continue
                
            # Sufficiency
            expl_edge_index = hom_edge[:, expl_edges > 0.5] if isinstance(expl_edges, torch.Tensor) else hom_edge
            with torch.no_grad():
                drug_emb_sub = model.forward({'drug': None, 'protein': None}, expl_edge_index, num_drugs)
                sub_pred = model.predict_side_effects(drug_emb_sub, sample_pairs[i:i+1])
            suff = sufficiency(full_preds[i:i+1], sub_pred)
            suff_scores.append(suff)
            
            # Necessity
            complement_mask = ~(expl_edges > 0.5) if isinstance(expl_edges, torch.Tensor) else torch.ones(hom_edge.size(1), dtype=torch.bool, device=device)
            compl_edge_index = hom_edge[:, complement_mask]
            with torch.no_grad():
                drug_emb_compl = model.forward({'drug': None, 'protein': None}, compl_edge_index, num_drugs)
                compl_pred = model.predict_side_effects(drug_emb_compl, sample_pairs[i:i+1])
            nec = necessity(full_preds[i:i+1], compl_pred)
            nec_scores.append(nec)

            # Fidelity+
            fid_p = fidelity_plus(full_preds[i:i+1], compl_pred)
            fid_plus_scores.append(fid_p)

            # Fidelity-
            fid_m = fidelity_minus(full_preds[i:i+1], sub_pred)
            fid_minus_scores.append(fid_m)

            # Sparsity
            d1, d2 = sample_pairs[i].tolist()
            _, subgraph_mask = get_k_hop_subgraph(hom_edge.cpu(), [d1, d2], k=2)
            expl_sparsity = sparsity(expl_edges, subgraph_mask)
            sparsity_scores.append(expl_sparsity)

            # Strict connectedness and hop distance
            is_strict, strict_hop = check_connectivity_and_dist(expl_edges, d1, d2, hom_edge)
            conn_strict_scores.append(float(is_strict))
            if is_strict:
                hop_strict_scores.append(float(strict_hop))
            
            # Lenient connectedness and hop distance
            is_lenient, lenient_hop = check_lenient_connectivity_and_dist(expl_edges, d1, d2, hom_edge)
            conn_lenient_scores.append(float(is_lenient))
            if is_lenient:
                hop_lenient_scores.append(float(lenient_hop))
            
        print(f"  Avg Sufficiency: {np.mean(suff_scores) if suff_scores else 0.0:.4f}")
        print(f"  Avg Necessity: {np.mean(nec_scores) if nec_scores else 0.0:.4f}")
        print(f"  Avg Fidelity+: {np.mean(fid_plus_scores) if fid_plus_scores else 0.0:.4f}")
        print(f"  Avg Fidelity-: {np.mean(fid_minus_scores) if fid_minus_scores else 0.0:.4f}")
        print(f"  Avg Sparsity: {np.mean(sparsity_scores) if sparsity_scores else 0.0:.4f}")
        print(f"  Strict Connected %: {np.mean(conn_strict_scores) * 100.0 if conn_strict_scores else 0.0:.1f}% | Hop: {np.mean(hop_strict_scores) if hop_strict_scores else -1.0:.2f}")
        print(f"  Lenient Connected %: {np.mean(conn_lenient_scores) * 100.0 if conn_lenient_scores else 0.0:.1f}% | Hop: {np.mean(hop_lenient_scores) if hop_lenient_scores else -1.0:.2f}")
        
    print("\nAll pipeline components tested successfully on bidirectional graph!")

if __name__ == "__main__":
    test_pipeline()
