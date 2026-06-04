"""
Diagnostic script to analyze the topology, mapping coverage, and sparsity
of the protein-protein interaction (PPI) network.
"""

import torch
import numpy as np
from pathlib import Path
import yaml
import sys
sys.path.append(".")

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    graph_path = Path(config["data"]["graph_dir"]) / "hetero_data.pt"
    if not graph_path.exists():
        print(f"Graph file not found at {graph_path}. Please run 02_build_kg.py first.")
        return

    data = torch.load(graph_path, weights_only=False)
    
    num_proteins = data['protein'].num_nodes
    ppi_edge_index = data['protein', 'interacts', 'protein'].edge_index
    num_ppi_edges = ppi_edge_index.shape[1]

    # Convert to adjacency list (undirected)
    from collections import defaultdict
    adj = defaultdict(set)
    for u, v in ppi_edge_index.t().tolist():
        adj[u].add(v)
        adj[v].add(u)

    # Calculate connected components
    visited = set()
    components = []
    
    for node in range(num_proteins):
        if node not in visited:
            component = []
            queue = [node]
            visited.add(node)
            while queue:
                curr = queue.pop(0)
                component.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            components.append(component)

    component_sizes = [len(c) for c in components]
    largest_cc_size = max(component_sizes) if component_sizes else 0
    isolated_proteins = sum(1 for size in component_sizes if size == 1)
    connected_proteins = num_proteins - isolated_proteins
    
    # Calculate density
    # Possible undirected edges: V * (V - 1) / 2
    # In PyG, interacts is stored as directed edges or bidirectional edges?
    # PyG edge_index for interacts has num_ppi_edges
    possible_edges = num_proteins * (num_proteins - 1) / 2
    actual_undirected_edges = num_ppi_edges // 2  # Assuming stored bidirectionally
    density = actual_undirected_edges / possible_edges if possible_edges > 0 else 0.0
    avg_degree = (2 * actual_undirected_edges) / num_proteins if num_proteins > 0 else 0.0

    print("==================================================")
    print("STRING PPI Graph Diagnostic Report")
    print("==================================================")
    print(f"Total Proteins in Graph:     {num_proteins}")
    print(f"Total PPI Edges in PyG:      {num_ppi_edges}")
    print(f"Actual Undirected Edges:     {actual_undirected_edges}")
    print(f"Average Protein Degree:      {avg_degree:.4f}")
    print(f"PPI Edge Density:            {density:.8f}")
    print(f"Connected Components Count:  {len(components)}")
    print(f"Largest Component Size:      {largest_cc_size} ({largest_cc_size/num_proteins*100:.2f}%)")
    print(f"Connected Proteins (deg >=1): {connected_proteins} ({connected_proteins/num_proteins*100:.2f}%)")
    print(f"Isolated Proteins (deg = 0): {isolated_proteins} ({isolated_proteins/num_proteins*100:.2f}%)")
    print("==================================================")

if __name__ == "__main__":
    main()
