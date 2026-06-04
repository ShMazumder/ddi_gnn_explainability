"""
Generate publication-ready visualisations from the faithfulness results and explanations.

Produces:
  - Bar chart comparing faithfulness metrics across methods
  - Example subgraph visualisations for selected drug pairs
  - Training curves (if logged)
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pathlib import Path
import yaml


def plot_faithfulness_comparison(results_path, output_dir):
    """Bar chart comparing sufficiency and necessity across explanation methods."""
    df = pd.read_csv(results_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Sufficiency
    sns.barplot(data=df, x='method', y='sufficiency', ax=axes[0], hue='method', palette='viridis', legend=False)
    axes[0].set_title('Sufficiency by Method')
    axes[0].set_ylabel('Sufficiency Score')
    axes[0].set_xlabel('')
    axes[0].set_ylim(0, 1)

    # Necessity
    sns.barplot(data=df, x='method', y='necessity', ax=axes[1], hue='method', palette='magma', legend=False)
    axes[1].set_title('Necessity by Method')
    axes[1].set_ylabel('Necessity Score')
    axes[1].set_xlabel('')

    plt.tight_layout()
    fig.savefig(output_dir / "faithfulness_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved faithfulness comparison to {output_dir / 'faithfulness_comparison.png'}")


def plot_example_subgraph(explanations, pair_idx, drug_names, protein_names, output_dir):
    """Visualise an explanation subgraph for a single drug pair."""
    G = nx.Graph()

    expl = explanations[pair_idx]
    if expl is None:
        print(f"No explanation for pair {pair_idx}")
        return

    nodes = expl.get('nodes', [])
    edges = expl.get('edges', [])

    for n in nodes:
        node_type = 'drug' if n < len(drug_names) else 'protein'
        name = drug_names[n] if node_type == 'drug' else protein_names[n - len(drug_names)]
        G.add_node(n, label=name, node_type=node_type)

    for src, dst in edges:
        G.add_edge(src, dst)

    if len(G.nodes) == 0:
        print(f"Empty subgraph for pair {pair_idx}")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)

    drug_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'drug']
    protein_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'protein']

    nx.draw_networkx_nodes(G, pos, nodelist=drug_nodes, node_color='#4CAF50', node_size=500, ax=ax, label='Drug')
    nx.draw_networkx_nodes(G, pos, nodelist=protein_nodes, node_color='#2196F3', node_size=300, ax=ax, label='Protein')
    nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax)

    labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)

    ax.legend(scatterpoints=1)
    ax.set_title(f'Explanation Subgraph – Pair {pair_idx}')
    plt.tight_layout()
    fig.savefig(output_dir / f"subgraph_pair_{pair_idx}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved subgraph for pair {pair_idx}")


def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    results_dir = Path("results")
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Plot faithfulness comparison
    faithfulness_path = results_dir / "faithfulness_results.csv"
    if faithfulness_path.exists():
        plot_faithfulness_comparison(faithfulness_path, figures_dir)
    else:
        print("No faithfulness results found. Run 05_faithfulness.py first.")

    # Load drug/protein names
    graph_dir = Path(config["data"]["graph_dir"])
    drug_names = []
    protein_names = []
    if (graph_dir / "drug_list.txt").exists():
        drug_names = (graph_dir / "drug_list.txt").read_text().strip().split('\n')
    if (graph_dir / "protein_list.txt").exists():
        protein_names = (graph_dir / "protein_list.txt").read_text().strip().split('\n')

    # Plot example subgraphs
    explanations_path = results_dir / "explanations.pt"
    if explanations_path.exists():
        all_explanations = torch.load(explanations_path)
        # Use the first method's explanations for visualisation
        first_method = list(all_explanations.keys())[0]
        explanations = all_explanations[first_method]
        for i in range(min(3, len(explanations))):
            plot_example_subgraph(explanations, i, drug_names, protein_names, figures_dir)
    else:
        print("No explanations found. Run 04_explain_methods.py first.")

    print(f"\nAll figures saved to {figures_dir}")


if __name__ == "__main__":
    main()
