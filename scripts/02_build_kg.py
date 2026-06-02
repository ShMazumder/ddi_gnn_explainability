"""
Build heterogeneous knowledge graph from downloaded files.
Outputs a PyTorch Geometric HeteroData object.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml
import pickle


def parse_drugbank(xml_path):
    """Extract drug-protein edges and drug-drug interactions from DrugBank XML."""
    import xml.etree.ElementTree as ET
    print("Parsing DrugBank XML...")
    ns = {'db': 'http://www.drugbank.ca'}
    tree = ET.parse(xml_path)
    root = tree.getroot()

    drug_protein = []
    drug_list = []
    protein_set = set()

    for drug in root.findall("db:drug", ns):
        db_id = drug.find("db:drugbank-id[@primary='true']", ns)
        if db_id is None:
            continue
        drug_id = db_id.text
        drug_list.append(drug_id)

        for target in drug.findall("db:targets/db:target", ns):
            uniprot = target.find(".//db:uniprot-id", ns)
            if uniprot is not None:
                drug_protein.append((drug_id, uniprot.text))
                protein_set.add(uniprot.text)

    return drug_list, list(protein_set), drug_protein


def process_twosides(twosides_path, drug_to_idx):
    """Create drug pair -> side effect multi-hot labels."""
    df = pd.read_csv(twosides_path)
    # Assume columns: drug1, drug2, side_effect
    drug_pairs = {}
    side_effect_set = set()
    for _, row in df.iterrows():
        d1, d2 = row['drug1'], row['drug2']
        if d1 not in drug_to_idx or d2 not in drug_to_idx:
            continue
        pair = tuple(sorted((drug_to_idx[d1], drug_to_idx[d2])))
        se = row['side_effect']
        side_effect_set.add(se)
        if pair not in drug_pairs:
            drug_pairs[pair] = set()
        drug_pairs[pair].add(se)

    side_effect_list = list(side_effect_set)
    se_to_idx = {se: i for i, se in enumerate(side_effect_list)}

    # Build sparse label matrix
    num_pairs = len(drug_pairs)
    num_se = len(side_effect_list)
    labels = np.zeros((num_pairs, num_se), dtype=np.float32)
    pair_list = list(drug_pairs.keys())
    for i, (pair, ses) in enumerate(drug_pairs.items()):
        for se in ses:
            labels[i, se_to_idx[se]] = 1.0

    return pair_list, labels, side_effect_list


def build_graph():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    raw_dir = Path(config["data"]["raw_dir"])
    graph_dir = Path(config["data"]["graph_dir"])
    graph_dir.mkdir(parents=True, exist_ok=True)

    # Parse DrugBank
    drug_list, protein_list, drug_protein = parse_drugbank(raw_dir / "drugbank.xml")
    drug_to_idx = {d: i for i, d in enumerate(drug_list)}
    protein_to_idx = {p: i for i, p in enumerate(protein_list)}

    # Process STRING
    string_path = raw_dir / "9606.protein.links.v12.0.txt"
    string_df = pd.read_csv(string_path, sep=" ")
    string_df = string_df[string_df['combined_score'] >= 700]
    # Map STRING IDs (e.g., "9606.ENSP000...") to UniProt (requires mapping file – simplified)
    # For demo, we assume STRING IDs are already UniProt (they are not; real script needs mapping)
    ppi_edges = []
    for _, row in string_df.iterrows():
        p1 = row['protein1'].split('.')[-1]
        p2 = row['protein2'].split('.')[-1]
        if p1 in protein_to_idx and p2 in protein_to_idx:
            ppi_edges.append((protein_to_idx[p1], protein_to_idx[p2]))

    # Process TWOSIDES
    pair_list, labels, side_effects = process_twosides(raw_dir / "TWOSIDES.csv", drug_to_idx)
    side_effect_to_idx = {se: i for i, se in enumerate(side_effects)}
    num_side_effects = len(side_effects)

    # Build HeteroData
    from torch_geometric.data import HeteroData
    data = HeteroData()

    data['drug'].num_nodes = len(drug_list)
    data['protein'].num_nodes = len(protein_list)
    data['side_effect'].num_nodes = num_side_effects

    # Edges: drug-protein
    drug_protein_edges = [(drug_to_idx[d], protein_to_idx[p]) for d, p in drug_protein if d in drug_to_idx and p in protein_to_idx]
    data['drug', 'binds', 'protein'].edge_index = torch.tensor(drug_protein_edges, dtype=torch.long).t().contiguous()

    # Edges: protein-protein
    data['protein', 'interacts', 'protein'].edge_index = torch.tensor(ppi_edges, dtype=torch.long).t().contiguous()

    # DDI labels: store as edge attribute on drug-drug edges
    # Create a drug-drug edge for each unique pair
    drug_pair_edges = torch.tensor(pair_list, dtype=torch.long).t().contiguous()  # shape (2, num_pairs)
    data['drug', 'ddi', 'drug'].edge_index = drug_pair_edges
    data['drug', 'ddi', 'drug'].side_effect_label = torch.tensor(labels, dtype=torch.float)  # (num_pairs, num_side_effects)

    # Save
    torch.save(data, graph_dir / "hetero_data.pt")
    with open(graph_dir / "drug_list.txt", "w") as f:
        f.write("\n".join(drug_list))
    with open(graph_dir / "protein_list.txt", "w") as f:
        f.write("\n".join(protein_list))
    with open(graph_dir / "side_effects.txt", "w") as f:
        f.write("\n".join(side_effects))

    print(f"Graph saved to {graph_dir / 'hetero_data.pt'}")
    print(f"Drugs: {len(drug_list)}, Proteins: {len(protein_list)}, Side effects: {num_side_effects}")
    print(f"Drug-protein edges: {len(drug_protein_edges)}, PPI edges: {len(ppi_edges)}, DDI pairs: {len(pair_list)}")


if __name__ == "__main__":
    build_graph()
