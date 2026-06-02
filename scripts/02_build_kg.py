"""
Build heterogeneous knowledge graph from downloaded files.
Outputs a PyTorch Geometric HeteroData object.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import csv
import gzip


def parse_drugbank(xml_path):
    """Extract drug-protein edges, drug list, proteins, and RxCUI mapping stramingly."""
    import xml.etree.ElementTree as ET
    print("Parsing DrugBank XML streamingly...")
    ns = {'db': 'http://www.drugbank.ca'}
    
    drug_protein = []
    drug_list = []
    protein_set = set()
    rxcui_to_db = {}
    
    context = ET.iterparse(xml_path, events=('end',))
    for event, elem in context:
        if elem.tag == '{http://www.drugbank.ca}drug':
            db_id_elem = elem.find("db:drugbank-id[@primary='true']", ns)
            if db_id_elem is not None:
                drug_id = db_id_elem.text
                drug_list.append(drug_id)
                
                # Extract RxCUI mapping
                ext_ids = elem.find("db:external-identifiers", ns)
                if ext_ids is not None:
                    for ext_id in ext_ids.findall("db:external-identifier", ns):
                        res = ext_id.find("db:resource", ns)
                        ident = ext_id.find("db:identifier", ns)
                        if res is not None and res.text == 'RxCUI' and ident is not None:
                            rxcui_to_db[ident.text] = drug_id
                            break
                
                # Extract targets
                for target in elem.findall("db:targets/db:target", ns):
                    uniprot = target.find(".//db:uniprot-id", ns)
                    if uniprot is not None:
                        drug_protein.append((drug_id, uniprot.text))
                        protein_set.add(uniprot.text)
            
            elem.clear() # Free XML memory
            
    print(f"Parsed {len(drug_list)} drugs, {len(protein_set)} proteins, and mapped {len(rxcui_to_db)} RxCUIs.")
    return drug_list, list(protein_set), drug_protein, rxcui_to_db


def process_twosides(twosides_path, drug_to_idx, rxcui_to_db, limit_se=200):
    """Create drug pair -> side effect multi-hot labels using low-memory streaming."""
    print("Processing TWOSIDES csv streamingly in a single pass...")
    
    drug_pairs = {}
    side_effect_counts = {}
    
    with open(twosides_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        # Map headers
        col_to_idx = {col.lower().strip(): idx for idx, col in enumerate(header)}
        d1_idx = col_to_idx.get('drug_1_rxnorn_id')
        d2_idx = col_to_idx.get('drug_2_rxnorm_id')
        se_idx = col_to_idx.get('condition_concept_name')
        
        if d1_idx is None: d1_idx = 0
        if d2_idx is None: d2_idx = 2
        if se_idx is None: se_idx = 5
        
        for row in reader:
            if not row or len(row) <= max(d1_idx, d2_idx, se_idx):
                continue
            
            rxcui1, rxcui2 = row[d1_idx], row[d2_idx]
            d1_db = rxcui_to_db.get(rxcui1)
            d2_db = rxcui_to_db.get(rxcui2)
            
            if d1_db in drug_to_idx and d2_db in drug_to_idx:
                pair = tuple(sorted((drug_to_idx[d1_db], drug_to_idx[d2_db])))
                se = row[se_idx]
                
                side_effect_counts[se] = side_effect_counts.get(se, 0) + 1
                if pair not in drug_pairs:
                    drug_pairs[pair] = set()
                drug_pairs[pair].add(se)
                
    # Filter for top side effects to keep the label tensor reasonable
    sorted_se = sorted(side_effect_counts.items(), key=lambda x: x[1], reverse=True)
    top_se = {se: idx for idx, (se, _) in enumerate(sorted_se[:limit_se])}
    side_effects_list = list(top_se.keys())
    
    print(f"Top {len(side_effects_list)} side effects selected.")
    
    # Filter drug pairs for only the top side effects
    filtered_pairs = {}
    for pair, ses in drug_pairs.items():
        valid_ses = [se for se in ses if se in top_se]
        if valid_ses:
            filtered_pairs[pair] = valid_ses
            
    num_pairs = len(filtered_pairs)
    num_se = len(side_effects_list)
    print(f"Mapped {num_pairs} valid drug pairs with {num_se} side effects.")
    
    labels = np.zeros((num_pairs, num_se), dtype=np.float32)
    pair_list = list(filtered_pairs.keys())
    for i, (pair, ses) in enumerate(filtered_pairs.items()):
        for se in ses:
            labels[i, top_se[se]] = 1.0
            
    return pair_list, labels, side_effects_list


def build_graph():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    raw_dir = Path(config["data"]["raw_dir"])
    graph_dir = Path(config["data"]["graph_dir"])
    graph_dir.mkdir(parents=True, exist_ok=True)

    # Parse DrugBank
    drug_list, protein_list, drug_protein, rxcui_to_db = parse_drugbank(raw_dir / "drugbank.xml")
    drug_to_idx = {d: i for i, d in enumerate(drug_list)}
    protein_to_idx = {p: i for i, p in enumerate(protein_list)}

    # Process STRING
    string_path = raw_dir / "9606.protein.links.v12.0.txt"
    ppi_edges = []
    if string_path.exists():
        print("Processing STRING protein links streamingly...")
        with open(string_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=' ')
            header = next(reader)
            
            for row in reader:
                if not row or len(row) < 3:
                    continue
                score = int(row[2])
                if score >= 700:
                    p1 = row[0].split('.')[-1]
                    p2 = row[1].split('.')[-1]
                    if p1 in protein_to_idx and p2 in protein_to_idx:
                        ppi_edges.append((protein_to_idx[p1], protein_to_idx[p2]))
    
    # Process TWOSIDES
    twosides_path = raw_dir / "TWOSIDES.csv"
    pair_list, labels, side_effects = process_twosides(twosides_path, drug_to_idx, rxcui_to_db, limit_se=10)
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
    data['protein', 'interacts', 'protein'].edge_index = torch.tensor(ppi_edges, dtype=torch.long).t().contiguous() if ppi_edges else torch.empty((2, 0), dtype=torch.long)

    # DDI labels: store as edge attribute on drug-drug edges
    drug_pair_edges = torch.tensor(pair_list, dtype=torch.long).t().contiguous()  # shape (2, num_pairs)
    data['drug', 'ddi', 'drug'].edge_index = drug_pair_edges
    data['drug', 'ddi', 'drug'].side_effect_label = torch.tensor(labels, dtype=torch.float)  # (num_pairs, num_side_effects)

    # Save
    torch.save(data, graph_dir / "hetero_data.pt")
    with open(graph_dir / "drug_list.txt", "w", encoding='utf-8') as f:
        f.write("\n".join(drug_list))
    with open(graph_dir / "protein_list.txt", "w", encoding='utf-8') as f:
        f.write("\n".join(protein_list))
    with open(graph_dir / "side_effects.txt", "w", encoding='utf-8') as f:
        f.write("\n".join(side_effects))

    print(f"Graph saved to {graph_dir / 'hetero_data.pt'}")
    print(f"Drugs: {len(drug_list)}, Proteins: {len(protein_list)}, Side effects: {num_side_effects}")
    print(f"Drug-protein edges: {len(drug_protein_edges)}, PPI edges: {len(ppi_edges)}, DDI pairs: {len(pair_list)}")


if __name__ == "__main__":
    build_graph()
