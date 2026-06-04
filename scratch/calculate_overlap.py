import torch
import numpy as np
from sklearn.model_selection import train_test_split
import yaml
from pathlib import Path

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    graph_path = Path(config["data"]["graph_dir"]) / "hetero_data.pt"
    if not graph_path.exists():
        print(f"Graph file not found at {graph_path}. Please run build_kg script first.")
        return

    data = torch.load(graph_path, weights_only=False)
    edge_index = data['drug', 'ddi', 'drug'].edge_index.numpy() # (2, num_pairs)
    num_pairs = edge_index.shape[1]
    
    indices = np.arange(num_pairs)
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
    
    # Train, Val, Test drug pairs
    train_pairs = edge_index[:, train_idx]
    val_pairs = edge_index[:, val_idx]
    test_pairs = edge_index[:, test_idx]
    
    # Unique drugs in each split
    train_drugs = set(train_pairs.flatten())
    val_drugs = set(val_pairs.flatten())
    test_drugs = set(test_pairs.flatten())
    
    print("==================================================")
    print("Drug Overlap & Split Statistics")
    print("==================================================")
    print(f"Total DDI Pairs:       {num_pairs}")
    print(f"Train Pairs count:     {train_pairs.shape[1]} (70%)")
    print(f"Val Pairs count:       {val_pairs.shape[1]} (15%)")
    print(f"Test Pairs count:      {test_pairs.shape[1]} (15%)")
    print(f"Unique drugs in Train: {len(train_drugs)}")
    print(f"Unique drugs in Val:   {len(val_drugs)}")
    print(f"Unique drugs in Test:  {len(test_drugs)}")
    
    # Overlap
    test_overlap_with_train = test_drugs.intersection(train_drugs)
    print(f"Test drugs overlapping with Train: {len(test_overlap_with_train)} ({len(test_overlap_with_train)/len(test_drugs)*100:.2f}%)")
    
    # How many test pairs have completely unseen drugs?
    unseen_pairs_count = 0
    partially_seen_pairs_count = 0
    fully_seen_pairs_count = 0
    
    for i in range(test_pairs.shape[1]):
        d1, d2 = test_pairs[0, i], test_pairs[1, i]
        d1_seen = d1 in train_drugs
        d2_seen = d2 in train_drugs
        if d1_seen and d2_seen:
            fully_seen_pairs_count += 1
        elif d1_seen or d2_seen:
            partially_seen_pairs_count += 1
        else:
            unseen_pairs_count += 1
            
    print(f"\nTest pairs classification based on training drug visibility:")
    print(f"  - Both drugs seen in training (Fully Seen):     {fully_seen_pairs_count} ({fully_seen_pairs_count/test_pairs.shape[1]*100:.2f}%)")
    print(f"  - One drug seen in training (Partially Unseen): {partially_seen_pairs_count} ({partially_seen_pairs_count/test_pairs.shape[1]*100:.2f}%)")
    print(f"  - Neither drug seen in training (Fully Unseen): {unseen_pairs_count} ({unseen_pairs_count/test_pairs.shape[1]*100:.2f}%)")
    print("==================================================")

if __name__ == "__main__":
    main()
