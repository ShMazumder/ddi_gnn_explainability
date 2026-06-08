import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import wilcoxon

def compute_rank_biserial(x, y):
    diff = x - y
    diff = diff[diff != 0]
    if len(diff) == 0:
        return 0.0
    ranks = np.argsort(np.abs(diff)) + 1
    pos_ranks = sum(ranks[diff > 0])
    neg_ranks = sum(ranks[diff < 0])
    total = pos_ranks + neg_ranks
    if total == 0:
        return 0.0
    return (pos_ranks - neg_ranks) / total

def main():
    csv_path = Path("results/raw_faithfulness_scores.csv")
    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Please run scripts/05_faithfulness.py first on the remote environment.")
        return
        
    df = pd.read_csv(csv_path)
    print("=== Loaded Raw Faithfulness Scores ===")
    print(f"Found {len(df)} total rows across {df['method'].nunique()} methods.")
    
    # We want to perform statistical significance tests on fidelity_plus scores for the 5 comparisons:
    # 1. PGExplainer vs. Attention
    # 2. PGExplainer vs. GNNExplainer
    # 3. PGExplainer vs. KEC
    # 4. KEC vs. Attention
    # 5. KEC vs. GNNExplainer
    
    methods = df['method'].unique()
    # Format scores by method as dataframes indexed by pair_idx
    scores_by_method = {}
    for m in methods:
        scores_by_method[m] = df[df['method'] == m][['pair_idx', 'fidelity_plus']].set_index('pair_idx')
        
    comparisons = [
        ("pgexplainer", "attention", "PGExplainer vs. Attention"),
        ("pgexplainer", "gnnexplainer", "PGExplainer vs. GNNExplainer"),
        ("attention", "gnnexplainer", "Attention vs. GNNExplainer"),
        ("pgexplainer", "kec", "PGExplainer vs. KEC"),
        ("kec", "attention", "KEC vs. Attention"),
        ("kec", "gnnexplainer", "KEC vs. GNNExplainer")
    ]
    
    # Perform Wilcoxon signed-rank tests
    raw_p_values = []
    effect_sizes = []
    
    for m1, m2, label in comparisons:
        if m1 not in scores_by_method or m2 not in scores_by_method:
            print(f"Missing data for comparison: {m1} vs {m2}")
            return
            
        # Join on pair_idx to keep only common pairs
        joined = scores_by_method[m1].join(scores_by_method[m2], lsuffix='_1', rsuffix='_2', how='inner')
        s1 = joined['fidelity_plus_1'].values
        s2 = joined['fidelity_plus_2'].values
        
        # Check if the difference is entirely zero
        if np.all(s1 == s2):
            p_val = 1.0
        else:
            try:
                # two-sided Wilcoxon signed-rank test
                stat, p_val = wilcoxon(s1, s2, alternative='two-sided')
            except ValueError as e:
                # If all differences are zero/etc.
                p_val = 1.0
                
        r = compute_rank_biserial(s1, s2)
        raw_p_values.append(p_val)
        effect_sizes.append(r)
        
    # Apply Holm-Bonferroni correction
    m_tests = len(comparisons)
    sorted_indices = np.argsort(raw_p_values)
    adjusted_p_values = [0.0] * m_tests
    
    # Holm-Bonferroni step-up/down
    prev_adj = 0.0
    for idx, rank_idx in enumerate(sorted_indices):
        p_raw = raw_p_values[rank_idx]
        multiplier = m_tests - idx
        p_adj = min(p_raw * multiplier, 1.0)
        p_adj = max(p_adj, prev_adj)
        adjusted_p_values[rank_idx] = p_adj
        prev_adj = p_adj
        
    # Generate Markdown Table
    print("\n=== Formatted Markdown Table for Section 5.6.1 ===")
    print("| Comparison | Metric | p-value | Effect Size (Rank-Biserial) | Significant? |")
    print("| :--- | :---: | :---: | :---: | :---: |")
    
    for idx, (m1, m2, label) in enumerate(comparisons):
        p_val = adjusted_p_values[idx]
        r = effect_sizes[idx]
        p_str = f"<0.001" if p_val < 0.001 else f"{p_val:.4f}"
        sig_str = "Yes" if p_val < 0.05 else "No"
        print(f"| **{label}** | Fidelity+ | {p_str} | {r:.4f} | {sig_str} |")
        
    print("\nDone. Copy the table above and paste it into paper/thesis_chapter_5_draft.md!")

if __name__ == "__main__":
    main()
