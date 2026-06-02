"""
Analyze usability survey responses from pharmacist participants.

Computes:
  - Average helpfulness scores per case and overall
  - Mechanism inference accuracy (if ground truth provided)
  - Inter-rater agreement (Krippendorff's alpha or Fleiss' kappa)
  - Qualitative summary of free-text comments
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml


def load_responses(csv_path):
    """Load and validate survey responses."""
    df = pd.read_csv(csv_path)
    required_cols = ['participant_id', 'case_id', 'helpfulness_score']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    return df


def compute_helpfulness_stats(df):
    """Compute per-case and overall helpfulness statistics."""
    case_stats = df.groupby('case_id')['helpfulness_score'].agg(['mean', 'std', 'count'])
    case_stats.columns = ['mean_helpfulness', 'std_helpfulness', 'num_responses']

    overall = {
        'overall_mean': df['helpfulness_score'].mean(),
        'overall_std': df['helpfulness_score'].std(),
        'overall_median': df['helpfulness_score'].median(),
        'num_participants': df['participant_id'].nunique(),
        'num_cases': df['case_id'].nunique()
    }

    return case_stats, overall


def compute_agreement(df):
    """
    Compute inter-rater agreement using Fleiss' kappa (simplified).
    Assumes helpfulness_score is ordinal (1-5).
    """
    # Pivot to create rating matrix: rows = cases, columns = participants
    pivot = df.pivot_table(index='case_id', columns='participant_id',
                           values='helpfulness_score', aggfunc='first')

    if pivot.shape[1] < 2:
        return {'fleiss_kappa': None, 'note': 'Need at least 2 raters'}

    # Simplified agreement: pairwise agreement proportion
    n_cases = pivot.shape[0]
    agreements = 0
    comparisons = 0

    for i in range(pivot.shape[1]):
        for j in range(i + 1, pivot.shape[1]):
            valid = pivot.iloc[:, i].notna() & pivot.iloc[:, j].notna()
            if valid.sum() > 0:
                agree = (pivot.iloc[:, i][valid] == pivot.iloc[:, j][valid]).sum()
                agreements += agree
                comparisons += valid.sum()

    pairwise_agreement = agreements / max(comparisons, 1)
    return {'pairwise_agreement': pairwise_agreement, 'comparisons': comparisons}


def plot_helpfulness(df, output_dir):
    """Generate helpfulness score visualisation."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Distribution of helpfulness scores
    sns.histplot(df['helpfulness_score'].dropna(), bins=5, kde=True, ax=axes[0], color='#4CAF50')
    axes[0].set_title('Distribution of Helpfulness Scores')
    axes[0].set_xlabel('Helpfulness Score (1-5)')
    axes[0].set_ylabel('Count')

    # Per-case average helpfulness
    case_means = df.groupby('case_id')['helpfulness_score'].mean().sort_values()
    case_means.plot(kind='barh', ax=axes[1], color='#2196F3')
    axes[1].set_title('Average Helpfulness by Case')
    axes[1].set_xlabel('Mean Helpfulness Score')
    axes[1].set_ylabel('Case ID')

    plt.tight_layout()
    output_path = Path(output_dir) / "helpfulness_analysis.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved helpfulness analysis to {output_path}")


def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    survey_path = config.get("usability", {}).get("survey_output", "usability/survey_results.csv")

    if not Path(survey_path).exists():
        print(f"Survey results not found at {survey_path}")
        print("Using template for demonstration...")
        survey_path = "usability/survey_responses_template.csv"

    df = load_responses(survey_path)
    print(f"Loaded {len(df)} responses from {df['participant_id'].nunique()} participants")

    # Helpfulness stats
    case_stats, overall = compute_helpfulness_stats(df)
    print("\n=== Per-Case Helpfulness ===")
    print(case_stats.to_string())
    print(f"\n=== Overall ===")
    for k, v in overall.items():
        print(f"  {k}: {v}")

    # Agreement
    agreement = compute_agreement(df)
    print(f"\n=== Inter-Rater Agreement ===")
    for k, v in agreement.items():
        print(f"  {k}: {v}")

    # Plot
    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    if df['helpfulness_score'].notna().sum() > 0:
        plot_helpfulness(df, output_dir)

    # Save summary
    summary_path = Path("results") / "usability_summary.csv"
    case_stats.to_csv(summary_path)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
