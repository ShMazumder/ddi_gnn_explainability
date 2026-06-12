import json
import yaml
import subprocess
from pathlib import Path
import pandas as pd


def load_metrics(path):
    if not Path(path).exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def save_config(config):
    with open("config.yaml", "w") as f:
        yaml.safe_dump(config, f)


def run_training(script_name):
    print(f"\nRunning {script_name} via subprocess...")
    result = subprocess.run([".venv/bin/python", script_name], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {script_name}:\n{result.stderr}")
    else:
        print(f"Completed {script_name} successfully.")


def main():
    # Read current config to restore later
    with open("config.yaml", "r") as f:
        original_config = yaml.safe_load(f)

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Check/Train GAT Baseline
    gat_metrics_path = results_dir / "test_metrics.json"
    if not gat_metrics_path.exists():
        print("GAT Baseline metrics not found. Running training...")
        config = original_config.copy()
        config["model"]["type"] = "gat"
        save_config(config)
        run_training("scripts/03_train_gat.py")
    else:
        print("GAT Baseline metrics found. Skipping training.")

    # 2. Check/Train GT Sparse Attention
    gt_sparse_metrics_path = results_dir / "test_metrics_gt_sparse.json"
    if not gt_sparse_metrics_path.exists():
        print("GT Sparse metrics not found. Running training...")
        config = original_config.copy()
        config["model"]["type"] = "gt"
        config["model"]["attention_type"] = "sparse"
        save_config(config)
        run_training("scripts/03_train_gt.py")
    else:
        print("GT Sparse metrics found. Skipping training.")

    # 3. Check/Train GT Linear Attention
    gt_linear_metrics_path = results_dir / "test_metrics_gt_linear.json"
    if not gt_linear_metrics_path.exists():
        print("GT Linear metrics not found. Running training...")
        config = original_config.copy()
        config["model"]["type"] = "gt"
        config["model"]["attention_type"] = "linear"
        save_config(config)
        run_training("scripts/03_train_gt.py")
    else:
        print("GT Linear metrics found. Skipping training.")

    # Restore original config
    save_config(original_config)

    # Load all metrics
    gat_metrics = load_metrics(gat_metrics_path)
    gt_sparse_metrics = load_metrics(gt_sparse_metrics_path)
    gt_linear_metrics = load_metrics(gt_linear_metrics_path)

    # Build Comparison Table
    models = []
    if gat_metrics:
        models.append({"Model": "Vanilla GAT (Baseline)", **gat_metrics})
    if gt_sparse_metrics:
        models.append({"Model": "Graph Transformer (Sparse Attention)", **gt_sparse_metrics})
    if gt_linear_metrics:
        models.append({"Model": "Graph Transformer (Linear Attention)", **gt_linear_metrics})

    if not models:
        print("No metrics found to compare.")
        return

    df = pd.DataFrame(models)
    
    # Format for markdown output
    headers = ["Model", "Test AUROC", "Test AUPRC", "Test F1", "Test Precision", "Test Recall"]
    # Reorder keys
    df = df[["Model", "test_auroc", "test_auprc", "test_f1", "test_precision", "test_recall"]]
    df.columns = headers

    markdown_table = df.to_markdown(index=False)
    
    print("\n==========================================================================================")
    print("Model Performance Comparison Table")
    print("==========================================================================================")
    print(markdown_table)
    print("==========================================================================================\n")

    # Save to comparison file
    with open(results_dir / "model_comparison.md", "w") as f:
        f.write("# Model Performance Comparison\n\n")
        f.write("Comparative results between Vanilla GAT and Graph Transformer models on the heterogeneous Clinical KG:\n\n")
        f.write(markdown_table)
        f.write("\n")
    print(f"Saved comparison table to {results_dir / 'model_comparison.md'}")


if __name__ == "__main__":
    main()
