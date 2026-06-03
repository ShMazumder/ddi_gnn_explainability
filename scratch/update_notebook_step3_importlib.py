import json

notebook_path = "notebooks/00_end_to_end_pipeline.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find step 3 code cell (contains scripts.02_build_kg)
cell_idx = None
for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code" and any("scripts.02_build_kg" in line for line in cell["source"]):
        cell_idx = idx
        break

if cell_idx is not None:
    print(f"Found Step 3 cell at index {cell_idx}")
    new_source = [
        "import importlib\n",
        "import sys\n",
        "from pathlib import Path\n",
        "import torch\n",
        "\n",
        "if \"..\" not in sys.path:\n",
        "    sys.path.append(\"..\")\n",
        "build_kg = importlib.import_module(\"scripts.02_build_kg\")\n",
        "importlib.reload(build_kg)\n",
        "\n",
        "# Build the graph streamingly using the optimized script\n",
        "build_kg.build_graph()\n",
        "\n",
        "# Load the built graph into the notebook\n",
        "graph_path = Path(\"../data/graph/hetero_data.pt\")\n",
        "data = torch.load(graph_path, weights_only=False)\n",
        "\n",
        "print(\"\\n=== Heterogeneous Graph Built and Loaded ===\")\n",
        "print(data)"
    ]
    nb["cells"][cell_idx]["source"] = new_source
    nb["cells"][cell_idx]["outputs"] = []
    nb["cells"][cell_idx]["execution_count"] = None
    
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Notebook Step 3 updated successfully with importlib.reload!")
else:
    print("Failed to find Step 3 cell in the notebook.")
