import json

notebook_path = "notebooks/00_end_to_end_pipeline.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cell_idx = None
for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code" and any("out_dim=num_side_effects" in line for line in cell["source"]) and any("HeteroGATDDI" in line for line in cell["source"]):
        cell_idx = idx
        break

if cell_idx is not None:
    print(f"Found Step 4 cell at index {cell_idx}")
    
    # We will modify the source to define num_side_effects
    source = cell["source"]
    # Check if already defined
    if not any("num_side_effects = " in line for line in source):
        insert_idx = None
        for l_idx, line in enumerate(source):
            if "num_proteins = data['protein'].num_nodes" in line:
                insert_idx = l_idx + 1
                break
                
        if insert_idx is not None:
            source.insert(insert_idx, "num_side_effects = data['side_effect'].num_nodes\n")
            nb["cells"][cell_idx]["source"] = source
            
            with open(notebook_path, "w", encoding="utf-8") as f:
                json.dump(nb, f, indent=1)
            print("Notebook Step 4 updated successfully!")
        else:
            print("Failed to find insertion point in Step 4 cell source.")
    else:
        print("num_side_effects is already defined in Step 4 cell.")
else:
    print("Failed to find Step 4 cell in the notebook.")
