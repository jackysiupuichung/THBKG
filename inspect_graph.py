import torch
from torch_geometric.data import HeteroData

# Load the graph
graph_path = "/Users/pui.chungsiu/Documents/opentarget_het_graph/output/graph/hetero_graph_with_features.pt"
print(f"Loading graph from: {graph_path}")
data = torch.load(graph_path)

print("\n" + "="*80)
print("GRAPH STRUCTURE")
print("="*80)

# Print node types
print("\nNode types:")
for node_type in data.node_types:
    print(f"  - {node_type}: {data[node_type].num_nodes} nodes")

# Print edge types
print("\nEdge types:")
for edge_type in data.edge_types:
    src, rel, dst = edge_type
    num_edges = data[edge_type].edge_index.shape[1] if hasattr(data[edge_type], 'edge_index') else 0
    print(f"  - {src} --[{rel}]--> {dst}: {num_edges} edges")

print("\n" + "="*80)
print("CHECKING FOR CLINICAL TRIAL DATA")
print("="*80)

# Check for clinical trial related edge types
clinical_trial_edges = []
for edge_type in data.edge_types:
    src, rel, dst = edge_type
    if 'trial' in rel.lower() or 'clinical' in rel.lower():
        clinical_trial_edges.append(edge_type)
        print(f"\n✓ Found clinical trial edge: {src} --[{rel}]--> {dst}")
        
        # Check edge attributes
        edge_data = data[edge_type]
        print(f"  Attributes:")
        for key in edge_data.keys():
            if key != 'edge_index':
                val = edge_data[key]
                if torch.is_tensor(val):
                    print(f"    - {key}: shape {val.shape}, dtype {val.dtype}")
                else:
                    print(f"    - {key}: {type(val)}")

if not clinical_trial_edges:
    print("\n✗ No clinical trial edges found")

# Check all edge attributes for phase/outcome related data
print("\n" + "="*80)
print("SEARCHING ALL EDGE ATTRIBUTES FOR PHASE/OUTCOME DATA")
print("="*80)

for edge_type in data.edge_types:
    src, rel, dst = edge_type
    edge_data = data[edge_type]
    
    # Check for phase or outcome related attributes
    has_phase_outcome = False
    for key in edge_data.keys():
        if key == 'edge_index':
            continue
        if any(keyword in key.lower() for keyword in ['phase', 'outcome', 'trial', 'clinical']):
            if not has_phase_outcome:
                print(f"\n✓ Found in {src} --[{rel}]--> {dst}:")
                has_phase_outcome = True
            
            val = edge_data[key]
            if torch.is_tensor(val):
                print(f"  - {key}: shape {val.shape}, dtype {val.dtype}")
                # Sample values
                if val.numel() > 0:
                    print(f"    Sample values: {val[:5] if len(val.shape) == 1 else val[:5, :]}")
            else:
                print(f"  - {key}: {type(val)}, value: {val}")

# Check node attributes for clinical trial data
print("\n" + "="*80)
print("CHECKING NODE ATTRIBUTES")
print("="*80)

for node_type in data.node_types:
    node_data = data[node_type]
    has_trial_data = False
    
    for key in node_data.keys():
        if any(keyword in key.lower() for keyword in ['phase', 'outcome', 'trial', 'clinical']):
            if not has_trial_data:
                print(f"\n✓ Found in {node_type} nodes:")
                has_trial_data = True
            
            val = node_data[key]
            if torch.is_tensor(val):
                print(f"  - {key}: shape {val.shape}, dtype {val.dtype}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Total node types: {len(data.node_types)}")
print(f"Total edge types: {len(data.edge_types)}")
print(f"Clinical trial related edges: {len(clinical_trial_edges)}")
