
import torch
import sys
import os

# Create scripts/debug_data_stats.py
def main():
    try:
        path = 'output/graph/hetero_graph_with_features_sample.pt'
        if not os.path.exists(path):
            path = 'output/graph/hetero_graph_with_features.pt'
            
        print(f"Loading {path}...")
        data = torch.load(path)
        
        print("\n=== Node Feature Statistics (x) ===")
        for nt in data.node_types:
            if hasattr(data[nt], 'x') and data[nt].x is not None:
                x = data[nt].x.float()
                print(f"Node: {nt}")
                print(f"  Shape: {x.shape}")
                print(f"  Range: [{x.min():.4f}, {x.max():.4f}]")
                print(f"  Mean:  {x.mean():.4f}")
                print(f"  Std:   {x.std():.4f}")
                if torch.isnan(x).any():
                    print("  ⚠️ WARNING: Contains NaNs!")
                if torch.isinf(x).any():
                    print("  ⚠️ WARNING: Contains Infs!")
            else:
                print(f"Node: {nt} (No features)")

        print("\n=== Edge Attribute Statistics (edge_attr/Targets) ===")
        for et in data.edge_types:
            if hasattr(data[et], 'edge_attr') and data[et].edge_attr is not None:
                ea = data[et].edge_attr.float()
                print(f"Edge: {et}")
                print(f"  Shape: {ea.shape}")
                print(f"  Range: [{ea.min():.4f}, {ea.max():.4f}]")
                print(f"  Mean:  {ea.mean():.4f}")
                print(f"  Std:   {ea.std():.4f}")
                if torch.isnan(ea).any():
                    print("  ⚠️ WARNING: Contains NaNs!")
                if torch.isinf(ea).any():
                    print("  ⚠️ WARNING: Contains Infs!")
                    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
