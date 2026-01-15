import os
import re
import pandas as pd
import numpy as np
from src.parsers.parser import EdgeParser

class GOOntologyParser(EdgeParser):
    def __init__(self, root_dir, schema_file, output_dir, node_store=None, static=False):
        super().__init__(root_dir, schema_file, output_dir, node_store, static=static)

    def parse(self, file_path=None):
        """
        Parse GO Ontology OBO format to extract terms (nodes) and is_a relations (edges).
        """
        if not file_path:
            go_dir = os.path.join(self.root_dir, "go_ontology")
            files = [os.path.join(go_dir, f) for f in os.listdir(go_dir) if f.endswith(".txt") or f.endswith(".obo")]
            if not files:
                print(f"⚠️ No GO Ontology files found in {go_dir}")
                return {}
            file_path = files[0]

        print(f"📦 Parsing GO Ontology: {file_path}")
        
        nodes = []
        edges = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Split by [Term] but keep the content
        term_blocks = re.split(r'\n\[Term\]\n', content)
        
        # Pattern for PMID extraction from def line
        pmid_pattern = re.compile(r'PMID:(\d+)')
        
        for block in term_blocks:
            if not block.strip():
                continue
            
            # Extract Term properties
            term_id_match = re.search(r'^id:\s*(GO:\d+)', block, re.MULTILINE)
            name_match = re.search(r'^name:\s*(.+)', block, re.MULTILINE)
            namespace_match = re.search(r'^namespace:\s*(.+)', block, re.MULTILINE)
            is_obsolete = re.search(r'^is_obsolete:\s*true', block, re.MULTILINE)
            
            if not term_id_match or is_obsolete:
                continue
                
            term_id = term_id_match.group(1)
            name = name_match.group(1) if name_match else None
            namespace = namespace_match.group(1) if namespace_match else None
            
            nodes.append({
                "id": term_id,
                "name": name,
                "namespace": namespace
            })
            
            # Extract hierarchal edges (is_a)
            # Example: is_a: GO:0048308 ! organelle inheritance
            parents = re.findall(r'^is_a:\s*(GO:\d+)', block, re.MULTILINE)
            
            # Extract PMIDs from def line for the edges
            # Example: def: "..." [GOC:mcc, PMID:10873824]
            def_line = re.search(r'^def:\s*(.+)', block, re.MULTILINE)
            pmids = []
            if def_line:
                pmids = pmid_pattern.findall(def_line.group(1))
            
            for parent_id in parents:
                edge = {
                    "sourceId": term_id,
                    "targetId": parent_id,
                    "source_type": "go",
                    "target_type": "go",
                    "relation": "is_subtype_of",
                    "datasourceId": "gene_ontology",
                    "score": 1.0,
                }
                
                edges.append(self._add_props(edge, {}, []))

        df_nodes = pd.DataFrame(nodes)
        df_edges = pd.DataFrame(edges)
        
        # Filter edges using node_store if available
        df_edges = self.validate(df_edges, None, "go_ontology_hierarchy")
        
        # Save Nodes (Update node_output directory)
        # Note: Usually NodeParser handles this, but here we do it as part of the specialized OBO parser
        nodes_out = os.path.join(os.path.dirname(self.output_dir), "nodes", "go_ontology_terms.parquet")
        os.makedirs(os.path.dirname(nodes_out), exist_ok=True)
        df_nodes.to_parquet(nodes_out, index=False)
        print(f"💾 Saved GO Nodes → {nodes_out} ({len(df_nodes)} rows)")
        
        # Save Edges
        out_name = self.output_name("go_hierarchy", {
            "relation_name": "is_subtype_of",
            "props": [
                "datasourceId=constant:gene_ontology",
                "source_type=constant:go",
                "target_type=constant:go"
            ]
        }, df_edges)
        out_path = os.path.join(self.output_dir, f"{out_name}.parquet")
        self.serialise(df_edges, out_path)
        
        return {"go_ontology_nodes": df_nodes, "go_ontology_edges": df_edges}
