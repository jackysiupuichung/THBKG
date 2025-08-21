#!/usr/bin/env python3
import argparse
from src.parsers.parser import NodeParser, EdgeParser
import pandas as pd

def run_pipeline(node_input, edge_input, node_schema, edge_schema, node_output, edge_output):
    print("🔹 Parsing nodes...")
    node_parser = NodeParser(node_input, node_schema, node_output, node_store=None)
    node_data, node_store = node_parser.parse()

    print("🔹 Parsing edges...")
    edge_parser = EdgeParser(edge_input, edge_schema, edge_output, node_store=node_store)
    edge_data = edge_parser.parse()

    print("✅ Pipeline finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge Graph Construction Pipeline")

    parser.add_argument("--node-input", required=True, help="Directory with node parquet files")
    parser.add_argument("--edge-input", required=True, help="Directory with edge parquet files")
    parser.add_argument("--node-schema", required=True, help="YAML schema for nodes")
    parser.add_argument("--edge-schema", required=True, help="YAML schema for edges")
    parser.add_argument("--node-output", required=True, help="Output directory for parsed node parquet files")
    parser.add_argument("--edge-output", required=True, help="Output directory for parsed edge parquet files")

    args = parser.parse_args()


    run_pipeline(
        node_input=args.node_input,
        edge_input=args.edge_input,
        node_schema=args.node_schema,
        edge_schema=args.edge_schema,
        node_output=args.node_output,
        edge_output=args.edge_output
    )
