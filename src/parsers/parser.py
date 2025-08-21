import os
import pandas as pd
import yaml
from glob import glob


class BaseParser:
    def __init__(self, root_dir: str, schema_file: str, output_dir: str, node_store: None):
        self.root_dir = root_dir
        self.output_dir = output_dir
        self.node_store = node_store or {}  # used by EdgeParser validation
        with open(schema_file, "r") as f:
            self.schema = yaml.safe_load(f)
        os.makedirs(self.output_dir, exist_ok=True)

    def deserialise(self, parquet_file: str) -> pd.DataFrame:
        return pd.read_parquet(parquet_file)

    def serialise(self, df: pd.DataFrame, out_path: str):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_parquet(out_path, index=False)
        print(f"💾 Saved → {out_path} ({len(df)} rows)")

    def parse(self):
        """
        Parse all schema-defined sources into one parquet per source.
        - If schema entry is a dict → single spec
        - If schema entry is a list → multiple specs (relations) for same source
        """
        all_data = {}

        for name, spec in self.schema.items():
            print(f"📦 Parsing: {name}")
            subdir_path = os.path.join(self.root_dir, name)
            if not os.path.exists(subdir_path):
                print(f"⚠️ No directory for {name}, skipping")
                continue

            # normalise spec to list
            specs = spec if isinstance(spec, list) else [spec]

            dfs = []
            for pq in glob(os.path.join(subdir_path, "*.parquet")):
                try:
                    df = self.deserialise(pq)

                    for sub_spec in specs:
                        df_sub = self.apply_spec(df.copy(), sub_spec, name)
                        df_sub = self.validate(df_sub, sub_spec, name)

                        # inject relation_name if given
                        if "relation_name" in sub_spec:
                            df_sub["relation"] = sub_spec["relation_name"]

                        dfs.append(df_sub)

                except Exception as e:
                    pd.set_option("display.max_columns", None)
                    pd.set_option("display.max_rows", None)
                    print(df.head(1))
                    print(f"⚠️ Error reading {pq}: {e}")
                    break

            if dfs:
                df_all = pd.concat(dfs, ignore_index=True)

                out_name = self.output_name(name, spec)
                if out_name in all_data:
                    before = len(all_data[out_name])
                    all_data[out_name] = pd.concat([all_data[out_name], df_all], ignore_index=True)
                    after = len(all_data[out_name])
                    print(f"🔗 Merged {name} into {out_name}: {before} → {after} rows")
                else:
                    all_data[out_name] = df_all

        # 🔹 Serialise once per unique output name
        for out_name, df in all_data.items():
            out_path = os.path.join(self.output_dir, f"{out_name}.parquet")
            self.serialise(df, out_path)

        return all_data

    # Must be implemented by child
    def apply_spec(self, df, spec, name): 
        raise NotImplementedError

    # Must be implemented by child
    def output_name(self, name, spec):
        raise NotImplementedError
    
    # Default: no validation (override in EdgeParser)
    def validate(self, df, spec, name):
        return df



class NodeParser(BaseParser):
    def apply_spec(self, df, spec, name):
        cols = {"id": spec.get("id"), "name": spec.get("name")}
        if "props" in spec:
            for p in spec["props"]:
                if p in df.columns:
                    cols[p] = p
        cols = {k: v for k, v in cols.items() if v in df.columns}
        df = df[list(cols.values())].rename(columns={v: k for k, v in cols.items()})
        # Ensure unique nodes based on 'id'
        df = df.drop_duplicates(subset=["id"])

        if name == "targets" and "biotype" in df.columns:
            df = df[df["biotype"] == "protein_coding"]
        
        return df

    def output_name(self, name, spec):
        return name  # node type
    
    def parse(self):
        node_dfs = super().parse()
        # build node_store = {node_type: set(ids)}
        node_store = {k: set(df["id"].astype(str)) for k, df in node_dfs.items() if "id" in df.columns}
        print("🔗 Node store built:")
        for k, v in node_store.items():
            print(f"  {k}: {len(v)} ids")
        return node_dfs, node_store


class EdgeParser(BaseParser):
    def apply_spec(self, df, spec, name):
        src_col = spec["source"]
        tgt_col = spec["target"]

        # Case 1: Normal edge (direct source-target columns)
        if tgt_col in df.columns and not tgt_col.startswith("pathways"):
            cols = {"source": src_col, "target": tgt_col, "relation": spec["relation"]}
            if "props" in spec:
                for p in spec["props"]:
                    if p in df.columns:
                        cols[p] = p
            return df[list(cols.values())].rename(columns={v: k for k, v in cols.items()})

        # Case 2: Expand pathway list-of-dicts
        if tgt_col == "pathways.id" and "pathways" in df.columns:
            expanded_edges = []
            for _, row in df.iterrows():
                pathways = row.get("pathways", [])
                if isinstance(pathways, list):
                    for pw in pathways:
                        if isinstance(pw, dict) and "id" in pw:
                            edge = {
                                "source": row.get(src_col),
                                "target": pw.get("id"),
                                "relation": spec.get("relation_name", spec["relation"])
                            }
                            # copy props
                            for p in spec.get("props", []):
                                if p in row:
                                    edge[p] = row[p]
                            expanded_edges.append(edge)
            return pd.DataFrame(expanded_edges)

        raise ValueError(f"Unsupported target {tgt_col} for {name}")

    def output_name(self, name, spec):
        return name  # one parquet per source dir
    
    def validate(self, df, spec, name):
        """Ensure sources/targets exist in node_store."""
        if not self.node_store:
            return df  # nothing to validate against

        before = len(df)
        src_valid = df["source"].astype(str).isin(set.union(*self.node_store.values()))
        tgt_valid = df["target"].astype(str).isin(set.union(*self.node_store.values()))

        # Extract invalid rows before filtering
        invalid_row = df[~(src_valid & tgt_valid)]
        # not_found_src = invalid_row.loc[~invalid_row["source"].astype(str).isin(set.union(*self.node_store.values())), "source"].unique()
        # not_found_tgt = invalid_row.loc[~invalid_row["target"].astype(str).isin(set.union(*self.node_store.values())), "target"].unique()

        # Keep only valid rows
        df = df[src_valid & tgt_valid]
        # after = len(df)

        # if after < before:
        #     print(f"originally there are {before} edges")
        #     print(f"⚠️ {before-after} edges removed during validation for {name}")
        #     print(f"Total invalid edges: {len(invalid_row)}")
        #     print("First 5 not found targets:", not_found_tgt[:5])
        #     print("Total not found targets:", len(not_found_tgt))
        #     print(f"Edges discarded due to missing target: {invalid_row.loc[~invalid_row['target'].astype(str).isin(set.union(*self.node_store.values()))].shape[0]}")

        return df


