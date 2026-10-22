import os
import pandas as pd
import numpy as np
import yaml
from glob import glob
import traceback
import requests
import time
import re
import urllib3
import xml.etree.ElementTree as ET
from .edge_extractor import extract_edge_props

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# PMID -> publication year (str), or None when PubMed has the PMID but no year.
# Request failures are never cached: a PMID is only stored once PubMed answered for it.
PUBMED_CACHE = {}
# Optional on-disk cache (parquet: pmid, year), loaded on first use and updated
# after every fetch, so a rebuild reuses earlier answers and records them.
PUBMED_CACHE_FILE = os.environ.get("PUBMED_YEAR_CACHE")
_PUBMED_CACHE_LOADED = False

EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
EFETCH_BATCH_SIZE = 200
PUBMED_RETRIES = 6
PUBMED_TIMEOUT = 120



REQUIRED_EDGE_COLS = ["sourceId", "targetId", "source_type", "target_type", "relation", "datasourceId", "score", "year"]
# Columns tried in order; the first non-null one dates the row. curationYear,
# studyYear and publicationYear are legacy (pre-26.03) columns. Europe PMC still
# carries publicationYear, the journal issue year; its publicationDate (first
# publication, online or print) goes first: it marks when the paper was out, and
# where the two are 2+ years apart PubMed mostly sides with the date.
YEAR_PRIORITY = ["resolvedTrialDate", "curationYear", "studyYear", "studyStartDate", "publicationDate",
                 "publicationYear", "evidenceDate"]
# Year of the Open Targets release (26.09). Later years are planned trial dates,
# placeholders (2099) or typos (2124), and are treated as missing.
MAX_EVIDENCE_YEAR = 2026


def _load_pubmed_cache():
    global _PUBMED_CACHE_LOADED
    if _PUBMED_CACHE_LOADED:
        return
    _PUBMED_CACHE_LOADED = True
    if PUBMED_CACHE_FILE and os.path.exists(PUBMED_CACHE_FILE):
        df = pd.read_parquet(PUBMED_CACHE_FILE)
        for pmid, year in zip(df["pmid"].astype(str), df["year"]):
            PUBMED_CACHE[pmid] = None if pd.isna(year) else str(year)
        print(f"📚 PubMed year cache: {len(df):,} PMIDs from {PUBMED_CACHE_FILE}")


def _save_pubmed_cache():
    if not PUBMED_CACHE_FILE:
        return
    os.makedirs(os.path.dirname(os.path.abspath(PUBMED_CACHE_FILE)), exist_ok=True)
    tmp = PUBMED_CACHE_FILE + ".tmp"
    pd.DataFrame({"pmid": list(PUBMED_CACHE.keys()), "year": list(PUBMED_CACHE.values())}).to_parquet(tmp, index=False)
    os.replace(tmp, PUBMED_CACHE_FILE)


def _parse_efetch_years(xml_bytes) -> dict:
    """{pmid: year or None} for every article in an efetch response (journal or book)."""
    root = ET.fromstring(xml_bytes)
    out = {}
    for article in list(root.iter("PubmedArticle")) + list(root.iter("PubmedBookArticle")):
        pmid = article.findtext(".//PMID")
        if not pmid:
            continue
        year = article.findtext(".//PubDate/Year")
        if year is None:
            medline = article.findtext(".//PubDate/MedlineDate")
            match = re.search(r"(\d{4})", medline or "")
            year = match.group(1) if match else None
        out[pmid.strip()] = year
    return out


def _efetch_batch(batch) -> dict:
    """Years for one batch of PMIDs, retried with backoff; raises if PubMed keeps failing."""
    last_err = None
    for attempt in range(PUBMED_RETRIES):
        try:
            res = requests.post(EFETCH_URL, data={"db": "pubmed", "id": ",".join(batch), "retmode": "xml"},
                                timeout=PUBMED_TIMEOUT, verify=False)
            res.raise_for_status()
            return _parse_efetch_years(res.content)
        except Exception as e:  # HTTP errors, timeouts, truncated or malformed XML
            last_err = e
            wait = min(60, 2 ** attempt)
            print(f"   ⚠️ PubMed efetch failed ({type(e).__name__}: {e}); retry {attempt + 1}/{PUBMED_RETRIES} in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"PubMed efetch failed {PUBMED_RETRIES} times for a batch of {len(batch)} PMIDs: {last_err}")


def fetch_pubmed_years(pubmed_ids):
    """
    Fetch publication years for PubMed IDs using NCBI E-utils (efetch).

    Results go into PUBMED_CACHE (and PUBMED_YEAR_CACHE on disk, if set). A PMID
    PubMed answered for without a year, or that PubMed does not return at all
    (withdrawn / invalid), is cached as None. A request that keeps failing raises
    instead of being cached: the 26.09 build cached one failed 10,000-PMID
    request as "no year", which silently dropped 105,422 IntAct and 48,582 GO
    rows.
    """
    _load_pubmed_cache()
    ids_to_fetch = sorted({str(pid) for pid in pubmed_ids} - PUBMED_CACHE.keys())
    if not ids_to_fetch:
        return PUBMED_CACHE
    print(f"📡 Fetching publication years for {len(ids_to_fetch):,} PMIDs from PubMed...")
    for i in range(0, len(ids_to_fetch), EFETCH_BATCH_SIZE):
        batch = ids_to_fetch[i : i + EFETCH_BATCH_SIZE]
        years = _efetch_batch(batch)
        for pid in batch:
            PUBMED_CACHE[pid] = years.get(pid)  # answered without this PMID -> None
        if (i // EFETCH_BATCH_SIZE) % 50 == 49:
            _save_pubmed_cache()
        time.sleep(0.4)  # NCBI: at most 3 requests/s without an API key
    _save_pubmed_cache()
    n_none = sum(PUBMED_CACHE[p] is None for p in ids_to_fetch)
    print(f"   ✅ {len(ids_to_fetch):,} PMIDs resolved ({n_none:,} without a year in PubMed)")
    return PUBMED_CACHE


class BaseParser:
    def __init__(self, root_dir: str, schema_file: str, output_dir: str, node_store: None, static: bool = False, debug: bool = False):
        self.root_dir = root_dir
        self.output_dir = output_dir
        self.node_store = node_store or {}  # used by EdgeParser validation
        self.static = static  # used by EdgeParser for year handling
        self.debug = debug  # if True, only read one file per datasource subdir
        with open(schema_file, "r") as f:
            self.schema = yaml.safe_load(f)
        os.makedirs(self.output_dir, exist_ok=True)

    def deserialise(self, parquet_file: str) -> pd.DataFrame:
        return pd.read_parquet(parquet_file)

    def serialise(self, df: pd.DataFrame, out_path: str) -> int:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_parquet(out_path, index=False)
        print(f"💾 Saved → {out_path} ({len(df)} rows)")
        return len(df)

    def parse(self):
        """
        Parse all schema-defined sources.
        """
        all_data = {}
        # tracks raw row counts before serialise filtering, keyed by datasource name
        raw_counts = {}
        # out_name -> columns to deduplicate on (spec key ``dedupe_on``), applied
        # in serialise after the year is known, keeping the earliest row
        dedupe = {}

        for name, spec in self.schema.items():
            print(f"📦 Processing schema entry: {name}")

            # normalise spec to list
            specs = spec if isinstance(spec, list) else [spec]

            # Use explicit input_dir from the first spec if available, otherwise fallback to name
            input_dir = specs[0].get("input_dir", name) if isinstance(specs[0], dict) else name
            subdir_path = os.path.join(self.root_dir, input_dir)

            if not os.path.exists(subdir_path):
                print(f"⚠️ No directory for {subdir_path}, skipping")
                continue

            parquet_files = glob(os.path.join(subdir_path, "*.parquet"))
            if self.debug and parquet_files:
                parquet_files = parquet_files[:1]
                print(f"🐛 [DEBUG] Limiting to 1 file: {parquet_files[0]}")
            for pq in parquet_files:
                try:
                    df = self.deserialise(pq)
                    for sub_spec in specs:
                        try:
                            df_sub = self.apply_spec(df.copy(), sub_spec, name)
                            df_sub = self.validate(df_sub, sub_spec, name)

                            if df_sub.empty:
                                continue

                            # If the spec used a column-driven `relation` (one row -> one
                            # of several relation values), split outputs by relation so each
                            # bucket gets its own file.
                            if "relation" in sub_spec and "relation" in df_sub.columns:
                                for relation_val, group_df in df_sub.groupby("relation"):
                                    temp_spec = sub_spec.copy()
                                    temp_spec["relation_name"] = relation_val
                                    out_name = self.output_name(name, temp_spec, group_df)
                                    if sub_spec.get("dedupe_on"):
                                        dedupe[out_name] = list(sub_spec["dedupe_on"])

                                    if out_name in all_data:
                                        all_data[out_name] = pd.concat([all_data[out_name], group_df], ignore_index=True)
                                    else:
                                        all_data[out_name] = group_df
                                    raw_counts[out_name] = raw_counts.get(out_name, 0) + len(group_df)
                            else:
                                # Normal flow for non-ChEMBL edges
                                out_name = self.output_name(name, sub_spec, df_sub)
                                if isinstance(sub_spec, dict) and sub_spec.get("dedupe_on"):
                                    dedupe[out_name] = list(sub_spec["dedupe_on"])

                                if out_name in all_data:
                                    all_data[out_name] = pd.concat([all_data[out_name], df_sub], ignore_index=True)
                                else:
                                    all_data[out_name] = df_sub
                                raw_counts[out_name] = raw_counts.get(out_name, 0) + len(df_sub)

                        except Exception as e:
                            # fatal: a swallowed error silently removes a relation
                            raise RuntimeError(f"error applying spec {sub_spec} to {pq}") from e
                except RuntimeError:
                    raise
                except Exception as e:
                    raise RuntimeError(f"error reading {pq}") from e

        # 🔹 Serialise once per unique output name, collect retained counts
        retained_counts = {}
        for out_name, df in all_data.items():
            out_path = os.path.join(self.output_dir, f"{out_name}.parquet")
            self._dedupe_on = dedupe.get(out_name)
            retained_counts[out_name] = self.serialise(df, out_path)
        self._dedupe_on = None

        # 🔹 Retention summary
        print("\n📊 Retention summary:")
        print(f"  {'datasource':<55} {'raw':>7}  {'retained':>8}  {'%':>6}")
        print(f"  {'-'*55}  {'-'*7}  {'-'*8}  {'-'*6}")
        for out_name in sorted(raw_counts):
            raw = raw_counts[out_name]
            kept = retained_counts.get(out_name) or 0
            pct = (kept / raw * 100) if raw else 0.0
            print(f"  {out_name:<55} {raw:>7,}  {kept:>8,}  {pct:>5.1f}%")

        return all_data

    # Must be implemented by child
    def apply_spec(self, df, spec, name): 
        raise NotImplementedError

    # Must be implemented by child
    def output_name(self, name, spec, df=None):
        raise NotImplementedError
    
    # Default: no validation (override in EdgeParser)
    def validate(self, df, spec, name):
        return df
    
    @staticmethod
    def normalise(val):
        """
        Ensure a value is always a list:
        - None/NaN -> []
        - list/tuple -> unchanged
        - numpy.ndarray -> converted to list
        - scalar -> [val]
        """
        if val is None:
            return []
        if isinstance(val, (list, tuple)):
            return list(val)
        if isinstance(val, np.ndarray):
            return val.tolist()
        # only call pd.isna on scalars
        try:
            if pd.isna(val):
                return []
        except Exception:
            pass
        return [val]



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

        # Molecules of every drugType are kept. Up to v26.09 only small molecules
        # with a SMILES string became nodes, which silently dropped ~1,150
        # trialled drugs (mostly antibodies and proteins) and every edge touching
        # them. Biologics are now featurised from their ChEMBL sequence
        # (preprocessing/node_features/molecule_multimodal.py), so the SMILES
        # requirement no longer applies.

        return df

    def output_name(self, name, spec, df=None):
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
    def _extract_literature(self, obj):
        """
        Extract PMID list from a nested object.
        Checks keys: literature, source, pubmedId.
        """
        lit_val = obj.get("literature") or obj.get("source") or obj.get("pubmedId")
        if not lit_val:
            return None
        
        def _clean(v):
            v_str = str(v).strip()
            if v_str.isdigit():
                return v_str
            # Handles PMID:123, pubmed: 123, PMID 123, etc.
            m = re.search(r'(?:PMID|pubmed)[\s:]*(\d+)', v_str, re.IGNORECASE)
            return m.group(1) if m else None

        if isinstance(lit_val, (list, np.ndarray, tuple)):
            pmids = [_clean(l) for l in lit_val if _clean(l)]
            return pmids if pmids else None
        
        pmid = _clean(lit_val)
        return [pmid] if pmid else None

    def _get_prop_value(self, spec, prop_name, df=None):
        """
        Extract a property value from spec (constant) or dataframe (first row).
        """
        # 1. Check if it's a constant in props
        props = spec.get("props", [])
        for p in props:
            if isinstance(p, str) and "=" in p:
                key, val = p.split("=", 1)
                if key == prop_name and "constant:" in val:
                    return val.split("constant:")[1]

        # 2. Check for relation_name (specific to EdgeParser)
        # We also check 'relation' if it's a constant (not likely but safe)
        if prop_name == "relation_name" and "relation_name" in spec:
            return spec["relation_name"]

        # 3. Fallback: inspect dataframe
        if df is not None and not df.empty and prop_name in df.columns:
            val = df[prop_name].iloc[0]
            if isinstance(val, (list, dict, np.ndarray)):
                return "complex"
            return str(val) if pd.notnull(val) else "unknown"

        return "unknown"

    @staticmethod
    def _expand_targets(raw_val):
        """Ensure targets are iterable list"""
        if isinstance(raw_val, np.ndarray):
            raw_val = raw_val.tolist()
        if not isinstance(raw_val, list):
            raw_val = [raw_val]
        return raw_val

    def _add_props(self, edge, row, props):
        datasource = row.get("datasourceId")

        extracted = extract_edge_props(row, props, datasource)
        edge.update(extracted)
                
        if self.static:
            # Static edges → no timestamps at all
            edge["year"] = np.nan
            return edge

        # Handle year
        if "year" in props:
            # Dynamic edges → pick best year candidate
            DATE_COLS = {"resolvedTrialDate", "studyStartDate", "publicationDate", "evidenceDate"}
            for col in YEAR_PRIORITY:
                if col in row and pd.notnull(row[col]):
                    if col in DATE_COLS:
                        try:
                            yr = pd.to_datetime(row[col], errors="coerce").year
                            if pd.notnull(yr):
                                edge["year"] = yr
                                return edge
                        except Exception:
                            continue
                    else:
                        edge["year"] = row[col]
                        return edge
            edge["year"] = np.nan  # fallback if no usable column

        return edge

    def apply_spec(self, df, spec, name):
        src_col = spec["sourceId"]
        tgt_col = spec["targetId"]

        # Decide relation value
        if "relation" in spec:
            relation_value = spec["relation"]
            relation_is_column = True
        elif "relation_name" in spec:
            relation_value = spec["relation_name"]
            relation_is_column = False
        else:
            raise ValueError(f"No relation or relation_name in spec for {name}")

        props = spec.get("props", [])

        # === Case 1: Direct column-to-column edges ===
        if tgt_col in df.columns and "." not in tgt_col:
            sample_val = df[tgt_col].iloc[0]
            if not isinstance(sample_val, (list, tuple, np.ndarray)):
                expanded_edges = []
                for _, row in df.iterrows():
                    edge = {
                        "sourceId": row[src_col],
                        "targetId": row[tgt_col],
                        "relation": row[relation_value] if relation_is_column else relation_value,
                    }
                    expanded_edges.append(self._add_props(edge, row, props))

                out = pd.DataFrame(expanded_edges)
                return out

        # === Case 2: Nested dict expansion (e.g. pathways.id) ===
        if "." in tgt_col:
            base_col, subfield = tgt_col.split(".", 1)
            if base_col in df.columns:
                expanded_edges = []
                for _, row in df.iterrows():
                    tgts = self._expand_targets(row.get(base_col, []))
                    for t in tgts:
                        if isinstance(t, dict) and subfield in t:
                            edge = {
                                "sourceId": row[src_col],
                                "targetId": t[subfield],
                                "relation": row[relation_value] if relation_is_column else relation_value,
                            }

                            edge = self._add_props(edge, row, props)
                            
                            lit = self._extract_literature(t)
                            if lit:
                                edge["literature"] = lit

                            expanded_edges.append(edge)

                out = pd.DataFrame(expanded_edges)
                return out

        # === Case 3: List-like targetId expansion ===
        if tgt_col in df.columns:
            expanded_edges = []
            for _, row in df.iterrows():
                tgts = self._expand_targets(row.get(tgt_col, []))
                for t in tgts:
                    if not t:
                        continue

                    # --- Case 3a: plain values (string, int, etc.)
                    if not isinstance(t, dict):
                        edge = {
                            "sourceId": row[src_col],
                            "targetId": t,
                            "relation": row[relation_value] if relation_is_column else relation_value,
                        }
                        expanded_edges.append(self._add_props(edge, row, props))

                    # --- Case 3b: dict with "id" (GO-style annotations)
                    elif "id" in t:
                        edge = {
                            "sourceId": row[src_col],
                            "targetId": t["id"],
                            "relation": row[relation_value] if relation_is_column else relation_value,
                        }
                        
                        lit = self._extract_literature(t)
                        if lit:
                            edge["literature"] = lit

                        expanded_edges.append(self._add_props(edge, row, props))
            if expanded_edges:
                out = pd.DataFrame(expanded_edges)
                return out

        raise ValueError(f"Unsupported targetId {tgt_col} for {name}")
    

    def output_name(self, name, spec, df=None):
        """
        Construct filename: {source_type}_{relation_name}_{datasourceId}_{target_type}
        """
        source_type = self._get_prop_value(spec, "source_type", df)
        
        relation_name = self._get_prop_value(spec, "relation_name", df)
        if relation_name == "unknown" and "relation" in spec:
            # If relation_name is unknown, check if 'relation' spec key gives us a column or value
            relation_name = self._get_prop_value(spec, "relation", df)
            
        datasourceId = self._get_prop_value(spec, "datasourceId", df)
        target_type = self._get_prop_value(spec, "target_type", df)

        base = f"{source_type}_{relation_name}_{target_type}_{datasourceId}"
        # Cleanup for filename-safe (lowercase and remove non-alphanumeric except underscore)
        clean_base = re.sub(r'[^a-zA-Z0-9_]', '_', base).lower()
        # Collapse multiple underscores
        clean_base = re.sub(r'_+', '_', clean_base).strip('_')
        return clean_base
    
    def validate(self, df, spec, name):
        """Ensure sources/targets exist in node_store."""
        if not self.node_store:
            return df  # nothing to validate against

        src_valid = df["sourceId"].astype(str).isin(set.union(*self.node_store.values()))
        tgt_valid = df["targetId"].astype(str).isin(set.union(*self.node_store.values()))

        df = df[src_valid & tgt_valid]
        return df


    def serialise(self, df, out_path):
        """
        Save edges to parquet while respecting static/dynamic schemas.
        Static edges do NOT require a year.
        Dynamic edges DO require a year.
        """

        # -----------------------------------
        # Determine required columns dynamically
        # -----------------------------------
        if self.static:
            required_cols = [c for c in REQUIRED_EDGE_COLS if c != "year"]
            print("🔹 Static edges: 'year' column not required.")
        else:
            required_cols = REQUIRED_EDGE_COLS

        # -----------------------------------
        # Build final column order
        # -----------------------------------
        if "literature" in df.columns:
            df["literature"] = df["literature"].apply(self.normalise)

        # -----------------------------------
        # Generalised PMID -> Year conversion
        # -----------------------------------
        if "literature" in df.columns:
            if "year" not in df.columns:
                df["year"] = np.nan
            
            mask = (df["year"].isna() | (df["year"] == 0)) & df["literature"].notna()
            if mask.any():
                all_pmids = set()
                for lits in df.loc[mask, "literature"]:
                    all_pmids.update(str(l) for l in lits if str(l).isdigit())
                
                if all_pmids:
                    fetch_pubmed_years(list(all_pmids))
                    
                    def get_min_year(lits):
                        years = [PUBMED_CACHE.get(str(l)) for l in lits if str(l) in PUBMED_CACHE]
                        int_years = [int(y) for y in years if y and str(y).isdigit()]
                        return min(int_years) if int_years else np.nan
                    
                    df.loc[mask, "year"] = df.loc[mask, "literature"].apply(get_min_year)

        # -----------------------------------
        # Check required columns exist
        # -----------------------------------
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            print(f"⚠️ Skipping save for {out_path}, missing columns: {missing_cols}")
            return 0

        if not self.static:
            future = pd.to_numeric(df["year"], errors="coerce") > MAX_EVIDENCE_YEAR
            if future.any():
                df.loc[future, "year"] = np.nan
                print(f"   {os.path.basename(out_path)}: {int(future.sum()):,} rows dated after "
                      f"{MAX_EVIDENCE_YEAR} treated as undated")

        # -----------------------------------
        # Drop rows missing required fields
        # -----------------------------------
        df = df.dropna(subset=required_cols)

        # Spec-level dedupe (``dedupe_on``), now that every row has its year:
        # keep the earliest row per key
        dedupe_on = getattr(self, "_dedupe_on", None)
        if dedupe_on:
            missing = [c for c in dedupe_on if c not in df.columns]
            if missing:
                raise ValueError(f"dedupe_on columns {missing} not in {out_path} (add them to the spec props)")
            n0 = len(df)
            df = df.sort_values("year", kind="stable").drop_duplicates(dedupe_on, keep="first")
            print(f"🔁 {os.path.basename(out_path)}: deduplicated on {dedupe_on}, {n0:,} → {len(df):,} rows "
                  f"(earliest year kept)")
        
        # # -----------------------------------
        # # For static edges → completely remove year column
        # # -----------------------------------
        # if self.static and "year" in df.columns:
        #     df = df.drop(columns=["year"])

        # -----------------------------------
        # Normalise list-like props
        # -----------------------------------
        if "literature" in df.columns:
            df["literature"] = df["literature"].apply(self.normalise)

        # -----------------------------------
        # Build final column order
        # -----------------------------------
        col_order = required_cols + \
                    [c for c in df.columns if c not in required_cols]

        df = df[col_order]

        # -----------------------------------
        # Save parquet
        # -----------------------------------
        if len(df):
            bad_mask = ~df["targetId"].apply(lambda x: isinstance(x, (str, int, float)))
            bad_rows = df[bad_mask]

            if len(bad_rows):
                print("❌ ERROR: Non-scalar targetId detected!")
                print(bad_rows.head(20))
                print("Full types:", bad_rows["targetId"].apply(type).value_counts())
                raise ValueError("Non-scalar targetId encountered. See debug output above.")
            print(df.head())
            df.to_parquet(out_path, index=False)
            print(f"💾 Saved → {out_path} ({len(df)} rows)")
            return len(df)
        else:
            print(f"⚠️ No valid edges left for {out_path}, skipping save.")
            return 0
