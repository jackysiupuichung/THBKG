import numpy as np
import pandas as pd


def extract_edge_props(row, props, datasource):
    """
    Extract edge properties safely, including datasource-specific logic.
    Returns a dict of extracted properties.
    """
    extracted = {}

    # ----------------------------------------------------
    # Handle constant props (score=constant:1.0)
    # ----------------------------------------------------
    for p in props:
        if isinstance(p, str) and "=" in p and "constant:" in p:
            key, val = p.split("=", 1)
            val = val.split("constant:")[1]
            extracted[key] = float(val) if key == "score" else val
    # remove constants from list so we don't process twice
    props = [p for p in props if not (isinstance(p, str) and "=" in p and "constant:" in p)]

    # ----------------------------------------------------
    # Datasource-specific logic (CHEMBL)
    # ----------------------------------------------------
    if datasource == "chembl":
        for p in props:
            if p not in row:
                continue

            val = row[p]

            # Special list fields
            if p == "studyStopReasonCategories":
                if isinstance(val, (list, np.ndarray)):
                    extracted[p] = str(list(val))
                continue

            # studyStopReason
            if p == "studyStopReason":
                if pd.notnull(val):
                    extracted[p] = str(val)
                continue

            # clinicalStatus is scalar
            if p == "clinicalStatus":
                if pd.notnull(val):
                    extracted[p] = str(val)
                continue

            # generic safe fallback
            if pd.notnull(val):
                if isinstance(val, np.ndarray):
                    val = val.tolist()
                if isinstance(val, (list, dict)):
                    val = str(val)
                extracted[p] = val
        return extracted

    # ----------------------------------------------------
    # Generic extraction (ALL OTHER DATASOURCES)
    # ----------------------------------------------------
    for p in props:
        if p not in row:
            continue
        val = row[p]
        if pd.isna(val):
            continue

        if isinstance(val, np.ndarray):
            val = val.tolist()
        if isinstance(val, (list, dict)):
            val = str(val)

        extracted[p] = val

    return extracted
