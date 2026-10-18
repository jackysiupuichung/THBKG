"""Write extraction results as parquet, nested JSON and a markdown brief."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

OT = "https://platform.opentargets.org"
PUBMED = "https://pubmed.ncbi.nlm.nih.gov"
_NCT = re.compile(r"^nct\d{8}$", re.IGNORECASE)


def node_url(node_type: str, acc: str) -> str | None:
    return {
        "target": f"{OT}/target/{acc}",
        "disease": f"{OT}/disease/{acc}",
        "molecule": f"{OT}/drug/{acc}",
        "reactome": f"https://reactome.org/content/detail/{acc}",
        "go": f"https://amigo.geneontology.org/amigo/term/{acc}",
    }.get(node_type)


def edge_page(src_type, src, dst_type, dst) -> str | None:
    """Open Targets evidence page for a target-disease edge."""
    if (src_type, dst_type) == ("target", "disease"):
        return f"{OT}/evidence/{src}/{dst}"
    if (src_type, dst_type) == ("disease", "target"):
        return f"{OT}/evidence/{dst}/{src}"
    return None


def record_links(pmids, details: dict | None) -> list[str]:
    """Trial registry link first (when the record is a trial), then papers."""
    links = []
    nct = (details or {}).get("clinicalReportId")
    if isinstance(nct, str) and _NCT.match(nct):
        links.append(f"https://clinicaltrials.gov/study/{nct.upper()}")
    lit = list(pmids) + list((details or {}).get("literature") or [])
    for p in dict.fromkeys(str(p) for p in lit):
        if p.isdigit():
            links.append(f"{PUBMED}/{p}/")
    return links


def summarise(ev: pd.DataFrame) -> pd.DataFrame:
    """One row per (query edge, stored relation, datasource)."""
    keys = ["query_id", "hop", "edge_type", "datasourceId"]
    cols = keys + ["n_rows", "n_evidence_ids", "n_pmids", "year_min", "year_max",
                   "score_max", "score_mean"]
    if ev.empty:
        return pd.DataFrame(columns=cols)
    g = ev.groupby(keys, sort=True, dropna=False)
    out = g.agg(n_rows=("datasourceId", "size"),
                n_evidence_ids=("evidence_id", "nunique"),
                year_min=("year", "min"), year_max=("year", "max"),
                score_max=("score", "max"), score_mean=("score", "mean"))
    out["n_pmids"] = g["pmids"].apply(lambda s: len({p for ps in s for p in ps}))
    return out.reset_index()[cols]


def _year_span(lo, hi) -> str:
    if pd.isna(lo):
        return "undated"
    return f"{int(lo)}" if lo == hi else f"{int(lo)}-{int(hi)}"


def _records(sub: pd.DataFrame, max_records: int) -> list[dict]:
    sub = sub.sort_values(["score", "year"], ascending=[False, False],
                          na_position="last").head(max_records)
    recs = []
    for r in sub.itertuples(index=False):
        details = json.loads(r.details) if getattr(r, "details", None) else None
        rec = {
            "evidence_id": r.evidence_id,
            "stored_edge": [r.sourceId, r.targetId],
            "year": None if pd.isna(r.year) else int(r.year),
            "score": None if pd.isna(r.score) else float(r.score),
            "pmids": list(r.pmids),
            "links": record_links(r.pmids, details),
        }
        if "after_cutoff" in sub.columns:
            rec["after_cutoff"] = bool(r.after_cutoff)
        if details is not None:
            rec["details"] = details
        recs.append(rec)
    return recs


def build_json(queries, ev, summary, unresolved, names, settings,
               max_records: int) -> dict:
    def node(t, a):
        return {"type": t, "id": a, "name": names.get((t, a)), "url": node_url(t, a)}

    out = {"settings": settings, "queries": []}
    for qid, qrows in queries.groupby("query_id", sort=False):
        qrows = qrows.sort_values("hop")
        hops = []
        for qi, q in qrows.iterrows():
            ev_q = ev[ev["query_row"] == qi]
            rels = []
            for et, ev_r in ev_q.groupby("edge_type", sort=True):
                srcs = []
                sm = summary[(summary.query_id == qid) & (summary.hop == q.hop)
                             & (summary.edge_type == et)]
                for s in sm.itertuples(index=False):
                    sub = ev_r[ev_r["datasourceId"] == s.datasourceId]
                    srcs.append({
                        "datasourceId": s.datasourceId,
                        "n_rows": int(s.n_rows),
                        "n_evidence_ids": int(s.n_evidence_ids),
                        "n_pmids": int(s.n_pmids),
                        "years": _year_span(s.year_min, s.year_max),
                        "score_max": None if pd.isna(s.score_max) else float(s.score_max),
                        "records": _records(sub, max_records),
                    })
                rels.append({"edge_type": et, "n_rows": int(len(ev_r)),
                             "datasources": srcs})
            un = unresolved[unresolved["query_row"] == qi]
            hops.append({
                "hop": int(q.hop),
                "src": node(q.src_type, q.src_id),
                "dst": node(q.dst_type, q.dst_id),
                "requested_relation": q.relation,
                "input_edge_type": q.input_edge_type,
                "weight": None if q.weight is None or pd.isna(q.weight) else float(q.weight),
                "evidence_page": edge_page(q.src_type, q.src_id, q.dst_type, q.dst_id),
                "n_rows": int(len(ev_q)),
                "relations": rels,
                "note": un["reason"].iloc[0] if len(un) else (
                    None if len(ev_q) else "no evidence rows found"),
            })
        out["queries"].append({"query_id": qid, "hops": hops})
    return out


def _label(n: dict) -> str:
    return f"{n['name']} ({n['id']})" if n.get("name") else n["id"]


def build_markdown(doc: dict, top_records: int = 3) -> str:
    s = doc["settings"]
    when = f"evidence dated before {s['as_of']}" if s.get("as_of") else "all years"
    lines = [f"# Open Targets evidence, release {s['release']}", "",
             f"Scope: {when}; year policy `{s['year_policy']}`; "
             f"raw records `{s['raw']}`.", ""]
    for q in doc["queries"]:
        lines += [f"## {q['query_id']}", ""]
        if q["hops"]:
            chain = [_label(q["hops"][0]["src"])] + [
                f"[{h['requested_relation'] or '*'}] {_label(h['dst'])}" for h in q["hops"]]
            lines += ["Path: " + "  →  ".join(chain), ""]
        for h in q["hops"]:
            head = f"### Hop {h['hop'] + 1}: {_label(h['src'])} – {_label(h['dst'])}"
            if h.get("weight") is not None:
                head += f" (weight {h['weight']:.4g})"
            lines += [head, ""]
            if h.get("evidence_page"):
                lines += [f"Open Targets page: {h['evidence_page']}", ""]
            if not h["relations"]:
                lines += [f"_{h['note']}_", ""]
                continue
            lines += ["| relation | datasource | rows | evidence ids | PMIDs | years | max score |",
                      "|---|---|---|---|---|---|---|"]
            for r in h["relations"]:
                for d in r["datasources"]:
                    ms = "" if d["score_max"] is None else f"{d['score_max']:.3g}"
                    lines.append(f"| {r['edge_type']} | {d['datasourceId']} | {d['n_rows']} | "
                                 f"{d['n_evidence_ids']} | {d['n_pmids']} | {d['years']} | {ms} |")
            lines.append("")
            seen = set()
            for r in h["relations"]:
                for d in r["datasources"]:
                    shown = 0
                    for rec in d["records"]:
                        if shown == top_records:
                            break
                        bits = [d["datasourceId"], str(rec["year"] or "undated")]
                        if rec["score"] is not None:
                            bits.append(f"score {rec['score']:.3g}")
                        if rec.get("after_cutoff"):
                            bits.append("AFTER CUTOFF")
                        det = rec.get("details") or {}
                        for k in ("clinicalReportId", "clinicalStage", "trial_outcome_bucket",
                                  "drugId", "variantRsId", "studyId", "studyLocusId"):
                            if det.get(k):
                                bits.append(f"{k}={det[k]}")
                        line = f"- {' · '.join(bits)} {' '.join(rec['links'][:2])}".rstrip()
                        if line in seen:      # e.g. IntAct rows citing one paper
                            continue
                        seen.add(line)
                        shown += 1
                        lines.append(line)
            lines.append("")
    return "\n".join(lines)


def write_outputs(out_dir: str | Path, ev: pd.DataFrame, summary: pd.DataFrame,
                  doc: dict) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ev.drop(columns=["query_row"], errors="ignore").to_parquet(
        out / "evidence.parquet", index=False)
    summary.to_parquet(out / "summary.parquet", index=False)
    (out / "evidence.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False,
                                                  default=str))
    (out / "evidence.md").write_text(build_markdown(doc))
