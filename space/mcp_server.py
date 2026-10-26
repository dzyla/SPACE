"""MCP server exposing SPACE to AI agents.

Run:  ``python -m space mcp``  (stdio)  or  ``python -m space.mcp_server``.

Typical agent flow::

    find_variants(query=..., source="NCBI", email=...)   -> run_id + top variants
    get_variants(run_id, min_frequency=0.01)             -> filtered variant table
    get_variant(run_id, "A263T")                         -> carriers, conservation, context
    annotate_run(run_id, uniprot_accession="P04637")     -> dbSNP / ClinVar / features
    map_run_to_structure(run_id, pdb_id="7KI6")          -> surface exposure per variant

Runs are stored under ``$SPACE_RUNS_DIR`` (default ``./runs``) and are also
readable as resources: ``space://runs/{run_id}/{table}``.
"""

from __future__ import annotations

import functools
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
import requests
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ResourceError, ToolError
from mcp.types import ToolAnnotations

from space.core import annotate
from space.core.pipeline import SpaceRun, default_runs_dir, list_runs, run_pipeline
from space.core.sequences import extract_uniprot_accession

INSTRUCTIONS = """\
SPACE finds protein sequence variants (substitutions, insertions, deletions) across
NCBI Protein or UniProtKB entries, relative to a reference sequence.
Start with find_variants (or analyze_fasta), then drill down with get_variants /
get_variant using the returned run_id. Frequencies are carriers / sequences covering
the position; 'unique' frequencies count identical sequences once (prefer them when
the database is dominated by one outbreak or study). NCBI requires an email.
NCBI queries use Entrez syntax, e.g. '"Influenza A virus"[Organism] AND hemagglutinin[Title]';
UniProt queries use UniProt syntax, e.g. 'gene:TP53 AND organism_id:9606'."""

mcp = MCPServer("space", instructions=INSTRUCTIONS)

_EXPECTED = (ValueError, KeyError, FileNotFoundError, requests.RequestException)


def tool(**kwargs):
    """Register an MCP tool whose expected failures reach the client as readable errors.

    Unexpected exceptions stay generic (MCP default); bad input, unknown runs and
    upstream HTTP errors are re-raised as ToolError so the agent can correct itself.
    The undecorated function is returned so it stays directly callable.
    """
    def deco(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kw):
            try:
                return fn(*args, **kw)
            except _EXPECTED as exc:
                raise ToolError(str(exc)) from exc
        mcp.tool(**kwargs)(wrapped)
        return fn
    return deco


def resource(uri: str, **kwargs):
    def deco(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kw):
            try:
                return fn(*args, **kw)
            except _EXPECTED as exc:
                raise ResourceError(str(exc)) from exc
        mcp.resource(uri, **kwargs)(wrapped)
        return fn
    return deco


READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
NETWORK = ToolAnnotations(readOnlyHint=False, openWorldHint=True)

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,200}$")
TABLES = {"variants", "variant_events", "sequences", "positions", "hits", "al2co_scores",
          "variants_annotated", "metadata"}
VARIANT_FIELDS = ["Mutation", "HGVS", "Type", "Position", "Ref", "Alt", "Count", "Covering", "Frequency",
                  "Unique carriers", "Unique covering", "Unique frequency", "BLOSUM62", "Change"]
KNOWN_FIELDS = ["Known", "Clinical significance", "dbSNP", "ClinVar", "SIFT", "PolyPhen", "Features"]
STRUCTURE_FIELDS = ["Structure residue", "RSA", "Exposure", "B-factor / pLDDT"]
ANNOTATION_FIELDS = KNOWN_FIELDS + STRUCTURE_FIELDS
_KNOWN_EXTRA = KNOWN_FIELDS + ["Consequence", "Sources"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _root() -> Path:
    return default_runs_dir()


def _run_dir(run_id: str) -> Path:
    if not _RUN_ID.match(run_id or ""):
        raise ValueError(f"Invalid run_id {run_id!r}.")
    d = _root() / run_id
    if not (d / "run.json").exists():
        raise ValueError(f"Unknown run_id {run_id!r}. Call list_runs to see available runs.")
    return d


def _load(run_id: str) -> SpaceRun:
    return SpaceRun.load(_run_dir(run_id))


def _records(df: pd.DataFrame, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    if limit:
        df = df.head(limit)
    return json.loads(df.to_json(orient="records", double_precision=4))


def _annotated(run: SpaceRun) -> pd.DataFrame:
    """Variant summary with any saved annotations (UniProt / structure) merged in."""
    p = run.run_dir / "tables" / "variants_annotated.tsv"
    if p.exists():
        return pd.read_csv(p, sep="\t", keep_default_na=False, na_values=[""])
    return run.variants.summary


def _variant_rows(df: pd.DataFrame, limit: int) -> List[Dict[str, Any]]:
    cols = [c for c in VARIANT_FIELDS + ANNOTATION_FIELDS if c in df]
    return _records(df[cols], limit)


def _run_overview(run: SpaceRun, top: int) -> Dict[str, Any]:
    s = _annotated(run)
    return {**run.summary(), "top_variants": _variant_rows(s, top),
            "next_steps": "get_variants / get_variant for details; annotate_run for dbSNP/ClinVar; "
                          "map_run_to_structure for surface exposure."}


def _save_annotated(run: SpaceRun, df: pd.DataFrame) -> None:
    df.to_csv(run.run_dir / "tables" / "variants_annotated.tsv", sep="\t", index=False)


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------

@tool(annotations=NETWORK)
def find_variants(
    query: str,
    source: Literal["NCBI", "UniProt"] = "NCBI",
    email: Optional[str] = None,
    max_seqs: int = 500,
    reference_id: Optional[str] = None,
    reference_sequence: Optional[str] = None,
    min_identity: float = 0.0,
    min_coverage: float = 30.0,
    deletion_threshold: float = 20.0,
    remove_pdb: bool = True,
    top: int = 25,
) -> Dict[str, Any]:
    """Search NCBI Protein or UniProtKB, align all hits to a reference and call variants.

    The reference defaults to a curated entry (Swiss-Prot / RefSeq) or the most common
    full-length sequence; pass reference_id (a record ID from the results) or
    reference_sequence to override. Use min_identity (e.g. 50) to drop unrelated hits.
    Returns run_id, run statistics and the most frequent variants.
    """
    if source == "NCBI" and not (email or os.environ.get("SPACE_EMAIL")):
        raise ValueError("NCBI requires an email address (argument 'email' or $SPACE_EMAIL).")
    run = run_pipeline(
        query=query, source=source, email=email or os.environ.get("SPACE_EMAIL"), max_seqs=max_seqs,
        reference=reference_sequence, reference_id=reference_id, min_identity=min_identity,
        min_coverage=min_coverage, deletion_threshold=deletion_threshold, remove_pdb=remove_pdb,
    )
    return _run_overview(run, top)


@tool(annotations=ToolAnnotations(readOnlyHint=False, openWorldHint=False))
def analyze_fasta(
    fasta: str,
    reference_sequence: Optional[str] = None,
    reference_id: Optional[str] = None,
    label: str = "fasta",
    min_identity: float = 0.0,
    min_coverage: float = 30.0,
    deletion_threshold: float = 20.0,
    top: int = 25,
) -> Dict[str, Any]:
    """Call variants for sequences supplied as FASTA text (no database search)."""
    from space.core.pipeline import new_run_dir

    run_dir = new_run_dir(label)
    path = run_dir / "input.fasta"
    path.write_text(fasta)
    run = run_pipeline(fasta_path=str(path), reference=reference_sequence, reference_id=reference_id,
                       min_identity=min_identity, min_coverage=min_coverage,
                       deletion_threshold=deletion_threshold, run_dir=run_dir)
    run.params["query"] = label
    run.save()
    return _run_overview(run, top)


@tool(name="list_runs", annotations=READ_ONLY)
def list_saved_runs(limit: int = 20) -> List[Dict[str, Any]]:
    """List saved runs (newest first) with their run_id, query and reference."""
    return list_runs(_root())[:limit]


@tool(annotations=READ_ONLY)
def get_variants(
    run_id: str,
    min_frequency: float = 0.0,
    types: Optional[List[Literal["Substitution", "Insertion", "Deletion"]]] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
    unique: bool = False,
    known_only: bool = False,
    sort_by: Literal["frequency", "position", "blosum62"] = "frequency",
    limit: int = 50,
) -> Dict[str, Any]:
    """Filtered variant table for a run.

    min_frequency is a fraction (0.01 = 1%). unique=True uses frequencies over unique
    sequences. known_only requires annotate_run first. Positions are reference numbering.
    """
    run = _load(run_id)
    s = _annotated(run)
    freq = "Unique frequency" if unique else "Frequency"
    m = s[freq] >= min_frequency
    if types:
        m &= s["Type"].isin(types)
    if start is not None:
        m &= s["Position"] >= start
    if end is not None:
        m &= s["Position"] <= end
    if known_only:
        if "Known" not in s:
            raise ValueError("Run annotate_run first to mark known variants.")
        m &= s["Known"].astype(str).str.lower().isin(["true", "1"])
    out = s[m]
    key = {"frequency": [freq], "position": ["Position"], "blosum62": ["BLOSUM62"]}[sort_by]
    out = out.sort_values(key, ascending=sort_by != "frequency")
    return {"run_id": run_id, "total_matching": int(len(out)), "returned": min(limit, len(out)),
            "variants": _variant_rows(out, limit)}


@tool(annotations=READ_ONLY)
def get_variant(run_id: str, mutation: str, max_carriers: int = 50) -> Dict[str, Any]:
    """Details for one variant (e.g. 'A263T', 'K5del', 'A4_Y5insWW'): carriers, other
    variants at the same position, conservation, annotations and sample metadata."""
    run = _load(run_id)
    s = _annotated(run)
    hit = s[s["Mutation"].str.upper() == mutation.upper()]
    if hit.empty:
        digits = re.search(r"\d+", mutation)
        near = s[s["Position"] == int(digits.group())]["Mutation"].head(10).tolist() if digits else []
        raise ValueError(f"{mutation!r} not found in run {run_id}. Variants at that position: {near}")
    row = _records(hit[[c for c in VARIANT_FIELDS + ANNOTATION_FIELDS if c in hit]])[0]
    pos = int(row["Position"])
    p = run.positions.set_index("Position").loc[pos]
    carriers = str(hit.iloc[0]["Sequence IDs"]).split(";")
    detail: Dict[str, Any] = {
        "variant": row,
        "position": {"residue": p["Residue"], "al2co_score": None if pd.isna(p["al2co_score"]) else float(p["al2co_score"]),
                     "modal_residue_frequency": None if pd.isna(p.get("Conservation")) else float(p["Conservation"]),
                     "covering_sequences": int(p["Covering"])},
        "other_variants_at_position": s[(s["Position"] == pos) & (s["Mutation"] != row["Mutation"])]["Mutation"].tolist(),
        "carriers": carriers[:max_carriers],
        "n_carriers": len(carriers),
    }
    meta = annotate.load_metadata(run.run_dir)
    if meta is not None:
        sub = meta[meta["Sequence ID"].isin(carriers)]
        detail["carrier_metadata"] = {
            "hosts": sub["Host"].value_counts().head(10).to_dict(),
            "countries": sub["Country"].value_counts().head(10).to_dict(),
            "years": sub["Year"].dropna().astype(int).value_counts().sort_index().to_dict(),
        }
    return detail


@tool(annotations=READ_ONLY)
def get_positions(run_id: str, start: int = 1, end: Optional[int] = None,
                  only_variable: bool = False) -> Dict[str, Any]:
    """Per-position table (reference residue, al2co conservation, entropy, gap fraction,
    covering sequences, number of variants, max variant frequency)."""
    run = _load(run_id)
    p = run.positions
    p = p[(p["Position"] >= start) & (p["Position"] <= (end or len(run.reference)))]
    if only_variable:
        p = p[p["Variants"] > 0]
    return {"run_id": run_id, "reference_length": len(run.reference), "positions": _records(p, 1000)}


@tool(annotations=NETWORK)
def annotate_run(run_id: str, uniprot_accession: Optional[str] = None, limit: int = 25) -> Dict[str, Any]:
    """Annotate a run with UniProt features and known variants from the EBI Proteins API
    (dbSNP, ClinVar, SIFT, PolyPhen). The accession defaults to the reference's own UniProt
    accession when the run used a UniProt reference."""
    run = _load(run_id)
    acc = uniprot_accession or (extract_uniprot_accession(run.reference_id) if run.reference_id else None)
    if not acc:
        raise ValueError("Provide uniprot_accession (the reference is not a UniProt record).")
    known = annotate.known_variants(acc, run.reference)
    feats = annotate.uniprot_features(acc, run.reference)
    base = _annotated(run)
    base = base.drop(columns=[c for c in _KNOWN_EXTRA if c in base])  # re-annotation replaces old columns
    s = annotate.add_feature_overlaps(annotate.merge_known_variants(base, known), feats)
    _save_annotated(run, s)
    k = s[s["Known"]]
    return {"run_id": run_id, "accession": acc, "uniprot_name": annotate.uniprot_name(acc),
            "features": int(len(feats)), "known_variants_in_database": int(len(known)),
            "observed_variants_already_known": int(len(k)),
            "known_observed": _variant_rows(k, limit),
            "features_list": _records(feats[feats["Type"] != "Chain"], 100)}


@tool(annotations=NETWORK)
def map_run_to_structure(run_id: str, uniprot_accession: Optional[str] = None, pdb_id: Optional[str] = None,
                         limit: int = 25) -> Dict[str, Any]:
    """Map variants onto a structure (an RCSB pdb_id or the AlphaFold model of a UniProt
    accession; default: best RCSB hit at ≥90% identity). Adds structure residue numbers,
    relative solvent accessibility (RSA) and B-factor/pLDDT to every variant."""
    from space.core.structure import download_alphafold, download_pdb, map_to_structure, pdb_metadata, search_pdb

    run = _load(run_id)
    sdir = run.run_dir / "structure"
    if pdb_id:
        path, meta, sid = download_pdb(pdb_id, sdir), pdb_metadata(pdb_id), pdb_id
    elif uniprot_accession:
        path, meta = download_alphafold(uniprot_accession, sdir)
        sid = uniprot_accession
    else:
        hits = search_pdb(run.reference)
        if not hits:
            raise ValueError("No RCSB entry at ≥90% identity; pass uniprot_accession for AlphaFold.")
        sid = hits[0]
        path, meta = download_pdb(sid, sdir), pdb_metadata(sid)
    al = run.positions.set_index("Position")["al2co_score"].dropna().to_dict()
    sm = map_to_structure(path, run.reference, values=al, out_dir=sdir / "mapped", structure_id=sid,
                          metadata=meta)
    chain = sm.chains[sm.best_chain]
    base = _annotated(run)
    base = base.drop(columns=[c for c in STRUCTURE_FIELDS if c in base])
    s = annotate.add_structure_context(base, chain)
    _save_annotated(run, s)
    mapped = s[s["Structure residue"] != ""]
    return {
        "run_id": run_id, "structure": meta, "best_chain": sm.best_chain,
        "chains": [{"chain": c, "coverage": round(m.coverage, 1), "identity": round(m.identity, 1),
                    "pdb_path": m.pdb_path} for c, m in sm.chains.items()],
        "variants_mapped": int(len(mapped)), "variants_unmapped": int(len(s) - len(mapped)),
        "surface_variants": int((mapped["Exposure"] == "surface").sum()),
        "variants": _variant_rows(mapped, limit),
    }


@tool(annotations=NETWORK)
def fetch_sample_metadata(run_id: str, email: Optional[str] = None) -> Dict[str, Any]:
    """Download host, country and collection date for the sequences of an NCBI run
    (enables per-year trends and carrier metadata in get_variant)."""
    run = _load(run_id)
    if run.params.get("source") != "NCBI":
        raise ValueError("Sample metadata is only available for NCBI runs.")
    mail = email or run.params.get("email") or os.environ.get("SPACE_EMAIL")
    if not mail:
        raise ValueError("NCBI requires an email address.")
    meta = annotate.ncbi_source_metadata(list(run.selected["id"]), mail)
    annotate.save_metadata(run.run_dir, meta)
    years = meta["Year"].dropna()
    return {"run_id": run_id, "sequences": int(len(meta)),
            "years": [int(years.min()), int(years.max())] if not years.empty else None,
            "hosts": meta["Host"].replace("", pd.NA).dropna().value_counts().head(10).to_dict(),
            "countries": meta["Country"].replace("", pd.NA).dropna().value_counts().head(10).to_dict()}


@tool(annotations=READ_ONLY)
def variant_trend(run_id: str, mutations: List[str]) -> Dict[str, Any]:
    """Per collection year carrier fraction for up to 10 mutations (needs fetch_sample_metadata)."""
    run = _load(run_id)
    meta = annotate.load_metadata(run.run_dir)
    if meta is None:
        raise ValueError("Call fetch_sample_metadata first.")
    included = run.variants.per_sequence.query("Status == 'included'")["Sequence ID"]
    by_year = annotate.variants_by_year(run.variants.events, meta, mutations[:10], included=included)
    return {"run_id": run_id, "trend": _records(by_year)}


# ---------------------------------------------------------------------------
# resources
# ---------------------------------------------------------------------------

@resource("space://runs/{run_id}/{table}", mime_type="text/plain",
          description="A run table as TSV (variants, variant_events, sequences, positions, hits, "
                      "al2co_scores, variants_annotated, metadata) or 'msa' for the alignment (FASTA)")
def run_table(run_id: str, table: str) -> str:
    # One template serves both: a second 'space://runs/{run_id}/msa' template is
    # shadowed by this one in URI matching.
    if table == "msa":
        return (_run_dir(run_id) / "msa" / "msa_out.fasta").read_text()
    if table not in TABLES:
        raise ValueError(f"Unknown table {table!r}; choose from {sorted(TABLES)}.")
    p = _run_dir(run_id) / "tables" / f"{table}.tsv"
    if not p.exists():
        raise ValueError(f"Table {table!r} does not exist for this run yet.")
    return p.read_text()



def main(transport: str = "stdio") -> None:
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
