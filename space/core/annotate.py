"""External annotation of variants and positions.

* UniProt features (domains, sites, PTMs, ...) projected onto reference numbering
* Known variants from the EBI Proteins API (dbSNP, ClinVar, gnomAD/1000G,
  COSMIC...), with SIFT/PolyPhen predictions
* Per-sequence source metadata (host, country, collection date) from GenBank
* Structural context (pLDDT/B-factor, relative solvent accessibility)

All network calls are cached per process.
"""

from __future__ import annotations

import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import requests

from . import ProgressFn, log, report
from .align import position_map

UNIPROT_ENTRY = "https://rest.uniprot.org/uniprotkb/{acc}.json"
EBI_VARIATION = "https://www.ebi.ac.uk/proteins/api/variation/{acc}"

FEATURE_TYPES = (
    "Domain", "Region", "Motif", "Repeat", "Topological domain", "Transmembrane",
    "Signal", "Propeptide", "Peptide", "Chain", "Active site", "Binding site", "Site",
    "Disulfide bond", "Glycosylation", "Modified residue", "Lipidation", "Zinc finger",
    "Coiled coil", "Compositional bias",
)


@lru_cache(maxsize=128)
def uniprot_entry(accession: str, timeout: int = 30) -> Dict:
    r = requests.get(UNIPROT_ENTRY.format(acc=accession.split(".")[0]), timeout=timeout)
    r.raise_for_status()
    return r.json()


def uniprot_name(accession: str) -> str:
    """Recommended (or submitted) protein name; '' if unavailable."""
    try:
        desc = uniprot_entry(accession).get("proteinDescription", {})
    except Exception:  # noqa: BLE001
        return ""
    for key in ("recommendedName",):
        if desc.get(key):
            return desc[key].get("fullName", {}).get("value", "")
    subs = desc.get("submissionNames") or []
    return subs[0].get("fullName", {}).get("value", "") if subs else ""


def _to_reference(accession_seq: str, reference: str) -> Dict[int, int]:
    """UniProt sequence position -> reference position (identity if equal)."""
    if accession_seq == reference:
        return {i: i for i in range(1, len(reference) + 1)}
    return position_map(accession_seq, reference)


def _project_features(features: List[Dict], useq: str, reference: str,
                      types: Iterable[str] = FEATURE_TYPES) -> pd.DataFrame:
    """Project UniProt features into reference numbering, clipped to the mapped part."""
    pmap = _to_reference(useq, reference)
    wanted = set(types)
    rows = []
    for ft in features:
        if ft.get("type") not in wanted:
            continue
        loc = ft.get("location", {})
        s, e = loc.get("start", {}).get("value"), loc.get("end", {}).get("value")
        if s is None or e is None:
            continue
        mapped = [pmap[i] for i in range(int(s), int(e) + 1) if i in pmap]
        if not mapped:
            continue
        rows.append({"Type": ft["type"], "Description": ft.get("description", ""),
                     "Start": min(mapped), "End": max(mapped), "UniProt start": s, "UniProt end": e})
    return pd.DataFrame(rows, columns=["Type", "Description", "Start", "End", "UniProt start", "UniProt end"])


def uniprot_features(accession: str, reference: str, types: Iterable[str] = FEATURE_TYPES) -> pd.DataFrame:
    """UniProt features mapped into reference numbering."""
    entry = uniprot_entry(accession)
    return _project_features(entry.get("features", []), entry.get("sequence", {}).get("value", ""),
                             reference, types)


@lru_cache(maxsize=128)
def _variation_json(accession: str, timeout: int = 60) -> Dict:
    r = requests.get(EBI_VARIATION.format(acc=accession.split(".")[0]),
                     headers={"Accept": "application/json"}, timeout=timeout)
    if r.status_code == 404:
        return {}
    r.raise_for_status()
    return r.json()


KNOWN_COLUMNS = ["Position", "Ref", "Alt", "Mutation", "Consequence", "Clinical significance",
                 "dbSNP", "ClinVar", "SIFT", "PolyPhen", "Sources"]


def _parse_variation(data: Dict, reference: str) -> pd.DataFrame:
    """Single-residue variants from an EBI variation payload, in reference numbering.

    Variants whose wild type differs from the reference residue at the mapped
    position are skipped: relative to this reference they are a different change.
    """
    cols = KNOWN_COLUMNS
    if not data:
        return pd.DataFrame(columns=cols)
    pmap = _to_reference(data.get("sequence", ""), reference)
    rows = []
    for ft in data.get("features", []):
        alt = ft.get("alternativeSequence") or ft.get("mutatedType")
        wt = ft.get("wildType")
        if not alt or not wt or ft.get("begin") != ft.get("end") or len(alt) != 1:
            continue
        pos = pmap.get(int(ft["begin"]))
        if pos is None or pos > len(reference) or reference[pos - 1] != wt:
            continue
        xrefs = ft.get("xrefs", []) or []
        preds = {p.get("predAlgorithmNameType"): p for p in ft.get("predictions", []) or []}

        def _pred(name):
            p = preds.get(name)
            return f"{p.get('predictionValType', '')} ({p.get('score', '')})" if p else ""

        rows.append({
            "Position": pos, "Ref": wt, "Alt": alt, "Mutation": f"{wt}{pos}{alt}",
            "Consequence": ft.get("consequenceType", ""),
            "Clinical significance": "; ".join(sorted({c.get("type", "") for c in ft.get("clinicalSignificances") or []})),
            "dbSNP": ";".join(sorted({x["id"] for x in xrefs if x.get("name") == "dbSNP"})),
            "ClinVar": ";".join(sorted({x["id"] for x in xrefs if x.get("name") == "ClinVar"})),
            "SIFT": _pred("SIFT"), "PolyPhen": _pred("PolyPhen"),
            "Sources": ";".join(sorted({x.get("name", "") for x in xrefs})),
        })
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df
    agg = {c: (lambda s: "; ".join(sorted({v for v in s if v}))) for c in cols[4:]}
    return df.groupby(["Position", "Ref", "Alt", "Mutation"], as_index=False).agg(agg)


def known_variants(accession: str, reference: str) -> pd.DataFrame:
    """Single-residue variants reported for a UniProt entry, in reference numbering."""
    return _parse_variation(_variation_json(accession), reference)


def merge_known_variants(summary: pd.DataFrame, known: pd.DataFrame) -> pd.DataFrame:
    """Left-join known-variant annotations onto a variant summary (substitutions)."""
    if known is None or known.empty or summary.empty:
        out = summary.copy()
        for c in KNOWN_COLUMNS[4:]:
            out[c] = ""
        out["Known"] = False
        return out
    k = known.drop(columns=["Mutation"]).assign(Known=True)
    out = summary.merge(k, on=["Position", "Ref", "Alt"], how="left")
    out["Known"] = out["Known"].astype("boolean").fillna(False).astype(bool)
    return out


def add_structure_context(summary: pd.DataFrame, chain_mapping) -> pd.DataFrame:
    """Add pLDDT/B-factor and relative solvent accessibility for each variant position."""
    if chain_mapping is None:
        return summary
    out = summary.copy()
    if out.empty:
        return out.assign(**{"Structure residue": "", "B-factor / pLDDT": np.nan, "RSA": np.nan, "Exposure": ""})
    out["Structure residue"] = out["Position"].map(
        lambda p: "".join(map(str, chain_mapping.ref_to_residue[p])) if p in chain_mapping.ref_to_residue else ""
    )
    out["B-factor / pLDDT"] = out["Position"].map(chain_mapping.bfactor)
    out["RSA"] = out["Position"].map(chain_mapping.rsa)
    out["Exposure"] = np.select(
        [out["RSA"] >= 0.25, out["RSA"] < 0.25], ["surface", "buried"], default=""
    )
    return out


def add_feature_overlaps(summary: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Add a ``Features`` column listing UniProt features overlapping each variant."""
    if features is None or features.empty:
        return summary
    out = summary.copy()
    if out.empty:
        return out.assign(Features="")
    rows = features.to_dict("records")

    def hits(p):
        names = [f"{f['Type']}: {f['Description']}".rstrip(": ") for f in rows
                 if f["Start"] <= p <= f["End"] and f["Type"] != "Chain"]
        return "; ".join(dict.fromkeys(names))

    out["Features"] = out["Position"].map(hits)
    return out


# ---------------------------------------------------------------------------
# Sequence metadata
# ---------------------------------------------------------------------------

_YEAR = re.compile(r"(1[89]\d\d|20\d\d)")


METADATA_COLUMNS = ["Sequence ID", "Organism", "Strain", "Host", "Country", "Collection date", "Year"]
METADATA_FILE = "metadata.tsv"


def _parse_gp(handle) -> List[dict]:
    import warnings

    from Bio import BiopythonParserWarning, SeqIO

    # Harmless for metadata (feature bonds); filterwarnings is process-wide, unlike catch_warnings.
    warnings.filterwarnings("ignore", message="Dropping bond qualifier", category=BiopythonParserWarning)

    rows = []
    for rec in SeqIO.parse(handle, "genbank"):
        src = next((f for f in rec.features if f.type == "source"), None)
        q = src.qualifiers if src else {}
        date = (q.get("collection_date") or [""])[0]
        year = _YEAR.search(date)
        rows.append({
            "Sequence ID": rec.id, "Organism": rec.annotations.get("organism", ""),
            "Strain": (q.get("strain") or q.get("isolate") or [""])[0],
            "Host": (q.get("host") or [""])[0],
            "Country": (q.get("geo_loc_name") or q.get("country") or [""])[0].split(":")[0],
            "Collection date": date,
            "Year": int(year.group(1)) if year else None,
        })
    return rows


def ncbi_source_metadata(
    accessions: List[str],
    email: str,
    batch: int = 100,
    workers: Optional[int] = None,
    progress: ProgressFn = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> pd.DataFrame:
    """Organism / host / country / collection date for NCBI protein accessions.

    Batches are fetched concurrently (3 at a time, 8 with ``$NCBI_API_KEY``,
    staying under NCBI's request-rate limits), each retried once on failure.
    ``progress(fraction, message)`` is called after every batch;
    ``should_stop()`` returning True abandons the remaining batches and returns
    what has been fetched so far.
    """
    from Bio import Entrez

    Entrez.email = email
    if os.environ.get("NCBI_API_KEY"):
        Entrez.api_key = os.environ["NCBI_API_KEY"]
    workers = workers or (8 if Entrez.api_key else 3)
    accs = list(dict.fromkeys(a for a in accessions if a and not a.lower().startswith("pdb|")))
    chunks = [accs[i:i + batch] for i in range(0, len(accs), batch)]
    # Bio.Entrez spaces requests with an unsynchronised timestamp; serialise the
    # request start across threads so concurrent batches respect the rate limit.
    gate, min_gap = threading.Lock(), 0.11 if Entrez.api_key else 0.34
    last = [0.0]

    def fetch(chunk: List[str]) -> List[dict]:
        for attempt in range(2):
            if should_stop and should_stop():
                return []
            with gate:
                wait = last[0] + min_gap - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                last[0] = time.monotonic()
            try:
                handle = Entrez.efetch(db="protein", id=",".join(chunk), rettype="gp", retmode="text")
                try:
                    return _parse_gp(handle)
                finally:
                    handle.close()
            except Exception as exc:  # noqa: BLE001
                log.warning("GenBank metadata batch failed (attempt %d): %s", attempt + 1, exc)
                time.sleep(2)
        return []

    rows: List[dict] = []
    report(progress, 0.0, f"Fetching GenBank records for {len(accs)} sequences")
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(fetch, c) for c in chunks]
        done = 0
        for fut in as_completed(futures):
            rows += fut.result()
            done += 1
            report(progress, done / max(len(chunks), 1),
                   f"GenBank records: {min(done * batch, len(accs))} / {len(accs)}")
            if should_stop and should_stop():
                for f in futures:
                    f.cancel()
    df = pd.DataFrame(rows, columns=METADATA_COLUMNS)
    df["Year"] = df["Year"].astype("Int64")
    return df


def save_metadata(run_dir, meta: pd.DataFrame) -> Path:
    path = Path(run_dir) / "tables" / METADATA_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(path, sep="\t", index=False)
    return path


def load_metadata(run_dir) -> Optional[pd.DataFrame]:
    """Metadata saved with the run (by the app, CLI or MCP server), if any."""
    path = Path(run_dir) / "tables" / METADATA_FILE
    if not path.exists():
        return None
    meta = pd.read_csv(path, sep="\t", keep_default_na=False, na_values=[""],
                       dtype={c: str for c in METADATA_COLUMNS if c != "Year"})
    meta["Year"] = pd.to_numeric(meta["Year"], errors="coerce").astype("Int64")
    for c in METADATA_COLUMNS[1:-1]:
        if c in meta:
            meta[c] = meta[c].fillna("")
    return meta


_OS = re.compile(r"OS=(.+?)(?: OX=| GN=| PE=| SV=|$)")


def uniprot_header_metadata(records) -> pd.DataFrame:
    """Organism from UniProt FASTA headers (no network)."""
    rows = []
    for r in records:
        m = _OS.search(r.description)
        rows.append({"Sequence ID": r.id, "Organism": m.group(1) if m else ""})
    return pd.DataFrame(rows, columns=["Sequence ID", "Organism"])


def variants_by_year(events: pd.DataFrame, metadata: pd.DataFrame, mutations: List[str],
                     included: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Per-year carrier fraction for the given mutations (needs ``Year`` metadata).

    ``included``: IDs of sequences used for variant calling; others (excluded
    or filtered out) do not count towards a year's denominator.
    """
    if metadata is None or metadata.empty or "Year" not in metadata:
        return pd.DataFrame(columns=["Year", "Mutation", "Carriers", "Sequences", "Frequency"])
    meta = metadata.dropna(subset=["Year"]).copy()
    if included is not None:
        meta = meta[meta["Sequence ID"].isin(set(included))]
    meta["Year"] = meta["Year"].astype(int)
    per_year = meta.groupby("Year")["Sequence ID"].nunique().rename("Sequences")
    ev = events[events["Mutation"].isin(mutations)].merge(meta[["Sequence ID", "Year"]], on="Sequence ID")
    carriers = ev.groupby(["Year", "Mutation"])["Sequence ID"].nunique().rename("Carriers").reset_index()
    out = carriers.merge(per_year.reset_index(), on="Year")
    out["Frequency"] = out["Carriers"] / out["Sequences"]
    return out.sort_values(["Mutation", "Year"]).reset_index(drop=True)
