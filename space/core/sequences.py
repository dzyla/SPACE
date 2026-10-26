"""Sequence cleaning, FASTA I/O and database fetching (NCBI Entrez, UniProt)."""

from __future__ import annotations

import os
import re
import time
from io import StringIO
from urllib.parse import quote_plus
from pathlib import Path
from typing import Iterable, List, Optional

import requests
from Bio import Entrez, SeqIO
from Bio.SeqRecord import SeqRecord

from . import ProgressFn, log, report

STANDARD_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")

UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_PAGE_SIZE = 500
# Upper bound for "fetch everything" (max_seqs == -1) so a broad query cannot
# pull hundreds of thousands of records into memory.
FETCH_ALL_CAP = 20000

_HTTP = requests.Session()
_HTTP.headers.update({"User-Agent": "SPACE (https://github.com/dzyla/SPACE)"})


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_sequence(seq: str) -> str:
    """Normalise a raw protein sequence.

    Uppercases, drops everything that is not a letter (whitespace, digits,
    gaps, stop ``*``) and maps non-standard letters (U, O, B, Z, J, ...) to
    ``X``. Mapping instead of deleting is what keeps reference numbering
    intact: a selenocysteine at position 50 must still be position 50.
    """
    letters = re.sub(r"[^A-Za-z]", "", seq or "").upper()
    return "".join(c if c in STANDARD_AA else "X" for c in letters)


def parse_sequence_text(text: str) -> str:
    """Return the first sequence from pasted text (raw sequence or FASTA)."""
    if not text:
        return ""
    lines = text.strip().splitlines()
    if lines and lines[0].startswith(">"):
        body: List[str] = []
        for line in lines[1:]:
            if line.startswith(">"):
                break
            body.append(line)
        return clean_sequence("".join(body))
    return clean_sequence("".join(lines))


def clean_fasta(fasta_str: str) -> str:
    """Clean every sequence line of a FASTA string (headers kept as-is)."""
    if not fasta_str:
        return ""
    out = [
        line if line.startswith(">") else clean_sequence(line)
        for line in fasta_str.strip().splitlines()
    ]
    return "\n".join(out).strip()


# ---------------------------------------------------------------------------
# FASTA I/O
# ---------------------------------------------------------------------------

def read_fasta(path: str | os.PathLike, remove_pdb: bool = False) -> List[SeqRecord]:
    """Read protein records, dropping empty ones and (optionally) PDB chains."""
    records = [r for r in SeqIO.parse(str(path), "fasta") if len(r.seq) > 0]
    if remove_pdb:
        before = len(records)
        records = [r for r in records if not is_pdb_record(r)]
        log.info("Removed %d PDB-derived sequences.", before - len(records))
    if not records:
        log.warning("No protein sequences found in %s.", path)
    return records


def is_pdb_record(record: SeqRecord) -> bool:
    """NCBI deflines for PDB chains look like ``pdb|7KI4|A``."""
    return record.id.lower().startswith("pdb|")


def write_fasta(records: Iterable[SeqRecord], path: str | os.PathLike) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    SeqIO.write(list(records), str(path), "fasta")
    return str(path)


def extract_uniprot_accession(record_id: str) -> Optional[str]:
    """``sp|P12345|NAME_HUMAN`` -> ``P12345``; bare accessions pass through."""
    parts = record_id.split("|")
    if len(parts) >= 2 and parts[0] in ("sp", "tr"):
        return parts[1]
    m = re.fullmatch(
        r"([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})(?:\.\d+)?",
        record_id,
    )
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# NCBI
# ---------------------------------------------------------------------------

def _entrez_call(fn, retries: int = 3, **kwargs):
    """Run an Entrez call with retry/backoff (NCBI returns transient 429/5xx)."""
    delay = 1.0
    for attempt in range(retries):
        try:
            return fn(**kwargs)
        except Exception as exc:  # noqa: BLE001 — urllib/HTTP errors vary
            if attempt == retries - 1:
                raise
            log.warning("NCBI request failed (%s); retrying in %.0fs.", exc, delay)
            time.sleep(delay)
            delay *= 2


def _esearch_ids(term: str, retmax: int) -> List[str]:
    handle = _entrez_call(
        Entrez.esearch, db="protein", term=term, sort="relevance", idtype="acc", retmax=retmax
    )
    try:
        return list(Entrez.read(handle, ignore_errors=True).get("IdList", []))
    finally:
        handle.close()


def fetch_ncbi(
    query: str,
    out_path: str | os.PathLike,
    email: str,
    max_seqs: int = 100,
    refseq_only: bool = False,
    include_swissprot: bool = True,
    api_key: Optional[str] = None,
    progress: ProgressFn = None,
) -> str:
    """Search NCBI Protein and write matching sequences to ``out_path``.

    Results of the main query are merged with the Swiss-Prot-filtered subset
    so curated entries are never truncated away; relevance order is kept.
    ``api_key`` (or ``$NCBI_API_KEY``) raises the NCBI rate limit to 10 req/s.
    """
    if not email or "@" not in email:
        raise ValueError("NCBI Entrez requires a valid email address.")
    Entrez.email = email
    key = api_key or os.environ.get("NCBI_API_KEY")
    if key:
        Entrez.api_key = key

    limit = FETCH_ALL_CAP if max_seqs is None or max_seqs < 0 else int(max_seqs)
    retmax = min(max(limit * 3, 300), FETCH_ALL_CAP)

    term = database_query(query, "ncbi", refseq_only=refseq_only)
    report(progress, 0.02, "Searching NCBI Protein")
    ids = _esearch_ids(term, retmax)
    sp_ids: List[str] = []
    if include_swissprot:
        sp_ids = _esearch_ids(f"({query}) AND swissprot[filter]", retmax)
    log.info("NCBI: %d hits for query, %d Swiss-Prot hits.", len(ids), len(sp_ids))

    # Swiss-Prot hits first so curated entries survive truncation, then the rest.
    combined = list(dict.fromkeys(sp_ids + ids))[:limit]
    if not combined:
        raise ValueError(f"No NCBI protein entries found for query: {query!r}")

    batch = 200
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for start in range(0, len(combined), batch):
            chunk = combined[start:start + batch]
            report(progress, 0.05 + 0.95 * start / len(combined),
                   f"Downloading sequences {start + 1}–{start + len(chunk)} of {len(combined)}")
            handle = _entrez_call(
                Entrez.efetch, db="protein", id=",".join(chunk), rettype="fasta", retmode="text"
            )
            try:
                SeqIO.write(SeqIO.parse(handle, "fasta"), fh, "fasta")
            finally:
                handle.close()
    report(progress, 1.0, f"Fetched {len(combined)} sequences from NCBI")
    return os.path.abspath(out_path)


# ---------------------------------------------------------------------------
# UniProt
# ---------------------------------------------------------------------------

_NEXT_LINK = re.compile(r'<([^>]+)>;\s*rel="next"')


def fetch_uniprot(
    query: str,
    out_path: str | os.PathLike,
    max_seqs: int = 500,
    reviewed_only: bool = False,
    progress: ProgressFn = None,
    timeout: int = 60,
) -> str:
    """Search UniProtKB and write matching sequences to ``out_path``.

    Follows UniProt's cursor pagination (``Link: rel="next"``), so
    ``max_seqs`` is honoured beyond the 500-per-page API limit.
    """
    limit = FETCH_ALL_CAP if max_seqs is None or max_seqs < 0 else int(max_seqs)
    q = database_query(query, "uniprot", reviewed_only=reviewed_only)
    url: Optional[str] = UNIPROT_SEARCH_URL
    params: Optional[dict] = {
        "query": q, "format": "fasta", "size": min(UNIPROT_PAGE_SIZE, max(limit, 1)),
    }

    records: List[SeqRecord] = []
    total: Optional[int] = None
    report(progress, 0.02, "Searching UniProtKB")
    while url and len(records) < limit:
        resp = _HTTP.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        if total is None and resp.headers.get("X-Total-Results"):
            total = int(resp.headers["X-Total-Results"])
        records.extend(SeqIO.parse(StringIO(resp.text), "fasta"))
        target = min(limit, total or limit)
        report(progress, len(records) / max(target, 1),
               f"Downloaded {min(len(records), target)} of {target} UniProt sequences")
        m = _NEXT_LINK.search(resp.headers.get("Link", ""))
        url, params = (m.group(1), None) if m else (None, None)

    records = records[:limit]
    if not records:
        raise ValueError(f"No UniProt entries found for query: {query!r}")
    log.info("UniProt: fetched %d sequences (total matching: %s).", len(records), total)
    return os.path.abspath(write_fasta(records, out_path))


def database_query(query: str, source: str, refseq_only: bool = False, reviewed_only: bool = False) -> str:
    """The exact search term sent to ``source`` (filters appended)."""
    if source.lower() == "ncbi" and refseq_only:
        return f"({query}) AND refseq[filter]"
    if source.lower() == "uniprot" and reviewed_only:
        return f"({query}) AND reviewed:true"
    return query


def search_url(query: str, source: str, refseq_only: bool = False, reviewed_only: bool = False) -> str:
    """Web search page for the same term, to check hits before fetching."""
    term = quote_plus(database_query(query, source, refseq_only, reviewed_only))
    if source.lower() == "ncbi":
        return f"https://www.ncbi.nlm.nih.gov/protein/?term={term}"
    return f"https://www.uniprot.org/uniprotkb?query={term}"


def fetch_sequences(
    query: str,
    out_path: str | os.PathLike,
    source: str = "UniProt",
    email: Optional[str] = None,
    max_seqs: int = 100,
    refseq_only: bool = False,
    reviewed_only: bool = False,
    progress: ProgressFn = None,
) -> str:
    """Dispatch to :func:`fetch_ncbi` or :func:`fetch_uniprot`."""
    if source.lower() == "ncbi":
        return fetch_ncbi(query, out_path, email or "", max_seqs=max_seqs,
                          refseq_only=refseq_only, progress=progress)
    if source.lower() == "uniprot":
        return fetch_uniprot(query, out_path, max_seqs=max_seqs,
                             reviewed_only=reviewed_only, progress=progress)
    raise ValueError(f"Unknown data source {source!r}; use 'NCBI' or 'UniProt'.")
