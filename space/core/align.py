"""Pairwise alignment to the reference, filtering, and MSA (pyfamsa)."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Align import MultipleSeqAlignment, PairwiseAligner, substitution_matrices
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from pyfamsa import Aligner as FamsaAligner, Sequence as FamsaSequence

from . import REFERENCE_ID, ProgressFn, log, report
from .sequences import clean_sequence

# ---------------------------------------------------------------------------
# Pairwise alignment
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_aligner() -> PairwiseAligner:
    """BLOSUM62 aligner, EMBOSS-needle-like gaps, free end gaps.

    Free end gaps (semi-global) let fragments and N/C-terminal extensions
    align where they belong instead of being smeared across the reference.
    Note: substitution_matrix must not be combined with match/mismatch
    scores — setting those silently drops the matrix.
    """
    al = PairwiseAligner()
    al.mode = "global"
    al.substitution_matrix = substitution_matrices.load("BLOSUM62")
    al.open_gap_score = -10.0
    al.extend_gap_score = -0.5
    al.end_gap_score = 0.0
    return al


def align_pair(reference: str, query: str):
    """Best global alignment of two cleaned sequences (or None)."""
    try:
        return next(iter(get_aligner().align(reference, query)), None)
    except (ValueError, OverflowError):
        log.debug("pairwise alignment failed", exc_info=True)
        return None


def position_map(seq_a: str, seq_b: str) -> Dict[int, int]:
    """Map 1-based positions in ``seq_a`` to aligned 1-based positions in ``seq_b``."""
    aln = align_pair(clean_sequence(seq_a), clean_sequence(seq_b))
    if aln is None:
        return {}
    out: Dict[int, int] = {}
    for (a0, a1), (b0, b1) in zip(*aln.aligned):
        for i, j in zip(range(a0, a1), range(b0, b1)):
            out[i + 1] = j + 1
    return out


def pairwise_to_reference(
    records: Sequence[SeqRecord],
    reference: str,
    progress: ProgressFn = None,
) -> pd.DataFrame:
    """Align every record to the reference.

    Returns one row per record with:
      * ``identity`` — identical residues / alignment length (%, end gaps
        included; the score used for filtering, as in earlier SPACE versions)
      * ``aligned_identity`` — identical / aligned residue pairs (%)
      * ``coverage`` — reference residues covered by an aligned residue (%)
    """
    reference = clean_sequence(reference)
    if not reference:
        raise ValueError("Reference sequence is empty.")
    rows = []
    n = len(records)
    failed = 0
    for i, rec in enumerate(records):
        seq = clean_sequence(str(rec.seq))
        aln = align_pair(reference, seq) if seq else None
        if aln is None:
            failed += 1
            continue
        c = aln.counts()
        aligned = c.identities + c.mismatches
        rows.append({
            "id": rec.id,
            "description": rec.description,
            "sequence": seq,
            "length": len(seq),
            "identity": 100.0 * c.identities / aln.shape[1],
            "aligned_identity": 100.0 * c.identities / aligned if aligned else 0.0,
            "coverage": 100.0 * aligned / len(reference),
        })
        if i % 25 == 0 or i == n - 1:
            report(progress, (i + 1) / n, f"Aligned {i + 1} of {n} sequences to the reference")
    if failed:
        log.warning("Failed to align %d sequences to the reference.", failed)
    if not rows:
        raise ValueError("No sequences could be aligned to the reference.")
    return pd.DataFrame(rows)


def filter_hits(
    hits: pd.DataFrame,
    min_identity: float = 0.0,
    max_identity: float = 100.0,
    min_length: int = 0,
    max_length: int = 10**9,
    min_coverage: float = 0.0,
) -> pd.DataFrame:
    """Keep hits inside identity / length / coverage bounds (inclusive)."""
    mask = (
        hits["identity"].between(min_identity, max_identity)
        & hits["length"].between(min_length, max_length)
        & (hits["coverage"] >= min_coverage)
    )
    return hits[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Multiple sequence alignment
# ---------------------------------------------------------------------------

def safe_id(seq_id: str) -> str:
    """IDs safe for FASTA/CLUSTAL/Newick (no pipes, spaces, colons...)."""
    return re.sub(r"[^\w\-.]", "_", seq_id)


def _unique_ids(ids: Sequence[str], reserved: Sequence[str] = (REFERENCE_ID,)) -> List[str]:
    """Sanitised, collision-free row names (never equal to a reserved name)."""
    used = set(reserved)
    out = []
    for raw in ids:
        base = safe_id(raw)
        name, k = base, 1
        while name in used:
            k += 1
            name = f"{base}_{k}"
        used.add(name)
        out.append(name)
    return out


def run_msa(
    hits: pd.DataFrame,
    reference: str,
    out_dir: str | os.PathLike,
    threads: int = 0,
    progress: ProgressFn = None,
) -> Tuple[MultipleSeqAlignment, pd.DataFrame, Dict[str, str]]:
    """Align selected hits plus the reference with pyfamsa.

    The reference is added as ``reference_sequence``. Writes
    ``msa_in.fasta``, ``msa_out.fasta`` and ``msa_out.aln`` (CLUSTAL W header)
    into ``out_dir``.

    Returns ``(msa, hits_with_msa_id, files)``. ``hits_with_msa_id`` gains an
    ``msa_id`` column linking each original ID to its sanitised MSA row name.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hits = hits.copy()
    hits = hits.reset_index(drop=True)
    hits["msa_id"] = _unique_ids(list(hits["id"]))
    names = list(hits["msa_id"]) + [REFERENCE_ID]
    seqs = list(hits["sequence"]) + [clean_sequence(reference)]

    files = {
        "msa_in": str(out_dir / "msa_in.fasta"),
        "msa_fasta": str(out_dir / "msa_out.fasta"),
        "msa_clustal": str(out_dir / "msa_out.aln"),
    }
    SeqIO.write(
        [SeqRecord(Seq(s), id=n, description="") for n, s in zip(names, seqs)],
        files["msa_in"], "fasta",
    )

    report(progress, 0.1, f"Running pyfamsa on {len(names)} sequences")
    n_threads = int(threads) if threads and threads > 0 else min(os.cpu_count() or 4, 16)
    famsa = FamsaAligner(guide_tree="upgma", threads=n_threads)
    aligned = famsa.align([FamsaSequence(n.encode(), s.encode()) for n, s in zip(names, seqs)])
    by_name = {s.id.decode(): s.sequence.decode() for s in aligned}
    # Keep input order (reference last) — FAMSA reorders rows.
    msa = MultipleSeqAlignment(
        [SeqRecord(Seq(by_name[n]), id=n, description="") for n in names]
    )
    SeqIO.write(list(msa), files["msa_fasta"], "fasta")
    write_clustal(msa, files["msa_clustal"])
    report(progress, 1.0, "MSA complete")
    return msa, hits, files


def write_clustal(msa: MultipleSeqAlignment, path: str) -> str:
    """Write CLUSTAL with the classic ``CLUSTAL W`` header some parsers require."""
    from Bio import AlignIO

    AlignIO.write(msa, path, "clustal")
    with open(path, encoding="utf-8") as fh:
        lines = fh.readlines()
    if lines:
        lines[0] = "CLUSTAL W multiple sequence alignment\n"
    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(lines)
    return path


def msa_array(msa: MultipleSeqAlignment) -> np.ndarray:
    """MSA as an (n_seqs, n_columns) array of single uppercase characters."""
    return np.array([list(str(r.seq).upper()) for r in msa], dtype="U1")


def reference_row(msa: MultipleSeqAlignment, ref_id: str = REFERENCE_ID) -> int:
    for i, rec in enumerate(msa):
        if rec.id == ref_id:
            return i
    raise ValueError(f"Reference {ref_id!r} not found in the alignment.")


def reference_mapping(msa: MultipleSeqAlignment, ref_id: str = REFERENCE_ID) -> List[Optional[int]]:
    """Alignment column (0-based) -> reference residue number (1-based) or None."""
    ref = str(msa[reference_row(msa, ref_id)].seq)
    mapping: List[Optional[int]] = []
    n = 0
    for ch in ref:
        if ch == "-":
            mapping.append(None)
        else:
            n += 1
            mapping.append(n)
    return mapping
