"""Per-position conservation: al2co, residue frequencies, logo and consensus."""

from __future__ import annotations

import math
from collections import Counter
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from . import REFERENCE_ID, log
from .align import msa_array, reference_row

AA_ALPHABET = 20
GAP = "-"


def al2co_scores(msa: MultipleSeqAlignment, mapping: List[Optional[int]], **al2co_kwargs) -> pd.DataFrame:
    """Run al2co and return one row per alignment column.

    Columns: ``Column`` (1-based), ``Location`` (reference residue number or
    NaN for reference gaps), ``Residue`` (reference residue), ``al2co_score``.

    al2co normalises scores to z-scores, so it fails when every sequence is
    identical (zero variance); scores are then NaN and a warning is logged.
    """
    import al2co

    ref = str(msa[reference_row(msa)].seq)
    try:
        scores = al2co.run_al2co(_al2co_safe(msa), **al2co_kwargs)
    except Exception as exc:  # noqa: BLE001 — native binary fails in several ways
        log.warning(
            "al2co failed (%s); conservation scores are left empty. This happens "
            "when all sequences are identical (zero variance).", exc,
        )
        scores = {}
    rows = []
    for col, loc in enumerate(mapping, start=1):
        rows.append({
            "Column": col,
            "Location": np.nan if loc is None else loc,
            "Residue": ref[col - 1],
            "al2co_score": float(scores[col]) if col in scores else np.nan,
        })
    return pd.DataFrame(rows)


def _without_reference(msa: MultipleSeqAlignment):
    """MSA rows except the reference row, so the reference (which may be a
    pasted sequence or a duplicate of a database record) does not weigh in."""
    arr = msa_array(msa)
    keep = [i for i, r in enumerate(msa) if r.id != REFERENCE_ID] or list(range(len(msa)))
    return arr[keep]


def _al2co_safe(msa: MultipleSeqAlignment) -> MultipleSeqAlignment:
    """al2co rejects any letter outside the 20 standard residues (X, B, Z, U...),
    so ambiguous residues are passed to it as gaps (no data)."""
    table = str.maketrans({c: GAP for c in "BJOUXZ"})
    return MultipleSeqAlignment(
        [SeqRecord(Seq(str(r.seq).upper().translate(table)), id=r.id, description="") for r in msa]
    )


def column_profile(msa: MultipleSeqAlignment, mapping: List[Optional[int]]) -> pd.DataFrame:
    """Simple per-column statistics for reference-mapped columns.

    ``Conservation`` is the modal residue frequency over all sequences,
    ``Entropy`` is Shannon entropy (bits) over non-gap residues and
    ``Gap fraction`` the share of sequences with a gap.
    """
    arr = _without_reference(msa)
    n = arr.shape[0]
    rows = []
    for col, loc in enumerate(mapping):
        if loc is None:
            continue
        counts = Counter(arr[:, col])
        gaps = counts.pop(GAP, 0)
        non_gap = sum(counts.values())
        if non_gap:
            aa, top = counts.most_common(1)[0]
            probs = np.array(list(counts.values()), dtype=float) / non_gap
            entropy = max(0.0, float(-(probs * np.log2(probs)).sum()))
        else:
            aa, top, entropy = GAP, 0, 0.0
        rows.append({
            "Location": loc,
            "Column": col + 1,
            "Modal residue": aa,
            "Conservation": top / n,
            "Entropy": entropy,
            "Gap fraction": gaps / n,
        })
    return pd.DataFrame(rows)


def information_content(counts: dict, n: int) -> dict:
    """Per-residue logo heights (bits), Schneider et al. (1986) with small-sample correction.

    Gaps count as an observed state (lowering information) but are not
    returned as a stack element.
    """
    if n == 0:
        return {}
    e_n = (AA_ALPHABET - 1) / (2.0 * math.log(2) * n)
    probs = {k: v / n for k, v in counts.items()}
    h = -sum(p * math.log2(p) for p in probs.values() if p > 0)
    info = max(0.0, math.log2(AA_ALPHABET) - h - e_n)
    return {k: p * info for k, p in probs.items() if k != GAP}


def logo_and_consensus(
    msa: MultipleSeqAlignment, mapping: List[Optional[int]], identity_threshold: float = 0.6
) -> Tuple[pd.DataFrame, str]:
    """Return a (positions × residues) bits table and the consensus string.

    Consensus uses the modal non-gap residue when its frequency over all
    sequences is ≥ ``identity_threshold``, else ``X`` (``-`` if all gaps).
    """
    arr = _without_reference(msa)
    n = arr.shape[0]
    rows, consensus = [], []
    for col, loc in enumerate(mapping):
        if loc is None:
            continue
        counts = Counter(arr[:, col])
        non_gap = {k: v for k, v in counts.items() if k != GAP}
        if not non_gap:
            consensus.append(GAP)
            rows.append({"Location": loc})
            continue
        top = max(non_gap, key=non_gap.get)
        consensus.append(top if non_gap[top] / n >= identity_threshold else "X")
        rows.append({"Location": loc, **information_content(dict(counts), n)})
    bits = pd.DataFrame(rows).set_index("Location").fillna(0.0)
    return bits, "".join(consensus)
