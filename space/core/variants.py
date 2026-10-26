"""Variant calling against the reference row of an MSA.

For every sequence the *covered span* is the reference region between its
first and last aligned residue. Inside that span:

* reference residue vs. different residue  -> substitution (``I6G``, ``p.Ile6Gly``)
* reference residue vs. gap                -> deletion (``K5del`` / ``K5_Q8del``)
* reference gap vs. residue(s)             -> insertion (``Q10_R11insWW``)

Outside the span nothing is called and the sequence does not count towards
that position's denominator, so fragments contribute only where they have
data. Frequencies are ``carriers / sequences covering the position``.
Ambiguous residues (``X``) are treated as "no data" at that position.

Because databases often contain many identical sequences (one outbreak, one
study), every count is also reported over *unique* sequences.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from Bio.Align import MultipleSeqAlignment, substitution_matrices

from . import REFERENCE_ID, ProgressFn, report
from .align import msa_array, reference_row

THREE_LETTER = {
    "A": "Ala", "R": "Arg", "N": "Asn", "D": "Asp", "C": "Cys", "Q": "Gln", "E": "Glu",
    "G": "Gly", "H": "His", "I": "Ile", "L": "Leu", "K": "Lys", "M": "Met", "F": "Phe",
    "P": "Pro", "S": "Ser", "T": "Thr", "W": "Trp", "Y": "Tyr", "V": "Val", "X": "Xaa",
}

AA_CLASS = {
    **dict.fromkeys("DE", "acidic"),
    **dict.fromkeys("KRH", "basic"),
    **dict.fromkeys("STNQCY", "polar"),
    **dict.fromkeys("AVLIMFWP", "hydrophobic"),
    "G": "glycine",
}
_CHARGE = {"D": -1, "E": -1, "K": 1, "R": 1}

SUBSTITUTION, DELETION, INSERTION = "Substitution", "Deletion", "Insertion"

EVENT_COLUMNS = ["Sequence ID", "MSA ID", "Mutation", "HGVS", "Type", "Position", "End", "Ref", "Alt"]


def _three(seq: str) -> str:
    return "".join(THREE_LETTER.get(c, "Xaa") for c in seq)


@dataclass
class VariantCalls:
    """Result of :func:`call_variants`."""

    events: pd.DataFrame          # one row per (sequence, variant)
    summary: pd.DataFrame         # one row per distinct variant
    per_sequence: pd.DataFrame    # one row per non-reference sequence
    coverage: np.ndarray          # included sequences covering each reference position
    unique_coverage: np.ndarray   # same, counting identical sequences once
    reference: str
    params: Dict = field(default_factory=dict)

    @property
    def n_included(self) -> int:
        return int((self.per_sequence["Status"] == "included").sum())

    @property
    def n_unique(self) -> int:
        inc = self.per_sequence[self.per_sequence["Status"] == "included"]
        return int(inc["Unique group"].nunique())

    @property
    def excluded(self) -> Dict[str, str]:
        ex = self.per_sequence[self.per_sequence["Status"] == "excluded"]
        return dict(zip(ex["Sequence ID"], ex["Reason"]))

    @property
    def unique_mutations(self) -> Dict[str, str]:
        """Legacy view: ``{sequence_id: "A1B, C2D" | "No mutations"}`` for included sequences."""
        inc = self.per_sequence[self.per_sequence["Status"] == "included"]
        return {
            sid: (v if v else "No mutations")
            for sid, v in zip(inc["Sequence ID"], inc["Variants"])
        }

    @property
    def all_mutations_str(self) -> str:
        if self.summary.empty:
            return ""
        return "; ".join(self.summary.sort_values(["Position", "Mutation"])["Mutation"])

    def filter(
        self,
        min_frequency: float = 0.0,
        min_count: int = 1,
        types: Optional[List[str]] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        unique: bool = False,
    ) -> pd.DataFrame:
        """Convenience filter over :attr:`summary`."""
        df = self.summary
        if df.empty:
            return df
        freq_col = "Unique frequency" if unique else "Frequency"
        count_col = "Unique carriers" if unique else "Count"
        m = (df[freq_col] >= min_frequency) & (df[count_col] >= min_count)
        if types:
            m &= df["Type"].isin(types)
        if start is not None:
            m &= df["Position"] >= start
        if end is not None:
            m &= df["Position"] <= end
        return df[m].reset_index(drop=True)


def _runs(positions: List[int]) -> List[List[int]]:
    out: List[List[int]] = []
    for p in positions:
        if out and p == out[-1][-1] + 1:
            out[-1].append(p)
        else:
            out.append([p])
    return out


def call_variants(
    msa: MultipleSeqAlignment,
    mapping: List[Optional[int]],
    ref_id: str = REFERENCE_ID,
    id_map: Optional[Dict[str, str]] = None,
    deletion_threshold: float = 10.0,
    min_coverage: float = 0.0,
    skip_ambiguous: bool = True,
    progress: ProgressFn = None,
) -> VariantCalls:
    """Call substitutions, insertions and deletions for every MSA row vs. the reference.

    Args:
        msa: Alignment containing the reference row ``ref_id``.
        mapping: Column -> reference residue number (see ``reference_mapping``).
        id_map: Optional MSA row ID -> original database ID.
        deletion_threshold: Exclude sequences whose internal deletions exceed
            this percentage of their covered span (likely misaligned/garbage).
        min_coverage: Exclude sequences covering less than this percentage of
            the reference.
        skip_ambiguous: Treat ``X`` as missing data instead of calling ``A12X``.
    """
    arr = msa_array(msa)
    r_idx = reference_row(msa, ref_id)
    ref_cols = arr[r_idx]
    ref_is_res = ref_cols != "-"
    ref_seq = "".join(ref_cols[ref_is_res])
    L = len(ref_seq)
    pos_of_col = np.array([m or 0 for m in mapping], dtype=int)
    # anchor[col] = number of reference residues at or before col; for a
    # reference-gap column that is the residue the insertion follows.
    anchor = np.cumsum(ref_is_res)
    id_map = id_map or {}

    coverage = np.zeros(L, dtype=int)
    unique_coverage = np.zeros(L, dtype=int)
    seen_unique: Dict[str, int] = {}
    counted_groups: set = set()
    events: List[dict] = []
    per_seq: List[dict] = []
    n_rows = arr.shape[0]

    for row in range(n_rows):
        if row == r_idx:
            continue
        msa_id = msa[row].id
        sid = id_map.get(msa_id, msa_id)
        t = arr[row]
        t_res = t != "-"
        ungapped = "".join(t[t_res])
        group = seen_unique.setdefault(ungapped, len(seen_unique) + 1)
        record = {"Sequence ID": sid, "MSA ID": msa_id, "Unique group": group,
                  "Length": int(t_res.sum())}

        # The covered span runs between the first and last residues aligned to a
        # reference residue; anything outside (including residues sitting in
        # reference-gap columns at the ends) is an extension, not data.
        cols = np.flatnonzero(t_res & ref_is_res)
        if cols.size == 0:
            per_seq.append({**record, "Status": "excluded", "Reason": "no overlap with the reference",
                            "Coverage (%)": 0.0, "Internal deletions (%)": 0.0,
                            "Substitutions": 0, "Insertions": 0, "Deletions": 0, "Variants": ""})
            continue

        span = np.zeros_like(t_res)
        span[cols[0]:cols[-1] + 1] = True
        in_span = ref_is_res & span
        if skip_ambiguous:
            ref_amb = ref_cols == "X"
            ambiguous = in_span & ((t == "X") | ref_amb)
        else:
            ref_amb = ambiguous = np.zeros_like(span)
        covered = in_span & ~ambiguous
        deleted = in_span & ~t_res & ~ref_amb
        covered_flag = np.zeros(L + 2, dtype=bool)  # by reference position
        covered_flag[pos_of_col[covered]] = True
        cov_pct = 100.0 * covered.sum() / L
        del_pct = 100.0 * deleted.sum() / max(int(in_span.sum()), 1)

        reason = ""
        if cov_pct < min_coverage:
            reason = f"coverage {cov_pct:.1f}% < {min_coverage:g}%"
        elif del_pct > deletion_threshold:
            reason = f"internal deletions {del_pct:.1f}% > {deletion_threshold:g}%"
        if reason:
            per_seq.append({**record, "Status": "excluded", "Reason": reason,
                            "Coverage (%)": cov_pct, "Internal deletions (%)": del_pct,
                            "Substitutions": 0, "Insertions": 0, "Deletions": 0, "Variants": ""})
            continue

        covered_pos = pos_of_col[covered] - 1
        coverage[covered_pos] += 1
        if group not in counted_groups:
            counted_groups.add(group)
            unique_coverage[covered_pos] += 1

        seq_events: List[dict] = []
        # Substitutions
        for c in np.flatnonzero(covered & t_res & (t != ref_cols)):
            p, ref_aa, alt = int(pos_of_col[c]), ref_cols[c], t[c]
            seq_events.append({
                "Mutation": f"{ref_aa}{p}{alt}",
                "HGVS": f"p.{_three(ref_aa)}{p}{_three(alt)}",
                "Type": SUBSTITUTION, "Position": p, "End": p, "Ref": ref_aa, "Alt": alt,
            })
        # Deletions (contiguous reference runs)
        for run in _runs([int(p) for p in pos_of_col[deleted]]):
            a, b = run[0], run[-1]
            ra, rb = ref_seq[a - 1], ref_seq[b - 1]
            short = f"{ra}{a}del" if a == b else f"{ra}{a}_{rb}{b}del"
            hgvs = f"p.{_three(ra)}{a}del" if a == b else f"p.{_three(ra)}{a}_{_three(rb)}{b}del"
            seq_events.append({"Mutation": short, "HGVS": hgvs, "Type": DELETION,
                               "Position": a, "End": b, "Ref": ref_seq[a - 1:b], "Alt": "-"})
        # Insertions, grouped by the reference residue they follow. Residues
        # before position 1 or after the last residue are terminal extensions,
        # not insertions, and are ignored.
        ins: Dict[int, List[str]] = defaultdict(list)
        for c in np.flatnonzero(~ref_is_res & span & t_res):
            p = int(anchor[c])
            if 1 <= p < L and covered_flag[p] and covered_flag[p + 1]:
                ins[p].append(t[c])
        for p, residues in sorted(ins.items()):
            alt = "".join(residues)
            if skip_ambiguous and "X" in alt:
                continue  # an unsequenced stretch, not a confident insertion
            ra, rb = ref_seq[p - 1], ref_seq[p]
            seq_events.append({
                "Mutation": f"{ra}{p}_{rb}{p + 1}ins{alt}",
                "HGVS": f"p.{_three(ra)}{p}_{_three(rb)}{p + 1}ins{_three(alt)}",
                "Type": INSERTION, "Position": p, "End": p + 1, "Ref": "-", "Alt": alt,
            })

        seq_events.sort(key=lambda e: (e["Position"], e["Type"] != SUBSTITUTION))
        events.extend({"Sequence ID": sid, "MSA ID": msa_id, "_group": group, **e} for e in seq_events)
        per_seq.append({
            **record, "Status": "included", "Reason": "",
            "Coverage (%)": cov_pct, "Internal deletions (%)": del_pct,
            "Substitutions": sum(e["Type"] == SUBSTITUTION for e in seq_events),
            "Insertions": sum(e["Type"] == INSERTION for e in seq_events),
            "Deletions": sum(e["Type"] == DELETION for e in seq_events),
            "Variants": ", ".join(e["Mutation"] for e in seq_events),
        })
        if row % 50 == 0:
            report(progress, row / n_rows, f"Called variants for {row} of {n_rows} sequences")

    events_df = pd.DataFrame(events, columns=EVENT_COLUMNS + ["_group"])
    summary = _summarise(events_df, coverage, unique_coverage)
    per_sequence = pd.DataFrame(per_seq)
    report(progress, 1.0, "Variant calling complete")
    return VariantCalls(
        events=events_df.drop(columns="_group"),
        summary=summary,
        per_sequence=per_sequence,
        coverage=coverage,
        unique_coverage=unique_coverage,
        reference=ref_seq,
        params={"deletion_threshold": deletion_threshold, "min_coverage": min_coverage,
                "skip_ambiguous": skip_ambiguous},
    )


def _covering(cov: np.ndarray, start: int, end: int) -> int:
    # Substitutions: start == end. Deletions span start..end. Insertions sit
    # between start and end = start + 1, so both flanks must be covered.
    return int(cov[start - 1:end].min())


def _summarise(events: pd.DataFrame, cov: np.ndarray, ucov: np.ndarray) -> pd.DataFrame:
    cols = ["Mutation", "HGVS", "Type", "Position", "End", "Ref", "Alt", "Count", "Covering",
            "Frequency", "Unique carriers", "Unique covering", "Unique frequency",
            "BLOSUM62", "Change", "Sequence IDs"]
    if events.empty:
        return pd.DataFrame(columns=cols)
    grouped = events.groupby(["Mutation", "HGVS", "Type", "Position", "End", "Ref", "Alt"], sort=False)
    blosum = substitution_matrices.load("BLOSUM62")
    rows = []
    for (mut, hgvs, typ, start, end, ref, alt), g in grouped:
        n_cov = max(_covering(cov, start, end), 1)
        n_ucov = max(_covering(ucov, start, end), 1)
        count = g["MSA ID"].nunique()  # MSA rows are unique; database IDs may repeat
        ucount = g["_group"].nunique()
        if typ == SUBSTITUTION:
            score = float(blosum[ref][alt]) if ref in blosum.alphabet and alt in blosum.alphabet else np.nan
            change = describe_change(ref, alt)
        else:
            score = np.nan
            change = f"{len(alt)} aa inserted" if typ == INSERTION else f"{len(ref)} aa deleted"
        rows.append({
            "Mutation": mut, "HGVS": hgvs, "Type": typ, "Position": int(start), "End": int(end),
            "Ref": ref, "Alt": alt, "Count": count, "Covering": n_cov, "Frequency": count / n_cov,
            "Unique carriers": ucount, "Unique covering": n_ucov, "Unique frequency": ucount / n_ucov,
            "BLOSUM62": score, "Change": change,
            "Sequence IDs": ";".join(sorted(g["Sequence ID"].unique())),
        })
    return (pd.DataFrame(rows, columns=cols)
            .sort_values(["Frequency", "Count", "Position"], ascending=[False, False, True])
            .reset_index(drop=True))


def describe_change(ref: str, alt: str) -> str:
    """Human-readable physico-chemical change, e.g. ``polar → hydrophobic``."""
    a, b = AA_CLASS.get(ref, "other"), AA_CLASS.get(alt, "other")
    text = f"{a} → {b}" if a != b else f"within {a}"
    dq = _CHARGE.get(alt, 0) - _CHARGE.get(ref, 0)
    if dq:
        text += f", charge {dq:+d}"
    if "P" in (ref, alt) and ref != alt:
        text += ", proline change"
    if "C" in (ref, alt) and ref != alt:
        text += ", cysteine change"
    return text
