# alignment.py

from __future__ import annotations

import logging
import os
import random
import re
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Sequence as _Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from Bio import AlignIO, Phylo, SeqIO
from Bio.Align import MultipleSeqAlignment, PairwiseAligner, substitution_matrices
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from pyfamsa import Aligner as FamsaAligner, Sequence as FamsaSequence

from .utils import clean_fasta


# ---------------------------------------------------------------------------
# Pairwise alignment (global BLOSUM62), identity-based scoring
# ---------------------------------------------------------------------------

_pairwise_aligner: Optional[PairwiseAligner] = None


def _get_aligner() -> PairwiseAligner:
    """
    Lazily create a process-local global aligner. Safe with ProcessPool on all OSes.
    """
    global _pairwise_aligner
    if _pairwise_aligner is None:
        al = PairwiseAligner()
        al.substitution_matrix = substitution_matrices.load("BLOSUM62")
        al.mode = "global"
        al.match_score = 2
        al.mismatch_score = -1
        al.open_gap_score = -0.5
        al.extend_gap_score = -0.3
        _pairwise_aligner = al
    return _pairwise_aligner


def _align_one(args: Tuple[str, str, str]) -> Tuple[Optional[float], str, str]:
    """
    Align a single sequence to the reference and return identity %.
    Returns (score_or_None, seq_str, seq_id).
    """
    seq_str, seq_id, reference_seq = args
    try:
        aligner = _get_aligner()
        # Biopython returns an Alignment object; we take the first best alignment
        aln = next(iter(aligner.align(reference_seq, seq_str)), None)
        if aln is None:
            return None, seq_str, seq_id

        # Convert Alignment to aligned strings
        s1 = aln[0]
        s2 = aln[1]
        # Identity over aligned length (including gaps). This matches prior UI behavior.
        matches = sum(a == b for a, b in zip(s1, s2))
        identity_pct = (matches / len(s1)) * 100.0 if len(s1) else 0.0
        return identity_pct, seq_str, seq_id

    except Exception:
        # We keep the original seq info for logging/diagnostics
        return None, seq_str, seq_id


def perform_alignment(
    proteins_list: _Sequence[SeqRecord], reference_seq: str
) -> Tuple[List[List], str]:
    """
    Align all sequences against reference_seq and compute identity % scores.

    Returns:
        ([scores, sequences, id_list], reference_seq)
    """
    if not proteins_list:
        st.error("Protein list is empty.")
        return None, None

    seq_data = [
        (str(rec.seq).upper(), rec.id, reference_seq.upper())
        for rec in proteins_list
        if rec and rec.seq and len(rec.seq) > 0
    ]
    if not seq_data:
        st.error("No valid sequences to align.")
        return None, None

    # Sensible worker cap to avoid oversubscription
    import os as _os
    max_workers = min(max(1, (_os.cpu_count() or 4) - 1), 8)

    scores: List[float] = []
    sequences: List[str] = []
    id_list: List[str] = []

    min_score = float("inf")
    max_score = float("-inf")
    min_seq_length = float("inf")
    max_seq_length = float("-inf")

    progress = st.progress(0.0)
    total = len(seq_data)
    done = 0
    alignment_errors: List[Tuple[str, int]] = []

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_align_one, item) for item in seq_data]
            for fut in as_completed(futures):
                score, seq_str, seq_id = fut.result()
                if score is not None:
                    scores.append(score)
                    sequences.append(seq_str)
                    id_list.append(seq_id)

                    min_score = min(min_score, score)
                    max_score = max(max_score, score)
                    L = len(seq_str)
                    min_seq_length = min(min_seq_length, L)
                    max_seq_length = max(max_seq_length, L)
                else:
                    alignment_errors.append((seq_id, len(seq_str)))

                done += 1
                progress.progress(done / total)
    except Exception as e:
        st.error(f"An error occurred during alignment: {e}")
        st.text(traceback.format_exc())
        return None, None
    finally:
        progress.empty()

    if alignment_errors:
        st.warning(f"Failed to align {len(alignment_errors)} sequences.")

    if not scores:
        st.error("No successful alignments were performed.")
        return None, None

    # Persist range stats for downstream sliders
    st.session_state.min_score = min_score
    st.session_state.max_score = max_score
    st.session_state.min_seq_length = min_seq_length
    st.session_state.max_seq_length = max_seq_length

    return [scores, sequences, id_list], reference_seq.upper()


# ---------------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------------

def _fix_id(seq_id: str) -> str:
    return re.sub(r"[^\w\-]", "_", seq_id)


def filter_sequences(
    result: List[List],
    filtering_score: float = 10.0,
    score_max: float = float("inf"),
    sequence_len_min: int = 0,
    sequence_len_max: int = 10**9,
) -> Tuple[List[str], List[str], List[float]]:
    """
    Filter sequences by identity score range and length.
    """
    if not result:
        st.error("No alignment results to filter.")
        return [], [], []

    scores, sequences, ids = result

    high_score_seqs: List[str] = []
    id_array_selected: List[str] = []
    scores_final: List[float] = []

    for sc, seq, sid in zip(scores, sequences, ids):
        if filtering_score <= sc <= score_max:
            L = len(seq)
            if sequence_len_min <= L <= sequence_len_max:
                id_array_selected.append(_fix_id(sid))
                high_score_seqs.append(seq)
                scores_final.append(sc)

    return high_score_seqs, id_array_selected, scores_final


# ---------------------------------------------------------------------------
# MSA with pyfamsa + CLUSTAL export
# ---------------------------------------------------------------------------

def perform_msa_pyfamsa(
    high_score_seqs: List[str],
    id_array_selected: List[str],
    reference_seq: str,
    msa_infile: str,
    msa_outfile: str,
    threads: int = 8,
) -> Optional[str]:
    """
    Run pyfamsa MSA on filtered sequences + reference, write:
      - FASTA alignment to `msa_outfile`
      - CLUSTAL alignment to `msa_outfile.replace('.fasta', '.aln')`
    Returns `msa_outfile` on success, else None.
    """
    try:
        # Ensure parent dirs exist
        Path(msa_infile).parent.mkdir(parents=True, exist_ok=True)
        Path(msa_outfile).parent.mkdir(parents=True, exist_ok=True)

        # Append the reference sequence explicitly
        final_ids = list(id_array_selected) + ["reference_sequence"]
        final_seqs = list(high_score_seqs) + [reference_seq]

        # Pre-clean and build FAMSA sequences
        famsa_inputs = [
            FamsaSequence(_fix_id(name).encode(), clean_fasta(seq).encode())
            for name, seq in zip(final_ids, final_seqs)
        ]

        # Also save the input as FASTA (helpful for debugging and reproducibility)
        input_records = [
            SeqRecord(Seq(s.sequence.decode()), id=s.id.decode(), description="")
            for s in famsa_inputs
        ]
        SeqIO.write(input_records, msa_infile, "fasta")

        with st.spinner("Running pyfamsa alignment..."):
            famsa = FamsaAligner(guide_tree="upgma", threads=max(1, int(threads)))
            aligned = famsa.align(famsa_inputs)

        # Write FASTA alignment
        aligned_records = [
            SeqRecord(Seq(s.sequence.decode()), id=s.id.decode(), description="")
            for s in aligned
        ]
        SeqIO.write(aligned_records, msa_outfile, "fasta")

        # Export CLUSTAL .aln next to it
        aln_path = msa_outfile.replace(".fasta", ".aln")
        _write_clustal(msa_outfile, aln_path)

        return msa_outfile

    except Exception as e:
        st.error(f"An unexpected error occurred during MSA (pyfamsa): {e}")
        st.text(traceback.format_exc())
        return None


def _write_clustal(fasta_alignment: str, clustal_out: str) -> None:
    """
    Convert a FASTA alignment to CLUSTAL format and ensure a classic header line.
    """
    try:
        count = SeqIO.convert(fasta_alignment, "fasta", clustal_out, "clustal")
        # Ensure standard header line expected by some viewers/parsers
        with open(clustal_out, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if lines:
            lines[0] = "CLUSTAL W multiple sequence alignment\n\n"
        with open(clustal_out, "w", encoding="utf-8") as f:
            f.writelines(lines)
        st.write(f"Converted {count} records to CLUSTAL format.")
    except Exception as e:
        st.warning(f"Could not create CLUSTAL file: {e}")


# ---------------------------------------------------------------------------
# Alignment analysis / conservation (used later by AL2CO and plotting)
# ---------------------------------------------------------------------------

def analyze_alignment(alignment_file: str, reference_seq: str) -> pd.DataFrame:
    """
    Read a CLUSTAL alignment (*.aln) and compute simple per-column conservation
    relative to the 'reference_sequence'. Also stores a residue index mapping in
    st.session_state.alignment_mapping (alignment-index -> reference-residue-number).
    """
    try:
        alignment = AlignIO.read(alignment_file, "clustal")
    except Exception as e:
        st.error(f"An error occurred while reading the alignment file: {e}")
        raise

    # Locate the reference record
    ref_rec = next((rec for rec in alignment if rec.id == "reference_sequence"), None)
    if ref_rec is None:
        st.error("Reference sequence not found in the alignment.")
        raise ValueError("Reference sequence missing in alignment.")

    aln_len = alignment.get_alignment_length()
    ref_seq_str = str(ref_rec.seq)

    # Map each alignment column to a reference residue number (or None for gaps)
    mapping: List[Optional[int]] = []
    ref_num = 0
    for aa in ref_seq_str:
        if aa != "-":
            ref_num += 1
            mapping.append(ref_num)
        else:
            mapping.append(None)

    # Simple conservation score: frequency of the modal residue (ignoring gaps)
    conservation_rows = []
    for i in range(aln_len):
        mapped_ref = mapping[i]
        residue_num = np.nan if mapped_ref is None else mapped_ref

        column = alignment[:, i]
        residues = Counter(column)
        residues.pop("-", None)
        if residues:
            most_common = residues.most_common(1)[0][1]
            conservation = most_common / len(alignment)
        else:
            conservation = 0.0

        conservation_rows.append({"Location": residue_num, "Conservation": conservation})

    st.session_state.alignment_mapping = mapping
    return pd.DataFrame(conservation_rows)


# ---------------------------------------------------------------------------
# Phylogenetic tree utilities
# ---------------------------------------------------------------------------# ---------------------------------------------------------------------------
# Phylogenetic tree utilities
# ---------------------------------------------------------------------------
# --- small, private helpers (no name conflicts) ---
def _trim_alignment(
    alignment: MultipleSeqAlignment,
    gap_frac_threshold: float = 0.95,
    drop_fully_conserved: bool = True,
) -> MultipleSeqAlignment:
    """Remove gap-heavy and (optionally) fully conserved columns to speed up."""
    n = len(alignment)
    L = alignment.get_alignment_length()
    if n == 0 or L == 0:
        return alignment

    arr = np.array([list(str(rec.seq)) for rec in alignment], dtype="U1")  # (n, L)
    gap = (arr == "-")

    keep = np.ones(L, dtype=bool)
    keep &= (gap.mean(axis=0) < gap_frac_threshold)

    if drop_fully_conserved:
        for j in np.where(keep)[0]:
            col = arr[:, j]
            non_gap = col[col != "-"]
            if non_gap.size <= 1:
                keep[j] = False
            elif np.all(non_gap == non_gap[0]):
                keep[j] = False

    kept_idx = np.where(keep)[0]
    if kept_idx.size == 0:
        kept_idx = np.array([0])

    new_records = []
    for rec in alignment:
        seq = "".join(np.array(list(str(rec.seq)))[kept_idx])
        r = rec[:]  # shallow copy
        r.seq = rec.seq.__class__(seq)
        new_records.append(r)
    return MultipleSeqAlignment(new_records)


def _identity_condensed_distance(aln: MultipleSeqAlignment) -> np.ndarray:
    """Condensed pairwise distance: proportion of mismatches on non-gap positions."""
    n = len(aln)
    arr = np.array([list(str(rec.seq)) for rec in aln], dtype="U1")
    gap = (arr == "-")

    m = n * (n - 1) // 2
    dist = np.empty(m, dtype=float)

    def cidx(i, j):  # index into condensed form
        return i * n - (i * (i + 1)) // 2 + (j - i - 1)

    for i in range(n - 1):
        ai = arr[i]
        gi = gap[i]
        for j in range(i + 1, n):
            valid = ~(gi | gap[j])
            denom = int(np.sum(valid))
            d = 1.0 if denom == 0 else float(np.mean(ai[valid] != arr[j, valid]))
            dist[cidx(i, j)] = d
    return dist


def _build_upgma_figure(Z, labels: List[str], gl_threshold: int = 300) -> go.Figure:
    """Create a Plotly dendrogram from SciPy linkage (Scattergl for large N)."""
    from scipy.cluster.hierarchy import dendrogram  # local import to avoid global deps

    dn = dendrogram(Z, labels=labels, no_plot=True)
    icoord, dcoord, leaf_labels = dn["icoord"], dn["dcoord"], dn["ivl"]

    edge_x, edge_y = [], []
    for xs, ys in zip(icoord, dcoord):
        edge_x += [xs[0], xs[1], None, xs[2], xs[3], None]
        edge_y += [ys[0], ys[1], None, ys[2], ys[3], None]

    trace_cls = go.Scattergl if len(leaf_labels) >= gl_threshold else go.Scatter
    fig = go.Figure()
    fig.add_trace(trace_cls(x=edge_x, y=edge_y, mode="lines", line=dict(width=1), showlegend=False))
    leaf_x = list(range(1, len(leaf_labels) + 1))
    leaf_y = [0] * len(leaf_labels)
    fig.add_trace(
        trace_cls(
            x=leaf_x, y=leaf_y, mode="markers+text",
            text=leaf_labels, textposition="top center",
            marker=dict(size=5), showlegend=False,
            hovertext=leaf_labels, hoverinfo="text",
        )
    )
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(title="Distance", showgrid=True, zeroline=False),
        plot_bgcolor="white",
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


def generate_phylogenetic_tree(
    msa_file: str, use_random_subsample: bool = False, folder: str = "."
) -> Optional[str]:
    """
    N ≤ 200  -> Biopython NJ, save Newick (.nwk), return its path (backward compatible).
    N > 200  -> Trim columns, (optional) subsample to 200, UPGMA + Plotly HTML, return .html.
    """
    logger = logging.getLogger(__name__)
    try:
        Path(folder).mkdir(parents=True, exist_ok=True)

        aln = AlignIO.read(msa_file, "fasta")
        n = len(aln)
        if n == 0:
            raise ValueError("Alignment contains no sequences.")

        if n <= 200:
            # Original behavior (fast enough for small N)
            calculator = DistanceCalculator("identity")
            dm = calculator.get_distance(aln)
            constructor = DistanceTreeConstructor()
            tree = constructor.nj(dm)

            tree_path = os.path.join(folder, "phylogenetic_tree.nwk")
            Phylo.write(tree, tree_path, "newick")
            return tree_path

        # Large-N path
        aln = _trim_alignment(aln, gap_frac_threshold=0.95, drop_fully_conserved=True)

        if use_random_subsample and len(aln) > 200:
            rng = np.random.default_rng(0)
            # numpy returns numpy.int64 values; coerce to plain Python int for Biopython indexing
            idx_np = rng.choice(len(aln), size=200, replace=False)
            idx = [int(i) for i in (idx_np.tolist() if hasattr(idx_np, "tolist") else idx_np)]
            aln = MultipleSeqAlignment([aln[int(i)] for i in idx])

        dvec = _identity_condensed_distance(aln)

        # Local import avoids impacting the rest of your module
        from scipy.cluster.hierarchy import linkage
        Z = linkage(dvec, method="average")  # UPGMA

        labels = [rec.id for rec in aln]
        fig = _build_upgma_figure(Z, labels, gl_threshold=300)
        html_path = os.path.join(folder, "phylogenetic_tree.html")
        fig.write_html(html_path)
        return html_path

    except Exception as e:
        st.error(f"An error occurred while generating the phylogenetic tree: {e}")
        logger.exception("Tree generation failed")
        return None


def plot_phylogenetic_tree(tree_file: str, st_column: Optional = None):
    """
    If given an HTML (fast path), embed it.
    If given a Newick (small N path), draw a Plotly figure and save HTML next to it.
    """
    try:
        if tree_file.lower().endswith(".html"):
            html = Path(tree_file).read_text(encoding="utf-8")
            (st_column or st).html(html)
            return

        # Legacy path: Newick → manual Plotly drawing (kept from your original approach)
        tree = Phylo.read(tree_file, "newick")

        pos = {}
        y_counter = [0]

        def assign_positions(clade, x_current):
            bl = clade.branch_length or 0
            x = x_current + bl
            if clade.is_terminal():
                y = y_counter[0]
                pos[clade] = (x, y)
                y_counter[0] += 1
            else:
                child_ys = []
                for child in clade.clades:
                    assign_positions(child, x)
                    child_ys.append(pos[child][1])
                y = sum(child_ys) / max(1, len(child_ys))
                pos[clade] = (x, y)

        assign_positions(tree.root, 0)

        edge_x, edge_y = [], []

        def add_edges(clade):
            x0, y0 = pos[clade]
            for child in clade.clades:
                x1, y1 = pos[child]
                edge_x.extend([x0, x0, None]); edge_y.extend([y0, y1, None])
                edge_x.extend([x0, x1, None]); edge_y.extend([y1, y1, None])
                add_edges(child)

        add_edges(tree.root)

        node_x, node_y, node_labels = [], [], []

        def collect_nodes(clade):
            x, y = pos[clade]
            node_x.append(x); node_y.append(y)
            node_labels.append(clade.name if clade.is_terminal() and clade.name else "")
            for child in clade.clades:
                collect_nodes(child)

        collect_nodes(tree.root)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(color="black", width=1), hoverinfo="none", showlegend=False))
        fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text", marker=dict(symbol="circle", size=6), text=node_labels, textposition="middle right", hoverinfo="text", showlegend=False))
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white", width=800, height=600,
        )

        # Save HTML next to the Newick
        html_out = os.path.join(os.path.dirname(tree_file), "phylogenetic_tree.html")
        fig.write_html(html_out)

        (st_column or st).plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred while plotting the phylogenetic tree: {e}")