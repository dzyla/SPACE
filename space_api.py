#!/usr/bin/env python3
"""
space_api.py — A headless Python API for SPACE (Sequence Protein Alignment and
Conservation Engine).

This module lets you run the core SPACE pipeline from plain Python, a Jupyter
notebook, a scheduling script, or a REST service — WITHOUT the Streamlit web
UI. It installs a Streamlit stub (``headless_streamlit``) so the ``space.*``
modules can be imported and run headlessly, then exposes a clean, typed API
around the computational building blocks.

Requirements
------------
* Python 3.12 with the dependencies in ``requirements.txt`` (see README).
* Run from the SPACE repo root (so ``space/`` is importable) OR add the repo
  root to ``sys.path``.

Quick start
-----------
    import space_api as sa
    result = sa.run_pipeline(
        query="Hendra henipavirus F",
        email="dzyla@lji.org",
        data_source="UniProt",
        max_seqs=50,
        out_dir="runs/hendra_f",
    )
    print(result["al2co_df"].head())
    print(result["mutations_df"].head())

The pipeline returns a dict with the standard SPACE outputs (sequences, MSA
files, alignment, al2co scores, point mutations, conservation table,
phylogenetic tree, and optional PDB structural mapping). Individual steps are
also exposed so you can compose a custom workflow.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Install the headless Streamlit stub BEFORE importing space.*
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import headless_streamlit  # noqa: E402
headless_streamlit.install()

from space.alignment import (  # noqa: E402
    analyze_alignment,
    filter_sequences,
    generate_phylogenetic_tree,
    perform_alignment,
    perform_msa_pyfamsa,
)
from space.analysis import (  # noqa: E402
    assign_unique_colors,
    get_mutation_dataframe,
    list_unique_point_mutations,
    parse_mutations,
    run_al2co,
)
from space.fetch import (  # noqa: E402
    get_protein_sequences,
    search_and_save_protein_ncbi,
    search_and_save_protein_uniprot,
)
from space.pdb_processing import process_pdb_chain  # noqa: E402
from space.utils import clean_fasta  # noqa: E402
from space.visualization import (  # noqa: E402
    visualize_al2co_plotly,
    visualize_logo_and_consensus,
)

# Access the (stubbed) session_state so we can seed/read pipeline state.
_session = headless_streamlit.stub.session_state


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_sequences(
    query: str,
    email: str,
    data_source: str = "UniProt",
    max_seqs: int = 100,
    use_refseq: bool = False,
    out_dir: str = ".",
) -> str:
    """Fetch protein sequences for a query and return the FASTA path.

    Args:
        query: Search term (e.g. ``"Hendra henipavirus F"``).
        email: Email for NCBI Entrez (required when ``data_source=="NCBI"``).
        data_source: ``"NCBI"`` or ``"UniProt"``.
        max_seqs: Max sequences (-1 = all for NCBI; UniProt caps at 500).
        use_refseq: Only used for NCBI. Restrict to RefSeq.
        out_dir: Directory to write ``sequences.fasta``.
    """
    os.makedirs(out_dir, exist_ok=True)
    fasta_path = os.path.join(out_dir, "sequences.fasta")
    if data_source.lower() == "ncbi":
        return search_and_save_protein_ncbi(
            query, fasta_path, email, max_seqs=max_seqs, use_refseq=use_refseq
        )
    return search_and_save_protein_uniprot(query, fasta_path, max_seqs=max_seqs)


def _pick_reference(proteins_list, reference_seq: Optional[str]) -> str:
    """Choose / clean the reference sequence.

    If ``reference_seq`` is None, use the longest protein in the list.
    """
    if reference_seq:
        cleaned = clean_fasta(reference_seq)
        if cleaned:
            return cleaned.upper()
    # Fall back to the longest sequence
    if not proteins_list:
        raise ValueError("No proteins available to choose a reference sequence.")
    best = max(proteins_list, key=lambda r: len(r.seq))
    return str(best.seq).upper()


def run_pipeline(
    query: Optional[str] = None,
    email: str = "dzyla@lji.org",
    data_source: str = "UniProt",
    max_seqs: int = 100,
    use_refseq: bool = False,
    remove_pdb: bool = False,
    fasta_path: Optional[str] = None,
    reference_seq: Optional[str] = None,
    filtering_score: float = 10.0,
    sequence_len_min: int = 0,
    sequence_len_max: int = 10**9,
    max_threads: int = 8,
    deletion_threshold: float = 10.0,
    exclude_mutations_with_X: bool = True,
    out_dir: str = ".",
    map_to_structure: bool = False,
    uniprot_id: Optional[str] = None,
    sequence_score: float = 0.9,
) -> Dict:
    """Run the full SPACE pipeline and return a dict of results.

    Either provide ``query`` (to fetch sequences) OR ``fasta_path`` (to use an
    existing FASTA file). All other parameters have sensible defaults.

    Returns a dict with keys:
        fasta_path, proteins_list, reference_seq, result, filtered_seqs,
        msa_infile, msa_outfile, aln_file, al2co_df, conservation_df,
        mutations_df, unique_mutations, all_mutations_str, excluded_sequences,
        tree_file, pdb_result (if map_to_structure), out_dir
    """
    out_dir = str(Path(out_dir))
    os.makedirs(out_dir, exist_ok=True)

    # ---- Step 0: get sequences ----
    if fasta_path is None:
        if not query:
            raise ValueError("Provide either 'query' or 'fasta_path'.")
        fasta_path = fetch_sequences(
            query, email, data_source=data_source, max_seqs=max_seqs,
            use_refseq=use_refseq, out_dir=out_dir,
        )
    proteins_list = get_protein_sequences(fasta_path, remove_PDB=remove_pdb)
    _session["proteins_list"] = proteins_list

    # ---- Step 1: pairwise alignment vs reference ----
    ref_seq = _pick_reference(proteins_list, reference_seq)
    result, _ref = perform_alignment(proteins_list, ref_seq)
    if result is None:
        raise RuntimeError("Pairwise alignment failed (no valid sequences).")
    _session["result"] = result
    _session["reference_seq"] = _ref

    # ---- Step 2: filter ----
    high_score_seqs, id_array_selected, scores_final = filter_sequences(
        result,
        filtering_score=filtering_score,
        sequence_len_min=sequence_len_min,
        sequence_len_max=sequence_len_max,
    )

    # ---- Step 3: MSA (pyfamsa) ----
    msa_dir = os.path.join(out_dir, "msa")
    msa_infile = os.path.join(msa_dir, "msa_in.fasta")
    msa_outfile = os.path.join(msa_dir, "msa_out.fasta")
    msa_path = perform_msa_pyfamsa(
        high_score_seqs, id_array_selected, _ref,
        msa_infile, msa_outfile, threads=max_threads,
    )
    if msa_path is None:
        raise RuntimeError("MSA failed.")
    aln_file = msa_path.replace(".fasta", ".aln")

    # ---- Step 4: conservation + al2co ----
    conservation_df = analyze_alignment(aln_file, _ref)
    al2co_df = run_al2co(aln_file)
    _session["al2co_df"] = al2co_df
    _session["alignment_mapping"] = _session.get("alignment_mapping")

    # ---- Step 5: point mutations ----
    unique_mutations, excluded_sequences, excluded_count, mutation_summary, all_str = \
        list_unique_point_mutations(
            aln_file,
            "reference_sequence",
            _session.get("alignment_mapping"),
            deletion_threshold=deletion_threshold,
            exclude_mutations_with_X=exclude_mutations_with_X,
        )
    mutations_df = parse_mutations(unique_mutations)
    _session["unique_mutations"] = unique_mutations

    # ---- Step 6: phylogenetic tree ----
    tree_dir = os.path.join(out_dir, "tree")
    n_seqs = len(result[0]) if result else 0
    tree_path = None
    if n_seqs <= 200:
        tree_path = generate_phylogenetic_tree(msa_path, folder=tree_dir)

    # ---- Step 7 (optional): structural mapping ----
    pdb_result = None
    if map_to_structure:
        pdb_result = map_structure(
            al2co_df=al2co_df,
            reference_seq=_ref,
            uniprot_id=uniprot_id,
            sequence_score=sequence_score,
            out_dir=os.path.join(out_dir, "pdb"),
        )

    return {
        "fasta_path": fasta_path,
        "proteins_list": proteins_list,
        "reference_seq": _ref,
        "result": result,
        "filtered_seqs": high_score_seqs,
        "msa_infile": msa_infile,
        "msa_outfile": msa_path,
        "aln_file": aln_file,
        "al2co_df": al2co_df,
        "conservation_df": conservation_df,
        "mutations_df": mutations_df,
        "unique_mutations": unique_mutations,
        "all_mutations_str": all_str,
        "excluded_sequences": excluded_sequences,
        "tree_file": tree_path,
        "pdb_result": pdb_result,
        "out_dir": out_dir,
    }


def map_structure(
    al2co_df,
    reference_seq: str,
    uniprot_id: Optional[str] = None,
    own_pdb: Optional[str] = None,
    sequence_score: float = 0.9,
    out_dir: str = ".",
) -> Dict:
    """Map AL2CO conservation scores onto a 3D structure (PDB / AlphaFold).

    Args:
        al2co_df: DataFrame with a ``al2co_score`` column (from run_al2co).
        reference_seq: The reference protein sequence.
        uniprot_id: If given, fetch the AlphaFold structure for this UniProt ID.
        own_pdb: If given, use this local PDB file instead.
        sequence_score: Identity cutoff for RCSB sequence search (0-1).
        out_dir: Where to write labeled structures.
    """
    return process_pdb_chain(
        seq=reference_seq,
        al2co_score=al2co_df,
        sequence_score=sequence_score,
        uniprot_id=uniprot_id,
        own_pdb=own_pdb,
        save_dir=out_dir,
    )


def write_outputs(result: Dict, out_dir: Optional[str] = None) -> List[str]:
    """Write pipeline outputs (tables, plots, trees) to disk.

    Returns the list of written file paths.
    """
    out_dir = str(Path(out_dir or result.get("out_dir", ".")))
    written: List[str] = []

    if result.get("mutations_df") is not None and not result["mutations_df"].empty:
        p = os.path.join(out_dir, "tables", "point_mutations.tsv")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        result["mutations_df"].to_csv(p, sep="\t", index=False)
        written.append(p)

    if result.get("al2co_df") is not None:
        p = os.path.join(out_dir, "tables", "al2co_scores.tsv")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        result["al2co_df"].to_csv(p, sep="\t", index=False)
        written.append(p)
        # HTML plot
        try:
            visualize_al2co_plotly(result["al2co_df"], os.path.join(out_dir, "al2co"))
            written.append(os.path.join(out_dir, "al2co", "al2co_plot.html"))
        except Exception as e:  # noqa: BLE001
            print(f"  [space_api] al2co plot failed: {e}", file=sys.stderr)

    if result.get("msa_outfile") and result.get("conservation_df") is not None \
            and result.get("alignment_mapping_stub") is not None:
        pass  # logo requires alignment mapping; handled by caller if needed

    return written


# Convenience alias
save_results = write_outputs


if __name__ == "__main__":
    # Self-test: run a tiny pipeline on a small FASTA with no network.
    import tempfile
    import pandas as pd
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    from Bio import SeqIO

    tmp = tempfile.mkdtemp(prefix="space_api_test_")
    fasta = os.path.join(tmp, "tiny.fasta")
    seqs = [
        SeqRecord(Seq("ACDEFGHIKLMNPQRSTVWY"), id="ref"),
        SeqRecord(Seq("ACDEFGHIKLMNPQRSTVWY"), id="s1"),
        SeqRecord(Seq("ACDEFGHIKLMNPQSTVWY"), id="s2"),
        SeqRecord(Seq("ACDEFGHIKLMNPQRSTVWE"), id="s3"),
        SeqRecord(Seq("ACD-EFGHIKLMNPQRSTVWY"), id="s4"),
        SeqRecord(Seq("ACDEFGHIKLMNPQRSTVWF"), id="s5"),
    ]
    with open(fasta, "w") as f:
        SeqIO.write(seqs, f, "fasta")

    res = run_pipeline(
        fasta_path=fasta,
        reference_seq="ACDEFGHIKLMNPQRSTVWY",
        out_dir=tmp,
    )
    print("=== space_api self-test ===")
    print("reference_seq:", res["reference_seq"])
    print("al2co_df:\n", res["al2co_df"].to_string(index=False))
    print("mutations_df:\n", res["mutations_df"].to_string(index=False))
    print("tree_file:", res["tree_file"])
    assert res["al2co_df"] is not None and len(res["al2co_df"]) > 0
    print("OK: space_api self-test passed")