#!/usr/bin/env python3
"""
space_api.py — Python API for SPACE (Sequence Protein Alignment and Conservation Engine).

The computation lives in ``space.core`` and has no Streamlit dependency, so
this module is a thin, backward-compatible convenience layer for scripts and
notebooks.

Quick start
-----------
    import space_api as sa
    result = sa.run_pipeline(
        query='"Henipavirus hendraense"[Organism] AND fusion[Title]',
        email="you@example.org",
        data_source="NCBI",
        max_seqs=200,
        out_dir="runs/hendra_f",
    )
    run = result["run"]                      # space.core.pipeline.SpaceRun
    print(run.variants.filter(min_frequency=0.01))   # the variant table
    print(result["mutations_df"].head())     # legacy per-sequence mutations

Lower-level building blocks are re-exported below; see ``space.core`` for the
full API.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from space.core import annotate  # noqa: E402,F401
from space.core.align import filter_hits, pairwise_to_reference, reference_mapping, run_msa  # noqa: E402,F401
from space.core.conservation import al2co_scores, column_profile, logo_and_consensus  # noqa: E402,F401
from space.core.pipeline import SpaceRun, analyse, choose_reference, list_runs  # noqa: E402,F401
from space.core.pipeline import run_pipeline as _run_pipeline  # noqa: E402
from space.core.sequences import clean_sequence, fetch_sequences as _fetch, read_fasta  # noqa: E402,F401
from space.core.structure import (download_alphafold, download_pdb, map_to_structure,  # noqa: E402,F401
                                  pdb_metadata, search_pdb)
from space.core.variants import VariantCalls, call_variants  # noqa: E402,F401


def fetch_sequences(query: str, email: str = "", data_source: str = "UniProt", max_seqs: int = 100,
                    use_refseq: bool = False, out_dir: str = ".") -> str:
    """Fetch sequences for ``query`` into ``<out_dir>/sequences.fasta`` and return the path."""
    os.makedirs(out_dir, exist_ok=True)
    return _fetch(query, os.path.join(out_dir, "sequences.fasta"), source=data_source, email=email,
                  max_seqs=max_seqs, refseq_only=use_refseq)


def legacy_mutations_df(variants: VariantCalls) -> pd.DataFrame:
    """Per-sequence mutation table in the column layout of SPACE ≤ 2025."""
    ev = variants.events
    if ev.empty:
        return pd.DataFrame(columns=["Sequence ID", "Original Residue", "Position", "Mutated Residue",
                                     "Mutation Type", "Mutation", "HGVS"])
    return pd.DataFrame({
        "Sequence ID": ev["Sequence ID"],
        "Original Residue": ev["Ref"],
        "Position": ev["Position"],
        "Mutated Residue": ev["Alt"],
        "Mutation Type": ev["Type"],
        "Mutation": ev["Mutation"],
        "HGVS": ev["HGVS"],
    })


def map_structure(al2co_df: pd.DataFrame, reference_seq: str, uniprot_id: Optional[str] = None,
                  own_pdb: Optional[str] = None, pdb_id: Optional[str] = None, sequence_score: float = 0.9,
                  out_dir: str = ".") -> Dict:
    """Map al2co scores onto a structure.

    Source priority: ``own_pdb`` > ``pdb_id`` > AlphaFold model for
    ``uniprot_id`` > best RCSB hit at ``sequence_score`` identity.
    Returns ``{"metadata", "selected_pdb", "chain_data", "mapping"}``.
    """
    os.makedirs(out_dir, exist_ok=True)
    if own_pdb:
        path, meta, sid, source = own_pdb, {"id": "custom", "title": "User-supplied structure"}, "custom", "Upload"
    elif pdb_id:
        path, meta, sid, source = download_pdb(pdb_id, out_dir), pdb_metadata(pdb_id), pdb_id, "PDB"
    elif uniprot_id:
        path, meta = download_alphafold(uniprot_id, out_dir)
        sid, source = uniprot_id, "AlphaFold"
    else:
        hits = search_pdb(reference_seq, identity_cutoff=sequence_score)
        if not hits:
            raise ValueError("No PDB entries at this identity; pass uniprot_id for AlphaFold.")
        sid, source = hits[0], "PDB"
        path, meta = download_pdb(sid, out_dir), pdb_metadata(sid)
    al = al2co_df.dropna(subset=["Location"])
    values = dict(zip(al["Location"].astype(int), al["al2co_score"]))
    sm = map_to_structure(path, reference_seq, values=values, out_dir=Path(out_dir) / "mapped",
                          source=source, structure_id=sid, metadata=meta)
    return {
        "metadata": sm.metadata,
        "selected_pdb": sid,
        "chain_data": {
            c: {"alignment_score": m.score, "matched_length": len(m.ref_to_residue),
                "identity": m.identity, "coverage": m.coverage, "alignment": m.alignment_text,
                "residue_score_map": {m.ref_to_residue[p][0]: v for p, v in values.items()
                                      if p in m.ref_to_residue},
                "updated_pdb_path": m.pdb_path}
            for c, m in sm.chains.items()
        },
        "mapping": sm,
    }


def run_pipeline(
    query: Optional[str] = None,
    email: str = "",
    data_source: str = "UniProt",
    max_seqs: int = 100,
    use_refseq: bool = False,
    remove_pdb: bool = False,
    fasta_path: Optional[str] = None,
    reference_seq: Optional[str] = None,
    reference_id: Optional[str] = None,
    filtering_score: float = 10.0,
    sequence_len_min: int = 0,
    sequence_len_max: int = 10**9,
    min_coverage: float = 0.0,
    max_threads: int = 0,
    deletion_threshold: float = 10.0,
    exclude_mutations_with_X: bool = True,
    out_dir: Optional[str] = None,
    map_to_structure: bool = False,
    uniprot_id: Optional[str] = None,
    sequence_score: float = 0.9,
    build_tree: bool = True,
) -> Dict:
    """Run the full pipeline; returns a dict (legacy keys + ``run``).

    Keys: run, fasta_path, reference_seq, reference_id, hits, msa_infile,
    msa_outfile, aln_file, al2co_df, conservation_df, variants_df,
    mutations_df, unique_mutations, all_mutations_str, excluded_sequences,
    tree_file, pdb_result, out_dir.

    If no reference is given, a curated (Swiss-Prot/RefSeq) entry or the most
    common full-length sequence is used (previously: the longest sequence).
    """
    run = _run_pipeline(
        query=query, fasta_path=fasta_path, source=data_source, email=email, max_seqs=max_seqs,
        refseq_only=use_refseq, remove_pdb=remove_pdb, reference=reference_seq, reference_id=reference_id,
        min_identity=filtering_score, min_length=sequence_len_min, max_length=sequence_len_max,
        min_coverage=min_coverage, deletion_threshold=deletion_threshold,
        skip_ambiguous=exclude_mutations_with_X, build_phylo=build_tree, threads=max_threads,
        run_dir=out_dir,
    )
    pdb_result = None
    if map_to_structure:
        pdb_result = map_structure(
            run.al2co, run.reference, uniprot_id=uniprot_id, sequence_score=sequence_score,
            out_dir=str(run.run_dir / "pdb"))
    v = run.variants
    return {
        "run": run,
        "fasta_path": run.files.get("sequences"),
        "reference_seq": run.reference,
        "reference_id": run.reference_id,
        "hits": run.hits,
        "msa_infile": run.files.get("msa_in"),
        "msa_outfile": run.files.get("msa_fasta"),
        "aln_file": run.files.get("msa_clustal"),
        "al2co_df": run.al2co,
        "conservation_df": run.profile,
        "variants_df": v.summary,
        "mutations_df": legacy_mutations_df(v),
        "unique_mutations": v.unique_mutations,
        "all_mutations_str": v.all_mutations_str,
        "excluded_sequences": v.excluded,
        "tree_file": run.tree.newick_path if run.tree else None,
        "pdb_result": pdb_result,
        "out_dir": str(run.run_dir),
    }


def write_outputs(result: Dict, out_dir: Optional[str] = None) -> List[str]:
    """Tables are written by the pipeline itself; this returns their paths."""
    run: SpaceRun = result["run"]
    run.save()
    return [p for k, p in run.files.items() if p and k in
            ("variants", "variant_events", "sequences", "positions", "hits", "al2co")]


save_results = write_outputs


if __name__ == "__main__":
    import tempfile

    from Bio import SeqIO
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord

    tmp = tempfile.mkdtemp(prefix="space_api_test_")
    fasta = os.path.join(tmp, "tiny.fasta")
    ref = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
    seqs = [ref, ref, ref[:5] + "G" + ref[6:], ref[:10] + "WW" + ref[10:], ref[:20] + ref[22:]]
    SeqIO.write([SeqRecord(Seq(s), id=f"s{i}", description="") for i, s in enumerate(seqs)], fasta, "fasta")
    res = run_pipeline(fasta_path=fasta, reference_seq=ref, out_dir=tmp, deletion_threshold=20)
    print(res["variants_df"][["Mutation", "HGVS", "Count", "Covering", "Frequency"]].to_string(index=False))
    assert {"I6G", "R10_Q11insWW"} <= set(res["variants_df"]["Mutation"])
    print("OK: space_api self-test passed")
