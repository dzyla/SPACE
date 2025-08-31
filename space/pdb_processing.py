# pdb_processing.py

from __future__ import annotations

import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from Bio.Align import PairwiseAligner, substitution_matrices
from biopandas.pdb import PandasPdb

from .utils import clean_fasta, download_alphafold_pdb, get_protein_data

# --------------------------------------------------------------------
# Global pairwise aligner (BLOSUM62, global) for chain matching
# --------------------------------------------------------------------
_ALIGNER: Optional[PairwiseAligner] = None


def _get_aligner() -> PairwiseAligner:
    global _ALIGNER
    if _ALIGNER is None:
        al = PairwiseAligner()
        al.substitution_matrix = substitution_matrices.load("BLOSUM62")
        al.mode = "global"
        al.match_score = 2
        al.mismatch_score = -1
        al.open_gap_score = -0.5
        al.extend_gap_score = -0.3
        _ALIGNER = al
    return _ALIGNER


def _fix_id(seq_id: str) -> str:
    return re.sub(r"[^\w\-]", "_", seq_id)


# 3-letter -> 1-letter AA map (includes common non-standard fallbacks)
_AA3_TO_1: Dict[str, str] = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G",
    "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
    "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S", "THR": "T", "VAL": "V",
    "TRP": "W", "TYR": "Y",
    # Tolerant fallbacks:
    "SEC": "U", "PYL": "O", "ASX": "B", "GLX": "Z", "XLE": "J", "XAA": "X",
}


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# --------------------------------------------------------------------
# RCSB sequence search (replaces rcsbsearchapi)
# --------------------------------------------------------------------
def _rcsb_sequence_search(
    seq: str,
    identity_cutoff: float = 0.9,
    evalue_cutoff: float = 10.0,
    max_hits: int = 500,
    timeout: int = 25,
) -> List[str]:
    """
    Query RCSB Search v2 API for PDB entries matching 'seq' by sequence identity.

    Returns a list of PDB IDs (uppercase).
    """
    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    # Construct the official sequence query payload
    payload = {
        "query": {
            "type": "terminal",
            "service": "sequence",
            "parameters": {
                "evalue_cutoff": evalue_cutoff,
                "identity_cutoff": float(identity_cutoff),
                "sequence_type": "protein",
                "target": "pdb_protein_sequence",
                "value": seq,
            },
        },
        "request_options": {
            "return_all_hits": True,
            "results_content_type": ["experimental"],  # better to bias real PDB entries
            "sort": [{"sort_by": "score", "direction": "desc"}],
        },
        "return_type": "entry",
    }

    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        result_set = data.get("result_set", [])
        ids = [str(item.get("identifier", "")).upper() for item in result_set]
        # Deduplicate and limit
        ids = [i for i in ids if i]
        seen = set()
        uniq = []
        for i in ids:
            if i not in seen:
                uniq.append(i)
                seen.add(i)
        return uniq[:max_hits]
    except Exception as e:
        st.warning(f"RCSB sequence search failed: {e}")
        return []


def _download_rcsb_pdb(pdb_id: str, save_dir: str) -> str:
    """
    Download a PDB file from RCSB and save under save_dir as selected_<ID>.pdb
    """
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    _ensure_dir(save_dir)
    out = os.path.join(save_dir, f"selected_{pdb_id}.pdb")
    with open(out, "w", encoding="utf-8") as f:
        f.write(r.text)
    return out


def _fetch_pdb_metadata(pdb_id: str) -> Dict[str, str]:
    """
    Try RCSB data API first; on failure, try PDBe summary as a fallback.
    """
    # RCSB
    try:
        url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        j = r.json()
        title = j.get("struct", {}).get("title", "N/A")
        authors = ", ".join(j.get("citation", [{}])[0].get("rcsb_authors", []))
        date = j.get("rcsb_accession_info", {}).get("initial_deposition_date", "N/A")
        return {"pdb_id": pdb_id, "title": title, "authors": authors, "date": date}
    except Exception:
        pass

    # PDBe fallback
    try:
        url = f"https://www.ebi.ac.uk/pdbe/api/pdb/entry/summary/{pdb_id}"
        r = requests.get(url, timeout=15, headers={"Accept": "application/json"})
        r.raise_for_status()
        j = r.json()
        # PDBe returns { "<pdbid>": [ {...} ] }
        lst = j.get(pdb_id.lower()) or j.get(pdb_id.upper()) or []
        item = lst[0] if lst else {}
        title = item.get("title", "N/A")
        return {"pdb_id": pdb_id, "title": title, "authors": "N/A", "date": "N/A"}
    except Exception:
        return {"pdb_id": pdb_id, "title": "N/A", "authors": "N/A", "date": "N/A"}


def _extract_chain_sequence(atom_df: pd.DataFrame, chain_id: str) -> str:
    """
    Build a 1-letter sequence for a given chain from ATOM records.
    """
    chain_atoms = atom_df[atom_df["chain_id"] == chain_id]
    # order by residue number + insertion code for stable sequences
    chain_atoms = chain_atoms.sort_values(by=["residue_number", "insertion"])
    residues = chain_atoms["residue_number"].unique()

    seq_chars: List[str] = []
    for res_no in residues:
        row = chain_atoms[chain_atoms["residue_number"] == res_no].iloc[0]
        res3 = (row["residue_name"] or "").strip().upper()
        seq_chars.append(_AA3_TO_1.get(res3, "X"))
    return "".join(seq_chars)


def process_pdb_chain(
    seq: str,
    al2co_score: pd.DataFrame,
    sequence_score: float = 0.9,
    default_score: float = -1.0,
    frequency_threshold: float = 0.1,  # kept for UI logic elsewhere
    uniprot_id: Optional[str] = None,
    own_pdb: Optional[str] = None,
    save_dir: Optional[str] = None,
    st_column: Optional = None,
) -> dict:
    """
    Find/prepare a structure, align each chain to the reference seq, map AL2CO
    scores onto B-factors, and write per-chain PDBs into save_dir/pdb_processing.

    Returns:
        {
          "metadata": {pdb_id,title,authors,date},
          "selected_pdb": <id or 'custom_pdb'>,
          "chain_data": {
              chain_id: {
                  "alignment_score": float,
                  "alignment": str,                    # human-readable
                  "residue_score_map": Dict[int,float],
                  "updated_pdb_path": str              # absolute path
              }, ...
          }
        }
    """
    logger = logging.getLogger(__name__)
    if save_dir is None:
        save_dir = os.getcwd()
    _ensure_dir(save_dir)

    # Validate reference sequence: allow only standard amino acids
    if not re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+", seq):
        st.error("Invalid sequence. Only standard amino acids are supported.")
        raise ValueError("Invalid sequence. Only standard amino acids are supported.")

    # Decide on structure source
    pdb_filename: Optional[str] = None
    selected_pdb: str = "custom_pdb"
    metadata: Dict[str, str] = {"pdb_id": "N/A", "title": "N/A", "authors": "N/A", "date": "N/A"}

    try:
        if own_pdb:
            # Use uploaded file (already saved under session folder by app)
            pdb_filename = own_pdb
            selected_pdb = "custom_pdb"
            metadata = {
                "pdb_id": "Custom PDB",
                "title": "User-supplied structure",
                "authors": "User",
                "date": "N/A",
            }

        elif uniprot_id:
            # AlphaFold predicted structure (strip version if present, e.g., O89342.2 -> O89342)
            af_id = str(uniprot_id).split(".")[0]
            try:
                path = download_alphafold_pdb(af_id, save_dir=save_dir)
            except TypeError:
                # Backward compatibility if utils fn lacks save_dir
                path = download_alphafold_pdb(af_id)
                if path and os.path.exists(path):
                    new_path = os.path.join(save_dir, f"selected_{af_id}.pdb")
                    try:
                        os.replace(path, new_path)
                        path = new_path
                    except Exception:
                        import shutil
                        shutil.copy2(path, new_path)
                        path = new_path

            if not path or not os.path.exists(path):
                raise ValueError("Failed to download AlphaFold PDB file.")
            pdb_filename = path
            selected_pdb = af_id

            # Metadata from UniProt (optional, best-effort)
            try:
                meta = get_protein_data(af_id) or {}
                title = (
                    meta.get("proteinDescription", {})
                    .get("recommendedName", {})
                    .get("fullName", {})
                    .get("value", "AlphaFold model")
                )
            except Exception:
                title = "AlphaFold model"
            metadata = {
                "pdb_id": f"{af_id} (AlphaFold)",
                "title": title,
                "authors": "AlphaFold DB",
                "date": "N/A",
            }

        else:
            # RCSB Search v2 REST (sequence)
            if sequence_score is None:
                st.warning("No sequence identity threshold provided. Using default value 0.9.")
                identity_cutoff = 0.9
            else:
                identity_cutoff = sequence_score
            try:
                hits = _rcsb_sequence_search(seq, identity_cutoff=identity_cutoff)
            except Exception as e:
                st.error(f"An error occurred during PDB processing: {e}")
                raise
            if not hits:
                raise ValueError(
                    "No PDB entries found at the chosen identity threshold. Try AlphaFold instead."
                )

            # Let user choose a PDB entry
            selector = st_column if st_column is not None else st
            selected_pdb = selector.selectbox("Select a PDB ID", hits)

            # Metadata + structure
            metadata = _fetch_pdb_metadata(selected_pdb)
            pdb_filename = _download_rcsb_pdb(selected_pdb, save_dir=save_dir)

    except Exception:
        # Error already surfaced to UI above in most cases
        raise

    # --- Parse and process chains ---
    try:
        ppdb = PandasPdb().read_pdb(pdb_filename)
        atom_df = ppdb.df["ATOM"]
    except Exception as e:
        st.error(f"An error occurred while parsing the PDB file: {e}")
        raise

    chains = atom_df["chain_id"].dropna().unique()
    if chains.size == 0:
        raise ValueError("No chains found in the PDB file.")

    chain_scores: Dict[str, float] = {}
    chain_align_text: Dict[str, str] = {}
    chain_sequences: Dict[str, str] = {}

    aligner = _get_aligner()
    for chain in chains:
        seq_chain = _extract_chain_sequence(atom_df, chain)
        if not seq_chain:
            st.warning(f"Chain `{chain}` has no detectable sequence. Skipping.")
            continue

        chain_sequences[chain] = seq_chain
        aln = next(iter(aligner.align(seq, seq_chain)), None)
        if aln is None:
            continue
        chain_scores[chain] = float(aln.score)
        try:
            chain_align_text[chain] = aln.format()
        except Exception:
            chain_align_text[chain] = str(aln)

    if not chain_scores:
        st.error("No valid chains with detectable sequences found in the PDB.")
        raise ValueError("No valid chains found.")

    # Where to write per-chain outputs
    out_dir = os.path.join(save_dir, "pdb_processing")
    _ensure_dir(out_dir)

    # Map AL2CO scores: alignment column index mapping to residue numbers (1-based)
    try:
        max_al2co = float(al2co_score["al2co_score"].max())
    except Exception:
        max_al2co = 1.0
    if default_score is None or default_score < 0:
        default_score = max_al2co

    chain_data: Dict[str, Dict] = {}
    chain_original_starting_position: Dict[str, int] = {}


    for chain, score in chain_scores.items():
        aln = next(iter(aligner.align(seq, chain_sequences[chain])), None)
        residue_map: Dict[int, int] = {}  # Maps reference seq pos (1-based) -> chain seq pos (1-based)
        ref_to_pdb_resnum: Dict[int, int] = {}  # Maps reference seq pos (1-based) -> PDB residue_number

        if aln is not None:
            aligned = getattr(aln, "aligned", None)
            # Get the residue numbers for the chain in order
            chain_atoms = atom_df[atom_df["chain_id"] == chain].copy()
            chain_atoms = chain_atoms.sort_values(by=["residue_number", "insertion"])
            pdb_residue_numbers = chain_atoms["residue_number"].unique()
            # Build chain seq pos (1-based) -> PDB residue_number mapping
            chainseqpos_to_resnum = {i+1: resnum for i, resnum in enumerate(pdb_residue_numbers)}

            # Now build reference seq pos -> PDB residue_number mapping via alignment
            if (
                aligned is not None
                and len(aligned) >= 2
                and getattr(aligned[0], "size", 0) > 0
                and getattr(aligned[1], "size", 0) > 0
            ):
                for (ref_s, ref_e), (tar_s, tar_e) in zip(aligned[0], aligned[1]):
                    for ref_idx, tgt_idx in zip(range(ref_s, ref_e), range(tar_s, tar_e)):
                        residue_map[ref_idx + 1] = tgt_idx + 1  # 1-based
                        pdb_resnum = chainseqpos_to_resnum.get(tgt_idx + 1)
                        if pdb_resnum is not None:
                            ref_to_pdb_resnum[ref_idx + 1] = pdb_resnum

        resi_scores: Dict[int, float] = {}
        for _, row in al2co_score.dropna(subset=["Location"]).iterrows():
            try:
                loc = int(row["Location"])
            except Exception:
                continue
            pdb_resnum = ref_to_pdb_resnum.get(loc)
            if pdb_resnum is not None:
                try:
                    resi_scores[pdb_resnum] = float(row["al2co_score"])
                except Exception:
                    resi_scores[pdb_resnum] = default_score

        # Update B-factors for this chain using PDB residue numbers
        for res_no in chain_atoms["residue_number"].unique():
            chain_atoms.loc[chain_atoms["residue_number"] == res_no, "b_factor"] = resi_scores.get(
                res_no, default_score
            )

        # Compose new PandasPdb subset (remove HETATM)
        selected_ppdb = PandasPdb()
        selected_ppdb.df["ATOM"] = chain_atoms
        if "HETATM" in ppdb.df:
            # het = ppdb.df["HETATM"]
            # if het is not None and not het.empty:
            #     selected_ppdb.df["HETATM"] = het[het["chain_id"] == chain]
            pass

        out_pdb = os.path.join(out_dir, f"selected_al2co_labeled_chain_{chain}.pdb")
        selected_ppdb.to_pdb(path=out_pdb, records=["ATOM"], gz=False)

        # For compatibility, keep residue_offset as before (first reference seq pos mapped)
        chain_original_starting_position = list(residue_map.keys())[0] if residue_map else None

        chain_data[chain] = {
            "alignment_score": score,
            "alignment": chain_align_text.get(chain, ""),
            "residue_score_map": resi_scores,
            "updated_pdb_path": out_pdb,
            "residue_offset": chain_original_starting_position
        }

    return {
        "metadata": metadata,
        "selected_pdb": selected_pdb,
        "chain_data": chain_data,
    }
