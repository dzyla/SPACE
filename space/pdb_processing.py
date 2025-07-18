# pdb_processing.py

import os
import re
from typing import List, Optional
import requests
import traceback
from collections import Counter

import numpy as np
import pandas as pd
from Bio.Align import substitution_matrices
from Bio import AlignIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import PairwiseAligner
from biopandas.pdb import PandasPdb
from Bio import Align

import streamlit as st
from stmol import showmol
import py3Dmol

from .utils import clean_fasta, download_alphafold_pdb, get_protein_data

# Initialize a global aligner configured for global alignment using BLOSUM62
aligner = PairwiseAligner()
aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
aligner.mode = "global"  # Ensure global alignment
aligner.match_score = 2
aligner.mismatch_score = -1
aligner.open_gap_score = -0.5
aligner.extend_gap_score = -0.3


def fix_id(seq_id: str) -> str:
    """
    Cleans the sequence ID by replacing spaces and special characters with underscores.

    Args:
        seq_id (str): Original sequence ID.

    Returns:
        str: Cleaned sequence ID.
    """
    return re.sub(r"[^\w\-]", "_", seq_id)


def process_pdb_chain(
    seq: str,
    al2co_score: pd.DataFrame,
    sequence_score: float = 0.9,
    default_score: float = -1.0,
    frequency_threshold: float = 0.1,
    uniprot_id: str = None,
    own_pdb: str = None,
    st_column: Optional = None,
    output_dir: str = "PDB_processing",
) -> dict:
    """
    Processes a protein sequence to find matching PDB chains, maps al2co scores,
    and replaces B-factors in the PDB files with the mapped scores for all chains.

    Args:
        seq (str): Reference protein sequence (1-letter code).
        al2co_score (pd.DataFrame): DataFrame mapping reference positions to al2co scores.
        sequence_score (float, optional): Minimum sequence identity score for the search. Defaults to 0.9.
        default_score (float, optional): Score to assign if a target residue is not aligned. Defaults to -1.0.
        frequency_threshold (float, optional): Threshold for annotating less frequent scores. Defaults to 0.1.
        uniprot_id (str, optional): UniProt ID for AlphaFold models. Defaults to None.
        own_pdb (str, optional): Path to user-provided PDB file. Defaults to None.

    Returns:
        dict: Contains metadata, PDB information, alignment details,
              paths to updated PDB files, and the mapped scores DataFrame for all chains.

    Raises:
        Exception: If any error occurs during processing.
    """

    # Ensure default_score is set correctly
    default_score = al2co_score["al2co_score"].max()

    # Validate the input sequence
    if not re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+", seq):
        st.error("Invalid sequence. Only standard amino acids are supported.")
        raise ValueError("Invalid sequence. Only standard amino acids are supported.")

    # Determine PDB source
    if own_pdb:
        # Use user-provided PDB file
        pdb_filename = own_pdb
        selected_pdb = "Custom PDB"
        metadata = {
            "pdb_id": "Custom PDB",
            "title": "Custom PDB",
            "authors": "User",
            "date": "N/A",
        }
    elif uniprot_id:
        # Download AlphaFold PDB
        if download_alphafold_pdb(uniprot_id, save_dir=output_dir):
            pdb_filename = f"selected.pdb"
            selected_pdb = uniprot_id
            uniprot_meta_data = get_protein_data(uniprot_id)
            uniprot_name = uniprot_meta_data.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', 'N/A')

            metadata = {
                "pdb_id": f'{uniprot_id} (UniProt)',
                "title": uniprot_name,
                "authors": "AlphaFold Server",
                "date": "N/A",
            }
        else:
            raise ValueError("Failed to download AlphaFold PDB file. No PDB found.")
    else:
        # Perform sequence search using RCSB Search API
        try:
            from rcsbsearchapi.search import SequenceQuery

            results = SequenceQuery(seq, 10, sequence_score)
            pdb_ids = [x.split("_")[0] for x in results("polymer_entity")]
        except Exception as e:
            st.error(f"An error occurred during PDB search: {e}")
            raise e

        if not pdb_ids:
            raise ValueError(
                "No PDB entries found matching the provided sequence and score threshold. Try AlphaFold instead."
            )

        # Allow user to select a PDB ID from search results via Streamlit
        selected_pdb = st_column.selectbox("Select a PDB ID", pdb_ids)

        try:
            pdb_data_url = f"https://data.rcsb.org/rest/v1/core/entry/{selected_pdb}"
            data_response = requests.get(pdb_data_url)
            data_response.raise_for_status()
            metadata_json = data_response.json()
            metadata = {
                "pdb_id": selected_pdb,
                "title": metadata_json.get("struct", {}).get("title", "N/A"),
                "authors": ", ".join(
                    metadata_json.get("citation", [{}])[0].get("rcsb_authors", [])
                ),
                "date": metadata_json.get("rcsb_accession_info", {}).get(
                    "initial_deposition_date", "N/A"
                ),
            }
        except requests.exceptions.RequestException:
            metadata = {
                "pdb_id": selected_pdb,
                "title": "N/A",
                "authors": "N/A",
                "date": "N/A",
            }
            st.warning("Failed to fetch PDB metadata.")

        # Download the Selected PDB File
        try:
            pdb_url = f"https://files.rcsb.org/download/{selected_pdb}.pdb"
            response = requests.get(pdb_url)
            response.raise_for_status()
            pdb_content = response.text
            pdb_filename = f"selected.pdb"
            with open(pdb_filename, "w") as file:
                file.write(pdb_content)
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to download PDB file: {e}")
            raise e

    # Parse the PDB using BioPandas
    try:
        ppdb = PandasPdb().read_pdb(pdb_filename)
        atom_df = ppdb.df["ATOM"]
    except Exception as e:
        st.error(f"An error occurred while parsing the PDB file: {e}")
        raise e

    # Extract unique chains
    chains = atom_df["chain_id"].unique()

    chain_scores = {}
    chain_alignments = {}
    chain_sequences = {}

    # Amino acid 3-letter to 1-letter mapping
    aa_3to1 = {
        "ALA": "A",
        "CYS": "C",
        "ASP": "D",
        "GLU": "E",
        "PHE": "F",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LYS": "K",
        "LEU": "L",
        "MET": "M",
        "ASN": "N",
        "PRO": "P",
        "GLN": "Q",
        "ARG": "R",
        "SER": "S",
        "THR": "T",
        "VAL": "V",
        "TRP": "W",
        "TYR": "Y",
        "SEC": "U",
        "PYL": "O",
        "ASX": "B",
        "GLX": "Z",
        "XLE": "J",
        "XAA": "X",
    }

    # Initialize the aligner
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5

    for chain in chains:
        # Filter atoms for the current chain
        chain_atoms = atom_df[atom_df["chain_id"] == chain]

        # Sort by residue number and insertion code
        chain_atoms = chain_atoms.sort_values(by=["residue_number", "insertion"])

        # Get unique residues
        residues = chain_atoms["residue_number"].unique()

        sequence = ""
        for res_num in residues:
            res = chain_atoms[chain_atoms["residue_number"] == res_num].iloc[0]
            resname = res["residue_name"].strip()
            aa = aa_3to1.get(resname, "X")  # Use 'X' for unknown amino acids
            sequence += aa

        if not sequence:
            st.warning(f"Chain `{chain}` has no detectable sequence. Skipping.")
            continue  # Skip chains with no detectable sequence

        # Store the sequence for later use
        chain_sequences[chain] = sequence

        # Perform pairwise alignment
        alignments = aligner.align(seq, sequence)
        top_alignment = next(iter(alignments), None)

        if top_alignment:
            score = top_alignment.score
            chain_scores[chain] = score
            chain_alignments[chain] = top_alignment

    if not chain_scores:
        st.error("No valid chains with detectable sequences found in the PDB.")
        raise ValueError("No valid chains found.")

    # Initialize a directory to save updated PDB files
    os.makedirs(output_dir, exist_ok=True)

    chain_data = {}

    for chain in chain_scores.keys():
        alignment = chain_alignments[chain]
        alignment_score = chain_scores[chain]

        # Map residues between reference and target sequences
        residue_map = {}
        for (ref_start, ref_end), (target_start, target_end) in zip(
            alignment.aligned[0], alignment.aligned[1]
        ):
            for ref_idx, target_idx in zip(
                range(ref_start, ref_end), range(target_start, target_end)
            ):
                residue_map[ref_idx + 1] = target_idx + 1  # 1-based indexing

        resi_scores = {}
        for resi in al2co_score["Location"]:
            mapped_residue = residue_map.get(resi, None)
            if mapped_residue is not None:
                try:
                    score = al2co_score[al2co_score["Location"] == resi][
                        "al2co_score"
                    ].values[0]
                    resi_scores[mapped_residue] = score
                except Exception:
                    resi_scores[mapped_residue] = default_score

        # Update B-factors in the chain
        chain_atoms = atom_df[atom_df["chain_id"] == chain].copy()
        unique_residues = chain_atoms["residue_number"].unique()

        for res_num in unique_residues:
            score = resi_scores.get(res_num, default_score)
            chain_atoms.loc[
                chain_atoms["residue_number"] == res_num, "b_factor"
            ] = score

        # Include HETATM records if any
        selected_ppdb = PandasPdb()
        selected_ppdb.df["ATOM"] = chain_atoms
        if "HETATM" in ppdb.df:
            chain_hetatm = ppdb.df["HETATM"][
                ppdb.df["HETATM"]["chain_id"] == chain
            ]
            selected_ppdb.df["HETATM"] = chain_hetatm

        # Save the updated PDB file for the chain
        updated_pdb_path = os.path.join(
            output_dir, f"selected_al2co_labeled_chain_{chain}.pdb"
        )
        selected_ppdb.to_pdb(
            path=updated_pdb_path, records=["ATOM", "HETATM"], gz=False
        )

        # Store the data in chain_data
        chain_data[chain] = {
            "alignment_score": alignment_score,
            "alignment": alignment,
            "residue_score_map": resi_scores,
            "updated_pdb_path": updated_pdb_path,
        }

    # Clean up the original downloaded PDB file if it was downloaded
    if not own_pdb:
        try:
            os.remove(pdb_filename)
        except OSError:
            pass  # If file doesn't exist, ignore

    return {
        "metadata": metadata,
        "selected_pdb": selected_pdb,
        "chain_data": chain_data,
    }



def extract_uniprot_ids(names_list: List[str], include_version: bool = True) -> List[str]:
    """
    Extracts all UniProt IDs from a list of names, allowing multiple IDs per string.

    Parameters:
        names_list (List[str] or str): List of name strings from NCBI search or a single string.
        include_version (bool): 
            - If True, includes the version number in the ID (e.g., "O89342.2").
            - If False, excludes the version number (e.g., "O89342").

    Returns:
        List[str]: List of extracted UniProt IDs.
    """
    if not isinstance(names_list, list):
        if isinstance(names_list, str):
            names_list = [names_list]
        else:
            raise ValueError("names_list must be a list or a string")
    
    uniprot_ids = []
    
    # Updated regex to find all matches within a string
    pattern = re.compile(r'sp[|_]([A-Za-z0-9]+)(?:[._-](\d+))?[|_]')
    
    for name in names_list:
        matches = pattern.findall(name)
        for match in matches:
            id_part, version_part = match
            if version_part and include_version:
                uniprot_id = f"{id_part}.{version_part}"
            else:
                uniprot_id = id_part
            uniprot_ids.append(uniprot_id)
    
    return uniprot_ids


def download_alphafold_pdb(accession: str, save_dir: str = '.') -> Optional[str]:
    """
    Checks if an AlphaFold model is available for the given UniProt accession and downloads the PDB file if available.

    Parameters:
        accession (str): The UniProt accession number (e.g., 'Q9UQF0').
        save_dir (str): Directory to save the downloaded PDB file. Defaults to the current directory.

    Returns:
        Optional[str]: The file path to the downloaded PDB file if successful, else None.
    """
    # Validate input
    if not isinstance(accession, str) or not accession.strip():
        raise ValueError("Accession must be a non-empty string.")

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Define the API endpoint
    api_url = f'https://alphafold.ebi.ac.uk/api/prediction/{accession}'

    try:
        # Make the GET request to the AlphaFold API
        response = requests.get(api_url, headers={'accept': 'application/json'}, timeout=10)

        # Check if the request was successful
        if response.status_code != 200:
            print(f"Failed to fetch data for accession '{accession}'. HTTP Status Code: {response.status_code}")
            return None

        # Parse the JSON response
        data = response.json()

        # Check if the response contains any entries
        if not isinstance(data, list) or len(data) == 0:
            print(f"No AlphaFold models found for accession '{accession}'.")
            return None

        # Iterate through the entries to find the desired model
        # For simplicity, we'll take the latest version based on 'latestVersion'
        latest_entry = max(data, key=lambda x: x.get('latestVersion', 0))

        # Extract the PDB URL
        pdb_url = latest_entry.get('pdbUrl')
        if not pdb_url:
            print(f"PDB URL not available for accession '{accession}'.")
            return None

        # Extract model ID and version for naming
        entry_id = latest_entry.get('entryId', f"{accession}_model")
        model_version = latest_entry.get('latestVersion', 'unknown')

        # Define the filename
        pdb_filename = f"selected.pdb"
        pdb_filepath = os.path.join(pdb_filename)

        # Download the PDB file
        pdb_response = requests.get(pdb_url, stream=True, timeout=20)

        if pdb_response.status_code == 200:
            with open(pdb_filepath, 'wb') as f:
                for chunk in pdb_response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return pdb_filepath
        else:
            print(f"Failed to download PDB file. HTTP Status Code: {pdb_response.status_code}")
            return None

    except requests.exceptions.Timeout:
        print("Request timed out. Please try again later.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching the PDB file: {e}")
        return None
    except ValueError as ve:
        print(f"JSON decoding failed: {ve}")
        return None


def get_protein_data(accession):
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
    response = requests.get(url)
    return response.json()