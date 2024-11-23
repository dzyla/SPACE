# fetch.py

import os
import requests
import traceback
import streamlit as st
from Bio import Entrez, SeqIO
from typing import Optional
from .utils import clean_fasta

def search_and_save_protein_ncbi(
    query: str,
    filename: str,
    email: str,
    max_seqs: int = -1,
    use_refseq: bool = False,
) -> str:
    
    Entrez.email = email
    try:
        if not use_refseq:
            handle_original = Entrez.esearch(
                db="protein",
                term=query,
                sort="relevance",
                idtype="acc",
                retmax=10000000,
            )
        else:
            modified_query = query + " AND refseq[filter]"
            print(modified_query)
            handle_original = Entrez.esearch(
                db="protein",
                term=modified_query,
                sort="relevance",
                idtype="acc",
                retmax=10000000,
            )
        
        search_results_original = Entrez.read(handle_original, ignore_errors=True)
        id_list_original = search_results_original.get("IdList", [])
        handle_original.close()

        st.success(f"Found {len(id_list_original)} IDs from original NCBI search.")

        swissprot_query = f"{query} AND swissprot[filter]"
        handle_swissprot = Entrez.esearch(
            db="protein",
            term=swissprot_query,
            sort="relevance",
            idtype="acc",
            retmax=99999999,
        )
        search_results_swissprot = Entrez.read(handle_swissprot, ignore_errors=True)
        id_list_swissprot = search_results_swissprot.get("IdList", [])
        handle_swissprot.close()

        st.success(f"Found {len(id_list_swissprot)} IDs from Swiss-Prot filtered search.")

        combined_id_set = set(id_list_original) | set(id_list_swissprot)
        combined_id_list = list(combined_id_set)

        st.info(f"Total unique IDs after combining both searches: {len(combined_id_list)}")

        if max_seqs != -1 and max_seqs < len(combined_id_list):
            combined_id_list = combined_id_list[:max_seqs]
            st.info(f"Using first {len(combined_id_list)} IDs after applying max_seqs.")

        if not combined_id_list:
            st.error("No PDB entries found after combining both searches.")
            raise ValueError("No PDB entries found.")

    except Exception as e:
        st.error(f"An error occurred while searching for proteins: {e}")
        st.error(traceback.format_exc())
        raise e

    batch_size = 500

    try:
        with open(filename, "w") as combined_fasta:
            for start in range(0, len(combined_id_list), batch_size):
                end = min(len(combined_id_list), start + batch_size)
                st.write(f"Fetching sequences for IDs {start + 1} to {end}...")
                handle_fetch = Entrez.efetch(
                    db="protein",
                    id=combined_id_list[start:end],
                    rettype="fasta",
                    retmode="text"
                )
                records = SeqIO.parse(handle_fetch, "fasta")
                SeqIO.write(records, combined_fasta, "fasta")
                handle_fetch.close()
        print(f"Sequences saved to {filename}.")
        return os.path.abspath(filename)
    except Exception as e:
        st.error(f"An error occurred while fetching or writing sequences: {e}")
        st.error(traceback.format_exc())
        raise e

def search_and_save_protein_uniprot(
    query: str, filename: str, max_seqs: int = 500
) -> Optional[str]:
    base_url = "https://rest.uniprot.org/uniprotkb/search?query="
    
    if max_seqs > 500:
        st.warning("UniProt limits the number of sequences to 500 per request. Fetching the first 500 sequences.")
        max_seqs = 500
    
    query_url = f"{base_url}{requests.utils.quote(query)}&format=fasta&size=500"
    try:
        response = requests.get(query_url)
        response.raise_for_status()
        sequences = response.text
        with open(filename, "w") as f:
            f.write(sequences)
        st.success(f"Sequences fetched from UniProt and saved to {filename}.")
        return filename
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch data from UniProt: {e}")
        raise e
    except Exception as e:
        st.error(f"An unexpected error occurred while fetching UniProt data: {e}")
        raise e

def get_protein_sequences(file_path: str, remove_PDB: bool = False) -> list:
    try:
        # Parse the protein sequences from the FASTA file
        proteins = list(SeqIO.parse(file_path, "fasta"))
        
        # Count initial number of sequences
        initial_count = len(proteins)

        # Filter out sequences starting with "pdb" if remove_PDB is True
        if remove_PDB:
            proteins = [protein for protein in proteins if not protein.id.lower().startswith("pdb")]
            removed_count = initial_count - len(proteins)  # Calculate the number of removed sequences
            st.info(f"Removed {removed_count} sequences from PDB.")
        
        # Display a warning if no sequences are found after filtering
        if not proteins:
            st.warning("No protein sequences found in the provided FASTA file.")
        
        return proteins

    except FileNotFoundError:
        st.error(f"The file {file_path} was not found.")
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    except Exception as e:
        st.error(f"An error occurred while reading the FASTA file: {e}")
        raise e
