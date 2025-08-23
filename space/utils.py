# utils.py

import os
import re
import zipfile
import io
import requests
import traceback
import pandas as pd
from shutil import which
import streamlit as st

def get_executable_path(executable_name: str, default_path: str) -> str:
    from shutil import which

    path = which(executable_name)
    if path is None:
        if os.path.exists(default_path):
            return default_path
        else:
            st.error(
                f"{executable_name} is not found in PATH. Please install {executable_name} or provide its path."
            )
            raise FileNotFoundError(
                f"{executable_name} not found in PATH or at {default_path}."
            )
    return path

def get_clustalo_path(default_path: str) -> str:
    return get_executable_path("clustalo", default_path)

def get_al2co_path(default_path: str) -> str:
    return get_executable_path("al2co", default_path)

def clean_fasta(fasta_str: str) -> str:
    if not fasta_str:
        return ""
    lines = fasta_str.strip().splitlines()
    cleaned_lines = []
    removed_characters = []
    valid_aas = set("ACDEFGHIKLMNPQRSTVWY")

    for line in lines:
        if line.startswith(">"):
            cleaned_lines.append(line)
        else:
            sequence = re.sub(r"[^A-Za-z]", "", line).upper()
            cleaned_sequence = "".join([aa for aa in sequence if aa in valid_aas])
            removed_characters.extend([aa for aa in sequence if aa not in valid_aas])
            cleaned_lines.append(cleaned_sequence)

    cleaned_fasta = "\n".join(cleaned_lines).strip()
    if removed_characters:
        print('Fastas with non-standard amino acids were cleaned. Removed characters: "{}"'.format("".join(set(removed_characters))))
    return cleaned_fasta

def create_zip_file(file_paths: list, zip_name: str) -> io.BytesIO:
    if not file_paths:
        return None

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in file_paths:
            try:
                if os.path.isfile(file_path):
                    zip_file.write(file_path, arcname=os.path.basename(file_path))
                else:
                    st.warning(f"File does not exist and will be skipped: {file_path}")
            except Exception as e:
                st.warning(f"Failed to add {file_path} to zip: {e}")

    zip_buffer.seek(0)
    return zip_buffer

def download_alphafold_pdb(accession: str, save_dir: str = '.') -> str:
    if not isinstance(accession, str) or not accession.strip():
        raise ValueError("Accession must be a non-empty string.")

    os.makedirs(save_dir, exist_ok=True)
    api_url = f'https://alphafold.ebi.ac.uk/api/prediction/{accession}'

    try:
        response = requests.get(api_url, headers={'accept': 'application/json'}, timeout=10)
        if response.status_code != 200:
            print(f"Failed to fetch data for accession '{accession}'. HTTP Status Code: {response.status_code}")
            return None

        data = response.json()
        if not isinstance(data, list) or len(data) == 0:
            print(f"No AlphaFold models found for accession '{accession}'.")
            return None

        latest_entry = max(data, key=lambda x: x.get('latestVersion', 0))
        pdb_url = latest_entry.get('pdbUrl')
        if not pdb_url:
            print(f"PDB URL not available for accession '{accession}'.")
            return None

        pdb_filename = f"selected.pdb"
        pdb_filepath = os.path.join(save_dir, pdb_filename)

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

def get_protein_data(accession: str) -> dict:
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        st.warning(f"Failed to fetch data for UniProt ID: {accession}")
        return {}


def clean_string_for_path(string: str) -> str:
    cleaned_string = re.sub(r'[\\/:*?<>|]', '', string)
    cleaned_string = re.sub(r' ', '_', cleaned_string)
    cleaned_string = cleaned_string.lower()
    return cleaned_string