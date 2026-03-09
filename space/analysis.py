# analysis.py

import os
import re
import traceback
import pandas as pd
import numpy as np
from collections import Counter
from typing import List, Optional
import subprocess
import streamlit as st
from .utils import clean_fasta
from Bio import AlignIO, Align, SeqIO
import plotly.express as px

import al2co

def run_al2co(alignment_file: str) -> pd.DataFrame:
    if "alignment_mapping" not in st.session_state:
        st.error("Alignment mapping not found. Cannot map al2co locations.")
        raise ValueError("Alignment mapping not found in session state.")

    alignment_mapping = st.session_state.alignment_mapping

    try:
        alignment = AlignIO.read(alignment_file, "clustal")
        with st.spinner("Running al2co..."):
            scores = al2co.run_al2co(alignment)

        data = []
        for pos, score in scores.items():
            if pos < 1 or pos > len(alignment_mapping):
                continue
            
            mapped_residue_num = alignment_mapping[pos - 1]
            mapped_location = mapped_residue_num if mapped_residue_num else np.nan
            residue = alignment[0, pos - 1] 

            data.append(
                {
                    "Location": mapped_location,
                    "Residue": residue,
                    "al2co_score": float(score),
                }
            )

        if data:
            df = pd.DataFrame(data)
            return df
        else:
            st.error("No valid data found in al2co output.")
            raise ValueError("No valid data in al2co output.")
    except Exception as e:
        st.error(f"An error occurred while running al2co: {e}")
        raise e

def list_unique_point_mutations(
    alignment_file: str,
    reference_sequence_id: str,
    alignment_mapping: list,
    deletion_threshold: float = 10,
    exclude_mutations_with_X: bool = False,
) -> tuple:
    try:
        alignment = AlignIO.read(alignment_file, "clustal")
    except Exception as e:
        st.error(f"An error occurred while reading the alignment file: {e}")
        raise e

    reference_record = None
    for record in alignment:
        if record.id == reference_sequence_id:
            reference_record = record
            break
    if not reference_record:
        st.error(
            f"Reference sequence '{reference_sequence_id}' not found in the alignment."
        )
        raise ValueError(
            f"Reference sequence '{reference_sequence_id}' missing in alignment."
        )

    unique_mutations = {}
    excluded_sequences = {}
    mutation_summary = set()
    alignment_length = alignment.get_alignment_length()

    for record in alignment:
        if record.id == reference_sequence_id:
            continue

        mutations = []
        deletion_count = 0
        total_residues = 0

        for i in range(alignment_length):
            ref_residue = reference_record.seq[i]
            test_residue = record.seq[i]
            mapped_ref_num = alignment_mapping[i]

            if ref_residue != "-":
                total_residues += 1
                if test_residue == "-":
                    deletion_count += 1
                else:
                    if ref_residue != test_residue:
                        mutation = f"{ref_residue}{mapped_ref_num}{test_residue}"
                        if exclude_mutations_with_X and "X" in mutation:
                            continue
                        mutations.append(mutation)
            else:
                if test_residue != "-":
                    mutation = f"ins{mapped_ref_num}{test_residue}"
                    if exclude_mutations_with_X and "X" in mutation:
                        continue
                    mutations.append(mutation)

        deletion_percentage = (
            (deletion_count / total_residues) * 100 if total_residues > 0 else 0
        )

        if deletion_percentage > deletion_threshold:
            excluded_sequences[record.id] = f"{deletion_percentage:.2f}% deletions"
            continue

        mutation_str = ", ".join(mutations) if mutations else "No mutations"
        unique_mutations[record.id] = mutation_str

        mutation_summary.update(mutations)

    all_mutations_str = "; ".join(
        sorted(mutation_summary, key=lambda x: extract_position(x))
    )

    excluded_count = len(excluded_sequences)
    return (
        unique_mutations,
        excluded_sequences,
        excluded_count,
        mutation_summary,
        all_mutations_str,
    )

def extract_position(mutation: str) -> int:
    match = re.search(r"\d+", mutation)
    return int(match.group()) if match else 0

def parse_mutations(unique_mutations: dict) -> pd.DataFrame:
    mutation_data = []

    for seq_id, mutations in unique_mutations.items():
        if mutations == "No mutations":
            continue

        mutation_list = mutations.split(", ")
        for mutation in mutation_list:
            if mutation.startswith("ins"):
                match = re.match(r"ins(\d+)(\w+)", mutation)
                if match:
                    position = int(match.group(1))
                    mutated_residue = match.group(2)
                    mutation_type = "Insertion"
                    original_residue = "-"
                else:
                    continue
            else:
                match = re.match(r"(\w)(\d+)(\w)", mutation)
                if match:
                    original_residue = match.group(1)
                    position = int(match.group(2))
                    mutated_residue = match.group(3)
                    mutation_type = "Substitution"
                else:
                    continue

            mutation_data.append(
                {
                    "Sequence ID": seq_id,
                    "Original Residue": original_residue,
                    "Position": position,
                    "Mutated Residue": mutated_residue,
                    "Mutation Type": mutation_type,
                    "Mutation": mutation,
                }
            )

    df_mutations = pd.DataFrame(mutation_data)
    return df_mutations

def get_mutation_dataframe() -> pd.DataFrame:
    unique_mutations = st.session_state.get("unique_mutations", {})
    if not unique_mutations:
        st.warning("No mutation data available to display.")
        return pd.DataFrame()

    df_mutations = parse_mutations(unique_mutations)
    if df_mutations.empty:
        st.warning("No valid mutations found to display.")
    return df_mutations

def assign_unique_colors(mut_df: pd.DataFrame) -> tuple:
    unique_mutations = mut_df["Mutation"].unique()
    color_palette = px.colors.qualitative.Plotly

    mutation_color_map = {
        mutation: color_palette[i % len(color_palette)]
        for i, mutation in enumerate(unique_mutations)
    }
    mut_df["Unique_Color"] = mut_df["Mutation"].map(mutation_color_map)

    return mut_df, mutation_color_map

