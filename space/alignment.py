# alignment.py

import re
import subprocess
import traceback
from typing import Optional
import streamlit as st
from Bio import AlignIO, Align, SeqIO
from Bio.Align import substitution_matrices
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import concurrent
import numpy as np
from scipy.ndimage import gaussian_filter1d
import pandas as pd
from .utils import clean_fasta
import plotly.graph_objects as go
import streamlit as st
from typing import Optional
import os
import traceback
from joblib import Parallel, delayed
import multiprocessing

import plotly.graph_objects as go
import streamlit as st
from typing import Optional
from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
import numpy as np
from Bio import AlignIO, Phylo
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor
from scipy.spatial.distance import pdist, squareform
import streamlit as st
import traceback
import plotly.graph_objects as go
import random
from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
from pyfamsa import Aligner, Sequence

import traceback
from typing import Optional
from Bio import Phylo, AlignIO
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
import streamlit as st
import logging


# Initialize a global aligner configured for global alignment using BLOSUM62
aligner = Align.PairwiseAligner()
aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
aligner.mode = "global"
aligner.match_score = 2
aligner.mismatch_score = -1
aligner.open_gap_score = -0.5
aligner.extend_gap_score = -0.3


def align_sequence(args: tuple) -> tuple:
    seq_str, seq_id, reference_seq = args
    try:
        alignment = aligner.align(reference_seq, seq_str)
        alignment = next(alignment)  # Get the first alignment
        if alignment:
            score = float(alignment.score)

            # calculate sequence identity
            seq1 = alignment[0]
            seq2 = alignment[1]
            matches = sum(aa1 == aa2 for aa1, aa2 in zip(seq1, seq2))
            score = matches / len(seq1) * 100

            return score, seq_str, seq_id
        else:
            return None, seq_str, seq_id
    except Exception as e:
        print(traceback.format_exc())
        return None, seq_str, seq_id


def perform_alignment(proteins_list: list, reference_seq: str) -> tuple:
    if not proteins_list:
        st.error("Protein list is empty.")
        return None, None

    scores, sequences, id_list = [], [], []
    min_score = float("inf")
    max_score = float("-inf")
    min_seq_length = float("inf")
    max_seq_length = float("-inf")

    seq_data = [
        (str(seq.seq).upper(), seq.id, reference_seq)
        for seq in proteins_list
        if len(seq.seq) > 0
    ]

    if not seq_data:
        st.error("No valid sequences to align.")
        return None, None

    max_workers = 4
    progress_bar = st.progress(0)
    total_sequences = len(seq_data)
    processed_sequences = 0
    alignment_errors = []

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(align_sequence, data): data for data in seq_data}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    score, seq_str, seq_id = result
                    if score is not None:
                        score = score  # / (len(seq_str)/len(reference_seq))  # Normalize score
                        scores.append(score)
                        sequences.append(seq_str)
                        id_list.append(seq_id)

                        min_score = min(min_score, score)
                        max_score = max(max_score, score)
                        seq_len = len(seq_str)
                        min_seq_length = min(min_seq_length, seq_len)
                        max_seq_length = max(max_seq_length, seq_len)
                    else:
                        alignment_errors.append([seq_id, len(seq_str), seq_str])

                processed_sequences += 1
                progress_bar.progress(processed_sequences / total_sequences)
    except Exception as e:
        st.error(f"An error occurred during alignment: {e}")
        print(traceback.format_exc())
        return None, None

    progress_bar.empty()

    if alignment_errors:
        st.warning(f"Failed to align {len(alignment_errors)} sequences.")

    if not scores:
        st.error("No successful alignments were performed.")
        return None, None

    st.session_state.min_score = min_score
    st.session_state.max_score = max_score
    st.session_state.min_seq_length = min_seq_length
    st.session_state.max_seq_length = max_seq_length

    return [scores, sequences, id_list], reference_seq


def filter_sequences(
    result: list,
    filtering_score: float = 10.0,
    score_max: float = float("inf"),
    sequence_len_min: int = 0,
    sequence_len_max: int = float("inf"),
) -> tuple:
    if not result:
        st.error("No alignment results to filter.")
        return [], [], []

    scores, sequences, ids = result

    high_score_seqs = []
    id_array_selected = []
    scores_final = []

    for score, seq, id_ in zip(scores, sequences, ids):
        if filtering_score <= score <= score_max:
            seq_len = len(seq)
            if sequence_len_min <= seq_len <= sequence_len_max:
                cleaned_id = re.sub(r"[^\w\-]", "_", id_)
                id_array_selected.append(cleaned_id)
                high_score_seqs.append(seq)
                scores_final.append(score)

    return high_score_seqs, id_array_selected, scores_final


def fix_id(seq_id: str) -> str:
    return re.sub(r"[^\w\-]", "_", seq_id)


def perform_msa(
    high_score_seqs: list,
    id_array_selected: list,
    reference_seq: str,
    msa_infile: str,
    msa_outfile: str,
    clustalo_path: str,
    max_threads: int,
) -> Optional[str]:
    try:
        final_seqs = high_score_seqs.copy()
        final_ids = id_array_selected.copy()
        final_seqs.append(reference_seq)
        final_ids.append("reference_sequence")

        sequences = [
            Sequence(fix_id(name).encode(), clean_fasta(seq).encode())
            for seq, name in zip(final_seqs, final_ids)
        ]

        # Optionally write input sequences to msa_infile
        input_records = [
            SeqRecord(
                Seq(sequence.sequence.decode()), id=sequence.id.decode(), description=""
            )
            for sequence in sequences
        ]

        SeqIO.write(input_records, msa_infile, "fasta")

        with st.spinner("Running pyfamsa alignment..."):
            aligner = Aligner(guide_tree="upgma", threads=max_threads)
            msa = aligner.align(sequences)

        aligned_records = [
            SeqRecord(
                Seq(sequence.sequence.decode()), id=sequence.id.decode(), description=""
            )
            for sequence in msa
        ]

        SeqIO.write(aligned_records, msa_outfile, "fasta")

        return msa_outfile

    except Exception as e:
        st.error(f"An unexpected error occurred during MSA: {e}")
        st.warning("Proceeding despite the error.")
        print(traceback.format_exc())
        return None


# def perform_msa_clustalo(
#     high_score_seqs: list,
#     id_array_selected: list,
#     reference_seq: str,
#     msa_infile: str,
#     msa_outfile: str,
#     clustalo_path: str,
#     max_threads: int,
# ) -> Optional[str]:
#     try:
#         final_seqs = high_score_seqs.copy()
#         final_ids = id_array_selected.copy()
#         final_seqs.append(reference_seq)
#         final_ids.append("reference_sequence")

#         records = [
#             SeqRecord(Seq(clean_fasta(seq)), id=fix_id(name), description="")
#             for seq, name in zip(final_seqs, final_ids)
#         ]

#         SeqIO.write(records, msa_infile, "fasta")

#         cmd = [
#             clustalo_path,
#             "-i",
#             msa_infile,
#             "-o",
#             msa_outfile,
#             "--force",
#             "--auto",
#             "--threads",
#             str(max_threads),
#             "--outfmt",
#             "fasta",
#             "--verbose",
#         ]

#         with st.spinner(f"Running Clustal Omega: {' '.join(cmd)}"):
#             process = subprocess.run(
#                 cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
#             )

#         if process.stderr:
#             print(f"Clustal Omega stderr: {process.stderr}")

#         if process.returncode == 0:
#             with st.expander("View Clustal Omega Output"):
#                 st.code(process.stdout)
#             return msa_outfile
#         else:
#             st.error(f"Clustal Omega failed with return code {process.returncode}.")
#             st.error(f"Error message: {process.stderr}")
#             return None

#     except FileNotFoundError:
#         st.error(f"Clustal Omega executable not found at: {clustalo_path}")
#         return None
#     except Exception as e:
#         st.error(f"An unexpected error occurred during MSA: {e}")
#         st.warning("Proceeding despite the error.")
#         print(traceback.format_exc())
#         return None


def perform_msa_and_fix_header(
    high_score_seqs: list,
    id_array_selected: list,
    reference_seq: str,
    msa_infile: str,
    msa_outfile: str,
    clustalo_path: str,
    max_threads: int,
) -> Optional[str]:
    # Perform MSA
    msa_outfile = perform_msa(
        high_score_seqs,
        id_array_selected,
        reference_seq,
        msa_infile,
        msa_outfile,
        clustalo_path,
        max_threads,
    )

    # Fix MSA header
    if msa_outfile:
        try:
            count = SeqIO.convert(
                msa_outfile, "fasta", msa_outfile.replace("fasta", "aln"), "clustal"
            )
            st.write(f"Converted {count} records to Clustal format.")

            with open(msa_outfile.replace("fasta", "aln"), "r") as f:
                lines = f.readlines()
                lines[0] = "CLUSTAL W multiple sequence alignment\n\n"
            with open(msa_outfile.replace("fasta", "aln"), "w") as f:
                f.writelines(lines)
        except Exception as e:
            st.error(f"An error occurred while fixing the MSA header: {e}")
            st.warning("Proceeding without fixing the MSA header.")
    return msa_outfile


def analyze_alignment(alignment_file: str, reference_seq: str) -> pd.DataFrame:
    try:
        alignment = AlignIO.read(alignment_file, "clustal")
    except Exception as e:
        st.error(f"An error occurred while reading the alignment file: {e}")
        raise e

    reference_record = None
    for record in alignment:
        if record.id == "reference_sequence":
            reference_record = record
            break

    if not reference_record:
        st.error("Reference sequence not found in the alignment.")
        raise ValueError("Reference sequence missing in alignment.")

    alignment_length = alignment.get_alignment_length()
    ref_seq = str(reference_record.seq)
    mapping = []
    ref_num = 0
    for aa in ref_seq:
        if aa != "-":
            ref_num += 1
            mapping.append(ref_num)
        else:
            mapping.append(None)

    conservation_scores = []
    for i in range(alignment_length):
        mapped_ref_num = mapping[i]
        if mapped_ref_num is None:
            residue_num = np.nan
        else:
            residue_num = mapped_ref_num

        column = alignment[:, i]
        residues = Counter(column)
        residues.pop("-", None)
        if residues:
            most_common = residues.most_common(1)[0][1]
            conservation = most_common / len(alignment)
        else:
            conservation = 0
        conservation_scores.append(
            {"Location": residue_num, "Conservation": conservation}
        )

    df = pd.DataFrame(conservation_scores)
    st.session_state.alignment_mapping = mapping

    return df


def generate_phylogenetic_tree(
    msa_file: str, use_random_subsample: bool = False
) -> Optional[str]:
    """
    Generates a phylogenetic tree from a Multiple Sequence Alignment (MSA) file using Biopython.
    Saves the tree in Newick format.

    Args:
        msa_file (str): Path to the MSA file in FASTA format.

    Returns:
        Optional[str]: Path to the saved Newick tree file if successful, else None.
    """
    try:
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        logger.info("Reading alignment from %s", msa_file)
        # Read the alignment using Biopython
        alignment = AlignIO.read(msa_file, "fasta")
        num_seqs = len(alignment)

        if use_random_subsample:
            if num_seqs == 0:
                logger.error("The alignment contains no sequences.")
                raise ValueError("Empty alignment.")

            num_seqs_to_keep = min(200, num_seqs)
            logger.info("Subsampling to %d sequences", num_seqs_to_keep)

            # Convert alignment to a list to ensure it's a sequence
            alignment_list = list(alignment)

            # Randomly subsample sequences without replacement
            subset = random.sample(alignment_list, num_seqs_to_keep)

            # Create a new MultipleSeqAlignment object with the subset
            alignment = MultipleSeqAlignment(subset)
            num_seqs = len(alignment)

            logger.info("Randomly subsampled to %d sequences", num_seqs)

        seq_length = alignment.get_alignment_length()
        ids = [record.id for record in alignment]

        logger.info("Number of sequences: %d", num_seqs)
        logger.info("Sequence length: %d", seq_length)

        # Compute distance matrix using Biopython's DistanceCalculator
        calculator = DistanceCalculator("identity")
        distance_matrix_biopython = calculator.get_distance(alignment)
        logger.info("Distance matrix computed")

        # Construct the tree using Neighbor-Joining method
        constructor = DistanceTreeConstructor()
        tree = constructor.nj(distance_matrix_biopython)
        logger.info("Phylogenetic tree constructed")

        # Save the tree in Newick format
        tree_file = "phylogenetic_tree.nwk"
        Phylo.write(tree, tree_file, "newick")
        logger.info("Phylogenetic tree saved to %s", tree_file)

        return tree_file

    except Exception as e:
        st.error(f"An error occurred while generating the phylogenetic tree: {e}")
        logger.error("Exception occurred", exc_info=True)
        print(traceback.format_exc())
        return None


def plot_phylogenetic_tree(tree_file: str, st_column: Optional = None):
    """
    Plots the phylogenetic tree using Plotly.

    Args:
        tree_file (str): Path to the phylogenetic tree file in Newick format.
        st_column (Optional): Streamlit column to plot the tree in.
    """
    try:
        from Bio import Phylo
        import plotly.graph_objects as go
        import streamlit as st

        # Read the tree
        tree = Phylo.read(tree_file, "newick")

        # Assign x and y coordinates to each node
        pos = {}
        y_counter = [0]  # Use list for mutable integer in nested function

        def assign_positions(clade, x_current):
            if clade.branch_length is None:
                branch_length = 0
            else:
                branch_length = clade.branch_length

            x = x_current + branch_length

            if clade.is_terminal():
                y = y_counter[0]
                pos[clade] = (x, y)
                y_counter[0] += 1
            else:
                child_ys = []
                for child in clade.clades:
                    assign_positions(child, x)
                    child_ys.append(pos[child][1])
                y = sum(child_ys) / len(child_ys)
                pos[clade] = (x, y)

        assign_positions(tree.root, 0)

        # Collect edges
        edge_x = []
        edge_y = []

        def add_edges(clade):
            x0, y0 = pos[clade]
            for child in clade.clades:
                x1, y1 = pos[child]
                # Vertical line from parent to child
                edge_x.extend([x0, x0, None])
                edge_y.extend([y0, y1, None])
                # Horizontal line from child to child's x position
                edge_x.extend([x0, x1, None])
                edge_y.extend([y1, y1, None])
                add_edges(child)

        add_edges(tree.root)

        # Collect node positions and labels
        node_x = []
        node_y = []
        node_labels = []

        def collect_nodes(clade):
            x, y = pos[clade]
            node_x.append(x)
            node_y.append(y)
            if clade.is_terminal():
                node_labels.append(clade.name if clade.name else "")
            else:
                node_labels.append("")
            for child in clade.clades:
                collect_nodes(child)

        collect_nodes(tree.root)

        # Create Plotly figure
        fig = go.Figure()

        # Add edges
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(color="black", width=1),
                hoverinfo="none",
                showlegend=False,
            )
        )

        # Add nodes
        fig.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                marker=dict(symbol="circle", size=6, color="#6872f7"),
                text=node_labels,
                textposition="middle right",
                hoverinfo="text",
                showlegend=False,
            )
        )

        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white",
            width=800,
            height=600,
        )
        fig.write_html("phylogenetic_tree.html")

        if st_column:
            st_column.plotly_chart(fig, use_container_width=True)
        else:
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred while plotting the phylogenetic tree: {e}")
        print(traceback.format_exc())
