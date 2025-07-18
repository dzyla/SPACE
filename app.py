# app.py

import datetime
import glob
import os
import platform
import traceback
import zipfile
import io

import streamlit as st
from typing import List, Optional

import py3Dmol
import stmol
from stmol import showmol
from collections import Counter
import pandas as pd

from space.alignment import *
from space.analysis import *
from space.pdb_processing import *
from space.utils import *
from space.visualization import *
from space.fetch import *

# -----------------------------------
# packages installation
# -----------------------------------
# pip install streamlit py3Dmol stmol pandas biopython plotly kaleido scipy biopandas ipython_genutils joblib seaborn rcsbsearchapi pyfamsa


# -----------------------------------
# Configuration
# -----------------------------------
st.set_page_config(
    page_title="SPACE",
    layout="wide",
    page_icon=":stars:",
    initial_sidebar_state="expanded",
    
)


# Function to create a zip file of given files
def create_zip_file(file_paths: List[str], zip_name: str) -> Optional[io.BytesIO]:
    """
    Creates a zip file in memory from the given list of file paths.

    Args:
        file_paths (List[str]): List of file paths to include in the zip.
        zip_name (str): The name of the resulting zip file.

    Returns:
        Optional[io.BytesIO]: The BytesIO object containing the zip file data, or None if no files.
    """
    if not file_paths:
        return None

    # Initialize a BytesIO buffer to hold the zip file in memory
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in file_paths:
            try:
                # Ensure the file exists before adding
                if os.path.isfile(file_path):
                    # arcname is used to specify the name within the zip archive
                    zip_file.write(file_path, arcname=os.path.basename(file_path))
                else:
                    st.warning(f"File does not exist and will be skipped: {file_path}")
            except Exception as e:
                st.warning(f"Failed to add {file_path} to zip: {e}")

    zip_buffer.seek(0)  # Reset buffer pointer to the beginning
    return zip_buffer


def main():
    st.title("Sequence Protein Alignment and Conservation Engine (SPACE) :stars:")
    welcome = st.empty()
    if "result" not in st.session_state:
        welcome.markdown(
            """
    Welcome to **Sequence Protein Alignment and Conservation Engine (SPACE)**, a tool for exploring and understanding protein sequences and their sequence conservation in 2D and 3D. 

    ## Description

    - **Fetch Protein Sequences:** Retrieve protein sequences from databases like **NCBI** and **UniProt** using your query of interest.
    - **Reference Sequence Selection:** Choose a reference sequence from the fetched data or provide your own sequence for alignment.
    - **Pairwise Alignment:** Perform pairwise alignments between the reference sequence to filter out sequences with low alignment scores.
    - **Multiple Sequence Alignment (MSA):** Perform alignments using **Clustal Omega** or **FAMSA** *(default)*, visualizing the results in an interactive plot.
    - **Conservation Scoring:** using **AL2CO** to assess the conservation levels across your aligned sequences.
    - **Mutation Analysis:** Identify and analyze unique point mutations within your protein sequences, providing insights into functional and structural impacts.
    - **Phylogenetic Tree Generation:** Visualize evolutionary relationships with dynamically generated phylogenetic trees.
    - **Structural Mapping:** Map conservation scores onto 3D protein structures using PDB files from **PDB**, **AlphaFold**, or your own uploaded PDB files.
    - **Download Results:** Easily download all your analysis results in a single ZIP file. PDB file has modified B-factor values based on the AL2CO conservation scores.

    ## Potential Applications

    - **Sequence conservation:** Identify conserved regions in target proteins to validate the epitope variability.
    - **Evolutionary Biology:** Explore the evolutionary conservation of proteins across different species to understand functional importance.
    - **Structural Biology:** Map conservation data onto protein structures to pinpoint functionally important sites for further study.

    ## How to cite
    - Zyla, D. **SPACE**: Sequence Protein Alignment and Conservation Engine. Streamlit app. [https://github.com/dzyla/SPACE]
    - Based on Zyla, D. et al. A neutralizing antibody prevents postfusion transition of measles virus fusion protein. Science384, eadm8693(2024).DOI:[10.1126/science.adm8693](https://www.science.org/stoken/author-tokens/ST-1959/full)

    ## References
    - [Clustal Omega](http://www.clustal.org/omega/)
    - [FAMSA](https://www.nature.com/articles/srep33964)
    - [pyFAMSA](https://github.com/althonos/pyfamsa)
    - [AL2CO](https://pubmed.ncbi.nlm.nih.gov/11524371/)
    - [Py3Dmol](https://github.com/avirshup/py3dmol)
    - [3Dmol.js](https://3dmol.csb.pitt.edu/)
    - [stMol](https://github.com/napoles-uach/stmol)
    - [UniProt](https://www.uniprot.org/)
    - [NCBI](https://www.ncbi.nlm.nih.gov/)
    - [AlphaFold](https://alphafold.ebi.ac.uk/)
    - [PDB](https://www.rcsb.org/)
    - [Entrez Programming Utilities](https://www.ncbi.nlm.nih.gov/home/develop/api/)
    - [Biopython](https://biopython.org/)
    - [Streamlit](https://streamlit.io/)


    """
        )
    else:
        welcome.empty()

    # Initialize session state
    if "result" not in st.session_state:
        st.session_state.result = None
        st.session_state.reference_seq = None
        st.session_state.msa_done = False
        st.session_state.filtered = False
        st.session_state.msa_df = None
        st.session_state.msa_image = None
        st.session_state.msa_letters = None
        st.session_state.msa_outfile = None
        st.session_state.al2co_output = None
        st.session_state.al2co_df = None
        st.session_state.alignment_mapping = None  # Initialize alignment mapping
        st.session_state.unique_mutations = {}
        st.session_state.excluded_sequences = {}
        st.session_state.mutation_summary = set()
        st.session_state.all_mutations_str = ""
        st.session_state.proteins_list = None
        st.session_state.phylogenetic_tree = None  # Initialize phylogenetic_tree
        st.session_state.user_pdb = None  # Initialize user-uploaded PDB

    # Sidebar inputs
    st.sidebar.header("Parameters")
    with st.sidebar.form("parameters_form"):
        email = st.text_input(
            "Enter your email for NCBI Entrez:",
            value="",
            help="Required for accessing NCBI databases.",
            autocomplete="email",
            placeholder="Your@email.com",
        )
        data_source = st.selectbox(
            "Select data source:", ["NCBI", "UniProt", "Upload FASTA"]
        )
        uploaded_fasta = None
        if data_source == "Upload FASTA":
            uploaded_fasta = st.file_uploader(
                "Upload FASTA file:", type=["fasta", "fa", "faa"]
            )
            query = ""
        else:
            query = st.text_input("Enter your query:", "Hendra henipavirus F")
        sc1, sc2 = st.columns(2)
        use_refseq = sc1.checkbox("Use RefSeq database", value=False)
        remove_PDB = sc2.checkbox("Remove PDB entries from MSA", value=False)
        max_seqs = st.number_input(
            "Enter maximum number of sequences to fetch (-1 for all):",
            min_value=-1,
            value=100,
            step=1,
        )

        # defining the default path for al2co executable
        # detect whether it is windows or linux
        system = platform.system()
        clustalo_default = './exceculatbles/amd64/clustalo'
        al2co_default = './exceculatbles/amd64/al2co'
        
        if system == "Windows":
            clustalo_default = (
                "./exceculatbles/win64/clustalo.exe"
                if os.path.exists(r"./exceculatbles/win64/clustalo.exe")
                else "clustalo"
            )
        else:
            os.chmod("./exceculatbles/amd64/clustalo", 0o777)
            clustalo_default = (
                "./exceculatbles/amd64/clustalo"
                if os.path.exists(r"./exceculatbles/amd64/clustalo")
                else "clustalo"
            )
        clustalo_path = clustalo_default
        
        # clustalo_path = st.text_input(
        #     "Enter the path to Clustal Omega executable:",
        #     value=clustalo_default,
        #     help="Provide the path if Clustal Omega is not in your system PATH.",
        # )
        
        if system == "Windows":
            al2co_default = (
                r"./exceculatbles/win64/al2co.exe"
                if os.path.exists(r"./exceculatbles/win64/al2co.exe")
                else "al2co"
            )

        else:
            os.chmod("./exceculatbles/amd64/al2co", 0o777)
            al2co_default = (
                r"./exceculatbles/amd64/al2co"
                if os.path.exists(r"./exceculatbles/amd64/al2co")
                else "al2co"
            )

        al2co_path = st.text_input(
            "Enter the path to al2co.exe executable:",
            value=al2co_default,
            help="Provide the path if al2co.exe is not in your system PATH.",
        )

        submit_button = st.form_submit_button(label="Fetch Sequences")

    if submit_button:
        if data_source != "Upload FASTA" and not email:
            st.sidebar.error("Email is required for NCBI Entrez access.")
        else:
            welcome.empty()
            try:
                clustalo_path_resolved = get_clustalo_path(clustalo_path)
                al2co_path_resolved = get_al2co_path(al2co_path)
                if clustalo_path_resolved and al2co_path_resolved:
                    filename = "sequences.fasta"
                    with st.spinner("Fetching sequences..."):
                        if data_source == "NCBI":
                            file_path = search_and_save_protein_ncbi(
                                query,
                                filename,
                                email,
                                max_seqs=max_seqs,
                                use_refseq=use_refseq,
                            )
                        elif data_source == "UniProt":
                            file_path = search_and_save_protein_uniprot(
                                query, filename, max_seqs=max_seqs
                            )
                        else:
                            if uploaded_fasta is None:
                                st.error("Please upload a FASTA file.")
                                file_path = None
                            else:
                                file_path = filename
                                with open(file_path, "wb") as f:
                                    f.write(uploaded_fasta.getbuffer())
                    if file_path:
                        proteins_list = get_protein_sequences(
                            file_path, remove_PDB=remove_PDB
                        )
                        st.session_state.proteins_list = proteins_list
                        st.session_state.reference_seq_input = None
                        st.session_state.result = None
                        st.session_state.reference_seq = None
                        st.session_state.msa_done = False
                        st.session_state.filtered = False
                        st.session_state.msa_df = None
                        st.session_state.msa_image = None
                        st.session_state.msa_letters = None
                        st.session_state.msa_outfile = None
                        st.session_state.al2co_output = None
                        st.session_state.al2co_df = None
                        st.session_state.alignment_mapping = (
                            None  # Reset alignment mapping
                        )
                        st.session_state.unique_mutations = {}
                        st.session_state.excluded_sequences = {}
                        st.session_state.mutation_summary = set()
                        st.session_state.all_mutations_str = ""
                        st.session_state.phylogenetic_tree = (
                            None  # Reset phylogenetic_tree
                        )
                        st.success("Sequences fetched and saved successfully.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
                print(traceback.format_exc())

    # Step 2: Select Reference Sequence if not provided
    if st.session_state.get("proteins_list") and not st.session_state.get("msa_done"):
        ref_seq_choice = st.radio(
            "Select reference sequence:",
            ["Select from list", "Use provided sequence"],
            horizontal=True,
        )

        if ref_seq_choice == "Select from list":
            st.header("Select Reference Sequence")

            # First, sort st.session_state.proteins_list based on the sequence ID
            sorted_protein_list = sorted(
                st.session_state.proteins_list, key=lambda seq: seq.id
            )

            # Now create the seq_options from the sorted list
            seq_options = [
                f"{seq.id} (Length: {len(seq.seq)})" for seq in sorted_protein_list
            ]

            # You can reverse the sorted list if needed
            seq_options = seq_options[::-1]  # Reverse the sorted seq_options
            sorted_protein_list = sorted_protein_list[
                ::-1
            ]  # Reverse the sorted proteins list as well

            # Present the sorted options in the selectbox
            selected_seq = st.selectbox("Select Reference Sequence:", seq_options)
            if selected_seq:
                selected_index = seq_options.index(selected_seq)
                st.session_state.reference_seq = str(
                    sorted_protein_list[selected_index].seq
                ).upper()
                st.success("Reference sequence selected.")
        else:
            st.header("Provide Reference Sequence")
            st.session_state.reference_seq = clean_fasta(
                st.text_area(
                    "Enter the reference sequence (1-letter code):", height=200
                )
                .strip()
                .upper()
            )
            if st.session_state.reference_seq:
                st.success("Reference sequence provided.")

        c1, c2 = st.columns(2)

        # Show Start Alignment button only after reference sequence is set
        if c1.button("Start Alignment") and st.session_state.get("reference_seq"):
            with st.spinner("Performing pairwise alignments..."):
                try:
                    result, reference_seq = perform_alignment(
                        st.session_state.proteins_list, st.session_state.reference_seq
                    )
                    st.session_state.result = result
                    st.session_state.reference_seq = reference_seq
                    st.session_state.msa_done = False
                    st.session_state.filtered = False
                    st.session_state.msa_df = None
                    st.session_state.msa_image = None
                    st.session_state.msa_letters = None
                    st.session_state.msa_outfile = None
                    st.session_state.al2co_output = None
                    st.session_state.al2co_df = None
                    st.session_state.phylogenetic_tree = None  # Reset phylogenetic_tree
                except Exception as e:
                    st.error(f"An error occurred during alignment: {e}")
                    print(traceback.format_exc())

    # **New Functionality: Upload Own MSA**
    # This needs more work to integrate with the existing code
    # if not st.session_state.get("msa_done"):
    #     st.header("Upload Your Own MSA")
    #     uploaded_msa = st.file_uploader(
    #         "Upload MSA File (FASTA or Clustal format):",
    #         type=["fasta", "fa", "aln", "clustal"],
    #     )
    #     if uploaded_msa:
    #         try:
    #             msa_content = uploaded_msa.read().decode("utf-8")
    #             msa_outfile = "uploaded_msa.fasta"
    #             with open(msa_outfile, "w") as f:
    #                 f.write(msa_content)
    #             st.session_state.msa_outfile = msa_outfile
    #             st.session_state.msa_done = True

    #             # Process the uploaded MSA
    #             msa_image, msa_letters = msa_to_image(msa_outfile)
    #             st.session_state.msa_image = msa_image
    #             st.session_state.msa_letters = msa_letters

    #             st.session_state.msa_df = analyze_alignment(
    #                 msa_outfile,
    #                 st.session_state.reference_seq,
    #             )

    #             # Run al2co.exe
    #             al2co_path_resolved = get_al2co_path(al2co_path)
    #             if al2co_path_resolved:
    #                 al2co_output = "al2co_output.txt"
    #                 al2co_result = run_al2co(
    #                     al2co_path_resolved,
    #                     msa_outfile,
    #                     al2co_output,
    #                 )
    #                 if al2co_result:
    #                     st.session_state.al2co_output = al2co_output
    #                     st.session_state.al2co_df = parse_al2co_output(al2co_output)

    #             # After al2co, perform mutation listing
    #             with st.spinner("Identifying point mutations..."):
    #                 (
    #                     unique_mutations,
    #                     excluded_sequences,
    #                     excluded_count,
    #                     mutation_summary,
    #                     all_mutations_str,
    #                 ) = list_unique_point_mutations(
    #                     msa_outfile,
    #                     "reference_sequence",  # Assuming the reference sequence ID is set as "reference_sequence"
    #                     st.session_state.alignment_mapping,
    #                     deletion_threshold=10,
    #                     exclude_mutations_with_X=True,
    #                 )
    #                 st.session_state.unique_mutations = unique_mutations
    #                 st.session_state.excluded_sequences = excluded_sequences
    #                 st.session_state.mutation_summary = mutation_summary
    #                 st.session_state.all_mutations_str = all_mutations_str
    #                 st.session_state.phylogenetic_tree = None  # Reset phylogenetic_tree

    #             st.success("MSA uploaded and processed successfully.")

    #         except Exception as e:
    #             st.error(f"An error occurred while processing the uploaded MSA: {e}")
    #             print(traceback.format_exc())

    # Step 3: Filter Sequences and Perform MSA
    if st.session_state.get("result") and not st.session_state.get("msa_done"):
        st.header("Filter Sequences")
        with st.form("Adjust Filtering Parameters"):
            st.subheader("Filter Parameters")

            min_seq_length = (
                int(st.session_state.min_seq_length)
                if "min_seq_length" in st.session_state
                else 0
            )
            max_seq_length = (
                int(st.session_state.max_seq_length)
                if "max_seq_length" in st.session_state
                else 1000
            )
            min_score = (
                float(st.session_state.min_score)
                if "min_score" in st.session_state
                else 0
            )
            max_score = (
                float(st.session_state.max_score)
                if "max_score" in st.session_state
                else 1000
            )

            # Determine the range based on the results
            all_scores = st.session_state.result[0] if st.session_state.result else []
            all_seq_lengths = (
                [len(seq) for seq in st.session_state.result[1]]
                if st.session_state.result
                else []
            )

            if all_scores:
                min_score = min(all_scores)
                max_score = max(all_scores)
            if all_seq_lengths:
                min_seq_length = min(all_seq_lengths)
                max_seq_length = max(all_seq_lengths)

            seq_len_range = st.slider(
                "Select sequence length range:",
                min_value=min_seq_length,
                max_value=max_seq_length,
                value=(min_seq_length, max_seq_length),
            )
            score_range = st.slider(
                "Select alignment score range:",
                min_value=min_score,
                max_value=max_score,
                value=(min_score, max_score),
            )

            st.markdown("---")
            st.subheader("Fasta sequence filtering")
            st.write(
                "Filtering sequences based on sequence length and alignment score to the reference sequence."
            )
            visualize_hexbin_plot(
                st.session_state.result,
                score_limit=score_range[0],
                score_max=score_range[1],
                seq_min=seq_len_range[0],
                seq_max=seq_len_range[1],
            )

            continue_button = st.form_submit_button("Continue with Alignment")

        if continue_button:
            with st.spinner("Filtering sequences and performing MSA..."):
                try:
                    high_score_seqs, id_array_selected, scores_final = filter_sequences(
                        st.session_state.result,
                        filtering_score=score_range[0],
                        score_max=score_range[1],
                        sequence_len_min=seq_len_range[0],
                        sequence_len_max=seq_len_range[1],
                    )

                    st.write(
                        f"After filtering, **{len(high_score_seqs)}** sequences will be used for MSA with FAMSA."
                    )

                    if high_score_seqs:
                        msa_infile = "msa_in.fasta"
                        msa_outfile = "msa_out.fasta"
                        clustalo_path_resolved = get_clustalo_path(clustalo_path)
                        max_threads = min(os.cpu_count(), 16) or 4
                        if clustalo_path_resolved:
                            msa_outfile = perform_msa_and_fix_header(
                                high_score_seqs,
                                id_array_selected,
                                st.session_state.reference_seq,
                                msa_infile,
                                msa_outfile,
                                clustalo_path_resolved,
                                max_threads,
                            )

                        if msa_outfile:
                            msa_image, msa_letters = msa_to_image(
                                msa_outfile.replace(".fasta", ".aln")
                            )
                            st.session_state.msa_done = True
                            st.session_state.msa_image = msa_image
                            st.session_state.msa_letters = msa_letters
                            st.session_state.msa_outfile = msa_outfile

                            st.session_state.msa_df = analyze_alignment(
                                msa_outfile.replace("fasta", "aln"),
                                st.session_state.reference_seq,
                            )

                            # Run al2co.exe
                            al2co_path_resolved = get_al2co_path(al2co_path)
                            if al2co_path_resolved:
                                al2co_output = "al2co_output.txt"
                                al2co_result = run_al2co(
                                    al2co_path_resolved,
                                    msa_outfile.replace("fasta", "aln"),
                                    al2co_output,
                                )
                                if al2co_result:
                                    st.session_state.al2co_output = al2co_output
                                    st.session_state.al2co_df = parse_al2co_output(
                                        al2co_output
                                    )

                            # After al2co, perform mutation listing
                            with st.spinner("Identifying point mutations..."):
                                (
                                    unique_mutations,
                                    excluded_sequences,
                                    excluded_count,
                                    mutation_summary,
                                    all_mutations_str,
                                ) = list_unique_point_mutations(
                                    msa_outfile.replace("fasta", "aln"),
                                    "reference_sequence",  # Assuming the reference sequence ID is set as "reference_sequence"
                                    st.session_state.alignment_mapping,
                                    deletion_threshold=10,  # You can make this a user parameter if desired
                                    exclude_mutations_with_X=True,  # You can make this a user parameter if desired
                                )
                                st.session_state.unique_mutations = unique_mutations
                                st.session_state.excluded_sequences = excluded_sequences
                                st.session_state.mutation_summary = mutation_summary
                                st.session_state.all_mutations_str = all_mutations_str
                                st.session_state.phylogenetic_tree = (
                                    None  # Reset phylogenetic_tree
                                )
                except Exception as e:
                    st.error(f"An error occurred during filtering and MSA: {e}")
                    print(traceback.format_exc())

    # Step 4: Display MSA Results, Conservation Analysis, al2co Output, and Mutations

    if st.session_state.get("msa_done"):
        try:
            plot_msa_image(st.session_state.msa_image, st.session_state.msa_letters)
        except Exception as e:
            st.error(f"An error occurred while plotting MSA image: {e}")
            print(traceback.format_exc())

        # Display al2co.exe Output
        if st.session_state.get("al2co_output"):
            st.header("Conservation Scoring with al2co.exe")
            with st.expander("View al2co.exe Output"):
                try:
                    with open(st.session_state.al2co_output, "r") as f:
                        al2co_content = f.read()
                    st.text_area("al2co.exe Output", value=al2co_content, height=300)
                except Exception as e:
                    print(traceback.format_exc())
                    st.error(
                        f"An error occurred while reading the al2co.exe output: {e}"
                    )

            # Plotting conservation scores from al2co.txt
            if st.session_state.get("al2co_df") is not None:
                st.subheader("Conservation Scores from al2co")
                plot_al2co_type = st.selectbox(
                    "Select plot type for al2co conservation scores:",
                    ["Plotly", "Seaborn"],
                )
                if plot_al2co_type == "Seaborn":
                    visualize_al2co_seaborn(st.session_state.al2co_df)
                else:
                    visualize_al2co_plotly(st.session_state.al2co_df)

    # Display Mutation Information
    if st.session_state.get("unique_mutations"):
        st.header("Point Mutations Analysis")

        st.subheader("Unique Mutations Per Sequence")
        mutations_df = pd.DataFrame(
            list(st.session_state.unique_mutations.items()),
            columns=["Sequence ID", "Mutations"],
        )
        st.dataframe(mutations_df, hide_index=True, use_container_width=True)

        st.subheader("All Unique Point Mutations")
        if st.session_state.all_mutations_str:
            st.text_area(
                f"All Unique Mutations ({len(st.session_state.all_mutations_str.split(';'))})",
                value=st.session_state.all_mutations_str,
                height=100,
            )
        else:
            st.write("No mutations to display.")

        # Save mutation data into CSV files
        mutations_df_export = get_mutation_dataframe()
        if not mutations_df_export.empty:
            mutations_df_export.to_csv("point_mutations.csv", index=False, sep="\t")
            st.success("Point mutations saved to `point_mutations.csv`.")

        # Create unique mutations DataFrame
        unique_mutations_df = mutations_df_export.drop_duplicates(subset=["Mutation"])
        if not unique_mutations_df.empty:
            unique_mutations_df.to_csv("unique_point_mutations.csv", index=False)
            st.success("Unique point mutations saved to `unique_point_mutations.csv`.")

        c1, c2 = st.columns(2)

        # **New Visualization: Point Mutations Scatter Plot**
        c1.subheader("Point Mutations Visualization")
        visualize_mutations_scatter(
            reference_seq=st.session_state.reference_seq,
            unique_mutations=st.session_state.unique_mutations,
            st_column=c1,
        )
        ##########################################
        # **New Functionality: Phylogenetic Tree**
        ##########################################

        c2.subheader("Phylogenetic Tree")
        if st.session_state.get("phylogenetic_tree") is None:
            if not len(st.session_state.result[0]) > 200:
                tree_file = generate_phylogenetic_tree(st.session_state.msa_outfile)
                if tree_file:
                    st.session_state.phylogenetic_tree = (
                        tree_file  # Store the tree file path
                    )
                    plot_phylogenetic_tree(tree_file, c2)
                else:
                    c2.warning("Phylogenetic tree could not be generated.")
            else:
                c2.warning(
                    f"Phylogenetic tree generation is disabled for more than 200 sequences. Number of sequences: {len(st.session_state.result[0])}"
                )
                if c2.checkbox("Generate tree with random sample of 200 sequences"):
                    tree_file = generate_phylogenetic_tree(
                        st.session_state.msa_outfile, use_random_subsample=True
                    )
                    if tree_file:
                        st.session_state.phylogenetic_tree = tree_file
                    if st.session_state.phylogenetic_tree:
                        plot_phylogenetic_tree(st.session_state.phylogenetic_tree, c2)
        else:
            plot_phylogenetic_tree(st.session_state.phylogenetic_tree, c2)

    ##########################################
    # Structural Mapping of the AL2CO Score
    ##########################################
    st.header("Structural Mapping of the AL2CO Score")

    if st.session_state.get("msa_done") and st.session_state.get("al2co_output"):
        pdb_server = st.radio(
            "Select PDB source:",
            ["PDB", "AlphaFold", "Upload Your Own PDB"],
            horizontal=True,
        )

        if pdb_server == "PDB":
            sequence_score = st.slider(
                "Sequence Identity Score for PDB search", 0.0, 1.0, 0.9, 0.01
            )
            uniprot_id = None
            selected_uniprot_id = None
        elif pdb_server == "AlphaFold":
            sequence_score = 1
            # Check if there is a UniProt ID in the filtered sequences
            uniprot_ids = extract_uniprot_ids(
                mutations_df["Sequence ID"].to_list(), include_version=False
            )
            if uniprot_ids:
                if len(uniprot_ids) > 1:
                    uniprot_names = []
                    for uniprot_id_item in uniprot_ids:
                        uniprot_meta_data = get_protein_data(uniprot_id_item)
                        if uniprot_meta_data:
                            uniprot_name = (
                                uniprot_meta_data.get("proteinDescription", {})
                                .get("recommendedName", {})
                                .get("fullName")
                                .get("value")
                            )
                            uniprot_kbId = uniprot_meta_data.get("uniProtkbId")
                            uniprot_names.append(f"{uniprot_kbId} - {uniprot_name}")
                    selected_uniprot_id = uniprot_ids[
                        uniprot_names.index(
                            st.selectbox("Select UniProt ID:", uniprot_names)
                        )
                    ]

                    # Extract the UniProt ID from the selected name
                    uniprot_id = selected_uniprot_id
                else:
                    selected_uniprot_id = uniprot_ids[0]
                    uniprot_id = selected_uniprot_id
            else:
                st.warning("No UniProt ID found in the filtered sequences.")
                uniprot_id = None
                selected_uniprot_id = None
        else:
            # **New Functionality: Upload Own PDB**
            st.session_state.user_pdb = st.file_uploader(
                "Upload Your Own PDB File:", type=["pdb", "ent"]
            )
            if st.session_state.user_pdb:
                pdb_file_path = "uploaded_structure.pdb"
                with open(pdb_file_path, "wb") as f:
                    f.write(st.session_state.user_pdb.getbuffer())
                st.success("PDB file uploaded successfully.")
                sequence_score = None
                uniprot_id = None
                selected_uniprot_id = None
            else:
                st.warning("Please upload a PDB file to proceed.")
                sequence_score = (
                    None  # Prevent further processing until PDB is uploaded
                )

        frequency_threshold = 0.1  # Needs further testing
        # st.slider(
        #    "Frequency Threshold for Residue Annotation", 0.0, 1.0, 0.1, 0.001
        # )

        # -----------------------------------------
        # 3. Process the PDB Chain
        # -----------------------------------------
        try:
            c1, c2 = st.columns([1, 2])
            c11, c12 = c1.columns(2)

            with st.spinner("Processing PDB Chains..."):
                if pdb_server == "Upload Your Own PDB" and st.session_state.user_pdb:
                    result = process_pdb_chain(
                        seq=st.session_state.reference_seq,
                        al2co_score=st.session_state.al2co_df,
                        frequency_threshold=frequency_threshold,
                        own_pdb=pdb_file_path,
                        st_column=c11,
                    )
                else:
                    result = process_pdb_chain(
                        seq=st.session_state.reference_seq,
                        al2co_score=st.session_state.al2co_df,
                        sequence_score=sequence_score,
                        frequency_threshold=frequency_threshold,
                        uniprot_id=uniprot_id,
                        st_column=c11,
                    )

            # Display Metadata
            c1.write("### PDB Metadata")
            c1.write(f"**PDB ID:** {result['metadata']['pdb_id']}")
            c1.write(f"**Structure Title:** {result['metadata']['title']}")
            c1.write(f"**Authors:** {result['metadata']['authors']}")
            c1.write(f'**Deposition date:** {result["metadata"]["date"]}')

            # Get the list of chains
            chain_ids = list(result["chain_data"].keys())

            # Ask user if they want to specify a different chain
            specify_chain = c12.checkbox("Specify a different chain per PDB")

            if specify_chain:
                # Allow user to select a chain from the list
                selected_chain = c12.selectbox("Select a chain to visualize", chain_ids)
                c1.write(
                    f"**Pairwise-aligment statistics for chain:** `{selected_chain}` with Alignment Score: `{result['chain_data'][selected_chain]['alignment_score']}`"
                )
            else:
                # Automatically select the chain with the highest alignment score
                selected_chain = max(
                    result["chain_data"],
                    key=lambda x: result["chain_data"][x]["alignment_score"],
                )
                c1.write(
                    f"**Best Matching Chain:** `{selected_chain}` with Alignment Score: `{result['chain_data'][selected_chain]['alignment_score']}`"
                )
            if (
                result["chain_data"][selected_chain]["alignment_score"]
                < len(st.session_state.reference_seq) * 0.2
            ):
                c1.warning("Low alignment score. Consider using a different chain.")

            # Get data for the selected chain
            chain_data = result["chain_data"][selected_chain]

            # Display Pairwise Sequence Alignment
            c1.subheader("Pairwise Sequence Alignment")
            with c1.expander("Show Alignment Details"):
                st.text(chain_data["alignment"])

            # Render the Selected Chain with B-factor Coloring
            c2.write("### Selected Chain Visualization")
            xyzview = py3Dmol.view()
            with open(chain_data["updated_pdb_path"], "r") as pdb_file:
                pdb_data = pdb_file.read()
            xyzview.addModel(pdb_data, "pdb")

            color_list = ["#ff2600", "#ffc04d", "#deddda"]
            unique_scores = sorted(set(chain_data["residue_score_map"].values()))
            number_of_colors = len(unique_scores)

            # Identify the most frequent score
            score_counter = Counter(chain_data["residue_score_map"].values())
            max_freq = max(score_counter.values())

            scores_not_none = [x for x in score_counter.keys() if x is not None]

            # Identify less frequent scores
            less_frequent_values = []
            for val, cnt in score_counter.items():
                if cnt < frequency_threshold * max_freq and val != max(scores_not_none):
                    less_frequent_values.append(val)

            annotation_residues = [
                (k, v)
                for k, v in chain_data["residue_score_map"].items()
                if v in less_frequent_values
            ]

            # Generate linear gradient colorscheme based on al2co scores
            gradient_values = generate_custom_gradient(color_list, number_of_colors)
            gradient_dict = dict(zip(unique_scores, gradient_values))

            xyzview.setStyle(
                {
                    "cartoon": {
                        "colorscheme": {
                            "prop": "b",
                            "gradient": "linear",
                            "colors": color_list,
                            "min": min(unique_scores),
                            "max": max(unique_scores),
                        }
                    }
                }
            )

            c3, c4 = c2.columns(2)
            annotation = c3.checkbox("Show Annotation")
            style = c4.radio("Select Style", ["Stick", "Sphere", "Cartoon"]).lower()

            for residue in annotation_residues:
                res_num, score = residue
                res_num = int(res_num)
                color = gradient_dict.get(
                    score, "#deddda"
                )  # Default color if not found
                if annotation:
                    xyzview.addLabel(
                        str(res_num),
                        {
                            "fontOpacity": 1,
                            "fontColor": "black",
                            "backgroundColor": color,
                        },
                        {"resi": res_num},
                    )
                xyzview.addStyle(
                    {"chain": selected_chain, "resi": res_num},
                    {style: {"color": color}},
                )

            xyzview.zoomTo()
            with c2:
                showmol(xyzview, width=1000, height=500)

        except Exception as e:
            print(traceback.format_exc())
            st.error(f"An error occurred during PDB processing: {e}")

    # File download section
    if st.session_state.get("msa_done"):
        st.header("Download Results")

        with open(f"{clean_string_for_path(query)}.txt", "w") as f:
            f.write(f"Query: {query}\n")
            f.write(
                f"Number of sequences fetched: {len(st.session_state.proteins_list)}\n"
            )
            f.write(
                f"Job ran at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )

        pdbs = glob.glob("./PDB_processing/*.pdb")
        # Collect all relevant file paths
        file_paths = [
            "sequences.fasta",
            "msa_in.fasta",
            "msa_out.fasta",
            "msa_out.aln",
            "al2co_output.txt",
            "msa_plot.html",
            "al2co_plot.html",
            "point_mutations.csv",
            "unique_point_mutations.csv",
            "mutations_scatter.html",
            "selected_al2co_labeled.pdb",
            "phylogenetic_tree.nwk",
            "phylogenetic_tree.html",
            f"{clean_string_for_path(query)}.txt",
        ]

        file_paths += pdbs
        # Remove any None or non-existing files
        file_paths = [file for file in file_paths if file and os.path.exists(file)]

        if file_paths:
            # Create zip in memory
            zip_buffer = create_zip_file(file_paths, "analysis_files.zip")
            if zip_buffer:
                st.download_button(
                    label="Download All Results as Zip",
                    data=zip_buffer,
                    file_name="analysis_files.zip",
                    mime="application/zip",
                    key="download_zip",
                )
        else:
            st.info("No files available for download.")

    # Option to redo alignment
    c1, c2 = st.columns([1, 1])
    st.sidebar.header("Rerun Pairwise Alignment")
    if st.sidebar.button("Redo Alignment"):
        st.session_state.msa_done = False
        st.session_state.filtered = False
        st.session_state.msa_df = None
        st.session_state.msa_image = None
        st.session_state.msa_letters = None
        st.session_state.msa_outfile = None
        st.session_state.al2co_output = None
        st.session_state.al2co_df = None
        st.session_state.unique_mutations = {}
        st.session_state.excluded_sequences = {}
        st.session_state.mutation_summary = set()
        st.session_state.all_mutations_str = ""
        st.session_state.phylogenetic_tree = None  # Reset phylogenetic_tree
        st.success("Alignment reset. You can perform a new alignment.")
        st.rerun()

    st.sidebar.markdown(
        """
Developed by [Dawid Zyla](mailto:dzyla@lji.org)
                     
Source code: [Github](https://github.com/dzyla/SPACE)
                    
"""
    )


if __name__ == "__main__":
    main()
