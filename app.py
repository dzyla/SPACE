# app.py

import datetime
import glob
import io
import os
import platform
import traceback
import uuid
import zipfile
from collections import Counter
from pathlib import Path
from typing import List, Optional

import pandas as pd
import plotly.express as px
import py3Dmol
import stmol
import streamlit as st
from stmol import showmol

from space.alignment import *
from space.analysis import *
from space.fetch import *
from space.pdb_processing import *
from space.utils import *
from space.visualization import *

# -----------------------------------
# Configuration
# -----------------------------------

st.set_page_config(
    page_title="SPACE",
    layout="wide",
    page_icon=":stars:",
    initial_sidebar_state="expanded",
)

# Create a per-session output directory
if "session_dir" not in st.session_state:
    session_id = uuid.uuid4().hex
    session_dir = Path(f"sessions/{session_id}")
    session_dir.mkdir(parents=True, exist_ok=True)
    st.session_state.session_dir = session_dir
else:
    session_dir = st.session_state.session_dir

def out_path(fname: str) -> str:
    """
    Prefix all filenames with the session directory.
    """
    return str(session_dir / fname)

def create_zip_file(file_paths: List[str], zip_name: str) -> Optional[io.BytesIO]:
    """
    Creates a zip file in memory from the given list of file paths.

    Args:
        file_paths (List[str]): List of file paths to include in the zip.
        zip_name (str): The name of the resulting zip file.

    Returns:
        Optional[io.BytesIO]: The BytesIO object containing the zip file data,
                              or None if no files.
    """
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

def main():
    st.title("Sequence Protein Alignment and Conservation Engine (SPACE) :stars:")
    welcome = st.empty()

    if "result" not in st.session_state:
        welcome.markdown(
            """
Welcome to **Sequence Protein Alignment and Conservation Engine (SPACE)**, a tool for exploring
and understanding protein sequences and their conservation in 2D and 3D.

## Description

- **Fetch Protein Sequences:** Retrieve protein sequences from databases like **NCBI** and **UniProt**.
- **Reference Sequence Selection:** Choose or provide a reference sequence for alignment.
- **Pairwise Alignment:** Filter sequences by alignment score.
- **Multiple Sequence Alignment (MSA):** Perform and visualize alignments with Clustal Omega or FAMSA.
- **Conservation Scoring:** Assess conservation using AL2CO.
- **Mutation Analysis:** Identify unique point mutations.
- **Phylogenetic Tree Generation:** Visualize evolutionary relationships.
- **Structural Mapping:** Map conservation scores onto 3D structures.
- **Download Results:** Bundle all outputs in a ZIP.

## How to cite

- Zyla, D. **SPACE**: Sequence Protein Alignment and Conservation Engine. Streamlit app.
- Based on Zyla, D. et al. *Science*(2024). DOI:10.1126/science.adm8693
"""
        )
    else:
        welcome.empty()

    # Initialize session state
    if "result" not in st.session_state:
        st.session_state.update({
            "result": None,
            "reference_seq": None,
            "msa_done": False,
            "filtered": False,
            "msa_df": None,
            "msa_image": None,
            "msa_letters": None,
            "msa_outfile": None,
            "al2co_output": None,
            "al2co_df": None,
            "alignment_mapping": None,
            "unique_mutations": {},
            "excluded_sequences": {},
            "mutation_summary": set(),
            "all_mutations_str": "",
            "proteins_list": None,
            "phylogenetic_tree": None,
            "user_pdb": None,
            "show_msa": False,
        })

    # Sidebar inputs
    st.sidebar.header("Parameters")
    with st.sidebar.form("parameters_form"):
        email = st.text_input(
            "Enter your email for NCBI Entrez:",
            value="",
            help="Required for accessing NCBI databases.",
            autocomplete="email",
            placeholder="you@example.com",
        )
        data_source = st.selectbox("Select data source:", ["NCBI", "UniProt"])
        st.info('Check the query hits [here](https://www.ncbi.nlm.nih.gov/protein/)')
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

        system = platform.system()
        if system == "Windows":
            clustalo_default = "clustalo"
            al2co_default = "al2co"
        else:
            clustalo_default = "./exceculatbles/amd64/clustalo"
            al2co_default = "./exceculatbles/amd64/al2co"
            os.chmod(clustalo_default, 0o777)
            os.chmod(al2co_default, 0o777)

        clustalo_path = clustalo_default
        al2co_path = st.text_input(
            "Enter the path to al2co executable:",
            value=al2co_default,
            help="Provide the path if al2co is not in your system PATH.",
        )

        submit_button = st.form_submit_button(label="Fetch Sequences")

    # Fetch sequences
    if submit_button:
        if data_source == "NCBI" and not email:
            st.sidebar.error("Email is required for NCBI Entrez access.")
        else:
            welcome.empty()
            try:
                clustalo_path_resolved = get_clustalo_path(clustalo_path)
                al2co_path_resolved = get_al2co_path(al2co_path)
                if clustalo_path_resolved and al2co_path_resolved:
                    fasta_path = out_path("sequences.fasta")
                    with st.spinner("Fetching sequences..."):
                        if data_source == "NCBI":
                            file_path = search_and_save_protein_ncbi(
                                query,
                                fasta_path,
                                email,
                                max_seqs=max_seqs,
                                use_refseq=use_refseq,
                            )
                        else:
                            file_path = search_and_save_protein_uniprot(
                                query, fasta_path, max_seqs=max_seqs
                            )
                    if file_path:
                        proteins_list = get_protein_sequences(
                            file_path, remove_PDB=remove_PDB
                        )
                        st.session_state.proteins_list = proteins_list
                        # Reset downstream state
                        for key in [
                            "result", "reference_seq", "msa_done", "filtered",
                            "msa_df", "msa_image", "msa_letters", "msa_outfile",
                            "al2co_output", "al2co_df", "alignment_mapping",
                            "unique_mutations", "excluded_sequences",
                            "mutation_summary", "all_mutations_str",
                            "phylogenetic_tree"
                        ]:
                            if key == "msa_done":
                                st.session_state[key] = False
                            elif key in ["unique_mutations", "excluded_sequences"]:
                                st.session_state[key] = {}
                            elif key == "mutation_summary":
                                st.session_state[key] = set()
                            elif key == "all_mutations_str":
                                st.session_state[key] = ""
                            else:
                                st.session_state[key] = None
                        st.success("Sequences fetched and saved successfully.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.error(traceback.format_exc())

    # Step 2: Reference Sequence Selection
    if st.session_state.get("proteins_list") and not st.session_state.get("msa_done"):
        ref_seq_choice = st.radio(
            "Select reference sequence:",
            ["Select from list", "Use provided sequence"],
            horizontal=True,
        )

        if ref_seq_choice == "Select from list":
            st.header("Select Reference Sequence")
            sorted_protein_list = sorted(
                st.session_state.proteins_list, key=lambda seq: seq.id
            )
            seq_options = [
                f"{seq.id} (Length: {len(seq.seq)})" for seq in sorted_protein_list
            ][::-1]
            sorted_protein_list = sorted_protein_list[::-1]

            selected_seq = st.selectbox("Select Reference Sequence:", seq_options)
            if selected_seq:
                idx = seq_options.index(selected_seq)
                st.session_state.reference_seq = str(
                    sorted_protein_list[idx].seq
                ).upper()
                st.success("Reference sequence selected.")
        else:
            st.header("Provide Reference Sequence")
            ref = st.text_area(
                "Enter the reference sequence (1-letter code):", height=200
            ).strip().upper()
            cleaned = clean_fasta(ref)
            st.session_state.reference_seq = cleaned
            if cleaned:
                st.success("Reference sequence provided.")

        c1, c2 = st.columns(2)
        if c1.button("Start Alignment") and st.session_state.get("reference_seq"):
            with st.spinner("Performing pairwise alignments..."):
                try:
                    result, reference_seq = perform_alignment(
                        st.session_state.proteins_list,
                        st.session_state.reference_seq,
                    )
                    st.session_state.result = result
                    st.session_state.reference_seq = reference_seq
                    # Reset downstream
                    for key in [
                        "msa_done", "filtered", "msa_df", "msa_image",
                        "msa_letters", "msa_outfile", "al2co_output",
                        "al2co_df", "phylogenetic_tree"
                    ]:
                        if key == "msa_done":
                            st.session_state[key] = False
                        else:
                            st.session_state[key] = None
                except Exception as e:
                    st.error(f"An error occurred during alignment: {e}")
                    st.error(traceback.format_exc())

    # Step 3: Filter & MSA
    if st.session_state.get("result") and not st.session_state.get("msa_done"):
        st.header("Filter Sequences")
        with st.form("Adjust Filtering Parameters"):
            st.subheader("Filter Parameters")
            all_scores = st.session_state.result[0]
            all_seq_lengths = [len(seq) for seq in st.session_state.result[1]]
            min_score = min(all_scores) if all_scores else 0
            max_score = max(all_scores) if all_scores else 1000
            min_seq_length = min(all_seq_lengths) if all_seq_lengths else 0
            max_seq_length = max(all_seq_lengths) if all_seq_lengths else 1000
            c1, c2 = st.columns(2)
            score_range = c1.slider(
                "Select alignment score range:",
                min_value=float(min_score),
                max_value=float(max_score),
                value=(float(min_score), float(max_score)),
            )
            seq_len_range = c2.slider(
                "Select sequence length range:",
                min_value=min_seq_length,
                max_value=max_seq_length,
                value=(min_seq_length, max_seq_length),
            )

            st.markdown("---")
            st.subheader("Fasta sequence filtering")
            st.write(
                "Filtering sequences based on sequence length and alignment score."
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
                        f"After filtering, **{len(high_score_seqs)}** sequences will be used for MSA."
                    )
                    if high_score_seqs:
                        msa_infile = out_path("msa_in.fasta")
                        msa_outfile = out_path("msa_out.fasta")
                        clustalo_resolved = get_clustalo_path(clustalo_path)
                        max_threads = min(os.cpu_count() or 4, 16)
                        if clustalo_resolved:
                            msa_outfile = perform_msa_and_fix_header(
                                high_score_seqs,
                                id_array_selected,
                                st.session_state.reference_seq,
                                msa_infile,
                                msa_outfile,
                                clustalo_resolved,
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

                            # Run al2co
                            al2co_resolved = get_al2co_path(al2co_path)
                            if al2co_resolved:
                                al2co_output = out_path("al2co_output.txt")
                                ok = run_al2co(
                                    al2co_resolved,
                                    msa_outfile.replace("fasta", "aln"),
                                    al2co_output,
                                )
                                if ok:
                                    st.session_state.al2co_output = al2co_output
                                    st.session_state.al2co_df = parse_al2co_output(
                                        al2co_output
                                    )

                            # Identify point mutations
                            (
                                unique_mutations,
                                excluded_sequences,
                                excluded_count,
                                mutation_summary,
                                all_mutations_str,
                            ) = list_unique_point_mutations(
                                msa_outfile.replace("fasta", "aln"),
                                "reference_sequence",
                                st.session_state.alignment_mapping,
                                deletion_threshold=10,
                                exclude_mutations_with_X=True,
                            )
                            st.session_state.unique_mutations = unique_mutations
                            st.session_state.excluded_sequences = excluded_sequences
                            st.session_state.mutation_summary = mutation_summary
                            st.session_state.all_mutations_str = all_mutations_str
                            st.session_state.phylogenetic_tree = None
                except Exception as e:
                    st.error(f"An error occurred during filtering and MSA: {e}")
                    st.error(traceback.format_exc())

    # Step 4: Display MSA & Conservation
    if st.session_state.get("msa_done"):
        try:
            show_msa = st.checkbox("Show MSA", value=st.session_state.show_msa)
            st.session_state.show_msa = show_msa
            if show_msa:
                plot_msa_image(
                    st.session_state.msa_image, st.session_state.msa_letters, folder=session_dir
                )
        except Exception as e:
            st.error(f"An error occurred while plotting MSA image: {e}")
            st.error(traceback.format_exc())

        if st.session_state.get("al2co_output"):
            st.header("Conservation Scoring with al2co.exe")
            with st.expander("View al2co.exe Output"):
                try:
                    content = open(st.session_state.al2co_output, "r").read()
                    st.text_area("al2co.exe Output", value=content, height=300)
                except Exception as e:
                    st.error(f"Error reading al2co output: {e}")
                    st.error(traceback.format_exc())

            if st.session_state.get("al2co_df") is not None:
                st.subheader("Conservation Scores from al2co")
                choice = st.selectbox(
                    "Select plot type for al2co conservation scores:",
                    ["Plotly", "Seaborn"],
                )
                if choice == "Seaborn":
                    visualize_al2co_seaborn(st.session_state.al2co_df, session_dir)
                else:
                    visualize_al2co_plotly(st.session_state.al2co_df, session_dir)

    # Point Mutations Analysis with switchable histogram plots
    if st.session_state.get("unique_mutations"):
        st.header("Point Mutations Analysis")

        # Unique mutations per sequence table
        st.subheader("Unique Mutations Per Sequence")
        muts_df = pd.DataFrame(
            list(st.session_state.unique_mutations.items()),
            columns=["Sequence ID", "Mutations"],
        )
        st.dataframe(muts_df, hide_index=True, use_container_width=True)

        # All unique mutations text
        st.subheader("All Unique Point Mutations")
        if st.session_state.all_mutations_str:
            count = len(st.session_state.all_mutations_str.split(";"))
            st.text_area(
                f"All Unique Mutations ({count})",
                value=st.session_state.all_mutations_str,
                height=100,
            )
        else:
            st.write("No mutations to display.")
        c1, c2 = st.columns(2)

        # Save CSVs
        muts_export = get_mutation_dataframe()
        if not muts_export.empty:
            csv1 = out_path("point_mutations.csv")
            muts_export.to_csv(csv1, index=False, sep="\t")
            st.success(f"Point mutations saved to `{csv1}`.")
        unique_muts_df = muts_export.drop_duplicates(subset=["Mutation"])
        if not unique_muts_df.empty:
            csv2 = out_path("unique_point_mutations.csv")
            unique_muts_df.to_csv(csv2, index=False)
            st.success(f"Unique point mutations saved to `{csv2}`.")

        # Switchable histogram plots
        c1.subheader("Mutation Frequency Analysis")
        plot_type = c1.radio(
            "Select plot type:",
            ["Mutation frequency histogram", "Mutations per residue index"],
            horizontal=True,
        )

        if plot_type == "Mutation frequency histogram":
            # Compute mutation counts
            counts = muts_export["Mutation"].value_counts().reset_index()

            counts.columns = ["Mutation", "Count"]
            
            # Show the mutation counts as a fraction of all sequences
            max_count = st.session_state.msa_image.shape[0]
            counts['Count'] = counts['Count']/max_count*100
            fig = px.bar(
                counts,
                x="Mutation",
                y="Count",
                title="Frequency of Each Point Mutation",
                labels={"Count": "Frequency (%)"},
            )
            c1.plotly_chart(fig, use_container_width=True)
        else:
            # Compute unique mutation types per residue index
            unique_muts_df["Index"] = unique_muts_df["Mutation"].str.extract(r"(\d+)").astype(int)
            idx_counts = (
                unique_muts_df.groupby("Index")
                .size()
                .reset_index(name="Unique Mutations Count")
            )
            fig = px.bar(
                idx_counts,
                x="Index",
                y="Unique Mutations Count",
                title="Unique Mutation Types per Residue Index",
            )
            c1.plotly_chart(fig, use_container_width=True)

        # Phylogenetic Tree
        c2.subheader("Phylogenetic Tree")
        if st.session_state.get("phylogenetic_tree") is None:
            seq_count = len(st.session_state.result[0])
            if seq_count <= 200:
                tree_file = generate_phylogenetic_tree(st.session_state.msa_outfile, folder=session_dir)
                if tree_file:
                    st.session_state.phylogenetic_tree = tree_file
                    plot_phylogenetic_tree(tree_file, c2)
                else:
                    c2.warning("Phylogenetic tree could not be generated.")
            else:
                c2.warning(
                    f"Tree generation disabled for >200 sequences (current: {seq_count})."
                )
                if c2.checkbox("Generate tree with random 200 seq sample"):
                    tree_file = generate_phylogenetic_tree(
                        st.session_state.msa_outfile, use_random_subsample=True, folder=session_dir
                    )
                    if tree_file:
                        st.session_state.phylogenetic_tree = tree_file
                        plot_phylogenetic_tree(tree_file, c2)
        else:
            plot_phylogenetic_tree(st.session_state.phylogenetic_tree, c2)

    # Structural Mapping
    st.header("Structural Mapping of the AL2CO Score")
    if st.session_state.get("msa_done") and st.session_state.get("al2co_output"):
        pdb_server = st.radio(
            "Select PDB source:",
            ["PDB", "AlphaFold", "Upload Your Own PDB"],
            horizontal=True,
        )
        sequence_score = None
        uniprot_id = None
        uploaded = None

        if pdb_server == "PDB":
            sequence_score = st.slider(
                "Sequence Identity Score for PDB search", 0.0, 1.0, 0.9, 0.01
            )
        elif pdb_server == "AlphaFold":
            sequence_score = 1.0
            uniprot_ids = extract_uniprot_ids(
                muts_df["Sequence ID"].tolist(), include_version=False
            )
            if uniprot_ids:
                if len(uniprot_ids) > 1:
                    names = []
                    for uid in uniprot_ids:
                        meta = get_protein_data(uid)
                        if meta:
                            name = (
                                meta["proteinDescription"]["recommendedName"]["fullName"]["value"]
                            )
                            names.append(f"{uid} - {name}")
                    sel = st.selectbox("Select UniProt ID:", names)
                    idx = names.index(sel)
                    uniprot_id = uniprot_ids[idx]
                else:
                    uniprot_id = uniprot_ids[0]
            else:
                st.warning("No UniProt ID found.")
        else:
            uploaded = st.file_uploader("Upload Your Own PDB File:", type=["pdb", "ent"])
            if uploaded:
                pdb_file = out_path("uploaded_structure.pdb")
                with open(pdb_file, "wb") as f:
                    f.write(uploaded.getbuffer())
                st.success("PDB file uploaded.")
                uniprot_id = None
            else:
                st.warning("Please upload a PDB file.")
                uniprot_id = None

        frequency_threshold = 0.1

        try:
            c1, c2 = st.columns([1, 2])
            c11, c12 = c1.columns(2)
            with st.spinner("Processing PDB Chains..."):
                if pdb_server == "Upload Your Own PDB" and uploaded:
                    result = process_pdb_chain(
                        seq=st.session_state.reference_seq,
                        al2co_score=st.session_state.al2co_df,
                        frequency_threshold=frequency_threshold,
                        own_pdb=pdb_file,
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
            # Metadata
            md = result["metadata"]
            c1.write(f"**PDB ID:** {md['pdb_id']}")
            c1.write(f"**Title:** {md['title']}")
            c1.write(f"**Authors:** {md['authors']}")
            c1.write(f"**Deposition date:** {md['date']}")

            chain_ids = list(result["chain_data"].keys())
            specify = c12.checkbox("Specify a different chain")
            if specify:
                selected_chain = c12.selectbox("Select chain", chain_ids)
                score = result["chain_data"][selected_chain]["alignment_score"]
                c1.write(f"Chain `{selected_chain}` (Score: {score})")
            else:
                selected_chain = max(
                    chain_ids,
                    key=lambda x: result["chain_data"][x]["alignment_score"],
                )
                score = result["chain_data"][selected_chain]["alignment_score"]
                c1.write(f"Best chain `{selected_chain}` (Score: {score})")
            if score < len(st.session_state.reference_seq) * 0.2:
                c1.warning("Low alignment score; consider another chain.")

            chain_data = result["chain_data"][selected_chain]
            c1.subheader("Pairwise Sequence Alignment")
            with c1.expander("Show Alignment Details"):
                st.code(chain_data["alignment"])

            # 3D visualization
            c2.write("### Selected Chain Visualization")
            xyzview = py3Dmol.view()
            pdb_str = open(chain_data["updated_pdb_path"], "r").read()
            xyzview.addModel(pdb_str, "pdb")

            color_list = ["#ff2600", "#ffc04d", "#deddda"]
            scores = sorted(set(chain_data["residue_score_map"].values()))
            num_colors = len(scores)
            counter = Counter(chain_data["residue_score_map"].values())
            max_freq = max(counter.values())
            less_freq = [
                val for val, cnt in counter.items()
                if cnt < frequency_threshold * max_freq and val != max(scores)
            ]
            annotation_residues = [
                (k, v) for k, v in chain_data["residue_score_map"].items()
                if v in less_freq
            ]
            gradient = generate_custom_gradient(color_list, num_colors)
            grad_dict = dict(zip(scores, gradient))

            xyzview.setStyle({
                "cartoon": {
                    "colorscheme": {
                        "prop": "b",
                        "gradient": "linear",
                        "colors": color_list,
                        "min": min(scores),
                        "max": max(scores),
                    }
                }
            })

            c3, c4 = c2.columns(2)
            annotate = c3.checkbox("Show Annotation")
            style = c4.radio("Select Style", ["Stick", "Sphere", "Cartoon"]).lower()

            for res_num, score in annotation_residues:
                idx = int(res_num)
                col = grad_dict.get(score, "#deddda")
                if annotate:
                    xyzview.addLabel(
                        str(idx),
                        {"fontOpacity": 1, "fontColor": "black", "backgroundColor": col},
                        {"resi": idx},
                    )
                xyzview.addStyle(
                    {"chain": selected_chain, "resi": idx},
                    {style: {"color": col}},
                )

            xyzview.zoomTo()
            with c2:
                showmol(xyzview, width=1000, height=500)

        except Exception as e:
            st.error(f"An error occurred during PDB processing: {e}")
            st.error(traceback.format_exc())

    # Download Results
    if st.session_state.get("msa_done"):
        st.header("Download Results")
        summary_path = out_path(f"{clean_string_for_path(query)}.txt")
        with open(summary_path, "w") as f:
            f.write(f"Query: {query}\n")
            f.write(f"Number of sequences fetched: {len(st.session_state.proteins_list)}\n")
            f.write(f"Run at: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")

        # Gather session files
        all_files = glob.glob(str(session_dir / "*"))
        all_files += glob.glob(str(session_dir / "*/*"))
        zip_buffer = create_zip_file(all_files, "analysis_files.zip")
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

    # Rerun Alignment
    st.sidebar.header("Rerun Pairwise Alignment")
    if st.sidebar.button("Redo Alignment"):
        for key in [
            "msa_done", "filtered", "msa_df", "msa_image", "msa_letters",
            "msa_outfile", "al2co_output", "al2co_df", "unique_mutations",
            "excluded_sequences", "mutation_summary", "all_mutations_str",
            "phylogenetic_tree"
        ]:
            if key == "msa_done":
                st.session_state[key] = False
            elif key in ["unique_mutations", "excluded_sequences"]:
                st.session_state[key] = {}
            elif key == "mutation_summary":
                st.session_state[key] = set()
            elif key == "all_mutations_str":
                st.session_state[key] = ""
            else:
                st.session_state[key] = None
        st.success("Alignment reset.")
        st.rerun()

    st.sidebar.markdown(
        """
Developed by [Dawid Zyla](mailto:dzyla@lji.org)  
Source code: [GitHub](https://github.com/dzyla/SPACE)
"""
    )

if __name__ == "__main__":
    main()
