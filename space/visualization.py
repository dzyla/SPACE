# visualization.py

import re
import numpy as np
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter1d
from Bio import AlignIO
from collections import Counter
import streamlit as st
from typing import List, Optional
import py3Dmol
from stmol import showmol
import math

def visualize_al2co_seaborn(al2co_df: pd.DataFrame, folder: str = None):
    """
    Visualizes al2co conservation scores using Seaborn.

    Args:
        al2co_df (pd.DataFrame): DataFrame with al2co conservation scores.
    """
    if al2co_df is None or al2co_df.empty:
        st.error("No al2co conservation data to visualize.")
        return

    sns.set(style="whitegrid", font_scale=1.3)
    fig, ax = plt.subplots(figsize=(12, 6))

    # Smooth the conservation scores
    ysmoothed = gaussian_filter1d(al2co_df["al2co_score"], sigma=2)

    # Define custom colormap
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_cmap", ["#ff2600", "#ffc04d", "#deddda"]
    )

    # Line plot for trend
    sns.lineplot(
        x=al2co_df["Location"], y=ysmoothed, color="black", label="Trend", ax=ax
    )

    # Scatter plot for conservation scores
    scatter = ax.scatter(
        al2co_df["Location"],
        al2co_df["al2co_score"],
        c=al2co_df["al2co_score"],
        cmap=cmap,
        s=50,
        label="Conservation Score",
    )

    ax.set_xlabel("Residue Position")
    ax.set_ylabel("Conservation Score")
    ax.set_title("Conservation Scores from al2co")
    ax.legend(loc="upper left", frameon=False)

    # Add color bar
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Conservation Score")

    if folder:
        os.makedirs(folder, exist_ok=True)
        fig.savefig(os.path.join(folder, "al2co_plot.png"))

    st.pyplot(fig)

def visualize_al2co_plotly(al2co_df: pd.DataFrame, folder: str = None):
    """
    Visualizes al2co conservation scores using Plotly with a customized colorscale.

    Args:
        al2co_df (pd.DataFrame): DataFrame containing 'Location', 'Residue', and 'al2co_score'.
    """
    if al2co_df is None or al2co_df.empty:
        st.error("No al2co conservation data to visualize.")
        return

    # Drop rows where Location is NaN (gaps in reference)
    al2co_plot = al2co_df.dropna(subset=["Location"])

    if al2co_plot.empty:
        st.error("No valid al2co conservation data after removing gaps.")
        return

    # Smooth the scores
    ysmoothed = gaussian_filter1d(al2co_plot["al2co_score"], sigma=2)

    # Define custom colorscale
    custom_colorscale = [
        [0.0, "#ff2600"],  # Red
        [0.5, "#ffc04d"],  # Orange
        [1.0, "#deddda"],  # Light Blue
    ]

    # Create Plotly figure
    fig = go.Figure()

    # Add smoothed trend line
    fig.add_trace(
        go.Scatter(
            x=al2co_plot["Location"],
            y=ysmoothed,
            mode="lines",
            name="Trend",
            line=dict(color="black"),
        )
    )

    # Add scatter plot for conservation scores
    fig.add_trace(
        go.Scatter(
            x=al2co_plot["Location"],
            y=al2co_plot["al2co_score"],
            mode="markers",
            name="Conservation Score",
            marker=dict(
                color=al2co_plot["al2co_score"],
                colorscale=custom_colorscale,
                size=8,
                colorbar=dict(title="Conservation Score"),
                showscale=True,
            ),
            hovertemplate=(
                "Residue: %{text}<br>"
                "Position: %{x}<br>"
                "Score: %{y:.3f}"
            ),
            text=al2co_plot["Residue"],
        )
    )

    # Update layout
    fig.update_layout(
        title="Conservation Scores from al2co",
        xaxis_title="Residue Position",
        yaxis_title="Conservation Score",
        hovermode="closest",
        height=600,
    )
    # Make sure the directory exists
    if folder:
        os.makedirs(folder, exist_ok=True)
        fig.write_html(os.path.join(folder, "al2co_plot.html"))

    st.plotly_chart(fig, use_container_width=True)

def visualize_hexbin_plot(
    result: list, score_limit: float, score_max: float, seq_min: int, seq_max: int
):
    """
    Visualizes the density of sequence lengths and alignment scores using a heatmap.

    Args:
        result (list): [scores, sequences, IDs].
        score_limit (float): Minimum alignment score.
        score_max (float): Maximum alignment score.
        seq_min (int): Minimum sequence length.
        seq_max (int): Maximum sequence length.
    """
    if not result:
        st.error("No data available for plotting.")
        return

    scores, sequences, ids = result
    filtered_scores = []
    filtered_seq_len = []

    for score, seq in zip(scores, sequences):
        seq_len = len(seq)
        if seq_min <= seq_len <= seq_max and score_limit <= score <= score_max:
            filtered_scores.append(score)
            filtered_seq_len.append(seq_len)

    if not filtered_scores:
        st.warning("No data points match the current filter criteria.")
        return

    # Create 2D histogram
    heatmap, xedges, yedges = np.histogram2d(filtered_seq_len, filtered_scores, bins=50)
    heatmap = heatmap.T  # Transpose for correct orientation

    fig = go.Figure(
        data=go.Heatmap(
            z=np.log(heatmap + 1),  # Added 1 to avoid log(0)
            x=xedges,
            y=yedges,
            colorscale="Portland",
            colorbar=dict(title="Density (log scale)"),  # Updated title to reflect log scale
            customdata=heatmap,                  # Pass original z values for hover
            hovertemplate=                       # Define custom hover template
                'X: %{x}<br>' +
                'Y: %{y}<br>' +
                'Density: %{customdata}<extra></extra>',  # Display original z
            zmin=np.log(heatmap + 1).min(),
            zmax=np.log(heatmap + 1).max(),
        )
    )
    fig.update_layout(
        title="Density Heatmap of Sequence Length and Alignment Score",
        xaxis_title="Sequence Length",
        yaxis_title="Sequence identity, %",
        height=600,
        xaxis=dict(
            showgrid=True,           # Enable vertical grid lines
        ),
        yaxis=dict(
            showgrid=True,           # Enable horizontal grid lines
        ),
        plot_bgcolor='white'         # Optional: Set background color for better contrast
    )

    st.plotly_chart(fig, use_container_width=True)

def msa_to_image(msa_file: str) -> tuple:
    """
    Converts Multiple Sequence Alignment (MSA) to numerical image data and amino acid array.

    Args:
        msa_file (str): Path to the MSA file in Clustal format.

    Returns:
        tuple: (msa_image as numpy.ndarray, msa_letters as numpy.ndarray)

    Raises:
        Exception: If any error occurs during conversion.
    """
    try:
        alignment = AlignIO.read(msa_file, "clustal")
    except Exception as e:
        st.error(f"An error occurred while reading the MSA file: {e}")
        raise e

    AA_CODES = {
        "-": 0,
        "A": 1,
        "C": 2,
        "D": 3,
        "E": 4,
        "F": 5,
        "G": 6,
        "H": 7,
        "I": 8,
        "K": 9,
        "L": 10,
        "M": 11,
        "N": 12,
        "P": 13,
        "Q": 14,
        "R": 15,
        "S": 16,
        "T": 17,
        "V": 18,
        "W": 19,
        "Y": 20,
        "X": 21,
        "B": 22,
        "J": 23,
        "O": 24,
        "Z": 25,
    }

    # Reverse mapping for hover information
    CODE_TO_AA = {v: k for k, v in AA_CODES.items()}

    msa_image = np.zeros((len(alignment), alignment.get_alignment_length()), dtype=int)
    msa_letters = np.empty(
        (len(alignment), alignment.get_alignment_length()), dtype=object
    )

    for i, record in enumerate(alignment):
        for j, aa in enumerate(str(record.seq)):
            code = AA_CODES.get(aa.upper(), 0)
            msa_image[i, j] = code
            msa_letters[i, j] = aa.upper()

    return msa_image, msa_letters


def plot_msa_image(msa_image: np.ndarray, msa_letters: np.ndarray, folder: str = './'):
    """
    Plots the Multiple Sequence Alignment (MSA) as a heatmap with amino acid hover information.

    Args:
        msa_image (numpy.ndarray): Numerical representation of MSA.
        msa_letters (numpy.ndarray): Array of amino acid letters.
    """
    if msa_image is None or msa_letters is None:
        st.error("No MSA image data to plot.")
        return

    # Ensure that msa_image and msa_letters have compatible shapes
    if msa_image.shape != msa_letters.shape:
        st.error("Mismatch between msa_image and msa_letters dimensions.")
        return

    # Convert msa_image and msa_letters to lists for Plotly
    msa_image_list = msa_image.tolist()
    msa_letters_list = msa_letters.tolist()

    # Create hover text by combining X, Y, and amino acid letter
    hover_text = [
        [f"Position: {x}<br>Sequence: {y}<br>Amino Acid: {aa}"
         for x, aa in enumerate(row, start=1)]
        for y, row in enumerate(msa_letters_list, start=1)
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=msa_image_list,
            text=hover_text,
            hoverinfo="text",
            colorscale="Spectral",
            showscale=False  # Hide the color scale to remove extra traces
        )
    )

    # Update hovertemplate to display X, Y, and Text
    # Note: Using hoverinfo="text" and pre-formatted hover_text, so no need for hovertemplate
    # However, if you prefer to use hovertemplate, you can adjust accordingly
    fig.update_traces(
        hovertemplate="%{text}<extra></extra>"  # <extra></extra> removes the trace info
    )

    fig.update_layout(
        title="Multiple Sequence Alignment View",
        xaxis_title="MSA Residue Position",
        yaxis_title="Sequence Number",
        xaxis=dict(ticks='', showticklabels=True),
        yaxis=dict(ticks='', showticklabels=True),
        plot_bgcolor='white'
    )

    # Save the plot as an HTML file (optional)
    fig.write_html(os.path.join(folder, "msa_plot.html"))

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)


def visualize_mutations_scatter(reference_seq: str, unique_mutations: dict, st_column: Optional = None):
    """
    Visualizes point mutations as an interactive scatter plot.

    Args:
        reference_seq (str): Reference protein sequence.
        unique_mutations (dict): Dictionary mapping sequence IDs to mutation strings.
    """
    # Create a DataFrame for the reference sequence
    ref_positions = list(range(1, len(reference_seq) + 1))
    ref_residues = list(reference_seq)
    ref_df = pd.DataFrame({"Position": ref_positions, "Residue": ref_residues})

    # Initialize lists to store mutation data
    mutation_data = []
    mutation_keeper = set()  # Keep track of unique mutations to avoid duplicates

    # Parse the unique mutations
    for seq_id, mut_str in unique_mutations.items():
        if mut_str == "No mutations":
            continue  # Skip sequences with no mutations
        mut_list = mut_str.split(", ")
        for mut in mut_list:
            if mut.startswith("ins"):
                # Insertion mutation
                match = re.match(r"ins(\d+)(\w+)", mut)
                if match:
                    pos = int(match.group(1))
                    mutated_residue = match.group(2)
                    mutation_type = "Insertion"
                    original_residue = "-"
                else:
                    continue  # Skip malformed mutations
            else:
                # Substitution mutation
                match = re.match(r"(\w)(\d+)(\w)", mut)
                if match:
                    original_residue = match.group(1)
                    pos = int(match.group(2))
                    mutated_residue = match.group(3)
                    mutation_type = "Substitution"
                else:
                    continue  # Skip malformed mutations

            if mut not in mutation_keeper:
                mutation_keeper.add(mut)
                mutation_data.append(
                    {
                        "Position": pos,
                        "Mutation": mut,
                        "Mutation Type": mutation_type,
                        "Original Residue": original_residue,
                        "Mutated Residue": mutated_residue,
                    }
                )

    if not mutation_data:
        st.info("No mutations to display.")
        return

    mut_df = pd.DataFrame(mutation_data)

    # Handle multiple mutations at the same position by adding a small y-offset
    mut_df["Mutation Count"] = mut_df.groupby("Position").cumcount()
    mut_df["Y Offset"] = mut_df["Mutation Count"] * 0.2  # 0.2 increments

    # Assign colors based on Mutation Count layers
    color_palette = px.colors.qualitative.Plotly  # Extendable palette
    max_layer = mut_df["Mutation Count"].max()
    required_colors = max_layer + 1  # Layers start at 0
    extended_palette = color_palette * ((required_colors // len(color_palette)) + 1)

    # Assign a color to each layer
    mut_df["Layer"] = mut_df["Mutation Count"]
    mut_df["Color"] = mut_df["Layer"].apply(
        lambda x: extended_palette[x % len(extended_palette)]
    )

    # Create the scatter plot
    fig = go.Figure()

    # Plot reference sequence residues
    fig.add_trace(
        go.Scattergl(
            x=ref_df["Position"],
            y=[0] * len(ref_df),
            mode="markers",
            marker=dict(color="#363636", size=8),
            name="Reference Residues",
            hovertemplate="Position: %{x}<br>Residue: %{text}",
            text=ref_df["Residue"],
        )
    )

    # Plot mutations with colors based on Y Offset layers
    fig.add_trace(
        go.Scattergl(
            x=mut_df["Position"],
            y=mut_df["Y Offset"]
            + 0.2,  # Offset to position mutations above the reference
            mode="markers",
            marker=dict(color=mut_df["Color"], size=10, symbol="circle"),
            text=mut_df["Mutation"],
            hovertemplate="Mutation: %{text}<br>Position: %{x}<br>",
            showlegend=False,  # We'll handle the legend separately
        )
    )

    # Add legend manually for Y Offset layers
    for layer in sorted(mut_df["Layer"].unique()):
        color = extended_palette[layer % len(extended_palette)]
        fig.add_trace(
            go.Scattergl(
                x=[None],  # Invisible markers
                y=[None],
                mode="markers",
                marker=dict(color=color, size=10, symbol="circle"),
                name=f"Layer {int(layer) +1}",
                hoverinfo="none",
            )
        )

    # Update layout for better aesthetics
    fig.update_layout(
        xaxis_title="Residue Position",
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        height=600,
        hovermode="closest",
    )
    # Remove default legend
    fig.update_layout(showlegend=False)

    # Save the mutations scatter plot as HTML
    fig.write_html("mutations_scatter.html")

    # Save the mutations scatter plot as PNG
    # try:
    #     fig.write_image("mutations_scatter.png")
    # except Exception as e:
    #     st.error(f"Failed to save mutations scatter plot as PNG: {e}")

    # Optimize performance by using Scattergl and limiting data if necessary
    if st_column:
        st_column.plotly_chart(fig, use_container_width=True)
    else:
        st.plotly_chart(fig, use_container_width=True)


def generate_custom_gradient(color_list: list, number_of_colors: int) -> list:
    """
    Generates a custom gradient list of colors transitioning through the specified color_list.

    Args:
        color_list (list of str): List of color hex codes (e.g., ["#ff2600", "#ffc04d", "#deddda"]).
        number_of_colors (int): Total number of colors to generate in the gradient.

    Returns:
        list of str: List of color hex codes forming the gradient.

    Raises:
        ValueError: If the number of colors requested is less than the number of specified colors.
    """
    if number_of_colors < len(color_list):
        return ["#deddda"]

    # Number of segments between colors
    num_segments = len(color_list) - 1

    # Calculate the number of colors per segment
    colors_per_segment = [number_of_colors // num_segments] * num_segments
    remainder = number_of_colors % num_segments
    for i in range(remainder):
        colors_per_segment[i] += 1

    gradient = []
    for i in range(num_segments):
        # Create a linear gradient between two consecutive colors
        cmap = mcolors.LinearSegmentedColormap.from_list(
            f"segment_{i}", [color_list[i], color_list[i + 1]]
        )
        # Generate colors for this segment
        segment_colors = [
            mcolors.to_hex(cmap(j / (colors_per_segment[i])))
            for j in range(colors_per_segment[i])
        ]
        gradient.extend(segment_colors)

    return gradient

def calculate_information_content(counts, total_seqs):
    """
    Calculate information content (bits) per amino acid at a single position.

    Uses the Schneider et al. (1986) formula with small-sample correction:
        R(l) = log2(s) - (H(l) + e_n)
    where s=20 (amino acid alphabet), H(l) is the observed Shannon entropy,
    and e_n = (s-1) / (2 * ln(2) * n) is the small-sample correction.

    Args:
        counts: dict mapping amino acid (str) -> count (int).
                Should include ALL residues at this column (including gaps
                counted as a character if desired, or excluded beforehand).
        total_seqs: total number of sequences in the alignment (denominator).

    Returns:
        dict mapping amino acid -> bits contribution (height in logo).
    """
    if total_seqs == 0:
        return {}

    s = 20  # amino acid alphabet size
    s_max = math.log2(s)  # ~4.322 bits

    # Small-sample correction (Schneider et al., 1986)
    e_n = (s - 1) / (2.0 * math.log(2) * total_seqs)

    # Observed probabilities
    probs = {k: v / total_seqs for k, v in counts.items()}

    # Observed Shannon entropy
    h_obs = -sum(p * math.log2(p) for p in probs.values() if p > 0)

    # Information content, clamped to >= 0
    info_content = max(0.0, s_max - h_obs - e_n)

    return {k: p * info_content for k, p in probs.items()}


# Standard WebLogo "chemistry" color scheme for amino acids.
# Groupings:
#   Acidic  (D, E)              -> Red
#   Basic   (R, K, H)           -> Blue
#   Polar   (S, T, N, Q, C, Y)  -> Green
#   Hydrophobic (A, V, L, I, P, W, F, M) -> Black
#   Glycine (G)                 -> Orange (structural breaker, special)
CHEMISTRY_COLORS = {
    # Acidic – red
    'D': '#CC0000', 'E': '#CC0000',
    # Basic – blue
    'R': '#0000CC', 'K': '#0000CC', 'H': '#0000CC',
    # Polar – green
    'S': '#00CC00', 'T': '#00CC00', 'N': '#00CC00', 'Q': '#00CC00',
    'C': '#00CC00', 'Y': '#00CC00',
    # Hydrophobic – black
    'A': '#000000', 'V': '#000000', 'L': '#000000', 'I': '#000000',
    'P': '#000000', 'W': '#000000', 'F': '#000000', 'M': '#000000',
    # Glycine – orange (structural breaker)
    'G': '#FF8C00',
    # Uncommon / ambiguous
    'X': '#808080', 'B': '#808080', 'Z': '#808080', 'J': '#808080',
}

# Color legend groups for the annotation below the logo
_CHEMISTRY_LEGEND = [
    ('Acidic (D, E)', '#CC0000'),
    ('Basic (R, K, H)', '#0000CC'),
    ('Polar (S, T, N, Q, C, Y)', '#00CC00'),
    ('Hydrophobic (A, V, L, I, P, W, F, M)', '#000000'),
    ('Glycine (G)', '#FF8C00'),
]


def visualize_logo_and_consensus(
    msa_file: str,
    alignment_mapping: list,
    identity_threshold: float,
    folder: str = None,
):
    """
    Generates and visualizes a production-grade sequence logo and consensus
    sequence from an MSA file.

    Information content is computed per the Schneider et al. (1986) formula
    with small-sample correction.  Gaps are treated as an observed state when
    computing entropy (they reduce the information content at that position)
    but are **not** drawn as stacked bars—only amino-acid letters are shown.

    Only alignment columns that map to a reference-sequence position (via
    *alignment_mapping*) are included.

    Args:
        msa_file:  Path to the FASTA MSA file.
        alignment_mapping:  List whose i-th element is the reference-sequence
            position for alignment column *i*, or ``None`` for insert columns.
        identity_threshold:  Fraction (0–1).  If the most-frequent amino acid
            at a position has a frequency below this value the consensus
            residue is set to ``'X'``.
        folder:  Optional directory to save the output HTML plot.
    """
    if not msa_file or not os.path.exists(msa_file):
        st.error("MSA file not found.")
        return

    try:
        alignment = AlignIO.read(msa_file, "fasta")
    except Exception as e:
        st.error(f"Error reading MSA file: {e}")
        return

    aln_len = alignment.get_alignment_length()
    n_seqs = len(alignment)

    # Read the alignment into a NumPy array for speed
    arr = np.array([list(str(rec.seq)) for rec in alignment], dtype="U1")

    # ── per-position computation ──────────────────────────────────────────
    logo_data: List[dict] = []   # each entry: {aa: bits, ...}
    consensus_seq: List[str] = []
    positions: List[int] = []

    for i in range(aln_len):
        mapped_ref = alignment_mapping[i]
        if mapped_ref is None:
            continue

        col = arr[:, i]
        counts_raw = Counter(col)

        # Normalise keys to uppercase; keep gaps as '-'
        counts_all: dict = {}
        for k, v in counts_raw.items():
            key = k.upper() if k != '-' else '-'
            counts_all[key] = counts_all.get(key, 0) + v

        non_gap_counts = {k: v for k, v in counts_all.items() if k != '-'}

        if not non_gap_counts:
            # Entire column is gaps
            consensus_seq.append('-')
            positions.append(mapped_ref)
            logo_data.append({})
            continue

        # ── Consensus ─────────────────────────────────────────────────────
        most_common_aa = max(non_gap_counts, key=non_gap_counts.get)
        most_common_count = non_gap_counts[most_common_aa]
        identity = most_common_count / n_seqs

        if identity >= identity_threshold:
            consensus_seq.append(most_common_aa)
        else:
            consensus_seq.append('X')

        # ── Information content ───────────────────────────────────────────
        # Use *all* counts (including gaps) so that gappy positions have
        # lower information content, matching the standard WebLogo approach.
        info_dict_full = calculate_information_content(counts_all, n_seqs)

        # Keep only non-gap entries for the visual stack
        info_dict = {k: v for k, v in info_dict_full.items() if k != '-'}

        logo_data.append(info_dict)
        positions.append(mapped_ref)

    # ── guard ─────────────────────────────────────────────────────────────
    if not logo_data:
        st.warning("No data extracted for the Sequence Logo.")
        return

    # ── Collect all amino acids that appear anywhere ──────────────────────
    all_aas: set = set()
    for d in logo_data:
        all_aas.update(d.keys())

    # ── Build Plotly stacked-bar logo ─────────────────────────────────────
    fig = go.Figure()

    for aa in sorted(all_aas):
        y_vals = []
        hover_texts = []
        bar_texts = []
        for pos, d in zip(positions, logo_data):
            bits = d.get(aa, 0)
            y_vals.append(bits)
            bar_texts.append(aa if bits > 0.05 else "")
            hover_texts.append(
                f"Position {pos}<br>"
                f"Amino acid: {aa}<br>"
                f"Contribution: {bits:.3f} bits"
            )

        fig.add_trace(go.Bar(
            x=positions,
            y=y_vals,
            name=aa,
            text=bar_texts,
            textposition='inside',
            insidetextanchor='middle',
            textfont=dict(family='Arial Black, Arial', size=11),
            marker_color=CHEMISTRY_COLORS.get(aa, '#808080'),
            marker_line_width=0,
            hovertext=hover_texts,
            hoverinfo='text',
        ))

    # ── Layout ────────────────────────────────────────────────────────────
    fig.update_layout(
        barmode='stack',
        bargap=0,
        title=dict(
            text=f"Sequence Logo — {n_seqs} sequences, {len(positions)} positions",
            font=dict(size=16),
        ),
        xaxis_title="Residue Position",
        yaxis_title="Information (bits)",
        yaxis=dict(
            range=[0, math.log2(20) + 0.1],  # 0 – 4.42
            dtick=1.0,
        ),
        plot_bgcolor='white',
        height=420,
        legend=dict(
            title="Amino Acid",
            orientation='h',
            yanchor='bottom',
            y=-0.35,
            xanchor='center',
            x=0.5,
            font=dict(size=10),
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Color legend ──────────────────────────────────────────────────────
    legend_html = ' &nbsp; '.join(
        f'<span style="color:{color}; font-weight:bold;">■</span> {label}'
        for label, color in _CHEMISTRY_LEGEND
    )
    st.markdown(
        f'<div style="text-align:center; font-size:0.85em; margin-top:-12px;">{legend_html}</div>',
        unsafe_allow_html=True,
    )

    if folder:
        os.makedirs(folder, exist_ok=True)
        fig.write_html(os.path.join(folder, "sequence_logo.html"))

    # ── Consensus ─────────────────────────────────────────────────────────
    st.write("**Consensus Sequence:**")
    consensus_str = "".join(consensus_seq)
    st.code(consensus_str, language="text")
