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
