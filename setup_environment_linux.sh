#!/bin/bash

# Bash script for Linux/macOS

# Install uv
echo "Installing uv on Linux/macOS..."
curl -LsSf https://astral.sh/uv/install.sh | sh

# Initialize uv
echo "Initializing uv..."
uv init

# Create a virtual environment called "space" with uv
echo "Creating virtual environment 'space'..."
uv venv space_env

# Activate the "space" environment
echo "Activating 'space' environment..."
source space_env/bin/activate || { echo "Failed to activate 'space' environment. Exiting."; exit 1; }

# Install required packages in the 'space' environment
echo "Installing required packages in 'space' environment..."
uv pip install streamlit py3Dmol stmol pandas biopython plotly kaleido scipy biopandas ipython_genutils joblib seaborn rcsbsearchapi

echo "Environment 'space' setup complete with all required packages."
