# SPACE

Sequence Protein Alignment and Conservation Engine (SPACE)

## Introduction

## Installation

### Using uv (super fast):
To install SPACE using uv execute the following command:

* Windows Powershell
    
    ```powershell ./setup_enviroment.ps1 ```

* Linux
    
    ```bash ./setup_enviroment_linux.sh ```

Make sure no conda environment is active before running the script.

### Using conda/pip:

If you have conda installed, you can create a new environment and install the required packages using the following commands:

```bash
conda create -n space python=3.12
conda activate space
pip install -r requirements.txt
```

or:

```bash
pip install streamlit py3Dmol stmol pandas biopython plotly kaleido scipy biopandas ipython_genutils joblib seaborn rcsbsearchapi
```

## Usage

To run SPACE execute the following command:

```bash 
streamlit run app.py
```

