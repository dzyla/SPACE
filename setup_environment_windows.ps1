# PowerShell script for Windows (generalized)

# Deactivate Conda if it's currently active
if (Test-Path env:CONDA_DEFAULT_ENV) {
    Write-Output "Deactivating Conda environment..."
    conda deactivate
    Remove-Item env:CONDA_DEFAULT_ENV -ErrorAction SilentlyContinue
    Remove-Item env:CONDA_PREFIX -ErrorAction SilentlyContinue
}

# Determine the user's cargo bin directory and add it to the PATH
$CargoBinPath = "$([System.Environment]::GetFolderPath('UserProfile'))\.cargo\bin"
$env:Path = "$CargoBinPath;$env:Path"

# Install uv
Write-Output "Installing uv on Windows..."
Invoke-Expression (Invoke-RestMethod -Uri https://astral.sh/uv/install.ps1)

# Initialize uv
Write-Output "Initializing uv..."
uv init

# Create a virtual environment called "space_env" with uv
Write-Output "Creating virtual environment 'space_env'..."
uv venv space_env

# Activate the "space_env" environment
Write-Output "Activating 'space_env' environment..."
# Check if virtual environment was created in the default location
if (Test-Path ".\space_env\Scripts\Activate.ps1") {
    . .\space_env\Scripts\Activate.ps1
} else {
    Write-Output "Failed to activate 'space_env' environment. Exiting."
    exit 1
}

# Install required packages in the 'space_env' environment
Write-Output "Installing required packages in 'space_env' environment..."
uv pip install streamlit py3Dmol stmol pandas biopython plotly kaleido scipy biopandas ipython_genutils joblib seaborn rcsbsearchapi

Write-Output "Environment 'space_env' setup complete with all required packages."
