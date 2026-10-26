# Create a SPACE environment with uv (Windows PowerShell). Run from the repository root.
$ErrorActionPreference = "Stop"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Output "Installing uv..."
    Invoke-Expression (Invoke-RestMethod -Uri https://astral.sh/uv/install.ps1)
    $env:Path = "$([System.Environment]::GetFolderPath('UserProfile'))\.local\bin;$env:Path"
}

Write-Output "Creating .venv (Python 3.12)..."
uv venv .venv --python 3.12

Write-Output "Installing requirements (al2co is compiled from source; needs a C compiler)..."
uv pip install --python .venv\Scripts\python.exe -r requirements.txt pytest

Write-Output ""
Write-Output "Done. Next steps:"
Write-Output "  .\.venv\Scripts\Activate.ps1"
Write-Output "  streamlit run app.py          # web app"
Write-Output "  python -m space --help        # command line"
Write-Output "  python -m space mcp           # MCP server (stdio)"
