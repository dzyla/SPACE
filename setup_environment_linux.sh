#!/bin/bash
# Create a SPACE environment with uv (Linux/macOS). Run from the repository root.
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Creating .venv (Python 3.12)..."
uv venv .venv --python 3.12

echo "Installing requirements (al2co is compiled from source; needs gcc)..."
uv pip install --python .venv/bin/python -r requirements.txt pytest

echo
echo "Done. Next steps:"
echo "  source .venv/bin/activate"
echo "  streamlit run app.py          # web app"
echo "  python -m space --help        # command line"
echo "  python -m space mcp           # MCP server (stdio)"
