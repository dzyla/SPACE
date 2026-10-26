# SPACE

**Sequence Protein Alignment and Conservation Engine** — find and interpret
protein variants across sequence databases.

SPACE fetches every sequence matching a query from **NCBI Protein** or
**UniProtKB**, aligns them to a reference, and reports each **substitution,
insertion and deletion** with:

- frequency over the sequences that actually cover the position, and over
  *unique* sequences (identical database entries counted once);
- conservation (al2co, entropy, sequence logo, consensus);
- known-variant annotation from the EBI Proteins API (dbSNP, ClinVar,
  SIFT/PolyPhen) and UniProt features (domains, sites, PTMs);
- structural context on an RCSB entry or AlphaFold model (surface exposure,
  pLDDT) with a 3D viewer;
- sample metadata from GenBank (host, country, collection year) and per-year
  variant trends.

It runs as a Streamlit web app, a Python API, a command-line tool and an
**MCP server** for AI agents.

## Install

```bash
./setup_environment_linux.sh          # or setup_environment_windows.ps1
# or manually:
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -r requirements.txt
```

`al2co` is compiled from source on install, so a C compiler is needed
(`packages.txt` lists the system packages for Streamlit Community Cloud).

## Web app

```bash
streamlit run app.py
```

1. Pick NCBI or UniProt and enter a query, e.g.
   `"Henipavirus hendraense"[Organism] AND fusion[Title]` (NCBI) or
   `gene:TP53 AND organism_id:9606` (UniProt). Fielded queries keep unrelated
   proteins out.
2. Choose the reference (a curated Swiss-Prot/RefSeq entry or the most common
   full-length sequence is pre-selected) and filter by identity, length and
   coverage.
3. Explore the variant report: lollipop landscape, variant table (select rows to
   highlight them in the structure, alignment and tree), conservation, 3D
   structure, alignment and tree, per-sequence calls, exports.

## Command line

```bash
python -m space run -q '"Henipavirus hendraense"[Organism] AND fusion[Title]' \
    -s NCBI -e you@example.org --min-identity 50 -o runs/hendra_f
python -m space variants runs/hendra_f --min-frequency 0.01 --unique
python -m space annotate runs/my_tp53_run --uniprot P04637
python -m space runs
```

## Python

```python
from space.core.pipeline import run_pipeline, SpaceRun

run = run_pipeline(query="gene:HBA1 AND organism_id:9606", source="UniProt", max_seqs=100)
run.variants.filter(min_frequency=0.01)    # variant table (DataFrame)
run.positions                               # per-position conservation and coverage
run = SpaceRun.load(run.run_dir)            # runs are saved and reloadable
```

`space_api.run_pipeline(...)` keeps the dictionary interface of earlier
versions (`al2co_df`, `mutations_df`, `tree_file`, ...) and adds `run` and
`variants_df`. See `examples/space_headless_example.ipynb`.

## MCP server

```bash
python -m space mcp            # stdio
```

Example client configuration (Claude Code / Claude Desktop):

```json
{
  "mcpServers": {
    "space": {
      "command": "/path/to/SPACE/.venv/bin/python",
      "args": ["-m", "space", "mcp"],
      "cwd": "/path/to/SPACE",
      "env": { "SPACE_RUNS_DIR": "/path/to/SPACE/runs", "SPACE_EMAIL": "you@example.org" }
    }
  }
}
```

Tools: `find_variants`, `analyze_fasta`, `list_runs`, `get_variants`,
`get_variant`, `get_positions`, `annotate_run`, `map_run_to_structure`,
`fetch_sample_metadata`, `variant_trend`. Run tables are exposed as resources
at `space://runs/{run_id}/{table}`.

## How variants are called

- Every sequence is aligned in one MSA (pyfamsa) together with the reference.
- A sequence's *covered span* runs from its first to its last aligned residue.
  Outside it nothing is called and the sequence does not count towards that
  position — fragments contribute data without diluting frequencies.
- Inside the span: residue ≠ reference → substitution (`I6G`, `p.Ile6Gly`);
  gap → deletion (`K5del`, `K5_Q8del`); residues in reference gaps →
  insertion (`A4_Y5insWW`). N/C-terminal extensions are not insertions.
- `X` and other ambiguous residues are missing data, never variants; they are
  kept in the sequence so reference numbering never shifts.
- Sequences whose internal deletions exceed a threshold (default 20% in the
  app) are excluded as likely misalignments.

## Tests

```bash
pytest                          # offline
SPACE_NETWORK_TESTS=1 pytest    # also hit NCBI / UniProt / EBI
```

## Citation

Zyla, D. **SPACE**: Sequence Protein Alignment and Conservation Engine.
Based on Zyla, D. *et al.* *Science* (2024). DOI: 10.1126/science.adm8693
