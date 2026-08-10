# SPACE — Sequence Protein Alignment and Conservation Engine

A complete reference for autonomous agents. This file describes the SPACE package:
how to set up the environment, how to launch the Streamlit app, how to use the
headless API, the pipeline stages, and the key pitfalls to avoid.

## 1. Environment setup

The `space` conda environment is already created at
`/home/dzyla/miniconda3/envs/space`. Use its Python interpreter directly:

```bash
/home/dzyla/miniconda3/envs/space/bin/python
```

Dependencies are listed in `requirements.txt` (biopython, pandas, numpy, scipy,
streamlit, plotly, biopandas, seaborn, pyfamsa, al2co from the dzyla/al2co git
repo, stmol/py3Dmol, ipython_genutils). Install with:

```bash
pip install -r requirements.txt
```

No reinstallation is required for this task.

## 2. Running the Streamlit app

From the SPACE repo root:

```bash
streamlit run app.py
```

`app.py` imports `space_api` and builds a UI around `run_pipeline`. The UI
requires the full Streamlit runtime (not the headless stub).

## 3. Headless API (`space_api.run_pipeline`)

The headless API is the recommended way to run SPACE from scripts or notebooks.
`space_api.py` installs a minimal Streamlit stub via `headless_streamlit.install()`
**before** importing any `space.*` module, so the whole pipeline runs without a
browser.

### Minimal example

```python
import space_api as sa

result = sa.run_pipeline(
    query="Hendra henipavirus F",
    email="dzyla@lji.org",
    data_source="UniProt",
    max_seqs=50,
    out_dir="runs/hendra_f",
)

print(result["al2co_df"].head())
print(result["mutations_df"].head())
```

### `run_pipeline` parameters

| Parameter | Description | Default |
|---|---|---|
| `query` | Search term (e.g. `"Hendra henipavirus F"`) | optional; use `fasta_path` instead if provided |
| `email` | NCBI Entrez email (required for NCBI) | `"dzyla@lji.org"` |
| `data_source` | `"NCBI"` or `"UniProt"` | `"UniProt"` |
| `max_seqs` | Max sequences (-1 = all for NCBI; UniProt caps at 500) | `100` |
| `use_refseq` | Restrict NCBI to RefSeq | `False` |
| `remove_pdb` | Strip PDB-style headers from sequences | `False` |
| `fasta_path` | Path to an existing FASTA file (bypasses fetch) | `None` |
| `reference_seq` | Explicit reference sequence; auto-pick longest if None | `None` |
| `filtering_score` | Min percent identity to keep a sequence | `10.0` |
| `sequence_len_min` | Min aligned length | `0` |
| `sequence_len_max` | Max aligned length | `10**9` |
| `max_threads` | pyfamsa threads | `8` |
| `deletion_threshold` | Max deletion % before excluding a sequence | `10.0` |
| `exclude_mutations_with_X` | Exclude X-containing mutations | `True` |
| `out_dir` | Output directory | `"."` |
| `map_to_structure` | Enable optional PDB/AlphaFold mapping | `False` |
| `uniprot_id` | UniProt ID for AlphaFold structure lookup | `None` |
| `sequence_score` | RCSB sequence identity cutoff (0-1) | `0.9` |

### Return dict keys

`run_pipeline` returns a dict with: `fasta_path`, `proteins_list`,
`reference_seq`, `result`, `filtered_seqs`, `msa_infile`, `msa_outfile`,
`aln_file`, `al2co_df`, `conservation_df`, `mutations_df`, `unique_mutations`,
`all_mutations_str`, `excluded_sequences`, `tree_file`, `pdb_result`, `out_dir`.

## 4. Pipeline stages

`run_pipeline` executes the following stages sequentially:

### Step 0 — Fetch sequences (`fetch_sequences`)
Calls `search_and_save_protein_ncbi` (needs `email`) or
`search_and_save_protein_uniprot`. Writes `sequences.fasta` into `out_dir`.

### Step 1 — Pairwise alignment (`perform_alignment`)
Aligns every fetched protein against the reference sequence (the longest by
default, or the one passed via `reference_seq`). Computes percent identity and
coverage. Stores `result`, `reference_seq`, and `alignment_mapping` in the
session state.

### Step 2 — Filter (`filter_sequences`)
Keeps sequences whose alignment score ≥ `filtering_score` and whose aligned
length falls within `[sequence_len_min, sequence_len_max]`.

### Step 3 — Multiple sequence alignment (`perform_msa_pyfamsa`)
Runs pyfamsa on the filtered sequences, writing `msa_in.fasta` and
`msa_out.fasta`. Converts the FASTA MSA to CLUSTAL `.aln` via
`_write_clustal` (BioPython `SeqIO.convert` + header normalisation).

### Step 4 — Conservation + al2co (`analyze_alignment`, `run_al2co`)
`analyze_alignment` builds the per-position conservation DataFrame and the
alignment mapping used to map MSA columns back to residue numbers. `run_al2co`
calls the native `al2co` library on the CLUSTAL `.aln` file to produce per-residue
conservation scores.

### Step 5 — Point mutations (`list_unique_point_mutations`, `parse_mutations`)
Compares each aligned sequence to the reference, records substitutions/insertions,
excludes sequences exceeding `deletion_threshold`, and parses mutations into a
DataFrame.

### Step 6 — Phylogenetic tree (`generate_phylogenetic_tree`)
Only run when the number of aligned sequences ≤ 200. Writes a tree file into
`out_dir/tree`.

### Step 7 — Structural mapping (`map_structure`)
Optional. Maps al2co conservation scores onto PDB or AlphaFold structures via
`process_pdb_chain` (UniProt ID → AlphaFold PDB, or a local PDB). Writes labeled
structures into `out_dir/pdb`.

## 5. Key pitfalls

### 5.1 Native al2co binary needs a CLUSTAL W header
`run_al2co` reads the alignment with `AlignIO.read(alignment_file, "clustal")`,
which requires a properly formatted CLUSTAL file. The pyfamsa MSA output is
FASTA, so `_write_clustal` converts it with BioPython's `SeqIO.convert(..., "clustal")`
and then **normalises the first line** to the canonical header
`CLUSTAL W multiple sequence alignment\n\n`. Without this header the BioPython
ClustalIO parser (and therefore al2co) will fail. Do not hand-write CLUSTAL
files; always use the pipeline's `_write_clustal` output or regenerate via
`perform_msa_pyfamsa`.

### 5.2 Identical sequences yield zero variance
If all aligned sequences are identical, every column has zero variance. al2co
returns NaN / zero conservation scores for such columns. This is mathematically
correct (no information at those positions) but produces an all-NaN al2co table.
Deduplicate identical sequences before running the pipeline if this is
undesirable.

### 5.3 Headless stub needs `streamlit.components.v1`
`headless_streamlit.install()` registers `streamlit.components.v1` (with `html`
and `iframe` no-ops) in `sys.modules` so third-party libs that assume a full
Streamlit tree — notably `stmol` (`from stmol import showmol`) — import cleanly
in a headless context. Always call `headless_streamlit.install()` **before**
importing any `space.*` module. The `space_api` module does this internally at
the top of `space_api.py`, so end users only need to `import space_api`.

### 5.4 NCBI requires a real email
When `data_source="NCBI"`, Entrez requires a valid email address. Pass
`email="dzyla@lji.org"` (or your own). UniProt does not require an email.

### 5.5 conda activate may be broken
Use the interpreter directly (`/home/dzyla/miniconda3/envs/space/bin/python`)
rather than relying on `conda activate`.

## 6. Reproducible one-liner

```bash
/home/dzyla/miniconda3/envs/space/bin/python -c "import space_api as sa; \
sa.run_pipeline(query='Hendra henipavirus F', email='dzyla@lji.org', \
data_source='UniProt', max_seqs=50, out_dir='runs/hendra_f')"
```