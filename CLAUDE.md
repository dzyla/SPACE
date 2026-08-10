# SPACE — Sequence Protein Alignment and Conservation Engine

This is a concise guide for agentic / code-editor use. For the full reference,
see `AGENTS.md`.

## What is SPACE?
SPACE runs a protein-sequence alignment & conservation pipeline: fetch sequences
(NCBI/UniProt), pairwise-align to a reference, filter, build a multiple sequence
alignment (pyfamsa), compute conservation scores (al2co), call point mutations,
optionally build a phylogenetic tree and map conservation onto a 3D structure.

## Interpreter
Use `/home/dzyla/miniconda3/envs/space/bin/python` (the `space` conda env).

## How to run
- **Streamlit UI:** `streamlit run app.py` from the repo root.
- **Headless API:** import `space_api` and call `space_api.run_pipeline(...)`.
  The API installs a headless Streamlit stub (`headless_streamlit.install()`) so
  the `space.*` modules can be imported without the UI.

## Pipeline stages (inside `run_pipeline`)
1. **Step 0 — fetch sequences** (`fetch_sequences`): NCBI Entrez (needs a valid
   email; default `dzyla@lji.org`) or UniProt. Writes `sequences.fasta`.
2. **Step 1 — pairwise alignment** (`perform_alignment`): align each protein to
   the reference; compute percent identity / coverage.
3. **Step 2 — filter** (`filter_sequences`): keep sequences above a score
   threshold and within a length range.
4. **Step 3 — MSA** (`perform_msa_pyfamsa`): run pyfamsa; writes `msa_in.fasta`
   and `msa_out.fasta`; convert to CLUSTAL `.aln`.
5. **Step 4 — conservation + al2co** (`analyze_alignment`, `run_al2co`).
6. **Step 5 — point mutations** (`list_unique_point_mutations`, `parse_mutations`).
7. **Step 6 — phylogenetic tree** (`generate_phylogenetic_tree`): only if
   ≤ 200 sequences.
8. **Step 7 — structural mapping** (`map_structure`): map al2co scores onto PDB /
   AlphaFold structures (optional, needs `map_to_structure=True`).

## Key pitfalls
- **CLUSTAL W header:** the native `al2co` library reads alignments via
  `AlignIO.read(..., "clustal")`. The MSA output is converted to CLUSTAL format
  and `_write_clustal` normalises the header to `CLUSTAL W multiple sequence
  alignment`. Without the proper header/structure BioPython's ClustalIO parser
  (and hence al2co) fails.
- **Identical sequences → zero variance:** if all input sequences are identical
  (e.g. several copies of the same sequence), every column has zero variance and
  al2co returns NaN / zero conservation scores.
- **Headless stub needs `streamlit.components.v1`:** `headless_streamlit.install()`
  registers `streamlit.components.v1` in `sys.modules` so third-party libs such as
  `stmol` import cleanly in a headless context. Always call `.install()` *before*
  importing `space.*`.
- **Email requirement:** NCBI Entrez requires a real email address; pass
  `email="dzyla@lji.org"` (or your own) when `data_source="NCBI"`.

## Minimal headless example
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