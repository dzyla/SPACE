# SPACE Usage Skill

Reusable skill for running the SPACE (Sequence Protein Alignment and Conservation
Engine) pipeline headlessly or via Streamlit.

## Interpreter
Use `/home/dzyla/miniconda3/envs/space/bin/python`.

## Quick start
```python
import space_api as sa

result = sa.run_pipeline(
    query="Hendra henipavirus F",
    email="dzyla@lji.org",
    data_source="UniProt",
    max_seqs=50,
    out_dir="runs/hendra_f",
)

# al2co conservation scores (per-residue)
print(result["al2co_df"].head())

# point-mutation table
print(result["mutations_df"].head())

# conservation summary DataFrame
print(result["conservation_df"].head())

# phylogenetic tree path (None if >200 sequences)
print(result["tree_file"])

# optional structural mapping result
print(result["pdb_result"])
```

## Pipeline stages (order of execution inside `run_pipeline`)
1. **Step 0** — `fetch_sequences`: NCBI Entrez (needs `email`) or UniProt → `sequences.fasta`.
2. **Step 1** — `perform_alignment`: pairwise alignment to reference; compute identity/coverage.
3. **Step 2** — `filter_sequences`: keep sequences above `filtering_score` within length range.
4. **Step 3** — `perform_msa_pyfamsa`: pyfamsa MSA → `msa_out.fasta`; convert to CLUSTAL `.aln`.
5. **Step 4** — `analyze_alignment` + `run_al2co`: conservation DataFrame + al2co scores.
6. **Step 5** — `list_unique_point_mutations` + `parse_mutations`: point-mutation table.
7. **Step 6** — `generate_phylogenetic_tree`: only if ≤ 200 sequences.
8. **Step 7** — `map_structure` (optional): map conservation onto PDB/AlphaFold.

## Key pitfalls
- **CLUSTAL W header:** native `al2co` reads via `AlignIO.read(..., "clustal")`.
  The MSA is converted to CLUSTAL by `_write_clustal` which normalises the header
  to `CLUSTAL W multiple sequence alignment`. Without the proper header al2co
  fails.
- **Identical sequences → zero variance:** all-identical inputs produce all-NaN
  al2co scores (zero variance). Deduplicate identical sequences beforehand.
- **Headless stub:** `headless_streamlit.install()` registers
  `streamlit.components.v1` so `stmol` imports work headlessly. `space_api`
  calls `.install()` internally at import time.
- **Email:** NCBI needs `email="dzyla@lji.org"`.

## Streamlit UI
```bash
streamlit run app.py
```

## Environment
`pip install -r requirements.txt`. No reinstallation needed.