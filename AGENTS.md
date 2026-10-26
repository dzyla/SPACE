# SPACE — reference for agents

SPACE finds protein variants across sequence databases: it fetches sequences
(NCBI Protein or UniProtKB), aligns them to a reference, calls substitutions,
insertions and deletions with coverage-aware frequencies, scores conservation,
annotates known variants and maps everything onto 3D structures.

## 1. Environment

```bash
./setup_environment_linux.sh        # creates .venv with requirements.txt
.venv/bin/python -m pytest          # 30+ offline tests
```

Python 3.12. `al2co` is built from `git+https://github.com/dzyla/al2co.git` and
needs a C compiler. `mcp` is only needed for the MCP server.

## 2. Architecture

| Module | Role |
|---|---|
| `space/core/sequences.py` | `clean_sequence`, `parse_sequence_text`, `read_fasta`, `fetch_ncbi` (retry, API key, Swiss-Prot merge), `fetch_uniprot` (cursor pagination) |
| `space/core/align.py` | BLOSUM62 semi-global `get_aligner`, `pairwise_to_reference` (identity / aligned identity / coverage), `filter_hits`, `run_msa` (pyfamsa → FASTA + CLUSTAL), `reference_mapping`, `position_map` |
| `space/core/conservation.py` | `al2co_scores`, `column_profile` (modal frequency, entropy, gaps), `logo_and_consensus` |
| `space/core/variants.py` | `call_variants` → `VariantCalls` (`events`, `summary`, `per_sequence`, `coverage`, `unique_coverage`, `filter()`) |
| `space/core/tree.py` | `build_tree` (NJ ≤ 200 leaves, UPGMA above, subsample > 600) → Newick |
| `space/core/structure.py` | `search_pdb`, `download_pdb` (.pdb → .cif fallback), `download_alphafold`, `map_to_structure` → `StructureMapping` (per-chain residue map incl. insertion codes, B-factor/pLDDT, RSA) |
| `space/core/annotate.py` | `uniprot_features`, `known_variants` (EBI Proteins API), `merge_known_variants`, `add_structure_context`, `ncbi_source_metadata`, `variants_by_year` |
| `space/core/pipeline.py` | `run_pipeline`, `analyse`, `choose_reference`, `SpaceRun` (save/load), `list_runs` |
| `space/viz.py` | Plotly figures (landscape, heatmap, logo, MSA, tree, timeline) + `structure_html` |
| `app.py` | Streamlit UI |
| `space/cli.py`, `space/__main__.py` | `python -m space run|variants|annotate|runs|app|mcp` |
| `space/mcp_server.py` | MCP server (mcp ≥ 2, `MCPServer`) |
| `space_api.py` | Backward-compatible dict API (`run_pipeline(...)["al2co_df"]` etc.) |

Core modules never import Streamlit. They log through the `"space"` logger and
take an optional `progress(fraction, message)` callback.

## 3. Pipeline

`run_pipeline(query | fasta_path, source, email, max_seqs, reference | reference_id,
min_identity, max_identity, min_length, max_length, min_coverage,
deletion_threshold, skip_ambiguous, build_phylo, threads, run_dir)`:

1. Fetch → `<run_dir>/sequences.fasta` (or copy `fasta_path`).
2. Reference: explicit sequence > `reference_id` > most common curated
   (Swiss-Prot `sp|`, RefSeq `NP_`/`YP_`) entry > most common sequence within
   10% of the median length.
3. Pairwise alignment to the reference → `hits` (identity, aligned identity, coverage).
4. Filter → MSA (pyfamsa, reference row `reference_sequence`).
5. al2co + column profile.
6. Variant calling (see §4).
7. Tree (optional).
8. `SpaceRun.save()` writes `run.json` and `tables/*.tsv`.

Run directory:

```
run.json  sequences.fasta  msa/{msa_in.fasta,msa_out.fasta,msa_out.aln}
tables/{variants,variant_events,sequences,positions,hits,selected,al2co_scores,column_profile}.tsv
tree/phylogenetic_tree.nwk   structure/…   tables/variants_annotated.tsv, metadata.tsv (MCP/CLI annotations)
```

## 4. Variant semantics

- Covered span = first..last aligned residue of a sequence. No calls and no
  denominator contribution outside it.
- Substitution `I6G` / `p.Ile6Gly`; deletion `K5del`, `K5_Q8del`; insertion
  between residues p and p+1 `A4_Y5insWW` (Position = p). Terminal extensions are
  ignored.
- `X` (and any non-standard letter, which `clean_sequence` maps to `X`) is
  missing data.
- `Frequency = Count / Covering`; `Unique frequency = Unique carriers / Unique covering`
  (identical ungapped sequences collapsed).
- Excluded sequences: internal deletions > `deletion_threshold`% of the covered
  span, or coverage < `min_coverage`%.
- `summary` columns: Mutation, HGVS, Type, Position, End, Ref, Alt, Count,
  Covering, Frequency, Unique carriers, Unique covering, Unique frequency,
  BLOSUM62, Change, Sequence IDs.

## 5. MCP server

`python -m space mcp` (stdio) or `--transport streamable-http`. Environment:
`SPACE_RUNS_DIR` (default `./runs`), `SPACE_EMAIL` (NCBI default),
`NCBI_API_KEY`.

| Tool | Purpose |
|---|---|
| `find_variants` | search + full pipeline; returns run_id, stats, top variants |
| `analyze_fasta` | same, from FASTA text |
| `list_runs` | saved runs |
| `get_variants` | filter by frequency / type / range / known; sort |
| `get_variant` | carriers, same-position variants, conservation, carrier metadata |
| `get_positions` | per-position conservation and coverage |
| `annotate_run` | UniProt features + dbSNP/ClinVar/SIFT/PolyPhen |
| `map_run_to_structure` | RCSB / AlphaFold mapping; RSA, exposure, pLDDT per variant |
| `fetch_sample_metadata` | GenBank host / country / collection date (NCBI runs) |
| `variant_trend` | per-year carrier fraction |

Resources: `space://runs/{run_id}/{table}` (TSV), `space://runs/{run_id}/msa`.
Expected failures are raised as `ToolError` so the message reaches the client.

## 6. Pitfalls

- Never set `match_score`/`mismatch_score` on an aligner with a substitution
  matrix (Biopython drops the matrix).
- al2co accepts only the 20 standard residues and fails on zero-variance input;
  `_al2co_safe` converts ambiguous residues to gaps, scores become NaN with a
  logged warning when all sequences are identical.
- RCSB entries too large for PDB format are fetched as mmCIF; per-chain output is
  written as PDB (multi-letter chain IDs become `A`).
- `map_to_structure` skips chains below 40% identity (antibodies and other
  partners otherwise produce spurious alignments).
- After editing `space/*`, restart a running Streamlit server.
