# SPACE — Sequence Protein Alignment and Conservation Engine

Concise guide for agentic / code-editor use. Full reference: `AGENTS.md`.

## What it does
Find protein variants (substitutions, insertions, deletions) across NCBI Protein /
UniProtKB sequences relative to a reference, with frequencies, conservation, known-
variant annotation (dbSNP/ClinVar via EBI Proteins API), structure mapping and
GenBank sample metadata.

## Interpreter
Use the project venv: `.venv/bin/python` (create with `./setup_environment_linux.sh`).

## Layout
- `space/core/` — all computation, **no Streamlit**: `sequences` (clean/fetch),
  `align` (pairwise, MSA), `conservation` (al2co, logo), `variants` (caller),
  `tree`, `structure`, `annotate`, `pipeline` (`run_pipeline`, `SpaceRun`).
- `space/viz.py` — Plotly figure builders + py3Dmol HTML (no Streamlit).
- `app.py` — Streamlit UI on top of `space.core`.
- `space/cli.py` (`python -m space ...`), `space/mcp_server.py` (`python -m space mcp`).
- `space_api.py` — backward-compatible dict API used by the example notebook.
- `tests/` — offline tests; `SPACE_NETWORK_TESTS=1` enables live-service tests.

## Run
- App: `streamlit run app.py`
- Tests: `.venv/bin/python -m pytest`
- Python: `from space.core.pipeline import run_pipeline; run = run_pipeline(query=..., source="NCBI", email=...)`

## Key pitfalls
- **Never delete residues when cleaning.** `clean_sequence` maps U/B/Z/J/O to `X`
  so reference numbering stays correct; `X` is treated as missing data.
- **al2co rejects non-standard letters** (and fails on zero-variance alignments);
  `conservation._al2co_safe` passes ambiguous residues to it as gaps.
- **BLOSUM62 aligner:** never set `match_score`/`mismatch_score` on a
  `PairwiseAligner` that has a substitution matrix — Biopython silently drops it.
- **Frequencies** are carriers / sequences *covering* the position (`Covering`),
  not / all sequences. Unique-sequence frequencies collapse identical entries.
- **Insertions** are named between flanking residues (`A4_Y5insWW`); terminal
  extensions are not called.
- **Streamlit dev server caches imported modules** — restart it after editing
  `space/*`.
- NCBI requires a real email; `$NCBI_API_KEY` raises rate limits.
