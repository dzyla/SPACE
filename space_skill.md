# SPACE usage skill

Find protein variants across NCBI / UniProt sequences relative to a reference.

## Interpreter
`.venv/bin/python` in the repository root.

## Quick start (Python)
```python
from space.core.pipeline import run_pipeline
from space.core import annotate

run = run_pipeline(
    query='"Henipavirus hendraense"[Organism] AND fusion[Title]',
    source="NCBI", email="you@example.org", max_seqs=500,
    min_identity=50, min_coverage=30, deletion_threshold=20,
)
print(run.summary())
print(run.variants.filter(min_frequency=0.01, unique=True).head(20))

# Known variants / features (UniProt reference)
known = annotate.known_variants("P69905", run.reference)
table = annotate.merge_known_variants(run.variants.summary, known)
```

## Command line
```bash
python -m space run -q 'gene:HBA1 AND organism_id:9606' -s UniProt -o runs/hba
python -m space variants runs/hba --min-frequency 0.01 --format tsv
python -m space annotate runs/hba --uniprot P69905
```

## MCP
`python -m space mcp` — tools `find_variants`, `get_variants`, `get_variant`,
`annotate_run`, `map_run_to_structure`, `fetch_sample_metadata`, `variant_trend`.

## Reading results
- `Frequency` = carriers / sequences covering the position; prefer
  `Unique frequency` when one outbreak/study dominates the database.
- Names: `I6G`, `K5del`, `K5_Q8del`, `A4_Y5insWW` (+ HGVS `p.` column).
- `X` is missing data, never a variant.

## Pitfalls
- Use fielded queries; free text pulls in unrelated proteins (filter with
  `min_identity` otherwise).
- NCBI needs an email; set `NCBI_API_KEY` for large fetches.
- All-identical inputs give NaN al2co scores (zero variance) — expected.
