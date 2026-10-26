"""Command-line interface: ``python -m space <command>`` (or ``space`` once installed)."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _progress(fraction: float, message: str) -> None:
    print(f"\r[{fraction * 100:5.1f}%] {message[:90]:<90}", end="", file=sys.stderr, flush=True)


def _emit(df: pd.DataFrame, fmt: str) -> None:
    if fmt == "tsv":
        df.to_csv(sys.stdout, sep="\t", index=False)
    elif fmt == "json":
        print(df.to_json(orient="records", indent=2))
    else:
        with pd.option_context("display.max_rows", 200, "display.width", 200, "display.max_colwidth", 40):
            print(df.to_string(index=False))


def _resolve_run(arg: str):
    from space.core.pipeline import SpaceRun, default_runs_dir

    p = Path(arg)
    if not (p / "run.json").exists():
        p = default_runs_dir() / arg
    return SpaceRun.load(p)


SHOW = ["Mutation", "HGVS", "Type", "Count", "Covering", "Frequency", "Unique carriers", "Unique frequency",
        "BLOSUM62", "Change"]


def cmd_run(a) -> None:
    from space.core.pipeline import run_pipeline

    run = run_pipeline(
        query=a.query, fasta_path=a.fasta, source=a.source, email=a.email, max_seqs=a.max_seqs,
        refseq_only=a.refseq_only, reviewed_only=a.reviewed_only, remove_pdb=not a.keep_pdb,
        reference=a.reference, reference_id=a.reference_id, min_identity=a.min_identity,
        min_coverage=a.min_coverage, deletion_threshold=a.deletion_threshold,
        build_phylo=not a.no_tree, threads=a.threads, run_dir=a.out, progress=None if a.quiet else _progress,
    )
    if not a.quiet:
        print(file=sys.stderr)
    print(json.dumps(run.summary(), indent=2))
    top = run.variants.filter(min_frequency=a.min_frequency).head(a.top)
    if not top.empty:
        print(f"\nTop {len(top)} variants:")
        _emit(top[SHOW], "table")


def cmd_variants(a) -> None:
    run = _resolve_run(a.run)
    df = run.variants.filter(min_frequency=a.min_frequency, types=a.type, start=a.start, end=a.end,
                             unique=a.unique)
    _emit(df[SHOW + (["Sequence IDs"] if a.with_ids else [])].head(a.limit or None), a.format)


def cmd_annotate(a) -> None:
    from space.core import annotate

    run = _resolve_run(a.run)
    known = annotate.known_variants(a.uniprot, run.reference)
    feats = annotate.uniprot_features(a.uniprot, run.reference)
    s = annotate.add_feature_overlaps(annotate.merge_known_variants(run.variants.summary, known), feats)
    out = run.run_dir / "tables" / "variants_annotated.tsv"
    s.to_csv(out, sep="\t", index=False)
    print(f"{int(s['Known'].sum())} of {len(s)} variants are reported for {a.uniprot}; wrote {out}",
          file=sys.stderr)
    _emit(s[s["Known"]][["Mutation", "Frequency", "Clinical significance", "dbSNP", "Features"]], a.format)


def cmd_runs(a) -> None:
    from space.core.pipeline import list_runs

    _emit(pd.DataFrame(list_runs(a.root)), a.format)


def cmd_app(a) -> None:
    app = Path(__file__).resolve().parents[1] / "app.py"
    sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(app), *a.args]))


def cmd_mcp(a) -> None:
    from space.mcp_server import main

    main(transport=a.transport)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="space", description="SPACE: protein variants across sequence databases")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="fetch/read sequences, align, call variants")
    src = r.add_mutually_exclusive_group(required=True)
    src.add_argument("-q", "--query")
    src.add_argument("-f", "--fasta")
    r.add_argument("-s", "--source", default="NCBI", choices=["NCBI", "UniProt"])
    r.add_argument("-e", "--email", help="required for NCBI")
    r.add_argument("-n", "--max-seqs", type=int, default=500)
    r.add_argument("--reference", help="reference sequence (raw or FASTA)")
    r.add_argument("--reference-id", help="use this record from the results as reference")
    r.add_argument("--min-identity", type=float, default=0.0)
    r.add_argument("--min-coverage", type=float, default=30.0)
    r.add_argument("--deletion-threshold", type=float, default=20.0)
    r.add_argument("--refseq-only", action="store_true")
    r.add_argument("--reviewed-only", action="store_true")
    r.add_argument("--keep-pdb", action="store_true", help="keep PDB-derived sequences")
    r.add_argument("--no-tree", action="store_true")
    r.add_argument("--threads", type=int, default=0)
    r.add_argument("-o", "--out", help="run directory (default: $SPACE_RUNS_DIR or ./runs/<auto>)")
    r.add_argument("--top", type=int, default=20)
    r.add_argument("--min-frequency", type=float, default=0.0)
    r.add_argument("--quiet", action="store_true")
    r.set_defaults(func=cmd_run)

    v = sub.add_parser("variants", help="query the variant table of a saved run")
    v.add_argument("run", help="run directory or run ID")
    v.add_argument("--min-frequency", type=float, default=0.0)
    v.add_argument("--type", action="append", choices=["Substitution", "Insertion", "Deletion"])
    v.add_argument("--start", type=int)
    v.add_argument("--end", type=int)
    v.add_argument("--unique", action="store_true", help="frequencies over unique sequences")
    v.add_argument("--with-ids", action="store_true")
    v.add_argument("--limit", type=int, default=0)
    v.add_argument("--format", choices=["table", "tsv", "json"], default="table")
    v.set_defaults(func=cmd_variants)

    an = sub.add_parser("annotate", help="add UniProt features and known variants (dbSNP/ClinVar)")
    an.add_argument("run")
    an.add_argument("--uniprot", required=True, help="UniProt accession of the reference")
    an.add_argument("--format", choices=["table", "tsv", "json"], default="table")
    an.set_defaults(func=cmd_annotate)

    ls = sub.add_parser("runs", help="list saved runs")
    ls.add_argument("--root")
    ls.add_argument("--format", choices=["table", "tsv", "json"], default="table")
    ls.set_defaults(func=cmd_runs)

    ap = sub.add_parser("app", help="launch the Streamlit app")
    ap.add_argument("args", nargs=argparse.REMAINDER)
    ap.set_defaults(func=cmd_app)

    m = sub.add_parser("mcp", help="run the MCP server")
    m.add_argument("--transport", default="stdio", choices=["stdio", "streamable-http"])
    m.set_defaults(func=cmd_mcp)
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s", stream=sys.stderr)
    args.func(args)


if __name__ == "__main__":
    main()
