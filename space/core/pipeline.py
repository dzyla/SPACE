"""End-to-end pipeline and the persistent :class:`SpaceRun` result.

A run lives in its own directory::

    <run_dir>/
      run.json                 parameters, reference, file index
      sequences.fasta          fetched / supplied sequences
      msa/msa_in.fasta, msa_out.fasta, msa_out.aln
      tables/hits.tsv          pairwise identity / coverage for every sequence
      tables/variants.tsv      one row per distinct variant (the main output)
      tables/variant_events.tsv  one row per (sequence, variant)
      tables/sequences.tsv     per-sequence status, coverage, variant list
      tables/positions.tsv     per reference position: conservation, coverage
      tree/phylogenetic_tree.nwk
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
from Bio.SeqRecord import SeqRecord

from . import REFERENCE_ID, ProgressFn, log, report
from .align import filter_hits, pairwise_to_reference, reference_mapping, run_msa
from .conservation import al2co_scores, column_profile
from .sequences import (clean_sequence, extract_uniprot_accession, fetch_sequences, parse_sequence_text,
                        read_fasta)
from .tree import TreeResult, build_tree
from .variants import VariantCalls, call_variants

RUN_FILE = "run.json"


def default_runs_dir() -> Path:
    return Path(os.environ.get("SPACE_RUNS_DIR", "runs")).resolve()


def new_run_dir(label: str = "run", root: Optional[str | os.PathLike] = None) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "-", (label or "run").lower()).strip("-")[:40] or "run"
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = Path(root or default_runs_dir()) / f"{slug}-{stamp}-{uuid.uuid4().hex[:6]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Reference choice
# ---------------------------------------------------------------------------

def choose_reference(
    records: Sequence[SeqRecord],
    reference: Optional[str] = None,
    reference_id: Optional[str] = None,
) -> tuple[str, Optional[str]]:
    """Return ``(reference_sequence, reference_id_or_None)``.

    Priority: explicit sequence > record ID > automatic choice. The automatic
    choice prefers the most common Swiss-Prot / RefSeq entry, otherwise the
    most frequent sequence among those within 10% of the median length — in a
    database sample that is usually the wild type rather than a fusion or
    fragment.
    """
    if reference:
        seq = parse_sequence_text(reference)
        if not seq:
            raise ValueError("Provided reference sequence is empty after cleaning.")
        return seq, None
    if not records:
        raise ValueError("No sequences available to choose a reference from.")
    if reference_id:
        for r in records:
            if r.id == reference_id or r.id.split("|")[1:2] == [reference_id]:
                return clean_sequence(str(r.seq)), r.id
        raise ValueError(f"Reference ID {reference_id!r} not found among the sequences.")

    lengths = np.array([len(r.seq) for r in records])
    med = float(np.median(lengths))
    counts = Counter(clean_sequence(str(r.seq)) for r in records)
    # Swiss-Prot appears as sp|ACC|NAME in UniProt FASTA and as a bare UniProt
    # accession in NCBI; RefSeq curated proteins are NP_/YP_.
    curated = [r for r in records if r.id.startswith(("sp|", "NP_", "YP_"))
               or (r.id[:1].isalpha() and extract_uniprot_accession(r.id) == r.id.split(".")[0])]
    if curated:
        # Most common curated sequence; ties go to the one closest to the median length.
        best = max(curated, key=lambda r: (counts[clean_sequence(str(r.seq))], -abs(len(r.seq) - med)))
        return clean_sequence(str(best.seq)), best.id
    pool = [r for r in records if abs(len(r.seq) - med) <= 0.1 * med] or list(records)
    counts = Counter(clean_sequence(str(r.seq)) for r in pool)
    seq, _ = counts.most_common(1)[0]
    rid = next(r.id for r in pool if clean_sequence(str(r.seq)) == seq)
    return seq, rid


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------

@dataclass
class SpaceRun:
    run_dir: Path
    params: Dict
    reference: str
    reference_id: Optional[str]
    hits: pd.DataFrame
    selected: pd.DataFrame
    msa: MultipleSeqAlignment
    mapping: List[Optional[int]]
    al2co: pd.DataFrame
    profile: pd.DataFrame
    variants: VariantCalls
    tree: Optional[TreeResult] = None
    files: Dict[str, str] = field(default_factory=dict)
    created: str = field(default_factory=lambda: _dt.datetime.now().isoformat(timespec="seconds"))

    @property
    def run_id(self) -> str:
        return self.run_dir.name

    @property
    def positions(self) -> pd.DataFrame:
        """Per reference position: residue, conservation, coverage, variant load."""
        L = len(self.reference)
        df = pd.DataFrame({"Position": np.arange(1, L + 1), "Residue": list(self.reference)})
        al = self.al2co.dropna(subset=["Location"])[["Location", "al2co_score"]]
        al = al.assign(Location=al["Location"].astype(int)).rename(columns={"Location": "Position"})
        prof = self.profile.rename(columns={"Location": "Position"}).drop(columns=["Column"], errors="ignore")
        df = df.merge(al, on="Position", how="left").merge(prof, on="Position", how="left")
        df["Covering"] = self.variants.coverage
        df["Unique covering"] = self.variants.unique_coverage
        s = self.variants.summary
        if not s.empty:
            g = s.groupby("Position").agg(Variants=("Mutation", "count"), **{"Max frequency": ("Frequency", "max")})
            df = df.merge(g, on="Position", how="left")
        else:
            df["Variants"], df["Max frequency"] = 0, 0.0
        df["Variants"] = df["Variants"].fillna(0).astype(int)
        df["Max frequency"] = df["Max frequency"].fillna(0.0)
        return df

    def summary(self) -> Dict:
        v = self.variants
        s = v.summary
        return {
            "run_id": self.run_id,
            "query": self.params.get("query"),
            "source": self.params.get("source"),
            "reference_id": self.reference_id,
            "reference_length": len(self.reference),
            "sequences_fetched": int(len(self.hits)),
            "sequences_in_msa": int(len(self.selected)),
            "sequences_analysed": v.n_included,
            "unique_sequences": v.n_unique,
            "sequences_excluded": len(v.excluded),
            "distinct_variants": int(len(s)),
            "variant_positions": int(s["Position"].nunique()) if not s.empty else 0,
            "variants_ge_1pct": int((s["Frequency"] >= 0.01).sum()) if not s.empty else 0,
            "created": self.created,
            "run_dir": str(self.run_dir),
        }

    # -- persistence -------------------------------------------------------
    def save(self) -> Path:
        t = self.run_dir / "tables"
        t.mkdir(parents=True, exist_ok=True)
        self.hits.drop(columns=["sequence"], errors="ignore").to_csv(t / "hits.tsv", sep="\t", index=False)
        self.selected[["id", "msa_id"]].to_csv(t / "selected.tsv", sep="\t", index=False)
        self.variants.summary.to_csv(t / "variants.tsv", sep="\t", index=False)
        self.variants.events.to_csv(t / "variant_events.tsv", sep="\t", index=False)
        self.variants.per_sequence.to_csv(t / "sequences.tsv", sep="\t", index=False)
        self.al2co.to_csv(t / "al2co_scores.tsv", sep="\t", index=False)
        self.profile.to_csv(t / "column_profile.tsv", sep="\t", index=False)
        self.positions.to_csv(t / "positions.tsv", sep="\t", index=False)
        self.files.update({
            "variants": str(t / "variants.tsv"), "variant_events": str(t / "variant_events.tsv"),
            "sequences": str(t / "sequences.tsv"), "positions": str(t / "positions.tsv"),
            "hits": str(t / "hits.tsv"), "al2co": str(t / "al2co_scores.tsv"),
        })
        manifest = {
            "run_id": self.run_id, "created": self.created, "params": self.params,
            "reference": self.reference, "reference_id": self.reference_id,
            "variant_params": self.variants.params,
            "tree": None if self.tree is None else {
                "newick_path": self.tree.newick_path, "method": self.tree.method,
                "n_leaves": self.tree.n_leaves, "subsampled": self.tree.subsampled},
            # Paths are stored relative to the run directory so runs can be moved.
            "files": {k: _relative(v, self.run_dir) for k, v in self.files.items()},
        }
        if manifest["tree"]:
            manifest["tree"]["newick_path"] = _relative(manifest["tree"]["newick_path"], self.run_dir)
        (self.run_dir / RUN_FILE).write_text(json.dumps(manifest, indent=2, default=str))
        return self.run_dir

    @classmethod
    def load(cls, run_dir: str | os.PathLike) -> "SpaceRun":
        run_dir = Path(run_dir).resolve()
        m = json.loads((run_dir / RUN_FILE).read_text())
        files = {k: str((run_dir / v).resolve()) for k, v in m.get("files", {}).items()}
        t = run_dir / "tables"
        id_cols = {c: str for c in ("id", "msa_id", "Sequence ID", "MSA ID", "Sequence IDs", "description")}
        read = lambda name: pd.read_csv(t / name, sep="\t", keep_default_na=False,  # noqa: E731
                                        na_values=[""], dtype=id_cols)
        hits = read("hits.tsv")
        selected = read("selected.tsv")
        msa = AlignIO.read(files["msa_fasta"], "fasta")
        mapping = reference_mapping(msa)
        positions = read("positions.tsv")
        summary = read("variants.tsv")
        for col in ("Mutation", "HGVS", "Ref", "Alt", "Change", "Sequence IDs"):
            if col in summary:
                summary[col] = summary[col].fillna("").astype(str)
        per_seq = pd.read_csv(t / "sequences.tsv", sep="\t", keep_default_na=False,
                              dtype={"Sequence ID": str, "MSA ID": str, "Variants": str, "Reason": str})
        variants = VariantCalls(
            events=read("variant_events.tsv"), summary=summary, per_sequence=per_seq,
            coverage=positions["Covering"].to_numpy(int),
            unique_coverage=positions["Unique covering"].to_numpy(int),
            reference=m["reference"], params=m.get("variant_params", {}),
        )
        tree = None
        if m.get("tree"):
            tree = TreeResult(**{**m["tree"], "newick_path": str((run_dir / m["tree"]["newick_path"]).resolve())})
        return cls(run_dir=run_dir, params=m["params"], reference=m["reference"],
                   reference_id=m.get("reference_id"), hits=hits, selected=selected, msa=msa,
                   mapping=mapping, al2co=read("al2co_scores.tsv"), profile=read("column_profile.tsv"),
                   variants=variants, tree=tree, files=files, created=m.get("created", ""))


def _relative(path: str, root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def list_runs(root: Optional[str | os.PathLike] = None) -> List[Dict]:
    root = Path(root or default_runs_dir())
    out = []
    if not root.exists():
        return out
    for p in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        f = p / RUN_FILE
        if f.exists():
            try:
                m = json.loads(f.read_text())
                out.append({"run_id": m["run_id"], "created": m.get("created"),
                            "query": m["params"].get("query"), "source": m["params"].get("source"),
                            "reference_id": m.get("reference_id"), "run_dir": str(p)})
            except Exception:  # noqa: BLE001
                continue
    return out


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def analyse(
    hits: pd.DataFrame,
    reference: str,
    run_dir: str | os.PathLike,
    reference_id: Optional[str] = None,
    params: Optional[Dict] = None,
    deletion_threshold: float = 10.0,
    min_coverage: float = 0.0,
    skip_ambiguous: bool = True,
    build_phylo: bool = True,
    threads: int = 0,
    progress: ProgressFn = None,
) -> SpaceRun:
    """MSA -> conservation -> variants -> tree for already-filtered ``hits``."""
    run_dir = Path(run_dir).resolve()
    if hits.empty:
        raise ValueError("No sequences left after filtering.")
    sub = lambda a, b: (lambda f, msg: report(progress, a + (b - a) * f, msg))  # noqa: E731

    msa, selected, files = run_msa(hits, reference, run_dir / "msa", threads=threads, progress=sub(0.0, 0.5))
    mapping = reference_mapping(msa)
    report(progress, 0.55, "Scoring conservation (al2co)")
    al2co = al2co_scores(msa, mapping)
    profile = column_profile(msa, mapping)
    id_map = dict(zip(selected["msa_id"], selected["id"]))
    variants = call_variants(msa, mapping, id_map=id_map, deletion_threshold=deletion_threshold,
                             min_coverage=min_coverage, skip_ambiguous=skip_ambiguous,
                             progress=sub(0.6, 0.85))
    tree = None
    if build_phylo:
        report(progress, 0.9, "Building phylogenetic tree")
        try:
            tree = build_tree(msa, run_dir / "tree")
        except Exception as exc:  # noqa: BLE001
            log.warning("Tree construction failed: %s", exc)
    if tree:
        files["tree"] = tree.newick_path
    run = SpaceRun(run_dir=run_dir, params=dict(params or {}), reference=clean_sequence(reference),
                   reference_id=reference_id, hits=hits, selected=selected, msa=msa, mapping=mapping,
                   al2co=al2co, profile=profile, variants=variants, tree=tree, files=files)
    run.params.update({"deletion_threshold": deletion_threshold, "min_coverage": min_coverage,
                       "skip_ambiguous": skip_ambiguous})
    run.save()
    report(progress, 1.0, "Analysis complete")
    return run


def run_pipeline(
    query: Optional[str] = None,
    fasta_path: Optional[str] = None,
    source: str = "UniProt",
    email: Optional[str] = None,
    max_seqs: int = 100,
    refseq_only: bool = False,
    reviewed_only: bool = False,
    remove_pdb: bool = False,
    reference: Optional[str] = None,
    reference_id: Optional[str] = None,
    min_identity: float = 0.0,
    max_identity: float = 100.0,
    min_length: int = 0,
    max_length: int = 10**9,
    min_coverage: float = 0.0,
    deletion_threshold: float = 10.0,
    skip_ambiguous: bool = True,
    build_phylo: bool = True,
    threads: int = 0,
    run_dir: Optional[str | os.PathLike] = None,
    progress: ProgressFn = None,
) -> SpaceRun:
    """Fetch (or read) sequences, pick a reference, filter, align, call variants.

    ``min_coverage`` applies twice: to the pairwise pre-filter and to the
    per-sequence coverage of the MSA-based variant calls.
    """
    if not query and not fasta_path:
        raise ValueError("Provide either 'query' or 'fasta_path'.")
    run_dir = Path(run_dir).resolve() if run_dir else new_run_dir(query or Path(fasta_path).stem)
    run_dir.mkdir(parents=True, exist_ok=True)
    sub = lambda a, b: (lambda f, msg: report(progress, a + (b - a) * f, msg))  # noqa: E731

    seq_path = run_dir / "sequences.fasta"
    if fasta_path:
        if Path(fasta_path).resolve() != seq_path.resolve():
            seq_path.write_text(Path(fasta_path).read_text())
    else:
        fetch_sequences(query, seq_path, source=source, email=email, max_seqs=max_seqs,
                        refseq_only=refseq_only, reviewed_only=reviewed_only, progress=sub(0.0, 0.25))
    records = read_fasta(seq_path, remove_pdb=remove_pdb)
    ref_seq, ref_id = choose_reference(records, reference, reference_id)
    log.info("Reference: %s (%d aa)", ref_id or "user-supplied", len(ref_seq))

    hits = pairwise_to_reference(records, ref_seq, progress=sub(0.25, 0.45))
    selected = filter_hits(hits, min_identity, max_identity, min_length, max_length, min_coverage)
    log.info("%d of %d sequences pass the filters.", len(selected), len(hits))

    params = {"query": query, "source": "FASTA" if fasta_path else source, "email": email, "max_seqs": max_seqs,
              "refseq_only": refseq_only, "reviewed_only": reviewed_only, "remove_pdb": remove_pdb,
              "min_identity": min_identity, "max_identity": max_identity, "min_length": min_length,
              "max_length": max_length}
    run = analyse(selected, ref_seq, run_dir, reference_id=ref_id, params=params,
                  deletion_threshold=deletion_threshold, min_coverage=min_coverage,
                  skip_ambiguous=skip_ambiguous, build_phylo=build_phylo, threads=threads,
                  progress=sub(0.45, 1.0))
    run.hits = hits
    run.files["sequences"] = str(seq_path)
    run.save()
    return run
