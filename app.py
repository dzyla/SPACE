# app.py — SPACE Streamlit front-end (all computation lives in space.core)

from __future__ import annotations

import io
import logging
import shutil
import socket
import time
import uuid
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from space import viz
from space.core import annotate
from space.core.align import filter_hits, pairwise_to_reference
from space.core.conservation import logo_and_consensus
from space.core.pipeline import SpaceRun, analyse, choose_reference, list_runs, new_run_dir
from space.core.sequences import (extract_uniprot_accession, fetch_sequences, parse_sequence_text,
                                  read_fasta)
from space.core.structure import (download_alphafold, download_pdb, map_to_structure, pdb_metadata,
                                  search_pdb)

st.set_page_config(page_title="SPACE · protein variant explorer", layout="wide", page_icon="🧬",
                   initial_sidebar_state="expanded")
logging.basicConfig(level=logging.INFO)
# Bio.Entrez has no per-call timeout; a stalled socket would freeze the session.
socket.setdefaulttimeout(60)

SESSIONS_ROOT = Path("sessions")
SESSION_MAX_AGE_HOURS = 24
FREQ_STEPS = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5]
EXAMPLES = [
    ("Hendra virus F (NCBI)", "NCBI", '"Henipavirus hendraense"[Organism] AND fusion[Title]'),
    ("SARS-CoV-2 ORF8 (NCBI)", "NCBI", '"Severe acute respiratory syndrome coronavirus 2"[Organism] AND ORF8[Gene]'),
    ("Human TP53 orthologs (UniProt)", "UniProt", "gene:TP53 AND reviewed:true"),
]


# ---------------------------------------------------------------------------
# Session plumbing
# ---------------------------------------------------------------------------

def _prune_stale_sessions(root: Path, max_age_hours: float) -> None:
    """Public deployments have small, shared disks: drop session dirs older than a day."""
    if not root.exists():
        return
    cutoff = time.time() - max_age_hours * 3600
    for entry in root.iterdir():
        try:
            if entry.is_dir() and entry.stat().st_mtime < cutoff:
                shutil.rmtree(entry, ignore_errors=True)
        except OSError:
            continue


def session_dir() -> Path:
    if "session_dir" not in st.session_state:
        _prune_stale_sessions(SESSIONS_ROOT, SESSION_MAX_AGE_HOURS)
        d = SESSIONS_ROOT / uuid.uuid4().hex
        d.mkdir(parents=True, exist_ok=True)
        st.session_state.session_dir = d.resolve()
    return st.session_state.session_dir


def reset(*keys: str) -> None:
    for k in keys:
        st.session_state.pop(k, None)


@contextmanager
def surfaced_warnings():
    """Show warnings logged by space.core during a block as Streamlit warnings."""
    messages: List[str] = []

    class _Handler(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())

    handler = _Handler(level=logging.WARNING)
    logger = logging.getLogger("space")
    logger.addHandler(handler)
    try:
        yield messages
    finally:
        logger.removeHandler(handler)
        for m in dict.fromkeys(messages):
            st.warning(m, icon="⚠️")


def status_progress(status):
    bar = status.progress(0.0)

    def cb(fraction: float, message: str) -> None:
        bar.progress(fraction, text=message)
    return cb


def is_dark() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:  # noqa: BLE001
        return False


def chart(fig, key: Optional[str] = None):
    st.plotly_chart(fig, theme=None, key=key, config={"displaylogo": False})


def pct(x: float) -> str:
    return f"{x * 100:.1f}%" if x >= 0.001 or x == 0 else f"{x * 100:.2f}%"


# ---------------------------------------------------------------------------
# Sidebar: new analysis / saved runs
# ---------------------------------------------------------------------------

def _use_example(source: str, query: str) -> None:
    st.session_state.source_input = source
    st.session_state.query_input = query


def sidebar() -> None:
    sb = st.sidebar
    sb.markdown("### 🧬 SPACE")
    sb.caption("Find and interpret protein variants across sequence databases.")
    st.session_state.setdefault("source_input", "NCBI")
    with sb.form("fetch_form"):
        source = st.segmented_control("Database", ["NCBI", "UniProt"], key="source_input", required=True)
        query = st.text_area(
            "Search query", key="query_input", height=90,
            placeholder='"Henipavirus hendraense"[Organism] AND fusion[Title]',
            help="NCBI: Entrez syntax, e.g. `\"Influenza A virus\"[Organism] AND hemagglutinin[Title]`.\n\n"
                 "UniProt: query syntax, e.g. `gene:TP53 AND organism_id:9606`.\n\n"
                 "Fielded queries avoid unrelated proteins slipping in.",
        )
        email = st.text_input("Email (required by NCBI)", key="email_input", placeholder="you@example.org",
                              autocomplete="email")
        max_seqs = st.number_input("Maximum sequences (-1 = all, capped at 20,000)", min_value=-1,
                                   value=500, step=100)
        with st.expander("Options"):
            refseq_only = st.checkbox("NCBI: RefSeq only")
            remove_pdb = st.checkbox("Remove PDB-derived sequences", value=True)
            reviewed_only = st.checkbox("UniProt: reviewed (Swiss-Prot) only")
        submitted = st.form_submit_button("Fetch sequences", type="primary", width="stretch")

    if submitted:
        if not query.strip():
            sb.error("Enter a search query.")
        elif source == "NCBI" and "@" not in (email or ""):
            sb.error("NCBI requires a valid email address.")
        else:
            fetch(source, query.strip(), email, int(max_seqs), refseq_only, reviewed_only, remove_pdb)

    with sb.expander("…or upload a FASTA file"):
        up = st.file_uploader("FASTA", type=["fasta", "fa", "faa", "fas", "txt"], label_visibility="collapsed")
        if up is not None and st.button("Use uploaded sequences", width="stretch"):
            load_uploaded_fasta(up)

    runs = list_runs(session_dir() / "runs")
    if runs:
        sb.divider()
        sb.markdown("**Runs in this session**")
        labels = {r["run_id"]: f"{r['query'] or r['run_id']} · {r['created'][11:16]}" for r in runs}
        current = st.session_state.get("run")
        chosen = sb.selectbox("Open run", list(labels), format_func=labels.get,
                              index=list(labels).index(current.run_id) if current and current.run_id in labels else 0,
                              label_visibility="collapsed")
        if current is None or chosen != current.run_id:
            if sb.button("Open", width="stretch"):
                st.session_state.run = SpaceRun.load(Path(next(r["run_dir"] for r in runs if r["run_id"] == chosen)))
                reset("annotation", "metadata", "structure", "pdb_hits")
                st.rerun()

    sb.divider()
    sb.caption("Zyla, D. **SPACE**: Sequence Protein Alignment and Conservation Engine. "
               "Based on Zyla, D. *et al.* Science (2024) · [GitHub](https://github.com/dzyla/SPACE)")


def fetch(source, query, email, max_seqs, refseq_only, reviewed_only, remove_pdb) -> None:
    sdir = session_dir() / "fetch" / uuid.uuid4().hex[:8]
    with st.status(f"Fetching from {source}…", expanded=True) as status, surfaced_warnings():
        try:
            path = fetch_sequences(query, sdir / "sequences.fasta", source=source, email=email,
                                   max_seqs=max_seqs, refseq_only=refseq_only, reviewed_only=reviewed_only,
                                   progress=status_progress(status))
            records = read_fasta(path, remove_pdb=remove_pdb)
        except Exception as exc:  # noqa: BLE001
            status.update(label=f"Fetch failed: {exc}", state="error")
            return
        if not records:
            status.update(label="No sequences found.", state="error")
            return
        status.update(label=f"Fetched {len(records)} sequences", state="complete", expanded=False)
    reset("run", "hits", "hits_key", "annotation", "metadata", "structure", "pdb_hits")
    st.session_state.update(records=records, fasta_path=path, fetch_params={
        "query": query, "source": source, "email": email, "max_seqs": max_seqs,
        "refseq_only": refseq_only, "reviewed_only": reviewed_only, "remove_pdb": remove_pdb})


def load_uploaded_fasta(upload) -> None:
    sdir = session_dir() / "fetch" / uuid.uuid4().hex[:8]
    sdir.mkdir(parents=True, exist_ok=True)
    path = sdir / "sequences.fasta"
    path.write_bytes(upload.getbuffer())
    try:
        records = read_fasta(path)
    except Exception as exc:  # noqa: BLE001
        st.sidebar.error(f"Could not read FASTA: {exc}")
        return
    if len(records) < 2:
        st.sidebar.error("The file needs at least two protein sequences.")
        return
    reset("run", "hits", "hits_key", "annotation", "metadata", "structure", "pdb_hits")
    st.session_state.update(records=records, fasta_path=str(path), fetch_params={
        "query": upload.name, "source": "FASTA", "email": st.session_state.get("email_input", ""),
        "max_seqs": len(records), "refseq_only": False, "reviewed_only": False, "remove_pdb": False})
    st.rerun()


# ---------------------------------------------------------------------------
# Welcome
# ---------------------------------------------------------------------------

def welcome() -> None:
    st.title("Protein variants across sequence databases")
    st.markdown(
        "SPACE fetches every matching sequence from **NCBI** or **UniProt**, aligns them to a "
        "reference and reports each **substitution, insertion and deletion** with its frequency, "
        "conservation, known-variant annotations (dbSNP, ClinVar) and position on the 3D structure."
    )
    c = st.columns(3)
    for col, (label, source, query) in zip(c, EXAMPLES):
        with col.container(border=True):
            st.markdown(f"**{label}**")
            st.code(query, language=None, wrap_lines=True)
            st.button("Use this query", key=f"ex_{label}", on_click=_use_example, args=(source, query),
                      width="stretch")
    st.caption("Pick an example or write your own query in the sidebar, then press **Fetch sequences**.")
    with st.expander("How frequencies are computed"):
        st.markdown(
            "- Each sequence only counts at positions it actually covers, so fragments add data without "
            "diluting frequencies elsewhere.\n"
            "- **Unique frequency** counts identical sequences once — databases often hold many copies "
            "from one outbreak or study.\n"
            "- `X` and other ambiguous residues are treated as missing data, never as a variant.\n"
            "- Variant names: `I6G` (substitution), `K5del` / `K5_Q8del` (deletion), "
            "`A4_Y5insWW` (insertion between residues 4 and 5), plus HGVS protein notation."
        )


# ---------------------------------------------------------------------------
# Setup: reference + filters
# ---------------------------------------------------------------------------

def setup_stage(dark: bool) -> None:
    records = st.session_state.records
    params = st.session_state.fetch_params
    st.title("Set up the analysis")
    st.markdown(f"{len(records)} sequences from **{params['source']}** for `{params['query']}`")

    with st.container(border=True):
        st.markdown("#### 1 · Reference sequence")
        auto_seq, auto_id = choose_reference(records)
        mode = st.segmented_control("Reference", ["From results", "Paste sequence"], default="From results",
                                    required=True, label_visibility="collapsed")
        if mode == "From results":
            ids = [r.id for r in records]
            desc = {r.id: f"{r.id} · {len(r.seq)} aa · {r.description[len(r.id):].strip()[:70]}" for r in records}
            ref_id = st.selectbox("Reference record", ids, index=ids.index(auto_id) if auto_id in ids else 0,
                                  format_func=desc.get,
                                  help="Defaults to a curated (Swiss-Prot/RefSeq) entry or the most common "
                                       "full-length sequence — usually the wild type.")
            ref_seq, _ = choose_reference(records, reference_id=ref_id)
        else:
            ref_id = None
            ref_seq = parse_sequence_text(st.text_area("Reference (raw or FASTA)", height=110))
            if not ref_seq:
                st.info("Paste a reference sequence to continue.")
                return
        st.caption(f"Reference length: {len(ref_seq)} aa")

    key = (st.session_state.fasta_path, ref_seq)
    if st.session_state.get("hits_key") != key:
        with st.spinner("Aligning sequences to the reference…"), surfaced_warnings():
            st.session_state.hits = pairwise_to_reference(records, ref_seq)
        st.session_state.hits_key = key
    hits: pd.DataFrame = st.session_state.hits

    with st.container(border=True):
        st.markdown("#### 2 · Choose sequences")
        left, right = st.columns([2, 3])
        with left:
            lo_i, hi_i = float(hits["identity"].min()), float(hits["identity"].max())
            identity = st.slider("Identity to reference (%)", 0.0, 100.0,
                                 (max(0.0, np.floor(lo_i)), 100.0), 0.5,
                                 help="Identical residues / alignment length. Low values are usually "
                                      "other proteins that matched the query.")
            lo_l, hi_l = int(hits["length"].min()), int(hits["length"].max())
            length = st.slider("Length (aa)", lo_l, max(hi_l, lo_l + 1), (lo_l, max(hi_l, lo_l + 1)))
            coverage = st.slider("Minimum coverage of the reference (%)", 0, 100, 30,
                                 help="Fragments still count only at positions they cover.")
            with st.expander("Advanced"):
                deletion = st.slider("Exclude sequences with internal deletions above (%)", 0, 100, 20)
                skip_x = st.checkbox("Treat ambiguous residues (X) as missing data", value=True)
                tree = st.checkbox("Build phylogenetic tree", value=True)
        selected = filter_hits(hits, identity[0], identity[1], length[0], length[1], coverage)
        with right:
            chart(viz.hits_density(hits, identity, length, dark=dark), key="density")
        st.markdown(f"**{len(selected)}** of {len(hits)} sequences selected")
        start_run = st.button("Align and call variants", type="primary", disabled=selected.empty)

    if start_run:
        run_dir = new_run_dir(params["query"], root=session_dir() / "runs")
        (run_dir / "sequences.fasta").write_text(Path(st.session_state.fasta_path).read_text())
        with st.status("Running analysis…", expanded=True) as status, surfaced_warnings():
            try:
                run = analyse(
                    selected, ref_seq, run_dir, reference_id=ref_id,
                    params={**params, "min_identity": identity[0], "max_identity": identity[1],
                            "min_length": length[0], "max_length": length[1]},
                    deletion_threshold=deletion, min_coverage=coverage, skip_ambiguous=skip_x,
                    build_phylo=tree, progress=status_progress(status))
                run.hits = hits
                run.files["sequences"] = str(run_dir / "sequences.fasta")
                run.save()
            except Exception as exc:  # noqa: BLE001
                status.update(label=f"Analysis failed: {exc}", state="error")
                return
            status.update(label="Analysis complete", state="complete", expanded=False)
        reset("annotation", "metadata", "structure", "pdb_hits")
        st.session_state.run = run
        st.rerun()


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def reference_accession(run: SpaceRun) -> str:
    if run.reference_id:
        acc = extract_uniprot_accession(run.reference_id)
        if acc:
            return acc
    return ""


def annotated_summary(run: SpaceRun) -> pd.DataFrame:
    s = run.variants.summary
    ann = st.session_state.get("annotation")
    if ann:
        s = annotate.merge_known_variants(s, ann["known"])
        s = annotate.add_feature_overlaps(s, ann["features"])
    sm = st.session_state.get("structure")
    if sm:
        s = annotate.add_structure_context(s, sm.chains[sm.best_chain])
    return s


def results(run: SpaceRun, dark: bool) -> None:
    info = run.summary()
    head, actions = st.columns([5, 1])
    head.title("Variant report")
    head.markdown(f"`{info['query'] or run.run_id}` · reference **{run.reference_id or 'user sequence'}** "
                  f"({len(run.reference)} aa) · {info['source']}")
    if actions.button("New analysis", width="stretch"):
        reset("run")
        st.rerun()

    m = st.columns(5)
    m[0].metric("Sequences analysed", info["sequences_analysed"],
                help=f"{info['sequences_excluded']} excluded (see Sequences tab)")
    m[1].metric("Unique sequences", info["unique_sequences"])
    m[2].metric("Distinct variants", info["distinct_variants"])
    m[3].metric("Variable positions", info["variant_positions"])
    m[4].metric("Variants ≥ 1%", info["variants_ge_1pct"])

    tabs = st.tabs(["Variants", "Conservation", "Structure", "Alignment & tree", "Sequences", "Export"])
    with tabs[0]:
        selected = variants_tab(run, dark)
    with tabs[1]:
        conservation_tab(run, dark)
    with tabs[2]:
        structure_tab(run, selected, dark)
    with tabs[3]:
        alignment_tab(run, selected, dark)
    with tabs[4]:
        sequences_tab(run)
    with tabs[5]:
        export_tab(run)


def variants_tab(run: SpaceRun, dark: bool) -> List[str]:
    summary = annotated_summary(run)
    positions = run.positions
    if summary.empty:
        st.info("No variants: every analysed sequence matches the reference where it is covered.")
        return []

    c = st.columns([3, 2, 3, 1.4])
    basis = c[0].segmented_control("Count", ["All sequences", "Unique sequences"], default="All sequences",
                                   required=True, help="Unique: identical sequences counted once.")
    unique = basis == "Unique sequences"
    min_freq = c[1].select_slider("Minimum frequency", FREQ_STEPS, value=0.0, format_func=pct)
    types = c[2].pills("Type", ["Substitution", "Insertion", "Deletion"], selection_mode="multi",
                       default=["Substitution", "Insertion", "Deletion"])
    log_y = c[3].toggle("Log scale", value=True)
    L = len(run.reference)
    span = st.slider("Position range", 1, L, (1, L)) if L > 1 else (1, 1)

    freq_col = "Unique frequency" if unique else "Frequency"
    view = summary[
        (summary[freq_col] >= min_freq) & summary["Type"].isin(types or [])
        & summary["Position"].between(*span)
    ].sort_values([freq_col, "Position"], ascending=[False, True]).reset_index(drop=True)

    # Selections are row indices into the rendered table, so the widget key
    # includes the filter state: changing filters starts a fresh selection
    # instead of silently pointing at different mutations.
    table_key = "variant_table_" + uuid.uuid5(uuid.NAMESPACE_OID, repr(
        (run.run_id, unique, min_freq, tuple(types or []), tuple(span), tuple(summary.columns)))).hex[:12]
    sel_state = st.session_state.get(table_key)
    rows = list(sel_state.selection.rows) if sel_state and sel_state.selection else []
    selected = [view.loc[i, "Mutation"] for i in rows if i < len(view)]

    ann = st.session_state.get("annotation")
    feats = ann["features"] if ann else None
    chart(viz.variant_landscape(view, positions[positions["Position"].between(*span)], features=feats,
                                unique=unique, log_y=log_y, highlight=selected, dark=dark), key="landscape")
    st.caption("Stems show the most frequent variant at each position; the strip below is al2co "
               "conservation (dark = conserved). Select table rows to highlight variants everywhere.")

    cols = ["Mutation", "HGVS", "Type", "Position", "Count", "Covering", "Frequency", "Unique carriers",
            "Unique frequency", "BLOSUM62", "Change"]
    extra = [c for c in ("Known", "Clinical significance", "dbSNP", "SIFT", "PolyPhen", "Features",
                         "Exposure", "RSA", "B-factor / pLDDT") if c in view]
    st.dataframe(
        view[cols + extra], key=table_key, on_select="rerun", selection_mode="multi-row",
        hide_index=True, height=420,
        column_config={
            "Frequency": st.column_config.ProgressColumn("Frequency", format="percent", min_value=0, max_value=1),
            "Unique frequency": st.column_config.ProgressColumn("Unique freq.", format="percent",
                                                                min_value=0, max_value=1),
            "Covering": st.column_config.NumberColumn(help="Sequences covering this position"),
            "BLOSUM62": st.column_config.NumberColumn(format="%d", help="Substitution score; negative = unusual"),
            "RSA": st.column_config.NumberColumn(format="%.2f", help="Relative solvent accessibility"),
            "Known": st.column_config.CheckboxColumn(help="Reported in the EBI Proteins variation API"),
        },
    )
    st.caption(f"{len(view)} variants shown · select rows for details, structure and tree highlighting")

    if selected:
        variant_detail(run, summary, selected, dark)

    left, right = st.columns(2)
    with left.container(border=True):
        st.markdown("**Known variants & features (UniProt)**")
        acc = st.text_input("UniProt accession of the reference", value=reference_accession(run),
                            placeholder="e.g. P04637", key="annot_acc")
        if st.button("Annotate", disabled=not acc):
            with st.spinner("Querying UniProt and the EBI Proteins API…"), surfaced_warnings():
                try:
                    st.session_state.annotation = {
                        "accession": acc,
                        "features": annotate.uniprot_features(acc, run.reference),
                        "known": annotate.known_variants(acc, run.reference),
                    }
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Annotation failed: {exc}")
        if ann:
            known = summary["Known"].sum() if "Known" in summary else 0
            st.caption(f"{ann['accession']}: {len(ann['features'])} features, {len(ann['known'])} known "
                       f"single-residue variants; {known} of your variants are already reported.")
    with right.container(border=True):
        st.markdown("**Sample metadata (GenBank)**")
        meta = st.session_state.get("metadata")
        email = (st.session_state.get("fetch_params") or {}).get("email") or run.params.get("email")
        if run.params.get("source") != "NCBI":
            st.caption("Host, country and collection date are available for NCBI runs.")
        elif st.button("Fetch host / country / collection date", disabled=not email):
            with st.spinner("Downloading GenBank records…"), surfaced_warnings():
                st.session_state.metadata = annotate.ncbi_source_metadata(list(run.selected["id"]), email)
                st.rerun()
        if meta is not None and not meta.empty:
            st.caption(f"Metadata for {len(meta)} sequences · years "
                       f"{meta['Year'].min()}–{meta['Year'].max()}")

    if meta is not None and not meta.empty and meta["Year"].notna().any():
        tracked = selected[:8] or list(view["Mutation"].head(4))
        included = run.variants.per_sequence.query("Status == 'included'")["Sequence ID"]
        by_year = annotate.variants_by_year(run.variants.events, meta, tracked, included=included)
        if not by_year.empty:
            chart(viz.variant_timeline(by_year, dark=dark), key="timeline")

    with st.expander("Substitution heatmap"):
        chart(viz.substitution_heatmap(view, unique=unique, dark=dark), key="subheat")
    return selected


def variant_detail(run: SpaceRun, summary: pd.DataFrame, selected: List[str], dark: bool) -> None:
    mut = selected[0]
    row = summary.set_index("Mutation").loc[mut]
    carriers = row["Sequence IDs"].split(";")
    pos = int(row["Position"])
    p = run.positions.set_index("Position").loc[pos]
    with st.container(border=True):
        st.markdown(f"#### {mut} · `{row['HGVS']}`")
        c = st.columns(4)
        c[0].metric("Carriers", f"{row['Count']} / {row['Covering']}", help="Sequences carrying it / covering it")
        c[1].metric("Unique carriers", f"{row['Unique carriers']} / {row['Unique covering']}")
        c[2].metric("al2co at position", "–" if pd.isna(p["al2co_score"]) else f"{p['al2co_score']:.2f}")
        c[3].metric("Other variants here", int(p["Variants"]) - 1)
        st.caption(f"{row['Type']} · {row['Change']}" + (f" · BLOSUM62 {row['BLOSUM62']:.0f}"
                                                        if pd.notna(row["BLOSUM62"]) else ""))
        table = pd.DataFrame({"Sequence ID": carriers})
        meta = st.session_state.get("metadata")
        if meta is not None and not meta.empty:
            table = table.merge(meta, on="Sequence ID", how="left")
        hits = run.hits.drop_duplicates("id").set_index("id")
        if "description" in hits:
            table["Description"] = table["Sequence ID"].map(hits["description"])
        st.dataframe(table, hide_index=True, height=min(300, 36 * len(table) + 38))
        if len(selected) > 1:
            st.caption(f"Also selected: {', '.join(selected[1:])}")


def conservation_tab(run: SpaceRun, dark: bool) -> None:
    positions = run.positions
    chart(viz.conservation_plot(positions, dark=dark), key="cons")
    L = len(run.reference)
    if L < 3:
        return
    c = st.columns([3, 1])
    start = c[0].slider("Logo window start", 1, max(1, L - 1), 1)
    width = c[1].number_input("Window width", 20, 200, 60, step=10)
    threshold = st.slider("Consensus identity threshold", 0.0, 1.0, 0.6, 0.05,
                          help="Positions where the most common residue is rarer than this become X.")
    bits, consensus = logo_and_consensus(run.msa, run.mapping, threshold)
    chart(viz.sequence_logo(bits, start, min(L, start + int(width) - 1), dark=dark), key="logo")
    st.markdown("**Consensus**")
    st.code(consensus, language=None, wrap_lines=True)


@st.fragment
def structure_tab(run: SpaceRun, selected: List[str], dark: bool) -> None:
    sdir = run.run_dir / "structure"
    src = st.segmented_control("Structure source", ["AlphaFold", "PDB search", "Upload"], default="AlphaFold",
                               required=True)
    path, meta, label = None, None, None
    if src == "AlphaFold":
        c = st.columns([3, 1])
        acc = c[0].text_input("UniProt accession", value=reference_accession(run), placeholder="e.g. P04637")
        if c[1].button("Load model", disabled=not acc, width="stretch"):
            try:
                path, meta = download_alphafold(acc, sdir)
                label = acc
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
    elif src == "PDB search":
        c = st.columns([3, 1])
        ident = c[0].slider("Minimum sequence identity", 0.3, 1.0, 0.9, 0.05)
        if c[1].button("Search RCSB", width="stretch"):
            try:
                st.session_state.pdb_hits = search_pdb(run.reference, identity_cutoff=ident)
            except Exception as exc:  # noqa: BLE001
                st.error(f"RCSB search failed: {exc}")
        hits = st.session_state.get("pdb_hits")
        if hits is not None:
            if not hits:
                st.warning("No experimental structures at this identity. Try a lower cutoff or AlphaFold.")
            else:
                c = st.columns([3, 1])
                pid = c[0].selectbox(f"{len(hits)} entries", hits)
                if c[1].button("Load entry", width="stretch"):
                    try:
                        path, meta, label = download_pdb(pid, sdir), pdb_metadata(pid), pid
                    except Exception as exc:  # noqa: BLE001
                        st.error(str(exc))
    else:
        up = st.file_uploader("PDB or mmCIF file", type=["pdb", "ent", "cif"])
        if up is not None and st.button("Use uploaded structure"):
            sdir.mkdir(parents=True, exist_ok=True)
            suffix = ".cif" if up.name.lower().endswith(".cif") else ".pdb"
            path = str(sdir / f"uploaded{suffix}")
            Path(path).write_bytes(up.getbuffer())
            meta, label = {"id": up.name, "title": "Uploaded structure"}, up.name

    if path:
        try:
            al = run.positions.set_index("Position")["al2co_score"].dropna().to_dict()
            st.session_state.structure = map_to_structure(
                path, run.reference, values=al, out_dir=sdir / "mapped", source=src,
                structure_id=label, metadata=meta)
            st.rerun()  # refresh variant table structure columns
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not map the structure: {exc}")

    sm = st.session_state.get("structure")
    if not sm:
        st.info("Load a structure to see conservation and variants in 3D.")
        return

    md = sm.metadata
    st.caption(" · ".join(str(v) for k, v in md.items() if v and v != "N/A" and k != "authors"))
    chain_table = pd.DataFrame([{"Chain": c, "Coverage (%)": round(m.coverage, 1),
                                 "Identity (%)": round(m.identity, 1), "Residues": len(m.residues)}
                                for c, m in sm.chains.items()])
    c = st.columns([1.5, 2.6, 2.2, 2.2])
    chains = c[0].multiselect("Chains", list(sm.chains), default=[sm.best_chain])
    mode = c[1].segmented_control("Colour by", ["Conservation", "Variant frequency"], default="Conservation",
                                  required=True)
    style = c[2].segmented_control("Style", ["cartoon", "stick", "sphere"], default="cartoon", required=True)
    show = c[3].segmented_control("Label", ["Selected", "Top 20", "None"], default="Selected" if selected else "Top 20",
                                  required=True)

    positions = run.positions.set_index("Position")
    summary = run.variants.summary
    if mode == "Conservation":
        per_pos = positions["al2co_score"].dropna()
    else:
        per_pos = positions["Max frequency"] * 100
    finite = per_pos[np.isfinite(per_pos)]
    vmin, vmax = (float(finite.min()), float(finite.max())) if not finite.empty else (0.0, 1.0)
    if show == "Selected":
        label_muts = summary[summary["Mutation"].isin(selected)]
    elif show == "Top 20":
        label_muts = summary.head(20)
    else:
        label_muts = summary.iloc[0:0]

    payload = []
    for ch in chains:
        cm = sm.chains[ch]
        values = {cm.ref_to_residue[p]: float(v) for p, v in per_pos.items() if p in cm.ref_to_residue}
        default = vmax if mode == "Conservation" else vmin
        hl = []
        for pos, g in label_muts.groupby("Position"):
            if pos in cm.ref_to_residue:
                hl.append((cm.ref_to_residue[pos][0], ", ".join(g["Mutation"].head(3))))
        payload.append({"pdb": viz.pdb_with_values(cm.pdb_path, values, default), "vmin": vmin, "vmax": vmax,
                        "highlights": hl})
    if payload:
        st.iframe(viz.structure_html(payload, "conservation" if mode == "Conservation" else "frequency",
                                     style=style, label_positions=show != "None", dark=dark, height=560),
                  height=580)
        if mode == "Conservation":
            ramp = "faint = variable → strong blue = conserved"
        else:
            ramp = "grey = no variant → strongest orange = most frequent variant"
        st.caption(f"Colour: {ramp} ({vmin:.2f} to {vmax:.2f}). Unmapped residues use the "
                   f"{'conserved' if mode == 'Conservation' else 'no-variant'} colour.")
    st.dataframe(chain_table, hide_index=True)
    ctx = annotate.add_structure_context(summary, sm.chains[chains[0] if chains else sm.best_chain])
    if "RSA" in ctx:
        unmapped = int((ctx["Structure residue"] == "").sum())
        ctx = ctx[ctx["Structure residue"] != ""]
        n_surface = (ctx["Exposure"] == "surface").sum()
        st.markdown(f"**Structural context** · {n_surface} of {len(ctx)} modelled variants at surface-exposed "
                    f"residues" + (f" ({unmapped} fall outside the model)" if unmapped else ""))
        st.dataframe(ctx[["Mutation", "Frequency", "Structure residue", "RSA", "Exposure", "B-factor / pLDDT"]],
                     hide_index=True, height=300,
                     column_config={"Frequency": st.column_config.NumberColumn(format="percent"),
                                    "RSA": st.column_config.NumberColumn(format="%.2f")})


def alignment_tab(run: SpaceRun, selected: List[str], dark: bool) -> None:
    L = len(run.reference)
    if L < 3:
        st.info("The reference is too short to display an alignment window.")
        return
    carriers = []
    if selected:
        ev = run.variants.events
        carriers = list(ev.loc[ev["Mutation"] == selected[0], "MSA ID"].astype(str).unique())
    c = st.columns([3, 1, 2, 2])
    default_start = 1
    if selected:
        default_start = max(1, int(run.variants.summary.set_index("Mutation").loc[selected[0], "Position"]) - 30)
    start = c[0].slider("Window start", 1, max(1, L), min(default_start, L))
    width = c[1].number_input("Width", 20, 400, 80, step=20)
    only = c[2].toggle(f"Only carriers of {selected[0]}" if selected else "Only carriers of a selected variant",
                       disabled=not selected)
    diff = c[3].toggle("Differences from reference only", value=True)
    chart(viz.msa_view(run.msa, run.mapping, start, min(L, start + int(width) - 1),
                       row_ids=carriers if only else None, differences_only=diff, dark=dark), key="msa")
    if run.tree:
        note = " (random subset)" if run.tree.subsampled else ""
        st.caption(f"{run.tree.method} tree from MSA identity distances{note}.")
        chart(viz.tree_figure(run.tree.newick_path, highlight=carriers,
                              highlight_label=f"{selected[0]} carriers" if selected else "carriers", dark=dark),
              key="tree")
    else:
        st.info("No tree for this run.")


def sequences_tab(run: SpaceRun) -> None:
    ps = run.variants.per_sequence.copy()
    hits = run.hits.drop_duplicates("id").set_index("id")
    ps["Identity (%)"] = ps["Sequence ID"].map(hits["identity"])
    meta = st.session_state.get("metadata")
    if meta is not None and not meta.empty:
        ps = ps.merge(meta, on="Sequence ID", how="left")
    st.markdown(f"**{(ps['Status'] == 'included').sum()} analysed**, "
                f"{(ps['Status'] == 'excluded').sum()} excluded")
    st.dataframe(ps.drop(columns=["MSA ID"]), hide_index=True, height=480,
                 column_config={"Coverage (%)": st.column_config.NumberColumn(format="%.1f"),
                                "Internal deletions (%)": st.column_config.NumberColumn(format="%.1f"),
                                "Identity (%)": st.column_config.NumberColumn(format="%.1f")})
    with st.expander(f"All fetched sequences ({len(run.hits)})"):
        st.dataframe(run.hits.drop(columns=["sequence"], errors="ignore"), hide_index=True)


@st.cache_data(show_spinner=False, max_entries=8)
def _zip_run(run_dir: str, stamp: float) -> bytes:
    buf = io.BytesIO()
    root = Path(run_dir)
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(root.rglob("*")):
            if f.is_file():
                z.write(f, arcname=str(Path(root.name) / f.relative_to(root)))
    return buf.getvalue()


def export_tab(run: SpaceRun) -> None:
    # The run is saved when created/opened; re-saving on every rerun would
    # bust the ZIP cache below.
    stamp = (run.run_dir / "run.json").stat().st_mtime
    st.download_button("Download all results (ZIP)", _zip_run(str(run.run_dir), stamp),
                       file_name=f"{run.run_id}.zip", mime="application/zip", type="primary")
    s = annotated_summary(run)
    c = st.columns(3)
    c[0].download_button("Variants (TSV)", s.to_csv(sep="\t", index=False), f"{run.run_id}_variants.tsv")
    c[1].download_button("Per-sequence calls (TSV)", run.variants.per_sequence.to_csv(sep="\t", index=False),
                         f"{run.run_id}_sequences.tsv")
    c[2].download_button("Per-position table (TSV)", run.positions.to_csv(sep="\t", index=False),
                         f"{run.run_id}_positions.tsv")
    st.markdown("**Reuse this run from Python, the CLI or an MCP client**")
    st.code(f'from space.core.pipeline import SpaceRun\nrun = SpaceRun.load("{run.run_dir}")\n'
            f'run.variants.filter(min_frequency=0.01)', language="python")


# ---------------------------------------------------------------------------

def main() -> None:
    session_dir()
    sidebar()
    dark = is_dark()
    run = st.session_state.get("run")
    if run is not None:
        results(run, dark)
    elif st.session_state.get("records"):
        setup_stage(dark)
    else:
        welcome()


if __name__ == "__main__":
    main()
