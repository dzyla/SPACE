"""Plotly figure builders and 3D viewer HTML — no Streamlit.

Colour roles (validated with the dataviz palette checks):
  * variant type  -> categorical slots 1-3 (blue / orange / aqua) + marker shape
  * conservation  -> sequential blue (light = variable, dark = conserved)
  * frequency     -> sequential orange (second sequential context)
  * amino acids   -> WebLogo-style chemistry classes (letters carry identity)
Every builder takes ``dark`` and picks the matching selected steps.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------

LIGHT = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
             grid="#e1e0d9", axis="#c3c2b7", series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                                      "#e87ba4", "#008300", "#6250d6", "#e34948"])
DARK = dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781",
            grid="#2c2c2a", axis="#383835", series=["#3987e5", "#d95926", "#199e70", "#c98500",
                                                     "#d55181", "#008300", "#9085e9", "#e66767"])

SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
SEQ_ORANGE = ["#fde3d6", "#f9c3a8", "#f4a07b", "#eb6834", "#d4531f", "#b04015", "#86300f"]

TYPE_ORDER = ["Substitution", "Insertion", "Deletion"]
TYPE_SYMBOL = {"Substitution": "circle", "Insertion": "triangle-up", "Deletion": "square"}

CHEMISTRY_COLORS = {
    **dict.fromkeys("DE", "#E03131"),          # acidic
    **dict.fromkeys("RKH", "#1971C2"),         # basic
    **dict.fromkeys("STNQCY", "#2F9E44"),      # polar
    **dict.fromkeys("AVLIPWFM", "#52585E"),    # hydrophobic (deliberately achromatic)
    "G": "#E8590C",                            # glycine
    **dict.fromkeys("XBZJUO", "#ADB5BD"),
}
CHEMISTRY_LEGEND = [("Acidic", "#E03131"), ("Basic", "#1971C2"), ("Polar", "#2F9E44"),
                    ("Hydrophobic", "#52585E"), ("Glycine", "#E8590C")]
GAP_COLOR = {False: "#e1e0d9", True: "#2c2c2a"}

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def tokens(dark: bool = False) -> dict:
    return DARK if dark else LIGHT


def type_colors(dark: bool = False) -> Dict[str, str]:
    s = tokens(dark)["series"]
    return dict(zip(TYPE_ORDER, s[:3]))


def seq_ramp(colors: Sequence[str], dark: bool) -> List[str]:
    """Sequential ramp whose low end recedes into the surface: light→dark on a
    light surface, dark→light on a dark one."""
    return list(colors)[::-1] if dark else list(colors)


def _scale(colors: Sequence[str]) -> list:
    n = len(colors) - 1
    return [[i / n, c] for i, c in enumerate(colors)]


def _style(fig: go.Figure, dark: bool, title: Optional[str] = None, height: Optional[int] = None) -> go.Figure:
    t = tokens(dark)
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=t["ink"]), x=0, xanchor="left") if title else None,
        font=dict(family=FONT, size=12, color=t["ink2"]),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=t["surface"],
        margin=dict(l=56, r=16, t=48 if title else 16, b=44),
        hoverlabel=dict(bgcolor=t["surface"], bordercolor=t["axis"], font=dict(color=t["ink"], family=FONT)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", font=dict(color=t["ink2"])),
    )
    fig.update_xaxes(showgrid=False, linecolor=t["axis"], tickcolor=t["axis"], zeroline=False,
                     title_font=dict(color=t["ink2"]))
    fig.update_yaxes(gridcolor=t["grid"], gridwidth=1, linecolor=t["axis"], zeroline=False,
                     title_font=dict(color=t["ink2"]))
    if height:
        fig.update_layout(height=height)
    return fig


# ---------------------------------------------------------------------------
# Sequence selection
# ---------------------------------------------------------------------------

def hits_density(hits: pd.DataFrame, identity_range=None, length_range=None, dark: bool = False) -> go.Figure:
    """Each fetched sequence as a point (length vs identity); the filter box is drawn on top.

    Points inside the current filters use series 1, the rest are muted.
    """
    t = tokens(dark)
    x, y = hits["length"].to_numpy(), hits["identity"].to_numpy()
    inside = np.ones(len(hits), dtype=bool)
    if identity_range and length_range:
        inside = ((x >= length_range[0]) & (x <= length_range[1])
                  & (y >= identity_range[0]) & (y <= identity_range[1]))
    fig = go.Figure()
    for flag, name, color in ((False, "excluded", t["muted"]), (True, "selected", t["series"][0])):
        m = inside == flag
        if not m.any():
            continue
        d = hits[m]
        fig.add_trace(go.Scattergl(
            x=d["length"], y=d["identity"], mode="markers", name=f"{name} ({int(m.sum())})",
            marker=dict(size=8, color=color, opacity=0.75, line=dict(color=t["surface"], width=1)),
            customdata=np.stack([d["id"], d["coverage"]], axis=1),
            hovertemplate="%{customdata[0]}<br>%{x} aa · identity %{y:.1f}% · coverage "
                          "%{customdata[1]:.0f}%<extra></extra>",
        ))
    if identity_range and length_range:
        fig.add_shape(type="rect", x0=length_range[0], x1=length_range[1], y0=identity_range[0],
                      y1=identity_range[1], line=dict(color=t["ink2"], width=1), fillcolor="rgba(0,0,0,0)")
    pad = max(5, 0.02 * (x.max() - x.min()))
    fig.update_xaxes(title="Sequence length (aa)", range=[x.min() - pad, x.max() + pad])
    fig.update_yaxes(title="Identity to reference (%)", range=[max(0, y.min() - 3), 101])
    return _style(fig, dark, "Fetched sequences: length vs. identity to reference", 380)


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------

def variant_landscape(
    summary: pd.DataFrame,
    positions: pd.DataFrame,
    features: Optional[pd.DataFrame] = None,
    unique: bool = False,
    log_y: bool = True,
    highlight: Optional[Iterable[str]] = None,
    dark: bool = False,
) -> go.Figure:
    """Lollipop of variant frequency by position over conservation and feature tracks."""
    t = tokens(dark)
    tc = type_colors(dark)
    has_features = features is not None and not features.empty
    rows = 3 if has_features else 2
    heights = [0.72, 0.1, 0.18] if has_features else [0.86, 0.14]
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=heights)
    freq_col = "Unique frequency" if unique else "Frequency"
    count_col, cov_col = ("Unique carriers", "Unique covering") if unique else ("Count", "Covering")
    L = len(positions)
    highlight = set(highlight or [])

    if not summary.empty:
        s = summary.copy()
        s["_y"] = s[freq_col] * 100
        # stems: one per position up to the highest variant there
        top = s.groupby("Position")["_y"].max()
        sx, sy = [], []
        for p, v in top.items():
            sx += [p, p, None]
            sy += [1e-6 if log_y else 0, v, None]
        fig.add_trace(go.Scatter(x=sx, y=sy, mode="lines", line=dict(color=t["axis"], width=1),
                                 hoverinfo="skip", showlegend=False), row=1, col=1)
        for typ in TYPE_ORDER:
            d = s[s["Type"] == typ]
            if d.empty:
                continue
            sel = d["Mutation"].isin(highlight)
            fig.add_trace(go.Scatter(
                x=d["Position"], y=d["_y"], mode="markers", name=f"{typ} ({len(d)})",
                marker=dict(color=tc[typ], symbol=TYPE_SYMBOL[typ], size=np.where(sel, 14, 9),
                            line=dict(color=np.where(sel, t["ink"], t["surface"]).tolist(), width=np.where(sel, 2, 1.5).tolist())),
                customdata=np.stack([d["Mutation"], d["HGVS"], d[count_col], d[cov_col], d["Change"].fillna("")], axis=1),
                hovertemplate="<b>%{customdata[0]}</b> · %{customdata[1]}<br>"
                              "%{customdata[2]} of %{customdata[3]} sequences (%{y:.2f}%)<br>"
                              "%{customdata[4]}<extra></extra>",
            ), row=1, col=1)
        if highlight:
            hl = s[s["Mutation"].isin(highlight)]
            fig.add_trace(go.Scatter(x=hl["Position"], y=hl["_y"], mode="text", text=hl["Mutation"],
                                     textposition="top center", textfont=dict(color=t["ink"], size=11),
                                     hoverinfo="skip", showlegend=False), row=1, col=1)
    ylab = "Unique-sequence frequency (%)" if unique else "Frequency (%)"
    if log_y:
        lo = float(summary[freq_col].min() * 100) if not summary.empty else 0.01
        ticks = [v for v in (0.01, 0.1, 1, 10, 100) if v >= lo / 10]
        fig.update_yaxes(type="log", tickvals=ticks, ticktext=[f"{v:g}" for v in ticks],
                         range=[math.log10(lo / 2), math.log10(150)], row=1, col=1)
    fig.update_yaxes(title=ylab, row=1, col=1)

    # conservation strip
    cons = positions["al2co_score"] if "al2co_score" in positions else pd.Series(np.nan, index=positions.index)
    fig.add_trace(go.Heatmap(
        z=[cons.to_numpy()], x=positions["Position"], y=["al2co"], colorscale=_scale(seq_ramp(SEQ_BLUE, dark)),
        showscale=False, customdata=[positions["Residue"]],
        hovertemplate="Position %{x} (%{customdata})<br>al2co %{z:.2f}<extra></extra>",
    ), row=2, col=1)
    fig.update_yaxes(showgrid=False, row=2, col=1)

    if has_features:
        _feature_track(fig, features, row=3, dark=dark)
    fig.update_xaxes(range=[0.5, L + 0.5], row=rows, col=1, title="Reference position")
    fig = _style(fig, dark, None, 520 if has_features else 460)
    fig.update_layout(margin=dict(t=36))
    return fig


def _feature_track(fig: go.Figure, features: pd.DataFrame, row: int, dark: bool) -> None:
    t = tokens(dark)
    keep = features[~features["Type"].isin(["Chain"])].copy()
    lanes = ["Domain/region", "Site", "PTM"]
    lane_of = lambda typ: ("Site" if typ in ("Active site", "Binding site", "Site") else  # noqa: E731
                           "PTM" if typ in ("Glycosylation", "Modified residue", "Disulfide bond", "Lipidation")
                           else "Domain/region")
    keep["Lane"] = keep["Type"].map(lane_of)
    color = {"Domain/region": t["series"][0], "Site": t["series"][1], "PTM": t["series"][2]}
    for lane in lanes:
        d = keep[keep["Lane"] == lane]
        if d.empty:
            continue
        xs, ys, texts = [], [], []
        for f in d.itertuples():
            xs += [f.Start - 0.5, f.End + 0.5, None]
            ys += [lane, lane, None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color[lane], width=8),
                                 hoverinfo="skip", showlegend=False), row=row, col=1)
        mid = (d["Start"] + d["End"]) / 2
        fig.add_trace(go.Scatter(
            x=mid, y=[lane] * len(d), mode="markers", marker=dict(size=10, opacity=0),
            customdata=np.stack([d["Type"], d["Description"].fillna(""), d["Start"], d["End"]], axis=1),
            hovertemplate="%{customdata[0]}: %{customdata[1]}<br>%{customdata[2]}–%{customdata[3]}<extra></extra>",
            showlegend=False), row=row, col=1)
    fig.update_yaxes(showgrid=False, categoryorder="array", categoryarray=lanes[::-1], row=row, col=1)


def substitution_heatmap(summary: pd.DataFrame, unique: bool = False, max_positions: int = 150,
                         dark: bool = False) -> go.Figure:
    """Observed residue (rows) × reference position (columns), coloured by frequency."""
    t = tokens(dark)
    s = summary[summary["Type"] == "Substitution"].copy()
    freq_col = "Unique frequency" if unique else "Frequency"
    if s.empty:
        return _style(go.Figure(), dark, "No substitutions", 200)
    keep = s.groupby("Position")[freq_col].max().nlargest(max_positions).index
    s = s[s["Position"].isin(keep)]
    cols = sorted(s["Position"].unique())
    aas = list("ACDEFGHIKLMNPQRSTVWY")
    ref_at = s.drop_duplicates("Position").set_index("Position")["Ref"]
    z = np.full((len(aas), len(cols)), np.nan)
    ci = {p: i for i, p in enumerate(cols)}
    for alt, pos, f in zip(s["Alt"], s["Position"], s[freq_col]):
        if alt in aas:
            z[aas.index(alt), ci[pos]] = f * 100
    labels = [f"{ref_at[p]}{p}" for p in cols]
    ramp = [t["surface"]] + seq_ramp(SEQ_ORANGE[1:], dark)
    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=aas, colorscale=_scale(ramp), xgap=2, ygap=2, zmin=0,
        colorbar=dict(title=dict(text="%", side="right"), thickness=10, outlinewidth=0),
        hovertemplate="%{x} → %{y}<br>%{z:.2f}% of covering sequences<extra></extra>",
    ))
    fig.update_xaxes(type="category", tickangle=-60, title="Reference residue")
    fig.update_yaxes(title="Observed residue", autorange="reversed", showgrid=False)
    return _style(fig, dark, "Observed substitutions", 460)


def variant_timeline(by_year: pd.DataFrame, dark: bool = False) -> go.Figure:
    """Carrier fraction per collection year, one line per mutation (≤ 8)."""
    t = tokens(dark)
    fig = go.Figure()
    for i, (mut, d) in enumerate(by_year.groupby("Mutation", sort=False)):
        if i >= 8:
            break
        c = t["series"][i]
        fig.add_trace(go.Scatter(
            x=d["Year"], y=d["Frequency"] * 100, name=mut, mode="lines+markers",
            line=dict(color=c, width=2), marker=dict(size=8, color=c, line=dict(color=t["surface"], width=2)),
            customdata=np.stack([d["Carriers"], d["Sequences"]], axis=1),
            hovertemplate=f"<b>{mut}</b> · %{{x}}<br>%{{customdata[0]}} of %{{customdata[1]}} sequences (%{{y:.1f}}%)<extra></extra>",
        ))
    fig.update_xaxes(title="Collection year", dtick=1 if by_year["Year"].nunique() < 15 else None)
    fig.update_yaxes(title="Carriers (%)", rangemode="tozero")
    return _style(fig, dark, "Variant frequency by collection year", 360)


# ---------------------------------------------------------------------------
# Conservation
# ---------------------------------------------------------------------------

def conservation_plot(positions: pd.DataFrame, dark: bool = False) -> go.Figure:
    t = tokens(dark)
    d = positions.dropna(subset=["al2co_score"])
    fig = go.Figure()
    if d.empty:
        return _style(fig, dark, "al2co scores unavailable (identical sequences?)", 200)
    smooth = d["al2co_score"].rolling(9, center=True, min_periods=1).mean()
    fig.add_trace(go.Scatter(
        x=d["Position"], y=d["al2co_score"], mode="markers", name="al2co score",
        marker=dict(color=d["al2co_score"], colorscale=_scale(seq_ramp(SEQ_BLUE, dark)), size=7,
                    line=dict(color=t["surface"], width=1),
                    colorbar=dict(title=dict(text="al2co", side="right"), thickness=10, outlinewidth=0)),
        customdata=np.stack([d["Residue"], d["Conservation"].fillna(np.nan), d["Gap fraction"].fillna(np.nan)], axis=1),
        hovertemplate="%{customdata[0]}%{x}<br>al2co %{y:.2f}<br>modal residue %{customdata[1]:.0%}"
                      " · gaps %{customdata[2]:.0%}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(x=d["Position"], y=smooth, mode="lines", name="9-residue mean",
                             line=dict(color=t["ink2"], width=2), hoverinfo="skip"))
    fig.update_xaxes(title="Reference position")
    fig.update_yaxes(title="al2co conservation (z-score)")
    return _style(fig, dark, "Per-residue conservation", 380)


def sequence_logo(bits: pd.DataFrame, start: int, end: int, dark: bool = False) -> go.Figure:
    """Stacked logo, letters sorted by height at each position (tallest on top)."""
    t = tokens(dark)
    w = bits.loc[(bits.index >= start) & (bits.index <= end)]
    fig = go.Figure()
    if w.empty:
        return _style(fig, dark, "No positions in range", 200)
    vals = w.to_numpy()
    aas = np.array(w.columns)
    order = np.argsort(vals, axis=1)  # ascending -> trace k holds the k-th smallest
    for k in range(vals.shape[1]):
        idx = order[:, k]
        h = vals[np.arange(len(w)), idx]
        letters = aas[idx]
        if (h <= 1e-3).all():
            continue
        fig.add_trace(go.Bar(
            x=w.index, y=h, marker=dict(color=[CHEMISTRY_COLORS.get(a, "#ADB5BD") for a in letters], line_width=0),
            text=[a if v > 0.12 else "" for a, v in zip(letters, h)], textposition="inside",
            insidetextanchor="middle", textfont=dict(color="#ffffff", family=FONT, size=11),
            customdata=letters, hovertemplate="Position %{x}<br>%{customdata}: %{y:.2f} bits<extra></extra>",
            showlegend=False,
        ))
    for name, color in CHEMISTRY_LEGEND:
        fig.add_trace(go.Bar(x=[None], y=[None], name=name, marker_color=color))
    fig.update_layout(barmode="stack", bargap=0.08)
    fig.update_xaxes(title="Reference position", dtick=5 if end - start <= 80 else None)
    fig.update_yaxes(title="Information (bits)", range=[0, math.log2(20) + 0.1])
    return _style(fig, dark, f"Sequence logo · positions {start}–{end}", 340)


# ---------------------------------------------------------------------------
# Alignment and tree
# ---------------------------------------------------------------------------

_AA_INDEX = {aa: i + 1 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWYX")}


def msa_view(msa, mapping: List[Optional[int]], start: int, end: int, row_ids: Optional[List[str]] = None,
             max_rows: int = 200, differences_only: bool = True, dark: bool = False) -> go.Figure:
    """Chemistry-coloured MSA window between reference positions ``start``–``end``.

    With ``differences_only`` residues identical to the reference are drawn
    in a recessive tone so substitutions, gaps and insertions stand out.
    """
    cols = [i for i, p in enumerate(mapping) if p is not None and start <= p <= end]
    if not cols:
        return _style(go.Figure(), dark, "No columns in range", 200)
    c0, c1 = cols[0], cols[-1] + 1
    records = list(msa)
    ref = records[-1] if records[-1].id == "reference_sequence" else None
    others = [r for r in records if r is not ref]
    if row_ids:
        wanted = set(row_ids)
        others = [r for r in others if r.id in wanted] or others
    rows = ([ref] if ref else []) + others[:max_rows]
    letters = np.array([list(str(r.seq)[c0:c1].upper()) for r in rows])
    z = np.vectorize(lambda a: _AA_INDEX.get(a, 0))(letters)
    palette = [GAP_COLOR[dark]] + [CHEMISTRY_COLORS.get(a, "#ADB5BD") for a in "ACDEFGHIKLMNPQRSTVWYX"]
    same = np.zeros(letters.shape, dtype=bool)
    if differences_only and ref is not None:
        palette[0] = tokens(dark)["muted"]  # gaps must stand out against matching residues
        same = (letters == letters[0]) & (letters != "-")
        same[0] = False  # keep the reference row in colour
        z = np.where(same, len(palette), z)
        palette = palette + ["#e9e8e3" if not dark else "#262624"]
    n = len(palette)
    scale = []
    for i, c in enumerate(palette):
        scale += [[i / n, c], [(i + 1) / n, c]]
    refpos = [mapping[c] if mapping[c] is not None else "" for c in range(c0, c1)]
    xlab = [f"{p}" if p != "" else "·" for p in refpos]
    names = ["reference" if r is ref else r.id for r in rows]
    fig = go.Figure(go.Heatmap(
        z=z, x=list(range(c0, c1)), y=list(range(len(rows))), colorscale=scale, zmin=0, zmax=n, showscale=False,
        text=np.where(same, "·", letters),
        texttemplate="%{text}" if (c1 - c0) <= 120 and len(rows) <= 60 else None,
        textfont=dict(color="#ffffff" if not differences_only else tokens(dark)["ink2"], size=10,
                      family="ui-monospace, monospace"),
        customdata=np.array([[f"{nm}|{xlab[j]}" for j in range(c1 - c0)] for nm in names]),
        hovertemplate="%{customdata}<br>%{text}<extra></extra>", xgap=0, ygap=1,
    ))
    step = max(1, (c1 - c0) // 15)
    fig.update_xaxes(tickvals=list(range(c0, c1))[::step], ticktext=xlab[::step], title="Reference position")
    fig.update_yaxes(tickvals=list(range(len(rows))), ticktext=[nm[:24] for nm in names], autorange="reversed",
                     showgrid=False, tickfont=dict(size=9))
    fig = _style(fig, dark, f"Alignment · {len(rows)} of {len(records)} rows",
                 max(260, min(1600, 14 * len(rows) + 120)))
    fig.update_layout(margin=dict(l=12 + 6 * max(len(nm[:24]) for nm in names)))
    return fig


def tree_figure(newick_path: str, highlight: Optional[Iterable[str]] = None, highlight_label: str = "carriers",
                dark: bool = False) -> go.Figure:
    """Rectangular phylogram; ``highlight`` leaf names (MSA IDs) are coloured."""
    from Bio import Phylo

    t = tokens(dark)
    tree = Phylo.read(newick_path, "newick")
    tree.ladderize()
    depths = tree.depths(unit_branch_lengths=not any(c.branch_length for c in tree.find_clades()))
    leaves = tree.get_terminals()
    ypos = {leaf: i for i, leaf in enumerate(leaves)}

    def y_of(clade):
        if clade in ypos:
            return ypos[clade]
        ys = [y_of(c) for c in clade.clades]
        ypos[clade] = (min(ys) + max(ys)) / 2
        return ypos[clade]

    y_of(tree.root)
    ex, ey = [], []
    for clade in tree.find_clades(order="preorder"):
        x0 = depths[clade]
        for ch in clade.clades:
            ex += [x0, x0, depths[ch], None]
            ey += [ypos[clade], ypos[ch], ypos[ch], None]
    fig = go.Figure(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color=t["muted"], width=1),
                               hoverinfo="skip", showlegend=False))
    hl = set(highlight or [])
    names = [leaf.name or "" for leaf in leaves]
    is_hl = np.array([n in hl for n in names])
    groups = ([(np.flatnonzero(~is_hl), "other", t["muted"], 6),
               (np.flatnonzero(is_hl), highlight_label, t["series"][1], 9)]
              if hl else [(np.arange(len(leaves)), None, t["series"][0], 6)])
    for idx, label, color, size in groups:
        if idx.size == 0:
            continue
        fig.add_trace(go.Scattergl(
            x=[depths[leaves[i]] for i in idx], y=idx, mode="markers",
            name=f"{label} ({idx.size})" if label else None, showlegend=label is not None,
            marker=dict(size=size, color=color, line=dict(color=t["surface"], width=1)),
            text=[names[i] for i in idx], hovertemplate="%{text}<extra></extra>",
        ))
    fig.update_xaxes(title="Substitutions per site", showgrid=True, gridcolor=t["grid"])
    fig.update_yaxes(showticklabels=len(leaves) <= 60, tickvals=list(range(len(leaves))),
                     ticktext=[n[:28] for n in names], tickfont=dict(size=9), showgrid=False, autorange="reversed")
    return _style(fig, dark, f"Phylogenetic tree · {len(leaves)} sequences",
                  max(360, min(1400, 14 * len(leaves) + 120)))


# ---------------------------------------------------------------------------
# 3D structure (py3Dmol HTML)
# ---------------------------------------------------------------------------

def pdb_with_values(pdb_path: str, values_by_resnum: Dict[tuple, float], default: float = 0.0) -> str:
    """Return the PDB text with B-factors replaced by ``values_by_resnum[(resseq, icode)]``."""
    out = []
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")) and len(line) >= 66:
                key = (int(line[22:26]), line[26].strip())
                v = values_by_resnum.get(key, default)
                line = f"{line[:60]}{v:6.2f}{line[66:]}"
            out.append(line)
    return "".join(out)


def structure_html(
    chains: List[dict],
    color_mode: str = "conservation",
    style: str = "cartoon",
    label_positions: bool = True,
    dark: bool = False,
    width: int = 900,
    height: int = 560,
) -> str:
    """py3Dmol viewer HTML.

    ``chains``: list of dicts with ``pdb`` (text, B-factor = value to colour by),
    ``vmin``/``vmax``, and ``highlights`` = list of (resseq, label) to show as
    sticks with labels.
    """
    import py3Dmol

    t = tokens(dark)
    ramp = (seq_ramp(SEQ_BLUE, dark) if color_mode == "conservation"
            else [GAP_COLOR[dark]] + seq_ramp(SEQ_ORANGE[2:], dark))
    view = py3Dmol.view(width=width, height=height)
    view.setBackgroundColor(t["surface"])
    for i, ch in enumerate(chains):
        view.addModel(ch["pdb"], "pdb")
        scheme = {"prop": "b", "gradient": "linear", "min": ch["vmin"], "max": ch["vmax"], "colors": ramp}
        base = {"cartoon": {"colorscheme": scheme}} if style == "cartoon" else {style: {"colorscheme": scheme}}
        view.setStyle({"model": i}, base)
        for resi, label in ch.get("highlights", []):
            sel = {"model": i, "resi": int(resi)}
            view.addStyle(sel, {"stick": {"colorscheme": scheme, "radius": 0.3}})
            if label_positions:
                view.addLabel(label, {"fontColor": t["ink"], "backgroundColor": t["surface"],
                                      "backgroundOpacity": 0.85, "fontSize": 11, "borderThickness": 0},
                              {**sel, "atom": "CA"})
    view.zoomTo()
    return view._make_html()
