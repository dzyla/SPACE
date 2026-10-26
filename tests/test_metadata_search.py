"""Search-preview links and GenBank metadata fetching (offline, Entrez mocked)."""

from io import StringIO

import pandas as pd

from space import viz
from space.core import annotate
from space.core.sequences import database_query, search_url

GP = """LOCUS       {acc}                 5 aa            linear   VRL 01-JAN-2020
ACCESSION   {acc}
VERSION     {acc}.1
SOURCE      Hendra virus
  ORGANISM  Hendra virus
            Viruses.
FEATURES             Location/Qualifiers
     source          1..5
                     /organism="Hendra virus"
                     /host="Pteropus alecto"
                     /country="Australia: Queensland"
                     /collection_date="2013"
ORIGIN
        1 mktay
//
"""


def test_search_url_matches_fetched_term():
    assert database_query("x", "NCBI", refseq_only=True) == "(x) AND refseq[filter]"
    assert database_query("x", "UniProt", refseq_only=True) == "x"  # NCBI-only filter
    assert search_url('"Hendra"[Organism] AND F', "NCBI") == (
        "https://www.ncbi.nlm.nih.gov/protein/?term=%22Hendra%22%5BOrganism%5D+AND+F")
    assert search_url("gene:TP53", "UniProt", reviewed_only=True) == (
        "https://www.uniprot.org/uniprotkb?query=%28gene%3ATP53%29+AND+reviewed%3Atrue")


def _fake_efetch(calls):
    def efetch(db, id, rettype, retmode):  # noqa: A002
        accs = id.split(",")
        calls.append(accs)
        return StringIO("".join(GP.format(acc=a) for a in accs))
    return efetch


def test_metadata_progress_and_batches(monkeypatch):
    from Bio import Entrez

    calls, seen = [], []
    monkeypatch.setattr(Entrez, "efetch", _fake_efetch(calls))
    ids = [f"A{i:05d}" for i in range(25)] + ["pdb|1ABC|A", "A00000"]  # PDB skipped, duplicate collapsed
    meta = annotate.ncbi_source_metadata(ids, "t@example.org", batch=10, progress=lambda f, m: seen.append(f))
    assert len(meta) == 25 and sorted(len(c) for c in calls) == [5, 10, 10]
    assert seen[-1] == 1.0 and len(seen) == 4
    row = meta.iloc[0]
    assert row["Host"] == "Pteropus alecto" and row["Country"] == "Australia" and row["Year"] == 2013


def test_metadata_stop(monkeypatch):
    from Bio import Entrez

    calls = []
    monkeypatch.setattr(Entrez, "efetch", _fake_efetch(calls))
    meta = annotate.ncbi_source_metadata([f"A{i}" for i in range(50)], "t@example.org", batch=10,
                                         workers=1, should_stop=lambda: len(calls) >= 2)
    assert len(calls) == 2 and len(meta) == 20


def test_metadata_roundtrip(tmp_path):
    assert annotate.load_metadata(tmp_path) is None
    meta = pd.DataFrame({"Sequence ID": ["0123", "B"], "Organism": ["o", "o"], "Strain": ["", "s"],
                         "Host": ["", "h"], "Country": ["c", ""], "Collection date": ["2001", ""],
                         "Year": pd.array([2001, None], dtype="Int64")})
    annotate.save_metadata(tmp_path, meta)
    back = annotate.load_metadata(tmp_path)
    assert back["Sequence ID"].tolist() == ["0123", "B"]  # IDs stay strings
    assert str(back["Year"].dtype) == "Int64" and back["Year"].isna().tolist() == [False, True]
    assert back["Host"].tolist() == ["", "h"]


def test_dark_ramps_never_start_at_surface():
    for dark in (False, True):
        surface = viz.tokens(dark)["surface"]
        assert surface not in viz.frequency_ramp(dark)
        assert viz.seq_ramp(viz.SEQ_BLUE, True)[0] != viz.SEQ_BLUE[-1]  # darkest navy dropped in dark mode
    assert viz.chemistry_colors(True)["L"] != viz.chemistry_colors(False)["L"]
