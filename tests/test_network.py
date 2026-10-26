"""Live-service checks (skipped unless SPACE_NETWORK_TESTS=1)."""

import pytest

pytestmark = pytest.mark.network


def test_uniprot_pagination(tmp_path):
    from space.core.sequences import fetch_uniprot, read_fasta

    # 600 > one 500-record page: exercises the Link: rel="next" cursor.
    path = fetch_uniprot("gene:CYTB AND reviewed:false", tmp_path / "s.fasta", max_seqs=600)
    assert len(read_fasta(path)) == 600


def test_ncbi_fetch_and_metadata(tmp_path):
    from space.core.annotate import ncbi_source_metadata
    from space.core.sequences import fetch_ncbi, read_fasta

    path = fetch_ncbi('"Henipavirus hendraense"[Organism] AND fusion[Title]', tmp_path / "s.fasta",
                      email="dzyla@lji.org", max_seqs=20)
    recs = read_fasta(path, remove_pdb=True)
    meta = ncbi_source_metadata([r.id for r in recs], "dzyla@lji.org")
    assert len(meta) == len(recs) and meta["Year"].notna().any()


def test_known_variants_hba():
    from space.core.annotate import known_variants, uniprot_entry

    ref = uniprot_entry("P69905")["sequence"]["value"]
    kv = known_variants("P69905", ref)
    assert (kv["Mutation"] == "H113D").any()
