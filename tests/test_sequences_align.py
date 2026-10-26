"""Regression tests for sequence cleaning and pairwise alignment."""

import pytest
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from space.core.align import get_aligner, pairwise_to_reference, position_map
from space.core.sequences import clean_sequence, extract_uniprot_accession, parse_sequence_text


def test_clean_sequence_keeps_numbering():
    # U/B/Z/X used to be deleted, shifting every later residue number.
    assert clean_sequence("mkU xab*1") == "MKXXAX"
    assert len(clean_sequence("MKUXAB")) == 6


def test_parse_sequence_text_fasta_and_raw():
    assert parse_sequence_text(">x desc\nMKT\nAYI\n>y\nGGG") == "MKTAYI"
    assert parse_sequence_text("mkt ayi") == "MKTAYI"


@pytest.mark.parametrize("rid,acc", [
    ("sp|P69905|HBA_HUMAN", "P69905"),
    ("tr|A0A2K5|A0A2K5_X", "A0A2K5"),
    ("P69905.2", "P69905"),
    ("XP_012345.1", None),
])
def test_extract_uniprot_accession(rid, acc):
    assert extract_uniprot_accession(rid) == acc


def test_aligner_uses_blosum62():
    al = get_aligner()
    assert al.substitution_matrix is not None
    assert al.score("W", "W") == 11  # BLOSUM62 W/W, not a flat match score
    assert al.score("MKTAYIAKQR", "AYIAK") == al.score("AYIAK", "AYIAK")  # free end gaps


def test_pairwise_identity_and_coverage():
    ref = "MKTAYIAKQRQISFVKSHFSRQ"
    recs = [SeqRecord(Seq(ref), id="same"), SeqRecord(Seq("QRQISFVKSH"), id="frag")]
    hits = pairwise_to_reference(recs, ref).set_index("id")
    assert hits.loc["same", "identity"] == pytest.approx(100)
    assert hits.loc["frag", "aligned_identity"] == pytest.approx(100)
    assert hits.loc["frag", "coverage"] == pytest.approx(100 * 10 / len(ref))


def test_selenocysteine_sequence_aligns():
    ref = "MKTAYIAKQRU"
    hits = pairwise_to_reference([SeqRecord(Seq("MKTAYIAKQRU"), id="sec")], ref)
    assert len(hits) == 1


def test_position_map():
    m = position_map("MKTAYIAKQR", "GGMKTAYIAKQR")
    assert m[1] == 3 and m[10] == 12


def test_msa_ids_never_collide(tmp_path):
    import pandas as pd

    from space.core.align import run_msa

    ref = "MKTAYIAKQRQISFVKSHFSRQ"
    mut = "MKTAYGAKQRQISFVKSHFSRQ"
    hits = pd.DataFrame({"id": ["reference_sequence", "a_b", "a|b", "a_b_2"],
                         "sequence": [mut, ref, mut, ref]})
    msa, sel, _ = run_msa(hits, ref, tmp_path)
    names = [r.id for r in msa]
    assert len(set(names)) == len(names) == 5
    by = {r.id: str(r.seq).replace("-", "") for r in msa}
    assert by[sel.loc[0, "msa_id"]] == mut and by["reference_sequence"] == ref
    assert by[sel.loc[2, "msa_id"]] == mut
