"""Regression tests for variant calling (see code review: insertions named
``insNone``, deletions never reported, wrong frequency denominators)."""

import numpy as np
import pytest

from space.core.align import reference_mapping
from space.core.variants import call_variants

REF = "MKTAYIAKQR"  # 10 residues


def calls(msa_factory, rows, **kw):
    msa = msa_factory(rows + [("reference_sequence", rows_ref(rows))])
    return call_variants(msa, reference_mapping(msa), **kw)


def rows_ref(rows):
    # Reference row aligned to the same width; tests supply it via REF_ALN.
    return REF_ALN[len(rows[0][1])]


REF_ALN = {10: REF, 12: "MKTA--YIAKQR"}


def test_substitution_named_and_hgvs(msa_factory):
    vc = calls(msa_factory, [("s1", "MKTAYGAKQR"), ("s2", REF)])
    row = vc.summary.set_index("Mutation").loc["I6G"]
    assert row["HGVS"] == "p.Ile6Gly"
    assert row["Count"] == 1 and row["Covering"] == 2
    assert row["Frequency"] == pytest.approx(0.5)
    assert row["BLOSUM62"] == -4


def test_insertion_named_between_flanking_residues(msa_factory):
    vc = calls(msa_factory, [("s1", "MKTAWWYIAKQR"), ("s2", "MKTA--YIAKQR")])
    assert list(vc.summary["Mutation"]) == ["A4_Y5insWW"]
    row = vc.summary.iloc[0]
    assert row["HGVS"] == "p.Ala4_Tyr5insTrpTrp"
    assert row["Position"] == 4 and row["Type"] == "Insertion"
    assert not vc.summary["Mutation"].str.contains("None").any()


def test_internal_deletion_reported(msa_factory):
    vc = calls(msa_factory, [("s1", "MKT--IAKQR")], deletion_threshold=50)
    assert list(vc.summary["Mutation"]) == ["A4_Y5del"]
    assert vc.summary.iloc[0]["HGVS"] == "p.Ala4_Tyr5del"


def test_fragment_only_counts_where_it_covers(msa_factory):
    # s1 is a fragment covering positions 4-8 with a substitution at 6.
    vc = calls(msa_factory, [("s1", "---AYGAK--"), ("s2", REF), ("s3", "MKTAYIAKQK")])
    assert "s1" not in vc.excluded  # terminal gaps are not deletions
    sub = vc.summary.set_index("Mutation")
    assert sub.loc["I6G", "Covering"] == 3
    assert sub.loc["R10K", "Covering"] == 2  # s1 does not reach position 10
    assert vc.coverage.tolist() == [2, 2, 2, 3, 3, 3, 3, 3, 2, 2]
    assert not (vc.summary["Type"] == "Deletion").any()


def test_ambiguous_residue_is_missing_data(msa_factory):
    vc = calls(msa_factory, [("s1", "MKTAYXAKQR"), ("s2", REF)])
    assert vc.summary.empty
    assert vc.coverage[5] == 1  # only s2 informs position 6


def test_reference_row_never_counted(msa_factory):
    vc = calls(msa_factory, [("s1", REF)])
    assert vc.n_included == 1
    assert vc.coverage.max() == 1


def test_unique_frequency_collapses_identical_sequences(msa_factory):
    rows = [(f"dup{i}", "MKTAYGAKQR") for i in range(4)] + [("wt", REF)]
    vc = calls(msa_factory, rows)
    row = vc.summary.set_index("Mutation").loc["I6G"]
    assert row["Frequency"] == pytest.approx(0.8)
    assert row["Unique carriers"] == 1 and row["Unique covering"] == 2
    assert row["Unique frequency"] == pytest.approx(0.5)
    assert vc.n_unique == 2


def test_internal_deletion_threshold_excludes(msa_factory):
    vc = calls(msa_factory, [("s1", "M-----AKQR"), ("s2", REF)], deletion_threshold=10)
    assert "s1" in vc.excluded
    assert vc.coverage.max() == 1


def test_min_coverage_excludes(msa_factory):
    vc = calls(msa_factory, [("s1", "-----IAK--"), ("s2", REF)], min_coverage=50)
    assert "s1" in vc.excluded


def test_terminal_extension_not_an_insertion(msa_factory):
    msa = msa_factory([("s1", "GGMKTAYIAKQR"), ("reference_sequence", "--MKTAYIAKQR")])
    vc = call_variants(msa, reference_mapping(msa))
    assert vc.summary.empty


def test_legacy_views(msa_factory):
    vc = calls(msa_factory, [("s1", "MKTAYGAKQR"), ("s2", REF)])
    assert vc.unique_mutations == {"s1": "I6G", "s2": "No mutations"}
    assert vc.all_mutations_str == "I6G"


def test_id_map_restores_database_ids(msa_factory):
    msa = msa_factory([("sp_P1_X", "MKTAYGAKQR"), ("reference_sequence", REF)])
    vc = call_variants(msa, reference_mapping(msa), id_map={"sp_P1_X": "sp|P1|X"})
    assert vc.summary.iloc[0]["Sequence IDs"] == "sp|P1|X"


def test_insertion_with_ambiguous_residues_is_not_called(msa_factory):
    # Unsequenced stretches (X) placed in reference gaps are missing data, not insertions.
    vc = calls(msa_factory, [("s1", "MKTAXWYIAKQR"), ("s2", "MKTAWWYIAKQR")])
    assert list(vc.summary["Mutation"]) == ["A4_Y5insWW"]
    vc = calls(msa_factory, [("s1", "MKTAXWYIAKQR")], skip_ambiguous=False)
    assert list(vc.summary["Mutation"]) == ["A4_Y5insXW"]


def test_fragment_ending_in_reference_gap_is_not_an_insertion(msa_factory):
    # Review bug: trailing residues in a reference-gap column were called as an
    # insertion without covering the right flank -> Frequency 3.0.
    msa = msa_factory([(f"f{i}", "MKTAYIAKWW---") for i in range(3)]
                      + [("full", "MKTAYIAK--QRS"), ("reference_sequence", "MKTAYIAK--QRS")])
    vc = call_variants(msa, reference_mapping(msa))
    assert vc.summary.empty or (vc.summary["Frequency"] <= 1).all()
    assert not vc.summary["Mutation"].str.contains("ins").any()


def test_deletion_of_ambiguous_reference_residue_not_called(msa_factory):
    msa = msa_factory([(f"d{i}", "MKTA-IAKQR") for i in range(5)] + [("reference_sequence", "MKTAXIAKQR")])
    vc = call_variants(msa, reference_mapping(msa))
    assert vc.summary.empty


def test_frequency_never_exceeds_one(msa_factory):
    rows = [("a", "MKTAYGAKQR"), ("b", "---AYGAKQR"), ("c", "MKT-YGAKQR"), ("d", "MKTAWWYGAKQR"[:10])]
    msa = msa_factory(rows + [("reference_sequence", REF)])
    vc = call_variants(msa, reference_mapping(msa), deletion_threshold=50)
    assert (vc.summary["Frequency"] <= 1).all() and (vc.summary["Unique frequency"] <= 1).all()


def test_duplicate_database_ids_counted_per_row(msa_factory):
    msa = msa_factory([("x", "MKTAYGAKQR"), ("x_2", "MKTAYGAKQR"), ("reference_sequence", REF)])
    vc = call_variants(msa, reference_mapping(msa), id_map={"x": "x", "x_2": "x"})
    row = vc.summary.iloc[0]
    assert row["Count"] == 2 and row["Covering"] == 2
