"""End-to-end (offline) pipeline, persistence and structure mapping."""

import pandas as pd
import pytest
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from space.core.pipeline import SpaceRun, choose_reference, list_runs, run_pipeline
from space.core.structure import map_to_structure

REF = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQ"


def _variants():
    s = list(REF)
    s[5] = "G"            # I6G
    sub = "".join(s)
    ins = REF[:20] + "WW" + REF[20:]   # insertion after residue 20
    dele = REF[:30] + REF[33:]         # deletion of 31-33
    frag = REF[10:50]                  # fragment
    return [("wt1", REF), ("wt2", REF), ("sub1", sub), ("sub2", sub), ("ins", ins),
            ("del", dele), ("frag", frag), ("secU", REF[:40] + "U" + REF[41:])]


@pytest.fixture
def fasta(tmp_path):
    path = tmp_path / "in.fasta"
    SeqIO.write([SeqRecord(Seq(s), id=i, description="") for i, s in _variants()], path, "fasta")
    return path


def test_pipeline_end_to_end(fasta, tmp_path):
    run = run_pipeline(fasta_path=str(fasta), reference=REF, run_dir=tmp_path / "run",
                       deletion_threshold=20)
    s = run.variants.summary.set_index("Mutation")
    assert s.loc["I6G", "Count"] == 2
    assert s.loc["I6G", "Covering"] == 7  # the fragment starts at residue 11
    assert run.al2co["al2co_score"].notna().all()
    assert "S20_R21insWW" in s.index
    assert "E31_Q33del" in s.index  # REF[30:33] == "EVQ"
    # The U in secU is ambiguous (X) -> no call at 41, reference numbering intact.
    assert not s.index.str.contains("41X").any()
    assert len(run.positions) == len(REF)
    assert run.tree is not None


def test_run_roundtrip(fasta, tmp_path):
    run = run_pipeline(fasta_path=str(fasta), reference=REF, run_dir=tmp_path / "rt",
                       deletion_threshold=20, build_phylo=False)
    loaded = SpaceRun.load(run.run_dir)
    pd.testing.assert_frame_equal(
        run.variants.summary.reset_index(drop=True)[["Mutation", "Count", "Covering"]],
        loaded.variants.summary.reset_index(drop=True)[["Mutation", "Count", "Covering"]],
    )
    assert loaded.summary()["distinct_variants"] == run.summary()["distinct_variants"]
    assert run.run_id in {r["run_id"] for r in list_runs(tmp_path)}


def test_choose_reference_prefers_curated_modal_sequence():
    recs = [SeqRecord(Seq(REF), id="XP_1"), SeqRecord(Seq(REF), id="XP_2"),
            SeqRecord(Seq(REF[:-1] + "W"), id="sp|P1|X"), SeqRecord(Seq(REF * 3), id="fusion")]
    seq, rid = choose_reference(recs)
    assert rid == "sp|P1|X"
    recs = recs[:2] + [recs[3]]
    seq, rid = choose_reference(recs)
    assert seq == REF


def _ca_pdb(path, seq, start=1, icode_at=None):
    from Bio.SeqUtils import seq3

    lines = []
    num = start
    for i, aa in enumerate(seq):
        icode = " "
        if icode_at is not None and i == icode_at:
            num -= 1
            icode = "A"
        x = 3.8 * i
        lines.append(
            f"ATOM  {i + 1:5d}  CA  {seq3(aa).upper():>3s} A{num:4d}{icode}   "
            f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 50.00           C"
        )
        num += 1
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def test_structure_mapping_with_insertion_code(tmp_path):
    seq = REF[:30]
    pdb = _ca_pdb(tmp_path / "x.pdb", seq, start=100, icode_at=10)
    values = {p: float(p) for p in range(1, len(REF) + 1)}
    sm = map_to_structure(pdb, REF, values=values, out_dir=tmp_path / "out")
    cm = sm.chains["A"]
    assert cm.ref_to_residue[1] == (100, "")
    assert cm.ref_to_residue[11] == (109, "A")   # insertion code preserved
    assert cm.ref_to_residue[12] == (110, "")
    assert cm.identity == pytest.approx(100)
    # mapped value written to the B-factor column
    from Bio.PDB import PDBParser
    s = PDBParser(QUIET=True).get_structure("m", cm.pdb_path)
    res = s[0]["A"][(" ", 109, "A")]
    assert res["CA"].get_bfactor() == pytest.approx(11.0)
    assert cm.bfactor[1] == pytest.approx(50.0)


def test_choose_reference_curated_beats_length_pool():
    # Many short TrEMBL fragments must not push the canonical Swiss-Prot entry out.
    frags = [SeqRecord(Seq(REF[:40]), id=f"tr|F{i}|X") for i in range(6)]
    canon = SeqRecord(Seq(REF), id="sp|P69905|HBA_HUMAN")
    _, rid = choose_reference(frags + [canon])
    assert rid == "sp|P69905|HBA_HUMAN"


def test_load_keeps_string_ids_and_survives_move(tmp_path):
    import shutil

    fasta = tmp_path / "num.fasta"
    SeqIO.write([SeqRecord(Seq(s), id=str(i), description="") for i, (_, s) in enumerate(_variants())],
                fasta, "fasta")
    run = run_pipeline(fasta_path=str(fasta), reference=REF, run_dir=tmp_path / "a", build_phylo=False,
                       deletion_threshold=20)
    moved = shutil.move(str(run.run_dir), str(tmp_path / "b"))
    loaded = SpaceRun.load(moved)
    assert loaded.hits["id"].map(type).eq(str).all()
    assert loaded.selected["id"].map(type).eq(str).all()
    assert loaded.variants.events["Sequence ID"].map(type).eq(str).all()
    assert set(loaded.hits["id"]) >= set(loaded.variants.per_sequence["Sequence ID"])


def test_annotation_helpers_on_empty_summary():
    from space.core.annotate import add_structure_context, merge_known_variants
    from space.core.variants import call_variants

    empty = pd.DataFrame(columns=["Mutation", "Position", "Ref", "Alt", "Frequency"])
    assert "Known" in merge_known_variants(empty, pd.DataFrame(columns=["Position", "Ref", "Alt", "Mutation"]))

    class _CM:
        ref_to_residue, bfactor, rsa = {}, {}, {}
    out = add_structure_context(empty, _CM())
    assert {"Structure residue", "RSA", "Exposure"} <= set(out.columns)


def test_feature_projection_clips_to_reference():
    from space.core.annotate import _project_features

    useq = "MAAAAAAAAA" + REF[:30]           # reference starts at UniProt position 11
    feats = [{"type": "Domain", "description": "D", "location": {"start": {"value": 1}, "end": {"value": 25}}}]
    df = _project_features(feats, useq, REF[:30])
    assert (df.loc[0, "Start"], df.loc[0, "End"]) == (1, 15)


def test_known_variant_with_different_wild_type_is_skipped():
    from space.core.annotate import _parse_variation

    data = {"sequence": "MKTAY", "features": [
        {"begin": "2", "end": "2", "wildType": "K", "alternativeSequence": "R", "xrefs": []},
        {"begin": "3", "end": "3", "wildType": "T", "alternativeSequence": "A", "xrefs": []}]}
    df = _parse_variation(data, "MRTAY")    # reference already carries R at 2
    assert list(df["Mutation"]) == ["T3A"]


@pytest.mark.parametrize("bad", ["../../tmp/1ABC", "1ABC/../x", "ABCD", "1AB"])
def test_pdb_id_validation(bad, tmp_path):
    from space.core.structure import download_pdb

    with pytest.raises(ValueError):
        download_pdb(bad, tmp_path)
