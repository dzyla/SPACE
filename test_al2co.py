import os
import streamlit as st
import pandas as pd
import tempfile
from space.analysis import run_al2co
from Bio.Align import MultipleSeqAlignment
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import AlignIO

def test_run_al2co():
    # Setup dummy session state
    st.session_state.alignment_mapping = [20, 21, 22, 23, 24]
    
    # Create a dummy MSA file
    align1 = SeqRecord(Seq("ACDEF"), id="seq1")
    align2 = SeqRecord(Seq("ACDGF"), id="seq2")
    align3 = SeqRecord(Seq("ACD-F"), id="seq3")
    msa = MultipleSeqAlignment([align1, align2, align3])
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.aln', delete=False) as f:
        AlignIO.write(msa, f, "clustal")
        temp_aln = f.name
        
    try:
        # Run function
        df = run_al2co(temp_aln)
        
        # Check assertions
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert "Location" in df.columns
        assert "Residue" in df.columns
        assert "al2co_score" in df.columns
        assert df["Residue"].tolist() == ["A", "C", "D", "E", "F"]
        assert df["Location"].tolist() == [20, 21, 22, 23, 24]
        print("Success! al2co DataFrame generated:")
        print(df)
    finally:
        os.remove(temp_aln)

if __name__ == "__main__":
    test_run_al2co()
