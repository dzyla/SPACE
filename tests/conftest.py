import sys
from pathlib import Path

import pytest
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def make_msa(rows):
    """rows: list of (id, aligned_string); reference row id 'reference_sequence'."""
    return MultipleSeqAlignment([SeqRecord(Seq(s), id=i, description="") for i, s in rows])


@pytest.fixture
def msa_factory():
    return make_msa


def pytest_collection_modifyitems(config, items):
    import os

    if os.environ.get("SPACE_NETWORK_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="network test; set SPACE_NETWORK_TESTS=1 to run")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)
