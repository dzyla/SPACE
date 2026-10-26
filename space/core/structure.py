"""Find structures (RCSB / AlphaFold / upload), align chains to the reference
and map per-residue values (conservation, variant frequency) onto them."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import requests
from Bio.PDB import PDBIO, MMCIFParser, PDBParser
from Bio.PDB.Model import Model
from Bio.PDB.Structure import Structure
from Bio.PDB.DSSP import residue_max_acc
from Bio.PDB.SASA import ShrakeRupley
from Bio.SeqUtils import seq1

from . import log
from .align import align_pair
from .sequences import clean_sequence

RCSB_SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_FILES = "https://files.rcsb.org/download"
ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/api/prediction"

ResidueKey = Tuple[int, str]  # (residue number, insertion code)

_MODIFIED = {"MSE": "M", "SEC": "U", "PYL": "O", "HYP": "P", "SEP": "S", "TPO": "T", "PTR": "Y"}


@dataclass
class ChainMapping:
    chain_id: str
    sequence: str
    residues: List[ResidueKey]
    score: float
    identity: float                      # % identical over aligned pairs
    coverage: float                      # % of reference residues mapped
    alignment_text: str
    ref_to_residue: Dict[int, ResidueKey]
    bfactor: Dict[int, float] = field(default_factory=dict)   # ref pos -> original CA B-factor / pLDDT
    rsa: Dict[int, float] = field(default_factory=dict)       # ref pos -> relative solvent accessibility
    pdb_path: Optional[str] = None       # single-chain PDB with mapped values in B-factor

    def residue_numbers(self, ref_positions) -> List[int]:
        """Reference positions -> structure residue numbers (for viewers)."""
        return [self.ref_to_residue[p][0] for p in ref_positions if p in self.ref_to_residue]


@dataclass
class StructureMapping:
    source: str                          # "PDB" | "AlphaFold" | "Upload"
    structure_id: str
    path: str
    metadata: Dict[str, str]
    chains: Dict[str, ChainMapping]

    @property
    def best_chain(self) -> str:
        return max(self.chains, key=lambda c: (self.chains[c].coverage, self.chains[c].identity))


# ---------------------------------------------------------------------------
# Remote lookups
# ---------------------------------------------------------------------------

def search_pdb(sequence: str, identity_cutoff: float = 0.9, evalue_cutoff: float = 0.1,
               max_hits: int = 100, timeout: int = 30) -> List[str]:
    """RCSB sequence search; returns experimental entry IDs, best first."""
    payload = {
        "query": {"type": "terminal", "service": "sequence", "parameters": {
            "evalue_cutoff": evalue_cutoff, "identity_cutoff": float(identity_cutoff),
            "sequence_type": "protein", "value": clean_sequence(sequence),
        }},
        "request_options": {
            "paginate": {"start": 0, "rows": max_hits},
            "results_content_type": ["experimental"],
            "sort": [{"sort_by": "score", "direction": "desc"}],
        },
        "return_type": "entry",
    }
    r = requests.post(RCSB_SEARCH, json=payload, timeout=timeout)
    if r.status_code == 204:  # no hits
        return []
    r.raise_for_status()
    ids = [str(x.get("identifier", "")).upper() for x in r.json().get("result_set", [])]
    return list(dict.fromkeys(i for i in ids if i))


def pdb_metadata(pdb_id: str, timeout: int = 20) -> Dict[str, str]:
    meta = {"id": pdb_id, "title": "N/A", "authors": "N/A", "date": "N/A",
            "method": "N/A", "resolution": "N/A"}
    try:
        r = requests.get(f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}", timeout=timeout)
        r.raise_for_status()
        j = r.json()
        meta["title"] = j.get("struct", {}).get("title", "N/A")
        cit = (j.get("citation") or [{}])[0]
        meta["authors"] = ", ".join(cit.get("rcsb_authors", [])[:6]) or "N/A"
        meta["date"] = j.get("rcsb_accession_info", {}).get("initial_deposition_date", "N/A")[:10]
        meta["method"] = ", ".join(e.get("method", "") for e in j.get("exptl", [])) or "N/A"
        res = (j.get("rcsb_entry_info", {}) or {}).get("resolution_combined") or []
        meta["resolution"] = f"{res[0]:.2f} Å" if res else "N/A"
    except Exception as exc:  # noqa: BLE001
        log.info("RCSB metadata for %s unavailable: %s", pdb_id, exc)
    return meta


_PDB_ID = re.compile(r"^(?:[0-9][A-Za-z0-9]{3}|pdb_[0-9]{4}[0-9a-zA-Z]{4})$")
_UNIPROT_ACC = re.compile(r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})(?:-\d+)?$")


def download_pdb(pdb_id: str, out_dir: str | os.PathLike, timeout: int = 60) -> str:
    """Download an RCSB entry (.pdb, falling back to .cif for large entries)."""
    if not _PDB_ID.match(pdb_id or ""):
        raise ValueError(f"Invalid PDB ID {pdb_id!r} (expected e.g. 7KI6).")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for ext in ("pdb", "cif"):
        r = requests.get(f"{RCSB_FILES}/{pdb_id}.{ext}", timeout=timeout)
        if r.status_code == 200:
            path = os.path.join(out_dir, f"{pdb_id}.{ext}")
            Path(path).write_text(r.text)
            return path
    raise FileNotFoundError(f"Could not download {pdb_id} from RCSB.")


def download_alphafold(accession: str, out_dir: str | os.PathLike, timeout: int = 60) -> Tuple[str, Dict]:
    """Download the latest AlphaFold DB model for a UniProt accession."""
    acc = (accession or "").strip().split(".")[0]
    if not _UNIPROT_ACC.match(acc):
        raise ValueError(f"Invalid UniProt accession {accession!r} (expected e.g. P69905).")
    r = requests.get(f"{ALPHAFOLD_API}/{acc}", timeout=timeout)
    if r.status_code != 200:
        raise FileNotFoundError(f"No AlphaFold model for {acc} (HTTP {r.status_code}).")
    entries = [e for e in r.json() if e.get("uniprotAccession", acc) == acc] or r.json()
    entry = max(entries, key=lambda e: e.get("latestVersion", 0))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    url = entry.get("pdbUrl") or entry.get("cifUrl")
    f = requests.get(url, timeout=timeout)
    f.raise_for_status()
    path = os.path.join(out_dir, os.path.basename(url))
    Path(path).write_text(f.text)
    return path, {
        "id": f"{acc} (AlphaFold v{entry.get('latestVersion', '?')})",
        "title": entry.get("uniprotDescription", "AlphaFold model"),
        "authors": "AlphaFold DB", "date": entry.get("modelCreatedDate", "N/A"),
        "method": "Predicted", "resolution": f"mean pLDDT {entry.get('globalMetricValue', 'N/A')}",
    }


# ---------------------------------------------------------------------------
# Parsing and mapping
# ---------------------------------------------------------------------------

def load_model(path: str):
    """First model of a PDB or mmCIF file."""
    parser = MMCIFParser(QUIET=True) if path.lower().endswith((".cif", ".mmcif")) else PDBParser(QUIET=True)
    return parser.get_structure(Path(path).stem, path)[0]


def _is_polymer(res) -> bool:
    het = res.id[0]
    return (het == " " or res.get_resname() in _MODIFIED) and "CA" in res


def chain_sequence(chain) -> Tuple[str, List]:
    residues = [r for r in chain if _is_polymer(r)]
    seq = "".join(_MODIFIED.get(r.get_resname()) or seq1(r.get_resname(), undef_code="X") for r in residues)
    return clean_sequence(seq), residues


def _rsa(chain) -> Dict[Tuple, float]:
    """Relative solvent accessibility of each residue in the isolated chain."""
    try:
        ShrakeRupley().compute(chain, level="R")
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for r in chain:
        max_acc = residue_max_acc["Wilke"].get(r.get_resname())
        if max_acc and getattr(r, "sasa", None) is not None:
            out[r.id] = min(r.sasa / max_acc, 1.0)
    return out


def _polymer_copy(chain):
    """Detached copy of a chain holding only polymer residues."""
    pc = chain.copy()
    for res in [r for r in pc if not _is_polymer(r)]:
        pc.detach_child(res.id)
    return pc


def _save_chain(chain, path: str) -> None:
    """Write one chain as PDB (multi-character mmCIF chain IDs become 'A')."""
    structure, model = Structure("mapped"), Model(0)
    structure.add(model)
    if len(chain.id) != 1:
        chain.id = "A"
    model.add(chain)
    io = PDBIO()
    io.set_structure(structure)
    io.save(path)


def map_to_structure(
    path: str,
    reference: str,
    values: Optional[Mapping[int, float]] = None,
    out_dir: Optional[str | os.PathLike] = None,
    default_value: float = 0.0,
    min_coverage: float = 1.0,
    min_identity: float = 40.0,
    source: str = "Upload",
    structure_id: str = "custom",
    metadata: Optional[Dict[str, str]] = None,
) -> StructureMapping:
    """Align every protein chain to ``reference`` and map ``values`` onto it.

    ``values`` maps reference positions to numbers (e.g. al2co scores); they
    are written into the B-factor column of per-chain PDBs in ``out_dir`` so
    any viewer can colour by them. Chains covering < ``min_coverage`` % of the
    reference or aligning at < ``min_identity`` % identity are skipped.
    """
    reference = clean_sequence(reference)
    model = load_model(path)
    out_dir = Path(out_dir or Path(path).parent / "mapped")
    out_dir.mkdir(parents=True, exist_ok=True)
    chains: Dict[str, ChainMapping] = {}

    values = values or {}
    for chain in model:
        pc = _polymer_copy(chain)
        seq, residues = chain_sequence(pc)
        if len(seq) < 5:
            continue
        aln = align_pair(reference, seq)
        if aln is None:
            continue
        ref_to_obj = {}
        ident = 0
        for (a0, a1), (b0, b1) in zip(*aln.aligned):
            for i, j in zip(range(a0, a1), range(b0, b1)):
                ref_to_obj[i + 1] = residues[j]
                ident += reference[i] == seq[j]
        coverage = 100.0 * len(ref_to_obj) / len(reference)
        identity = 100.0 * ident / max(len(ref_to_obj), 1)
        # Unrelated chains (antibodies, partners) still produce a weak alignment.
        if coverage < min_coverage or identity < min_identity:
            continue
        rsa = _rsa(pc)
        cm = ChainMapping(
            chain_id=chain.id, sequence=seq,
            residues=[(r.id[1], r.id[2].strip()) for r in residues],
            score=float(aln.score),
            identity=identity,
            coverage=coverage,
            alignment_text=format(aln),
            ref_to_residue={p: (r.id[1], r.id[2].strip()) for p, r in ref_to_obj.items()},
            bfactor={p: float(r["CA"].get_bfactor()) for p, r in ref_to_obj.items()},
            rsa={p: rsa[r.id] for p, r in ref_to_obj.items() if r.id in rsa},
        )
        # Write the chain with mapped values in the B-factor column.
        res_value = {r.id: float(values[p]) for p, r in ref_to_obj.items()
                     if p in values and values[p] == values[p]}  # skip NaN
        for r in residues:
            v = res_value.get(r.id, default_value)
            for atom in r:
                atom.set_bfactor(v)
        cm.pdb_path = str(out_dir / f"{Path(path).stem}_chain_{chain.id}.pdb")
        _save_chain(pc, cm.pdb_path)
        chains[chain.id] = cm

    if not chains:
        raise ValueError("No chain in the structure aligns to the reference sequence.")
    return StructureMapping(source=source, structure_id=structure_id, path=path,
                            metadata=metadata or {"id": structure_id}, chains=chains)
