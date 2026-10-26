"""Phylogenetic trees from the MSA (NJ for small sets, UPGMA for large ones)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from Bio import Phylo
from Bio.Align import MultipleSeqAlignment
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor

from . import log
from .align import msa_array

NJ_MAX = 200        # Biopython NJ is O(n^3) in pure Python
TREE_MAX = 600      # beyond this a tree is unreadable; subsample deterministically


@dataclass
class TreeResult:
    newick_path: str
    method: str          # "neighbor-joining" | "UPGMA"
    n_leaves: int
    subsampled: bool


def identity_distances(msa: MultipleSeqAlignment) -> np.ndarray:
    """Square matrix of mismatch proportions over mutually non-gap columns."""
    arr = msa_array(msa).view(np.uint32).reshape(len(msa), -1)
    gap = arr == ord("-")
    n = arr.shape[0]
    d = np.zeros((n, n))
    for i in range(n - 1):
        valid = ~(gap[i] | gap[i + 1:])
        mism = (arr[i] != arr[i + 1:]) & valid
        denom = valid.sum(axis=1)
        row = np.where(denom > 0, mism.sum(axis=1) / np.maximum(denom, 1), 1.0)
        d[i, i + 1:] = row
        d[i + 1:, i] = row
    return d


def _linkage_to_newick(node, labels: List[str], parent_dist: float) -> str:
    if node.is_leaf():
        return f"{labels[node.id]}:{max(parent_dist - node.dist, 0) / 2:.6f}"
    left = _linkage_to_newick(node.get_left(), labels, node.dist)
    right = _linkage_to_newick(node.get_right(), labels, node.dist)
    return f"({left},{right}):{max(parent_dist - node.dist, 0) / 2:.6f}"


def build_tree(
    msa: MultipleSeqAlignment,
    out_dir: str | os.PathLike,
    max_leaves: int = TREE_MAX,
    seed: int = 0,
) -> Optional[TreeResult]:
    """Build a tree and write ``phylogenetic_tree.nwk`` to ``out_dir``."""
    n = len(msa)
    if n < 3:
        log.warning("A tree needs at least 3 sequences (got %d).", n)
        return None
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(out_dir, "phylogenetic_tree.nwk")

    subsampled = False
    if n > max_leaves:
        rng = np.random.default_rng(seed)
        keep = sorted(int(i) for i in rng.choice(n, size=max_leaves, replace=False))
        msa = MultipleSeqAlignment([msa[i] for i in keep])
        subsampled = True
        log.info("Tree: subsampled %d of %d sequences.", max_leaves, n)

    if len(msa) <= NJ_MAX:
        dm = DistanceCalculator("identity").get_distance(msa)
        tree = DistanceTreeConstructor().nj(dm)
        Phylo.write(tree, path, "newick")
        return TreeResult(path, "neighbor-joining", len(msa), subsampled)

    from scipy.cluster.hierarchy import linkage, to_tree
    from scipy.spatial.distance import squareform

    d = identity_distances(msa)
    root = to_tree(linkage(squareform(d, checks=False), method="average"))
    labels = [r.id for r in msa]
    newick = _linkage_to_newick(root, labels, root.dist) + ";"
    Path(path).write_text(newick)
    return TreeResult(path, "UPGMA", len(msa), subsampled)
