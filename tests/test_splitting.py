"""Tests for splitting: scaffold disjointness, full coverage, reproducibility."""
import numpy as np

from ppb_model.splitting import bemis_murcko_scaffold, random_split, scaffold_split

SMILES = [
    "c1ccccc1", "c1ccccc1C", "c1ccccc1CC",            # benzene scaffold family
    "c1ccncc1", "c1ccncc1C",                           # pyridine family
    "C1CCCCC1", "C1CCCCC1C",                           # cyclohexane family
    "CCO", "CCCO", "CCCCO",                            # acyclic (empty scaffold)
    "c1ccc2ccccc2c1", "c1ccc2ccccc2c1C",              # naphthalene family
]


def test_scaffold_split_covers_all_once():
    sp = scaffold_split(SMILES, 0.6, 0.2, 0.2, seed=42)
    all_idx = np.concatenate([sp["train"], sp["valid"], sp["test"]])
    assert sorted(all_idx.tolist()) == list(range(len(SMILES)))


def test_scaffold_disjoint():
    sp = scaffold_split(SMILES, 0.6, 0.2, 0.2, seed=42)
    scaffolds = {name: {bemis_murcko_scaffold(SMILES[i]) for i in idx}
                 for name, idx in sp.items()}
    assert scaffolds["train"].isdisjoint(scaffolds["valid"])
    assert scaffolds["train"].isdisjoint(scaffolds["test"])
    assert scaffolds["valid"].isdisjoint(scaffolds["test"])


def test_scaffold_split_reproducible():
    a = scaffold_split(SMILES, 0.6, 0.2, 0.2, seed=7)
    b = scaffold_split(SMILES, 0.6, 0.2, 0.2, seed=7)
    for k in a:
        assert np.array_equal(a[k], b[k])


def test_random_split_covers_all_once():
    sp = random_split(len(SMILES), 0.6, 0.2, 0.2, seed=1)
    all_idx = np.concatenate([sp["train"], sp["valid"], sp["test"]])
    assert sorted(all_idx.tolist()) == list(range(len(SMILES)))
