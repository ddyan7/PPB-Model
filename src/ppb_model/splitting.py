"""Data splitting: Bemis-Murcko scaffold split (primary) and random split (secondary).

The scaffold split assigns *whole scaffold groups* to a single partition, so no
scaffold - and therefore no near-analogue series - straddles train/val/test. This is
the leakage-resistant evaluation recommended in the modelling plan; random splitting is
kept only as a secondary, deliberately optimistic reference.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def bemis_murcko_scaffold(smiles: str, include_chirality: bool = False) -> str:
    """Return the generic Bemis-Murcko scaffold SMILES for a molecule.

    Acyclic molecules have no ring scaffold and yield an empty string; these are
    grouped together under the ``""`` key (standard convention).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    except Exception:  # noqa: BLE001
        return ""


def scaffold_split(
    smiles: list[str],
    train_frac: float = 0.70,
    valid_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Partition indices by scaffold group into train/val/test.

    Largest scaffold groups are placed first (into train), which produces stable,
    reproducible splits; ties are ordered by a seeded shuffle. Returns a dict of
    integer index arrays keyed ``"train"``, ``"valid"``, ``"test"``.
    """
    if abs(train_frac + valid_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("train/valid/test fractions must sum to 1.0")
    n = len(smiles)
    scaffolds: dict[str, list[int]] = defaultdict(list)
    for idx, smi in enumerate(smiles):
        scaffolds[bemis_murcko_scaffold(smi)].append(idx)

    rng = np.random.default_rng(seed)
    groups = list(scaffolds.values())
    # Sort by size desc; break ties with a seeded random key for reproducible shuffling.
    keys = rng.random(len(groups))
    order = sorted(range(len(groups)), key=lambda i: (-len(groups[i]), keys[i]))

    n_train_target = train_frac * n
    n_valid_target = valid_frac * n
    train_idx, valid_idx, test_idx = [], [], []
    for i in order:
        grp = groups[i]
        if len(train_idx) + len(grp) <= n_train_target or not train_idx:
            train_idx.extend(grp)
        elif len(valid_idx) + len(grp) <= n_valid_target or not valid_idx:
            valid_idx.extend(grp)
        else:
            test_idx.extend(grp)
    return {
        "train": np.array(sorted(train_idx), dtype=int),
        "valid": np.array(sorted(valid_idx), dtype=int),
        "test": np.array(sorted(test_idx), dtype=int),
    }


def random_split(
    n: int,
    train_frac: float = 0.70,
    valid_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Random index partition into train/val/test (secondary experiment)."""
    if abs(train_frac + valid_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("train/valid/test fractions must sum to 1.0")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(round(train_frac * n))
    n_valid = int(round(valid_frac * n))
    return {
        "train": np.array(sorted(perm[:n_train]), dtype=int),
        "valid": np.array(sorted(perm[n_train:n_train + n_valid]), dtype=int),
        "test": np.array(sorted(perm[n_train + n_valid:]), dtype=int),
    }


def assignment_frame(splits: dict[str, np.ndarray], record_ids: pd.Series) -> pd.DataFrame:
    """Build a tidy (record_id, split) frame from an index-based split dict."""
    rows = []
    for name, idx in splits.items():
        for i in idx:
            rows.append({"row_index": int(i), "record_id": int(record_ids.iloc[i]), "split": name})
    return pd.DataFrame(rows).sort_values("row_index").reset_index(drop=True)
