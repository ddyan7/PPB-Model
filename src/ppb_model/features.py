"""Molecular representations: physicochemical descriptors and fingerprints.

Representations:
    descriptors  - full RDKit descriptor set, cleaned with a **train-only** fitted
                   pipeline (impute -> drop constant -> drop highly correlated -> scale)
    morgan       - binary Morgan/ECFP fingerprint (radius, n_bits configurable)
    maccs        - 167-bit MACCS keys
    hybrid       - cleaned+scaled descriptors concatenated with the Morgan fingerprint

Fingerprints are deterministic functions of structure and need no fitting. Descriptor
cleaning is fitted on training rows only, to avoid information leakage.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys
from rdkit.Chem import rdFingerprintGenerator

# Curated, interpretable descriptor subset used for reporting/interpretation (Stage 13).
INTERPRETABLE_DESCRIPTORS = [
    "MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds",
    "RingCount", "NumAromaticRings", "FractionCSP3", "NumHeavyAtoms", "MolMR",
    "NumAliphaticRings", "NumSaturatedRings", "LabuteASA", "qed",
]


def _mols(smiles: list[str]) -> list[Chem.Mol | None]:
    return [Chem.MolFromSmiles(s) for s in smiles]


def compute_descriptor_matrix(smiles: list[str]) -> tuple[np.ndarray, list[str]]:
    """Compute the full RDKit descriptor matrix.

    Returns (X, names). Non-parseable molecules yield an all-NaN row (handled by the
    cleaner's imputer). Infinities are converted to NaN.
    """
    names = [name for name, _ in Descriptors.descList]
    rows = []
    for mol in _mols(smiles):
        if mol is None:
            rows.append([np.nan] * len(names))
            continue
        vals = []
        for _, fn in Descriptors.descList:
            try:
                vals.append(float(fn(mol)))
            except Exception:  # noqa: BLE001
                vals.append(np.nan)
        rows.append(vals)
    X = np.asarray(rows, dtype=float)
    X[~np.isfinite(X)] = np.nan
    return X, names


def morgan_matrix(smiles: list[str], radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Binary Morgan/ECFP fingerprint matrix (n_samples, n_bits)."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    out = np.zeros((len(smiles), n_bits), dtype=np.uint8)
    for i, mol in enumerate(_mols(smiles)):
        if mol is None:
            continue
        out[i] = gen.GetFingerprintAsNumPy(mol)
    return out


def maccs_matrix(smiles: list[str]) -> np.ndarray:
    """167-bit MACCS keys matrix (n_samples, 167)."""
    out = np.zeros((len(smiles), 167), dtype=np.uint8)
    for i, mol in enumerate(_mols(smiles)):
        if mol is None:
            continue
        fp = MACCSkeys.GenMACCSKeys(mol)
        out[i] = np.frombuffer(fp.ToBitString().encode(), "u1") - ord("0")
    return out


@dataclass
class DescriptorCleaner:
    """Train-only descriptor cleaning: impute -> drop constant -> decorrelate -> scale."""

    variance_threshold: float = 1e-8
    corr_threshold: float = 0.95
    keep_idx: np.ndarray | None = field(default=None, init=False)
    medians: np.ndarray | None = field(default=None, init=False)
    means: np.ndarray | None = field(default=None, init=False)
    stds: np.ndarray | None = field(default=None, init=False)
    feature_names_: list[str] | None = field(default=None, init=False)

    def fit(self, X: np.ndarray, names: list[str]) -> "DescriptorCleaner":
        medians = np.nanmedian(X, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        Xi = np.where(np.isnan(X), medians, X)

        variances = Xi.var(axis=0)
        keep = variances > self.variance_threshold

        # Correlation prune among surviving columns (greedy, keep earlier column).
        cols = np.where(keep)[0]
        if len(cols) > 1:
            corr = np.corrcoef(Xi[:, cols], rowvar=False)
            corr = np.nan_to_num(corr)
            drop_local = set()
            for a in range(len(cols)):
                if a in drop_local:
                    continue
                for b in range(a + 1, len(cols)):
                    if b in drop_local:
                        continue
                    if abs(corr[a, b]) > self.corr_threshold:
                        drop_local.add(b)
            for b in drop_local:
                keep[cols[b]] = False

        self.keep_idx = np.where(keep)[0]
        self.medians = medians
        kept = Xi[:, self.keep_idx]
        self.means = kept.mean(axis=0)
        stds = kept.std(axis=0)
        self.stds = np.where(stds > 0, stds, 1.0)
        self.feature_names_ = [names[i] for i in self.keep_idx]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.keep_idx is None:
            raise RuntimeError("DescriptorCleaner must be fitted before transform")
        Xi = np.where(np.isnan(X), self.medians, X)
        Xi[~np.isfinite(Xi)] = 0.0
        kept = Xi[:, self.keep_idx]
        return (kept - self.means) / self.stds
