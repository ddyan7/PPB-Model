"""RDKit-based structure standardisation.

Pipeline for each SMILES:
    1. Parse (record parse failures rather than dropping silently).
    2. RDKit ``Cleanup`` (sanitise, normalise functional groups, reionise).
    3. Keep the largest organic fragment (salt / counter-ion stripping to the parent).
    4. Neutralise charges where unambiguous (``Uncharger``).
    5. Emit canonical isomeric SMILES (stereochemistry preserved) and an InChIKey.

The InChIKey is used downstream as the structure identity key for duplicate and
conflict detection, because it is invariant to salt form and SMILES writing style.
"""
from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

# Silence RDKit's very chatty C++ logger; failures are captured explicitly below.
RDLogger.DisableLog("rdApp.*")


@dataclass(frozen=True)
class StandardisationResult:
    """Outcome of standardising a single SMILES string."""

    ok: bool
    canonical_smiles: str | None
    inchikey: str | None
    parent_multi_fragment: bool  # True if the input had >1 fragment (salt/mixture)
    error: str | None


def _largest_fragment(mol: Chem.Mol) -> tuple[Chem.Mol, bool]:
    """Return (largest-fragment mol, had_multiple_fragments)."""
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    had_multiple = len(frags) > 1
    if not had_multiple:
        return mol, False
    chooser = rdMolStandardize.LargestFragmentChooser()
    parent = chooser.choose(mol)
    return parent, True


def standardise_smiles(
    smiles: str,
    *,
    strip_salts: bool = True,
    neutralise_charges: bool = True,
    keep_stereo: bool = True,
) -> StandardisationResult:
    """Standardise one SMILES string.

    Args:
        smiles: input SMILES.
        strip_salts: keep only the largest organic fragment.
        neutralise_charges: apply RDKit Uncharger where unambiguous.
        keep_stereo: write isomeric (stereo-preserving) canonical SMILES.

    Returns:
        A :class:`StandardisationResult`. On failure ``ok`` is False and ``error``
        describes the problem; the caller decides how to record it (never silent).
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return StandardisationResult(False, None, None, False, "empty_or_non_string_smiles")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return StandardisationResult(False, None, None, False, "rdkit_parse_failed")

    try:
        mol = rdMolStandardize.Cleanup(mol)
        multi = False
        if strip_salts:
            mol, multi = _largest_fragment(mol)
            mol = rdMolStandardize.Cleanup(mol)
        else:
            multi = len(Chem.GetMolFrags(mol)) > 1
        if neutralise_charges:
            mol = rdMolStandardize.Uncharger().uncharge(mol)
        Chem.SanitizeMol(mol)
        canonical = Chem.MolToSmiles(mol, isomericSmiles=keep_stereo)
        inchikey = Chem.MolToInchiKey(mol) or None
    except Exception as exc:  # noqa: BLE001 - record any RDKit failure, do not crash the batch
        return StandardisationResult(False, None, None, False, f"standardise_error:{type(exc).__name__}")

    if not canonical:
        return StandardisationResult(False, None, None, multi, "empty_canonical_smiles")
    return StandardisationResult(True, canonical, inchikey, multi, None)
