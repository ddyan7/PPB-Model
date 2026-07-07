"""Tests for RDKit standardisation: salt stripping, parse failures, stereo preservation."""
from ppb_model.standardisation import standardise_smiles


def test_salt_stripped_to_parent():
    # Sodium acetate -> acetate/acetic-acid parent (single organic fragment).
    res = standardise_smiles("CC(=O)[O-].[Na+]")
    assert res.ok
    assert res.parent_multi_fragment
    assert "Na" not in res.canonical_smiles


def test_parse_failure_recorded_not_raised():
    res = standardise_smiles("this is not smiles")
    assert not res.ok
    assert res.error is not None
    assert res.canonical_smiles is None


def test_empty_input():
    res = standardise_smiles("")
    assert not res.ok


def test_stereo_preserved():
    res = standardise_smiles("C[C@H](N)C(=O)O")  # L-alanine
    assert res.ok
    assert "@" in res.canonical_smiles


def test_single_fragment_flagged_correctly():
    res = standardise_smiles("c1ccccc1")  # benzene, one fragment
    assert res.ok
    assert not res.parent_multi_fragment
    assert res.inchikey is not None
