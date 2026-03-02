"""
Molecular fingerprint utilities for GTM chemical space mapping.

Supported fingerprint types
---------------------------
ecfp4    : Morgan / ECFP4  (radius 2, 2048 bits) — default
ecfp6    : Morgan / ECFP6  (radius 3, 2048 bits)
fcfp4    : FCFP4 (feature-based Morgan, radius 2)
fcfp6    : FCFP6 (feature-based Morgan, radius 3)
maccs    : MACCS 166-bit structural keys
rdkit    : RDKit path fingerprint (2048 bits)
torsion  : Topological torsion fingerprint
atompair : Atom-pair fingerprint
"""

from __future__ import annotations
import numpy as np
from typing import Iterable, Optional
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, DataStructs
from rdkit.Chem.rdMolDescriptors import (
    GetHashedTopologicalTorsionFingerprintAsBitVect,
    GetHashedAtomPairFingerprintAsBitVect,
)

_FP_REGISTRY = {
    "ecfp4":      {"fn": "morgan", "kwargs": {"radius": 2, "nBits": 2048, "useFeatures": False}},
    "ecfp6":      {"fn": "morgan", "kwargs": {"radius": 3, "nBits": 2048, "useFeatures": False}},
    "ecfp4_1024": {"fn": "morgan", "kwargs": {"radius": 2, "nBits": 1024, "useFeatures": False}},
    "fcfp4":      {"fn": "morgan", "kwargs": {"radius": 2, "nBits": 2048, "useFeatures": True}},
    "fcfp6":      {"fn": "morgan", "kwargs": {"radius": 3, "nBits": 2048, "useFeatures": True}},
    "maccs":      {"fn": "maccs",  "kwargs": {}},
    "rdkit":      {"fn": "rdkit_fp", "kwargs": {"fpSize": 2048}},
    "torsion":    {"fn": "torsion", "kwargs": {"nBits": 2048}},
    "atompair":   {"fn": "atompair", "kwargs": {"nBits": 2048}},
}

AVAILABLE_FP_TYPES = list(_FP_REGISTRY.keys())


def list_fingerprints():
    return AVAILABLE_FP_TYPES.copy()


def _mol_to_fp_vector(mol, fp_type, fp_params):
    if mol is None:
        return None
    fn = fp_params["fn"]
    kw = fp_params["kwargs"]
    try:
        if fn == "morgan":
            try:
                from rdkit.Chem import rdFingerprintGenerator
                if kw.get("useFeatures", False):
                    gen = rdFingerprintGenerator.GetMorganGenerator(radius=kw["radius"], fpSize=kw["nBits"])
                else:
                    gen = rdFingerprintGenerator.GetMorganGenerator(radius=kw["radius"], fpSize=kw["nBits"])
                fp = gen.GetFingerprint(mol)
            except Exception:
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, radius=kw["radius"], nBits=kw["nBits"], useFeatures=kw["useFeatures"])
        elif fn == "maccs":
            fp = MACCSkeys.GenMACCSKeys(mol)
        elif fn == "rdkit_fp":
            fp = Chem.RDKFingerprint(mol, fpSize=kw["fpSize"])
        elif fn == "torsion":
            fp = GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=kw["nBits"])
        elif fn == "atompair":
            fp = GetHashedAtomPairFingerprintAsBitVect(mol, nBits=kw["nBits"])
        else:
            raise ValueError(f"Unknown fn '{fn}'")
        arr = np.zeros(fp.GetNumBits(), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception:
        return None


def compute_fingerprints(molecules, fp_type="ecfp4", as_smiles=True, return_valid_mask=False):
    """Compute fingerprint matrix from SMILES list or RDKit Mol list."""
    fp_type = fp_type.lower()
    if fp_type not in _FP_REGISTRY:
        raise ValueError(f"Unknown fp_type '{fp_type}'. Available: {AVAILABLE_FP_TYPES}")
    fp_params = _FP_REGISTRY[fp_type]
    molecules = list(molecules)
    fps, mask = [], np.zeros(len(molecules), dtype=bool)
    for i, entry in enumerate(molecules):
        mol = Chem.MolFromSmiles(str(entry)) if as_smiles else entry
        vec = _mol_to_fp_vector(mol, fp_type, fp_params)
        if vec is not None:
            fps.append(vec)
            mask[i] = True
    n_failed = (~mask).sum()
    if n_failed:
        print(f"  Warning: {n_failed}/{len(molecules)} molecules failed and were skipped.")
    if not fps:
        raise ValueError("No valid fingerprints could be computed.")
    X = np.vstack(fps).astype(np.float32)
    return (X, mask) if return_valid_mask else X


def compute_fingerprints_from_sdf(sdf_path, fp_type="ecfp4", return_names=False):
    """Load molecules from SDF and compute fingerprints."""
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=True)
    mols, names = [], []
    for mol in suppl:
        if mol is not None:
            mols.append(mol)
            names.append(mol.GetPropsAsDict().get("_Name", ""))
    print(f"  Loaded {len(mols)} molecules from {sdf_path}")
    X, mask = compute_fingerprints(mols, fp_type=fp_type, as_smiles=False, return_valid_mask=True)
    valid_names = [n for n, v in zip(names, mask) if v]
    return (X, valid_names) if return_names else X


def variance_filter(X, min_variance=0.01):
    """Remove near-constant fingerprint bits."""
    var = X.var(axis=0)
    keep = var >= min_variance
    print(f"  Variance filter: {keep.sum()}/{X.shape[1]} bits retained (threshold={min_variance})")
    return X[:, keep].astype(np.float32), keep


def pca_reduce(X, n_components=50, scale=True):
    """
    Optional PCA reduction before GTM training.
    Returns (X_pca, pca_object).  pca_object has a _gtm_scaler attribute.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    n_comp = min(n_components, X.shape[1], X.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=42)
    X_pca = pca.fit_transform(X)
    if scale:
        scaler = StandardScaler()
        X_pca = scaler.fit_transform(X_pca)
        pca._gtm_scaler = scaler
    else:
        pca._gtm_scaler = None
    var_exp = pca.explained_variance_ratio_.cumsum()[-1] * 100
    print(f"  PCA: {X.shape[1]}D -> {n_comp}D  ({var_exp:.1f}% variance explained)")
    return X_pca.astype(np.float32), pca


def apply_pca(X_new, pca, keep_mask=None):
    """Apply a fitted PCA (from pca_reduce) to new fingerprint data."""
    if keep_mask is not None:
        X_new = X_new[:, keep_mask]
    X_pca = pca.transform(X_new)
    if pca._gtm_scaler is not None:
        X_pca = pca._gtm_scaler.transform(X_pca)
    return X_pca.astype(np.float32)
