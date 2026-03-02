"""
gtm_chem — Generative Topographic Mapping for Chemical Space Analysis
=====================================================================

Quick start
-----------
>>> from gtm_chem import GTM, compute_fingerprints, pca_reduce, variance_filter
>>> from gtm_chem import plot_landscape, plot_activity, plot_overlay

>>> X_fps = compute_fingerprints(smiles_list, fp_type="ecfp4")
>>> X_filt, mask = variance_filter(X_fps)
>>> X_pca, pca = pca_reduce(X_filt, n_components=50)

>>> model = GTM(grid_size=20, rbf_size=5)
>>> model.fit(X_pca)
>>> coords, R = model.transform(X_pca)
>>> plot_landscape(model, X_pca, coords)

>>> model.save("my_gtm.pkl")
>>> model2 = GTM.load("my_gtm.pkl")

>>> # Project new molecules WITHOUT retraining
>>> X_new_fps = compute_fingerprints(new_smiles, fp_type="ecfp4")
>>> X_new_pca  = apply_pca(X_new_fps, pca, keep_mask=mask)
>>> coords_new, _ = model2.transform(X_new_pca)
>>> plot_overlay(model2, X_pca, X_new_pca, coords, coords_new)
"""

from .gtm import GTM
from .fingerprints import (
    compute_fingerprints,
    compute_fingerprints_from_sdf,
    variance_filter,
    pca_reduce,
    apply_pca,
    list_fingerprints,
    AVAILABLE_FP_TYPES,
)
from .visualization import (
    plot_landscape,
    plot_activity,
    plot_overlay,
    plot_multi_class,
    plot_uncertainty,
    plot_training_curve,
)

__all__ = [
    "GTM",
    "compute_fingerprints",
    "compute_fingerprints_from_sdf",
    "variance_filter",
    "pca_reduce",
    "apply_pca",
    "list_fingerprints",
    "AVAILABLE_FP_TYPES",
    "plot_landscape",
    "plot_activity",
    "plot_overlay",
    "plot_multi_class",
    "plot_uncertainty",
    "plot_training_curve",
]

__version__ = "0.2.0"
