"""
gtm_large_scale.py — GTM chemical space comparison for large libraries.

Designed for ~1M lib1 (reference) vs ~1M lib2 (new).

Pipeline
--------
1.  Subsample lib1 → compute ECFP4 → variance filter → fit PCA
2.  Train GTM on the subsample (model is then frozen)
3.  Project ALL lib1 compounds in chunks → save coords_lib1.npy
4.  Project ALL lib2 compounds in chunks → save coords_lib2.npy
5.  Generate density comparison figures (no scatter points)

Visualization modes
-------------------
  density_side      : side-by-side log-density maps
  logratio          : log2(lib2 / lib1) difference map
  contour_overlay   : contour lines of both sets on one axes
  coverage          : fraction of lib1 chemical space covered by lib2

Usage
-----
  python gtm_large_scale.py \\
      --lib1  zinc_lib1.smi \\
      --lib2 enamine_lib2.smi \\
      --fp ecfp4 \\
      --subsample 100000 \\
      --chunk    50000 \\
      --grid     25 \\
      --rbf      6 \\
      --pca      50 \\
      --outdir   gtm_output_large

Input format
------------
  Plain SMILES file: one SMILES per line (with or without name column).
  Gzipped files (.smi.gz, .smi.gz) are handled automatically.
  SDF files: use --format sdf for both inputs.
"""

import argparse
import gzip
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter

# ── package on path ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from gtm_chem import (
    GTM,
    compute_fingerprints,
    variance_filter,
    pca_reduce,
    apply_pca,
)
from gtm_chem.fingerprints import _FP_REGISTRY, _mol_to_fp_vector, AVAILABLE_FP_TYPES

warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def _open_file(path: str):
    """Open plain or gzipped text file."""
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")


def count_lines(path: str) -> int:
    """Fast line count without loading full file."""
    n = 0
    with _open_file(path) as fh:
        for _ in fh:
            n += 1
    return n


def iter_smiles(path: str):
    """Yield SMILES strings from a .smi / .smi.gz file (first token per line)."""
    with _open_file(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            yield line.split()[0]   # first token = SMILES, ignore name column


def iter_smiles_sdf(path: str):
    """Yield SMILES strings extracted from an SDF file via RDKit."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    opener = gzip.open if str(path).endswith(".gz") else open
    suppl = Chem.ForwardSDMolSupplier(path)
    for mol in suppl:
        if mol is not None:
            try:
                yield Chem.MolToSmiles(mol)
            except Exception:
                continue


def chunk_iterator(iterator, chunk_size: int):
    """Yield successive lists of length chunk_size from any iterator."""
    buf = []
    for item in iterator:
        buf.append(item)
        if len(buf) == chunk_size:
            yield buf
            buf = []
    if buf:
        yield buf


# ─────────────────────────────────────────────────────────────────────────────
# Chunked fingerprint + projection pipeline
# ─────────────────────────────────────────────────────────────────────────────

def project_file_chunked(
    path: str,
    model: GTM,
    pca,
    keep_mask: np.ndarray,
    fp_type: str = "ecfp4",
    chunk_size: int = 50_000,
    file_format: str = "smi",
    projection: str = "mean",
    coords_out: str = None,
    desc: str = "",
) -> np.ndarray:
    """
    Stream through a large SMILES/SDF file, compute fingerprints in chunks,
    apply variance filter + PCA, project with the FIXED GTM model,
    and accumulate latent coordinates.

    Returns (N_valid, 2) coordinate array.  Also saves to coords_out if given.
    """
    from rdkit import Chem
    fp_params = _FP_REGISTRY[fp_type]

    smiles_iter = iter_smiles_sdf(path) if file_format == "sdf" else iter_smiles(path)

    all_coords = []
    total_processed = 0
    total_valid = 0
    t0 = time.time()

    for i_chunk, smiles_chunk in enumerate(chunk_iterator(smiles_iter, chunk_size)):
        fps = []
        for smi in smiles_chunk:
            mol = Chem.MolFromSmiles(smi)
            vec = _mol_to_fp_vector(mol, fp_type, fp_params)
            if vec is not None:
                fps.append(vec)

        total_processed += len(smiles_chunk)

        if not fps:
            continue

        X_chunk = np.vstack(fps).astype(np.float32)
        X_chunk_pca = apply_pca(X_chunk, pca, keep_mask=keep_mask)
        coords_chunk, _ = model.transform(X_chunk_pca, projection=projection)
        all_coords.append(coords_chunk)
        total_valid += len(coords_chunk)

        elapsed = time.time() - t0
        rate = total_processed / elapsed
        print(
            f"  {desc:20s}  chunk {i_chunk+1:4d} | "
            f"processed {total_processed:>9,} | "
            f"valid {total_valid:>9,} | "
            f"{rate:,.0f} mol/s",
            end="\r",
            flush=True,
        )

    print()  # newline after \r progress

    if not all_coords:
        raise RuntimeError(f"No valid molecules found in {path}")

    coords = np.vstack(all_coords).astype(np.float32)
    print(f"  {desc}: {total_valid:,} / {total_processed:,} molecules projected "
          f"in {time.time()-t0:.1f}s")

    if coords_out:
        np.save(coords_out, coords)
        print(f"  Coords saved → {coords_out}")

    return coords


def subsample_fingerprints(
    path: str,
    n_subsample: int,
    fp_type: str,
    file_format: str = "smi",
    seed: int = 42,
) -> np.ndarray:
    """
    Reservoir-sample n_subsample SMILES from a file and compute fingerprints.
    Uses reservoir sampling so the file is read only once.
    """
    from rdkit import Chem
    fp_params = _FP_REGISTRY[fp_type]

    rng = np.random.default_rng(seed)
    reservoir = []   # list of np.ndarray fingerprint vectors
    n_seen = 0

    smiles_iter = iter_smiles_sdf(path) if file_format == "sdf" else iter_smiles(path)

    print(f"  Reservoir sampling {n_subsample:,} from {Path(path).name} ...")
    t0 = time.time()

    for smi in smiles_iter:
        mol = Chem.MolFromSmiles(smi)
        vec = _mol_to_fp_vector(mol, fp_type, fp_params)
        if vec is None:
            continue

        n_seen += 1
        if len(reservoir) < n_subsample:
            reservoir.append(vec)
        else:
            j = int(rng.integers(0, n_seen))
            if j < n_subsample:
                reservoir[j] = vec

        if n_seen % 100_000 == 0:
            print(f"    {n_seen:>9,} processed, {len(reservoir):,} in reservoir ...",
                  end="\r", flush=True)

    print()
    print(f"  Sampled {len(reservoir):,} / {n_seen:,} molecules in {time.time()-t0:.1f}s")

    if not reservoir:
        raise RuntimeError(f"No valid molecules found in {path}")

    return np.vstack(reservoir).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Density visualizations
# ─────────────────────────────────────────────────────────────────────────────

EXTENT = [-1, 1, -1, 1]

def _histogram2d(coords: np.ndarray, bins: int = 200, smooth: float = 1.5) -> np.ndarray:
    """
    Compute 2D histogram of latent coordinates, optionally Gaussian-smoothed.
    Returns (bins, bins) float array (counts, not normalised).
    """
    H, _, _ = np.histogram2d(
        coords[:, 0], coords[:, 1],
        bins=bins,
        range=[[-1, 1], [-1, 1]],
    )
    H = H.T   # transpose so x=cols, y=rows (matches imshow convention)
    if smooth > 0:
        H = gaussian_filter(H, sigma=smooth)
    return H


def _add_colorbar(ax, im, label=""):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.08)
    cb = plt.colorbar(im, cax=cax)
    cb.set_label(label, fontsize=9)
    return cb


def plot_density_side(
    coords_ref: np.ndarray,
    coords_new: np.ndarray,
    label_ref: str = "lib1",
    label_new: str = "lib2",
    bins: int = 200,
    smooth: float = 1.5,
    figsize: tuple = (16, 7),
    save_path: str = None,
):
    """Side-by-side log-density maps for both libraries."""
    H_ref = _histogram2d(coords_ref, bins=bins, smooth=smooth)
    H_new = _histogram2d(coords_new, bins=bins, smooth=smooth)

    vmax = max(H_ref.max(), H_new.max())
    norm = LogNorm(vmin=0.5, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, H, label, cmap in zip(
        axes,
        [H_ref, H_new],
        [label_ref, label_new],
        ["YlOrRd", "Blues"],
    ):
        im = ax.imshow(
            H, origin="lower", extent=EXTENT,
            cmap=cmap, norm=norm, aspect="auto",
            interpolation="bilinear",
        )
        _add_colorbar(ax, im, "Density (log scale)")
        ax.set_title(f"{label}\n(N={len(coords_ref if label==label_ref else coords_new):,})",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("GTM dimension 1", fontsize=10)
        ax.set_ylabel("GTM dimension 2", fontsize=10)

    fig.suptitle("GTM Chemical Space — Library Comparison", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    return fig, axes


def plot_logratio(
    coords_ref: np.ndarray,
    coords_new: np.ndarray,
    label_ref: str = "lib1",
    label_new: str = "lib2",
    bins: int = 200,
    smooth: float = 1.5,
    figsize: tuple = (9, 7),
    save_path: str = None,
):
    """
    Log2-ratio map: log2(lib2 / lib1).
    Blue  = lib1 enriched (region more populated in reference)
    Red   = lib2 enriched (region more populated in new set)
    White = equal coverage
    """
    H_ref = _histogram2d(coords_ref, bins=bins, smooth=smooth)
    H_new = _histogram2d(coords_new, bins=bins, smooth=smooth)

    # Normalise to density (per-cell fraction)
    H_ref_n = H_ref / H_ref.sum().clip(1e-12)
    H_new_n = H_new / H_new.sum().clip(1e-12)

    # Log-ratio; mask empty regions
    eps = 1e-9
    ratio = np.log2((H_new_n + eps) / (H_ref_n + eps))
    # Mask cells where BOTH densities are near zero
    mask = (H_ref < 0.5) & (H_new < 0.5)
    ratio_masked = np.where(mask, np.nan, ratio)

    absmax = np.nanpercentile(np.abs(ratio_masked), 98)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        ratio_masked, origin="lower", extent=EXTENT,
        cmap="RdBu_r", vmin=-absmax, vmax=absmax,
        aspect="auto", interpolation="bilinear",
    )
    _add_colorbar(ax, im, f"log₂({label_new} / {label_ref})")

    ax.set_xlabel("GTM dimension 1", fontsize=11)
    ax.set_ylabel("GTM dimension 2", fontsize=11)
    ax.set_title(
        f"Chemical Space Enrichment\nRed = {label_new} enriched  |  Blue = {label_ref} enriched",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    return fig, ax


def plot_contour_overlay(
    coords_ref: np.ndarray,
    coords_new: np.ndarray,
    label_ref: str = "lib1",
    label_new: str = "lib2",
    bins: int = 200,
    smooth: float = 2.0,
    n_levels: int = 8,
    figsize: tuple = (9, 8),
    save_path: str = None,
):
    """
    Filled contours for lib1, line contours for lib2 overlaid.
    Shows where the two spaces coincide and where they diverge.
    """
    H_ref = _histogram2d(coords_ref, bins=bins, smooth=smooth)
    H_new = _histogram2d(coords_new, bins=bins, smooth=smooth)

    # Percentile-based contour levels (skip zeros)
    def _levels(H, n):
        vals = H[H > 0]
        if len(vals) == 0:
            return np.linspace(0, H.max(), n)
        return np.percentile(vals, np.linspace(10, 99, n))

    x = np.linspace(-1, 1, bins)
    y = np.linspace(-1, 1, bins)

    fig, ax = plt.subplots(figsize=figsize)

    # Filled contour for reference (lib1)
    cf = ax.contourf(x, y, H_ref,
                     levels=_levels(H_ref, n_levels),
                     cmap="Blues", alpha=0.55)
    _add_colorbar(ax, cf, f"{label_ref} density")

    # Line contour for new set (lib2) — no fill, distinct colour
    cl = ax.contour(x, y, H_new,
                    levels=_levels(H_new, n_levels),
                    colors=["#d62728"], linewidths=1.2, alpha=0.85)
    ax.clabel(cl, inline=False, fontsize=0)   # suppress labels, keep lines clean

    # Legend proxy artists
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor="#4878d0", alpha=0.6, label=label_ref),
        Line2D([0], [0], color="#d62728", lw=2, label=label_new),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10, framealpha=0.9)

    ax.set_xlabel("GTM dimension 1", fontsize=11)
    ax.set_ylabel("GTM dimension 2", fontsize=11)
    ax.set_title("Chemical Space Overlap", fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    return fig, ax


def plot_coverage_map(
    coords_ref: np.ndarray,
    coords_new: np.ndarray,
    label_ref: str = "lib1",
    label_new: str = "lib2",
    bins: int = 200,
    smooth: float = 1.5,
    threshold: float = 0.05,   # fraction of peak density to call a cell "occupied"
    figsize: tuple = (10, 7),
    save_path: str = None,
):
    """
    Three-colour coverage map:
      Blue   = lib1 only
      Red    = lib2 only
      Purple = both (overlap)
      White  = neither (empty chemical space)

    Also prints a coverage summary table.
    """
    H_ref = _histogram2d(coords_ref, bins=bins, smooth=smooth)
    H_new = _histogram2d(coords_new, bins=bins, smooth=smooth)

    # Boolean occupancy masks (above threshold fraction of peak)
    occ_ref = H_ref > (threshold * H_ref.max())
    occ_new = H_new > (threshold * H_new.max())

    # RGBA image
    img = np.ones((*H_ref.shape, 4), dtype=np.float32)   # white everywhere

    # Both (purple)
    both = occ_ref & occ_new
    img[both] = [0.55, 0.15, 0.65, 0.85]

    # lib1 only (blue)
    ref_only = occ_ref & ~occ_new
    img[ref_only] = [0.23, 0.47, 0.78, 0.75]

    # lib2 only (red/coral)
    new_only = ~occ_ref & occ_new
    img[new_only] = [0.84, 0.15, 0.16, 0.75]

    total_occ = occ_ref.sum() + occ_new.sum() - both.sum()
    pct_both   = 100 * both.sum()    / max(occ_ref.sum(), 1)
    pct_new_in = 100 * both.sum()    / max(occ_new.sum(), 1)
    pct_unique = 100 * new_only.sum()/ max(occ_new.sum(), 1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img, origin="lower", extent=EXTENT, aspect="auto", interpolation="nearest")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(color=[0.23, 0.47, 0.78], alpha=0.8, label=f"{label_ref} only"),
        Patch(color=[0.84, 0.15, 0.16], alpha=0.8, label=f"{label_new} only"),
        Patch(color=[0.55, 0.15, 0.65], alpha=0.8, label="Both libraries"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=10, framealpha=0.95)

    coverage_text = (
        f"Coverage summary\n"
        f"  {label_ref} regions covered by {label_new}: {pct_both:.1f}%\n"
        f"  {label_new} in shared space: {pct_new_in:.1f}%\n"
        f"  {label_new} unique regions: {pct_unique:.1f}%"
    )
    ax.text(0.98, 0.98, coverage_text,
            transform=ax.transAxes, fontsize=8.5,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    ax.set_xlabel("GTM dimension 1", fontsize=11)
    ax.set_ylabel("GTM dimension 2", fontsize=11)
    ax.set_title("Chemical Space Coverage", fontsize=13, fontweight="bold")
    fig.tight_layout()

    # Print to console too
    print(f"\n  ── Coverage summary ──────────────────────────────────")
    print(f"  {label_ref} occupied cells : {occ_ref.sum():,}")
    print(f"  {label_new} occupied cells : {occ_new.sum():,}")
    print(f"  Shared cells              : {both.sum():,}")
    print(f"  {label_ref} covered by {label_new}  : {pct_both:.1f}%")
    print(f"  {label_new} unique regions     : {pct_unique:.1f}%")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    os.makedirs(args.outdir, exist_ok=True)
    print("=" * 65)
    print("  GTM Large-Scale Chemical Space Comparison")
    print(f"  lib1     : {args.lib1}")
    print(f"  lib2     : {args.lib2}")
    print(f"  train_on : {args.train_on}")
    print(f"  FP type  : {args.fp}   |  Grid: {args.grid}×{args.grid}"
          f"  |  Subsample: {args.subsample:,}")
    print("=" * 65)

    coords_lib1_path  = os.path.join(args.outdir, f"coords_{args.label_lib1}.npy")
    coords_lib2_path = os.path.join(args.outdir, f"coords_{args.label_lib2}.npy")
    model_path         = os.path.join(args.outdir, "gtm_model.pkl")
    pca_path           = os.path.join(args.outdir, "pca_model.pkl")
    mask_path          = os.path.join(args.outdir, "var_mask.npy")

    # ── Step 1: subsample lib1 → fit preprocessing + GTM ──────────────────
    if (os.path.exists(model_path) and os.path.exists(pca_path)
            and not args.retrain):
        print("\n── Loading saved model (use --retrain to force refit) ──────────")
        import pickle
        model = GTM.load(model_path)
        with open(pca_path, "rb") as f:
            pca = pickle.load(f)
        keep_mask = np.load(mask_path)
    else:
        print(f"\n── Step 1: Subsample {args.subsample:,} ({args.train_on}) + fit GTM ──────")
        if args.train_on == "lib1":
            X_sub = subsample_fingerprints(
                args.lib1, args.subsample, args.fp, args.format
            )
        elif args.train_on == "lib2":
            X_sub = subsample_fingerprints(
                args.lib2, args.subsample, args.fp, args.format
            )
        else:  # mix — equal halves from both libraries
            half = args.subsample // 2
            X_ref_sub = subsample_fingerprints(args.lib1, half, args.fp, args.format)
            X_new_sub = subsample_fingerprints(args.lib2, half, args.fp, args.format)
            X_sub = np.vstack([X_ref_sub, X_new_sub])
        print(f"  Subsample fingerprint matrix: {X_sub.shape}")

        print("\n── Step 2: Variance filter ──────────────────────────────────────")
        X_filt, keep_mask = variance_filter(X_sub, min_variance=0.005)
        np.save(mask_path, keep_mask)

        print("\n── Step 3: PCA ───────────────────────────────────────────────────")
        n_pca = min(args.pca, X_filt.shape[1], X_filt.shape[0] - 1)
        X_pca, pca = pca_reduce(X_filt, n_components=n_pca, scale=True)

        import pickle
        with open(pca_path, "wb") as f:
            pickle.dump(pca, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  PCA saved → {pca_path}")

        print("\n── Step 4: Train GTM ─────────────────────────────────────────────")
        model = GTM(
            grid_size=args.grid,
            rbf_size=args.rbf,
            rbf_width_scale=1.2,
            regularization=0.1,
            n_iter=args.n_iter,
            tol=1e-6,
            verbose=True,
        )
        model.fit(X_pca)
        model.save(model_path)

        # Plot convergence curve
        from gtm_chem.visualization import plot_training_curve
        fig, _ = plot_training_curve(model, show=False,
            save_path=os.path.join(args.outdir, "gtm_convergence.png"))
        plt.close(fig)

    # ── Step 5: Project lib1 ───────────────────────────────────────────────
    if os.path.exists(coords_lib1_path) and not args.reproject:
        print(f"\n── Loading lib1 coords (use --reproject to redo) ──────────────")
        coords_lib1 = np.load(coords_lib1_path)
        print(f"  Loaded {len(coords_lib1):,} coords from {coords_lib1_path}")
    else:
        print(f"\n── Step 5: Project ALL lib1 compounds ──────────────────────────")
        coords_lib1 = project_file_chunked(
            args.lib1, model, pca, keep_mask,
            fp_type=args.fp,
            chunk_size=args.chunk,
            file_format=args.format,
            coords_out=coords_lib1_path,
            desc="lib1",
        )

    # ── Step 6: Project lib2 CORE ─────────────────────────────────────────
    if os.path.exists(coords_lib2_path) and not args.reproject:
        print(f"\n── Loading lib2 coords ────────────────────────────────────────")
        coords_lib2 = np.load(coords_lib2_path)
        print(f"  Loaded {len(coords_lib2):,} coords from {coords_lib2_path}")
    else:
        print(f"\n── Step 6: Project ALL lib2 compounds ─────────────────────────")
        coords_lib2 = project_file_chunked(
            args.lib2, model, pca, keep_mask,
            fp_type=args.fp,
            chunk_size=args.chunk,
            file_format=args.format,
            coords_out=coords_lib2_path,
            desc="lib2",
        )

    # ── Step 7: Visualize ───────────────────────────────────────────────────
    print(f"\n── Step 7: Generating density visualizations ────────────────────")

    label_ref = args.label_lib1
    label_new = args.label_lib2

    fig, _ = plot_density_side(
        coords_lib1, coords_lib2,
        label_ref=label_ref, label_new=label_new,
        bins=args.bins, smooth=args.smooth,
        save_path=os.path.join(args.outdir, "density_side_by_side.png"),
    )
    plt.close(fig)

    fig, _ = plot_logratio(
        coords_lib1, coords_lib2,
        label_ref=label_ref, label_new=label_new,
        bins=args.bins, smooth=args.smooth,
        save_path=os.path.join(args.outdir, "logratio_map.png"),
    )
    plt.close(fig)

    fig, _ = plot_contour_overlay(
        coords_lib1, coords_lib2,
        label_ref=label_ref, label_new=label_new,
        bins=args.bins, smooth=args.smooth + 0.5,
        save_path=os.path.join(args.outdir, "contour_overlay.png"),
    )
    plt.close(fig)

    fig, _ = plot_coverage_map(
        coords_lib1, coords_lib2,
        label_ref=label_ref, label_new=label_new,
        bins=args.bins, smooth=args.smooth,
        threshold=args.coverage_threshold,
        save_path=os.path.join(args.outdir, "coverage_map.png"),
    )
    plt.close(fig)

    print(f"\n✓ All outputs in: {args.outdir}/")
    print("  density_side_by_side.png  — log-density, two panels")
    print("  logratio_map.png          — log2 enrichment map")
    print("  contour_overlay.png       — contour comparison")
    print("  coverage_map.png          — 3-colour coverage analysis")
    print(f"  coords_{args.label_lib1}.npy          — (N,2) latent coords, reusable")
    print(f"  coords_{args.label_lib2}.npy         — (N,2) latent coords, reusable")
    print("  gtm_model.pkl             — frozen GTM for future projections")
    print("  pca_model.pkl             — PCA for future projections")
    print("  var_mask.npy              — variance filter mask")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="GTM chemical space comparison — large scale (1M+ compounds)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input
    p.add_argument("--lib1",  required=True,
                   help="Path to lib1 SMILES file (.smi or .smi.gz)")
    p.add_argument("--lib2", required=True,
                   help="Path to lib2 SMILES file (.smi or .smi.gz)")
    p.add_argument("--format", default="smi", choices=["smi", "sdf"],
                   help="Input file format for both files")
    p.add_argument("--label-lib1",  default="lib1",
                   help="Display label for lib1 library")
    p.add_argument("--label-lib2", default="lib2",
                   help="Display label for lib2 library")
    p.add_argument("--train_on", choices=["lib1", "lib2", "mix"], default="lib1",)

    # GTM / preprocessing
    p.add_argument("--fp",         default="ecfp4", choices=AVAILABLE_FP_TYPES)
    p.add_argument("--subsample",  type=int, default=100_000,
                   help="Molecules to subsample from lib1 for GTM training")
    p.add_argument("--chunk",      type=int, default=50_000,
                   help="Chunk size for projection streaming")
    p.add_argument("--pca",        type=int, default=50,
                   help="PCA components")
    p.add_argument("--grid",       type=int, default=25,
                   help="GTM latent grid size (grid × grid nodes)")
    p.add_argument("--rbf",        type=int, default=6,
                   help="GTM RBF grid size (rbf × rbf basis functions)")
    p.add_argument("--n-iter",     type=int, default=200, dest="n_iter")

    # Visualization
    p.add_argument("--bins",               type=int,   default=200,
                   help="Histogram resolution (bins × bins grid)")
    p.add_argument("--smooth",             type=float, default=1.5,
                   help="Gaussian smoothing sigma for density maps")
    p.add_argument("--coverage-threshold", type=float, default=0.05,
                   dest="coverage_threshold",
                   help="Fraction of peak density to call a cell 'occupied'")

    # Output / control
    p.add_argument("--outdir",    default="gtm_output_large")
    p.add_argument("--retrain",   action="store_true",
                   help="Force GTM retrain even if saved model exists")
    p.add_argument("--reproject", action="store_true",
                   help="Force reprojection even if saved coords exist")

    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
