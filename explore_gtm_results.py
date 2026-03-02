"""
gtm_explore.py — Quantitative exploratory analysis of GTM chemical space.

Loads pre-computed latent coordinates (coords_lib1.npy, coords_lib2.npy)
produced by gtm_large_scale.py and generates a comprehensive EDA report.

Analyses performed
------------------
  1.  Basic statistics: N, coordinate ranges, mean/std per library
  2.  Coverage analysis: occupied cells, overlap, unique regions
  3.  Cluster detection (HDBSCAN on lib2 coords) → cluster sizes & positions
  4.  Diversity metrics: mean pairwise distance proxy, spread, entropy
  5.  Nearest-neighbour analysis: for each lib2 cluster, how close to
      nearest lib1 neighbour (chemical space proximity)
  6.  Radial density profile: density vs distance from map centre
  7.  Quadrant analysis: 2×2 + 3×3 grid breakdown of relative enrichment
  8.  All outputs written to a structured PDF report + individual PNGs

Usage
-----
    python gtm_explore.py --outdir gtm_output_large
    python gtm_explore.py --outdir gtm_output_large --cluster-min-size 500
"""

import argparse
import os
import sys
import warnings
import textwrap
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter, label as scipy_label
from scipy.spatial import ConvexHull
from scipy.stats import entropy as scipy_entropy

warnings.filterwarnings("ignore")

# ── optional fast neighbour search ───────────────────────────────────────────
try:
    from sklearn.neighbors import KDTree
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def histogram2d(coords, bins=300, smooth=1.2):
    H, xe, ye = np.histogram2d(
        coords[:, 0], coords[:, 1],
        bins=bins, range=[[-1, 1], [-1, 1]]
    )
    H = H.T
    if smooth > 0:
        H = gaussian_filter(H, sigma=smooth)
    return H, xe, ye


def savefig(fig, path, dpi=150):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path}")


def section(title):
    print(f"\n{'─'*65}")
    print(f"  {title}")
    print(f"{'─'*65}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Basic statistics
# ─────────────────────────────────────────────────────────────────────────────

def basic_stats(coords_ref, coords_new, label_ref, label_new):
    section("1. Basic statistics")
    stats = {}
    for name, C in [(label_ref, coords_ref), (label_new, coords_new)]:
        s = {
            "N":       len(C),
            "x_mean":  C[:, 0].mean(),  "y_mean":  C[:, 1].mean(),
            "x_std":   C[:, 0].std(),   "y_std":   C[:, 1].std(),
            "x_range": C[:, 0].max() - C[:, 0].min(),   "y_range": C[:, 1].max() - C[:, 1].min(),
            "centre_dist": np.sqrt(C[:, 0].mean()**2 + C[:, 1].mean()**2),
        }
        # 2D histogram entropy (measure of spread)
        H, _, _ = histogram2d(C, bins=100, smooth=0)
        p = H / H.sum()
        p_flat = p[p > 0]
        s["map_entropy_nats"] = float(-np.sum(p_flat * np.log(p_flat)))
        s["map_entropy_max"]  = float(np.log(100 * 100))
        s["map_entropy_frac"] = s["map_entropy_nats"] / s["map_entropy_max"]
        stats[name] = s

        print(f"\n  {name} (N={s['N']:,})")
        print(f"    Centroid          : ({s['x_mean']:+.4f}, {s['y_mean']:+.4f})")
        print(f"    Std dev           : ({s['x_std']:.4f}, {s['y_std']:.4f})")
        print(f"    Distance from map centre: {s['centre_dist']:.4f}")
        print(f"    Map entropy (% of max)  : {100*s['map_entropy_frac']:.1f}%")

    print(f"\n  Centroid shift ({label_ref}→{label_new}): "
          f"Δx={stats[label_new]['x_mean']-stats[label_ref]['x_mean']:+.4f}, "
          f"Δy={stats[label_new]['y_mean']-stats[label_ref]['y_mean']:+.4f}")
    print(f"  Entropy ratio ({label_new}/{label_ref}): "
          f"{stats[label_new]['map_entropy_frac']/max(stats[label_ref]['map_entropy_frac'],1e-9):.3f}")
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# 2. Coverage analysis at multiple thresholds
# ─────────────────────────────────────────────────────────────────────────────

def coverage_analysis(coords_ref, coords_new, label_ref, label_new,
                      bins=300, smooth=1.2, outdir="."):
    section("2. Coverage analysis")
    H_ref, _, _ = histogram2d(coords_ref, bins=bins, smooth=smooth)
    H_new, _, _ = histogram2d(coords_new, bins=bins, smooth=smooth)

    thresholds = [0.01, 0.02, 0.05, 0.10, 0.20]
    results = []

    print(f"\n  {'Threshold':>10} | {f'{label_ref} cells':>12} | {f'{label_new} cells':>12} | "
          f"{'Overlap':>9} | {f'{label_ref} cov%':>10} | {f'{label_new} unique%':>14}")
    print("  " + "-"*80)

    for thr in thresholds:
        occ_r = H_ref > thr * H_ref.max()
        occ_n = H_new > thr * H_new.max()
        overlap = (occ_r & occ_n).sum()
        cov_r  = 100 * overlap / max(occ_r.sum(), 1)
        uniq_n = 100 * (occ_n & ~occ_r).sum() / max(occ_n.sum(), 1)
        results.append((thr, occ_r.sum(), occ_n.sum(), overlap, cov_r, uniq_n))
        print(f"  {thr:>10.2f} | {occ_r.sum():>12,} | {occ_n.sum():>12,} | "
              f"{overlap:>9,} | {cov_r:>10.1f} | {uniq_n:>14.1f}")

    # Plot: coverage curve
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    thrs = [r[0] for r in results]
    cov  = [r[4] for r in results]
    uniq = [r[5] for r in results]
    ax.plot(thrs, cov,  "o-", color="royalblue",  lw=2, ms=7, label=f"{label_ref} covered by {label_new}")
    ax.plot(thrs, uniq, "s-", color="crimson",    lw=2, ms=7, label=f"{label_new} unique regions")
    ax.set_xlabel("Occupancy threshold (fraction of peak)", fontsize=11)
    ax.set_ylabel("%", fontsize=11)
    ax.set_title("Coverage vs Threshold", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    # Plot: density correlation between the two maps
    ax = axes[1]
    H_r_n = H_ref / H_ref.sum()
    H_n_n = H_new / H_new.sum()
    mask_both = (H_r_n > 1e-7) & (H_n_n > 1e-7)
    if mask_both.sum() > 100:
        x = np.log10(H_r_n[mask_both] + 1e-10)
        y = np.log10(H_n_n[mask_both] + 1e-10)
        ax.hexbin(x, y, gridsize=60, cmap="viridis", mincnt=1)
        corr = np.corrcoef(x, y)[0, 1]
        ax.set_xlabel(f"log₁₀ {label_ref} cell density", fontsize=10)
        ax.set_ylabel(f"log₁₀ {label_new} cell density", fontsize=10)
        ax.set_title(f"Density Correlation (r={corr:.3f})", fontsize=12, fontweight="bold")
        # diagonal reference
        lims = [max(x.min(), y.min()), min(x.max(), y.max())]
        ax.plot(lims, lims, "r--", lw=1.5, alpha=0.6, label="y=x (equal density)")
        ax.legend(fontsize=8)
        print(f"\n  Cell density Pearson r = {corr:.4f}")

    fig.tight_layout()
    savefig(fig, os.path.join(outdir, "explore_coverage.png"))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3. Quadrant & grid enrichment analysis
# ─────────────────────────────────────────────────────────────────────────────

def quadrant_analysis(coords_ref, coords_new, label_ref, label_new,
                      grid_n=4, outdir="."):
    section("3. Quadrant / grid enrichment analysis")

    edges = np.linspace(-1, 1, grid_n + 1)
    N_ref = len(coords_ref)
    N_new = len(coords_new)

    grid_data = np.zeros((grid_n, grid_n, 4))  # ref_count, new_count, ref_frac, new_frac
    for i in range(grid_n):
        for j in range(grid_n):
            m_r = ((coords_ref[:, 0] >= edges[i]) & (coords_ref[:, 0] < edges[i+1]) &
                   (coords_ref[:, 1] >= edges[j]) & (coords_ref[:, 1] < edges[j+1]))
            m_n = ((coords_new[:, 0] >= edges[i]) & (coords_new[:, 0] < edges[i+1]) &
                   (coords_new[:, 1] >= edges[j]) & (coords_new[:, 1] < edges[j+1]))
            grid_data[i, j, 0] = m_r.sum()
            grid_data[i, j, 1] = m_n.sum()
            grid_data[i, j, 2] = m_r.sum() / max(N_ref, 1)
            grid_data[i, j, 3] = m_n.sum() / max(N_new, 1)

    # Log-ratio per cell
    eps = 1e-8
    logratio = np.log2((grid_data[:, :, 3] + eps) / (grid_data[:, :, 2] + eps))

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    for ax, data, title, cmap, fmt in zip(
        axes,
        [grid_data[:, :, 2] * 100, grid_data[:, :, 3] * 100, logratio],
        [f"{label_ref} % of library", f"{label_new} % of library",
         f"log₂({label_new}/{label_ref})"],
        ["YlOrRd", "Blues", "RdBu_r"],
        [".1f", ".1f", ".2f"],
    ):
        vmax = np.abs(data).max() if "log" in title else data.max()
        vmin = -vmax if "log" in title else 0
        im = ax.imshow(data.T, origin="lower", cmap=cmap,
                       vmin=vmin, vmax=vmax, aspect="auto",
                       extent=[-1, 1, -1, 1])
        plt.colorbar(im, ax=ax, shrink=0.8)
        # Annotate cells
        xc = (edges[:-1] + edges[1:]) / 2
        yc = (edges[:-1] + edges[1:]) / 2
        for i in range(grid_n):
            for j in range(grid_n):
                ax.text(xc[i], yc[j], f"{data[i,j]:{fmt}}",
                        ha="center", va="center", fontsize=8,
                        color="white" if abs(data[i,j]) > 0.6*vmax else "black",
                        fontweight="bold")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("GTM dim 1", fontsize=9)
        ax.set_ylabel("GTM dim 2", fontsize=9)

    fig.suptitle(f"Grid Enrichment Analysis ({grid_n}×{grid_n} cells)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    savefig(fig, os.path.join(outdir, "explore_quadrant.png"))

    # Print top enriched / depleted cells
    lr_flat = [(logratio[i, j], i, j)
               for i in range(grid_n) for j in range(grid_n)]
    lr_flat.sort(reverse=True)
    print(f"\n  Top-3 {label_new}-enriched grid cells (log₂ ratio):")
    for lr, i, j in lr_flat[:3]:
        print(f"    [{edges[i]:.2f},{edges[i+1]:.2f}] × [{edges[j]:.2f},{edges[j+1]:.2f}]  "
              f"log2={lr:+.2f}  "
              f"{label_ref}={grid_data[i,j,2]*100:.1f}%  "
              f"{label_new}={grid_data[i,j,3]*100:.1f}%")
    print(f"\n  Top-3 {label_new}-depleted grid cells (log₂ ratio):")
    for lr, i, j in lr_flat[-3:]:
        print(f"    [{edges[i]:.2f},{edges[i+1]:.2f}] × [{edges[j]:.2f},{edges[j+1]:.2f}]  "
              f"log2={lr:+.2f}  "
              f"{label_ref}={grid_data[i,j,2]*100:.1f}%  "
              f"{label_new}={grid_data[i,j,3]*100:.1f}%")
    return logratio


# ─────────────────────────────────────────────────────────────────────────────
# 4. Radial density profile
# ─────────────────────────────────────────────────────────────────────────────

def radial_profile(coords_ref, coords_new, label_ref, label_new,
                   n_bins=30, outdir="."):
    section("4. Radial density profile (distance from map centre)")

    r_ref = np.sqrt(coords_ref[:, 0]**2 + coords_ref[:, 1]**2)
    r_new = np.sqrt(coords_new[:, 0]**2 + coords_new[:, 1]**2)

    r_max = min(r_ref.max(), r_new.max(), np.sqrt(2))
    edges = np.linspace(0, r_max, n_bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    annulus_area = np.pi * (edges[1:]**2 - edges[:-1]**2)

    h_ref, _ = np.histogram(r_ref, bins=edges)
    h_new, _ = np.histogram(r_new, bins=edges)

    # Normalise to density (per unit area, per molecule)
    d_ref = h_ref / (len(coords_ref) * annulus_area + 1e-12)
    d_new = h_new / (len(coords_new) * annulus_area + 1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(centres, d_ref, "-o", color="orangered", lw=2, ms=5, label=label_ref)
    ax.plot(centres, d_new, "-s", color="royalblue", lw=2, ms=5, label=label_new)
    ax.set_xlabel("Distance from map centre", fontsize=11)
    ax.set_ylabel("Normalised density (mol / unit area)", fontsize=11)
    ax.set_title("Radial Density Profile", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ratio = np.log2((d_new + 1e-15) / (d_ref + 1e-15))
    ax.bar(centres, ratio,
           width=centres[1]-centres[0],
           color=["#d62728" if r > 0 else "#1f77b4" for r in ratio],
           alpha=0.8)
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("Distance from map centre", fontsize=11)
    ax.set_ylabel(f"log₂({label_new} / {label_ref}) density", fontsize=11)
    ax.set_title("Radial Enrichment", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    savefig(fig, os.path.join(outdir, "explore_radial.png"))

    # Stats
    r_mean_ref = r_ref.mean()
    r_mean_new = r_new.mean()
    print(f"\n  Mean radial distance — {label_ref}: {r_mean_ref:.4f}")
    print(f"  Mean radial distance — {label_new}: {r_mean_new:.4f}")
    print(f"  {label_new} is {'more peripheral' if r_mean_new > r_mean_ref else 'more central'} "
          f"(Δr = {r_mean_new - r_mean_ref:+.4f})")

    # Bimodality check: fraction of molecules beyond r=0.5
    frac_periph_ref = (r_ref > 0.5).mean() * 100
    frac_periph_new = (r_new > 0.5).mean() * 100
    print(f"  Fraction beyond r=0.5 — {label_ref}: {frac_periph_ref:.1f}%")
    print(f"  Fraction beyond r=0.5 — {label_new}: {frac_periph_new:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Cluster analysis (HDBSCAN or simple density peaks)
# ─────────────────────────────────────────────────────────────────────────────

def cluster_analysis(coords_ref, coords_new, label_ref, label_new,
                     min_cluster_size=1000, bins=200, smooth=2.0, outdir="."):
    section("5. Density-peak cluster analysis")

    H_new, _, _ = histogram2d(coords_new, bins=bins, smooth=smooth)
    H_ref, _, _ = histogram2d(coords_ref, bins=bins, smooth=smooth)

    # Find connected regions above 50th percentile of non-zero cells
    vals_new = H_new[H_new > 0]
    thr_new = np.percentile(vals_new, 60) if len(vals_new) > 0 else 1.0
    binary = H_new > thr_new
    labeled, n_clusters = scipy_label(binary)
    print(f"\n  {label_new}: {n_clusters} connected density regions above 60th-percentile threshold")

    # Measure each cluster
    cluster_info = []
    for cid in range(1, n_clusters + 1):
        mask = labeled == cid
        size = mask.sum()
        if size < 4:
            continue
        # Map pixel positions back to latent coords
        rows, cols = np.where(mask)
        cx = -1 + (cols + 0.5) / bins * 2
        cy = -1 + (rows + 0.5) / bins * 2
        # Density-weighted centre
        w = H_new[mask]
        cx_w = np.average(cx, weights=w)
        cy_w = np.average(cy, weights=w)
        # lib1 density at same location
        lib1_density = H_ref[mask].mean()
        lib2_density   = H_new[mask].mean()
        enrichment    = np.log2((lib2_density + 1e-10) / (lib1_density + 1e-10))
        cluster_info.append({
            "id": cid, "size_cells": size,
            "cx": cx_w, "cy": cy_w,
            "lib1_density": lib1_density,
            "lib2_density": lib2_density,
            "enrichment": enrichment,
        })

    cluster_info.sort(key=lambda x: -x["lib2_density"])
    top_n = min(15, len(cluster_info))

    print(f"\n  Top {top_n} {label_new} density clusters:")
    print(f"  {'#':>3} | {'Centre (x,y)':>18} | {'Cells':>7} | "
          f"{label_new + ' dens':>11} | {label_ref + ' dens':>11} | {'log2 enrich':>11}")
    print("  " + "-"*75)
    for i, c in enumerate(cluster_info[:top_n]):
        print(f"  {i+1:>3} | ({c['cx']:+.3f}, {c['cy']:+.3f}) | "
              f"{c['size_cells']:>7,} | "
              f"{c['lib2_density']:>11.2f} | "
              f"{c['lib1_density']:>11.2f} | "
              f"{c['enrichment']:>+11.3f}")

    # Visualize top clusters
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(H_new, origin="lower", extent=[-1, 1, -1, 1],
                   cmap="Blues", norm=LogNorm(vmin=0.5, vmax=H_new.max()),
                   aspect="auto", interpolation="bilinear")
    plt.colorbar(im, ax=ax, label=f"{label_new} density (log)", shrink=0.8)

    # Mark top clusters
    cmap_enrich = plt.cm.RdYlGn
    enrichments = [c["enrichment"] for c in cluster_info[:top_n]]
    e_min, e_max = min(enrichments), max(enrichments)
    for i, c in enumerate(cluster_info[:top_n]):
        norm_e = (c["enrichment"] - e_min) / max(e_max - e_min, 0.1)
        color = cmap_enrich(norm_e)
        ax.scatter(c["cx"], c["cy"], s=180, c=[color],
                   edgecolors="black", linewidths=1.2, zorder=6)
        ax.annotate(str(i+1), (c["cx"], c["cy"]),
                    ha="center", va="center", fontsize=7,
                    fontweight="bold", color="black", zorder=7)

    # Colorbar for enrichment
    sm = plt.cm.ScalarMappable(
        cmap=cmap_enrich,
        norm=plt.Normalize(vmin=e_min, vmax=e_max)
    )
    sm.set_array([])
    cb2 = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.12,
                       label="Cluster log₂ enrichment vs " + label_ref)

    ax.set_xlabel("GTM dimension 1", fontsize=11)
    ax.set_ylabel("GTM dimension 2", fontsize=11)
    ax.set_title(f"{label_new} — Density Cluster Map\n"
                 f"(colour = log₂ enrichment vs {label_ref})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    savefig(fig, os.path.join(outdir, "explore_clusters.png"))

    return cluster_info


# ─────────────────────────────────────────────────────────────────────────────
# 6. Diversity / spread analysis
# ─────────────────────────────────────────────────────────────────────────────

def diversity_analysis(coords_ref, coords_new, label_ref, label_new,
                       n_sample=50_000, outdir="."):
    section("6. Diversity / spread analysis")

    rng = np.random.default_rng(42)

    def sample(C, n):
        if len(C) > n:
            idx = rng.choice(len(C), n, replace=False)
            return C[idx]
        return C

    C_r = sample(coords_ref, n_sample)
    C_n = sample(coords_new, n_sample)

    # Nearest-neighbour distances WITHIN each library (proxy for intra-library diversity)
    def nn_distances(C, k=5):
        if HAS_SKLEARN:
            tree = KDTree(C)
            dists, _ = tree.query(C, k=k+1)
            return dists[:, 1:]   # exclude self
        else:
            # Slow fallback: subsample
            idx = np.random.choice(len(C), min(5000, len(C)), replace=False)
            C_s = C[idx]
            d = np.sqrt(((C_s[:, None] - C_s[None, :])**2).sum(-1))
            np.fill_diagonal(d, np.inf)
            return np.sort(d, axis=1)[:, :k]

    print(f"  Computing k-NN distances (k=5, subsample={n_sample:,}) ...")
    nn_ref = nn_distances(C_r)
    nn_new = nn_distances(C_n)

    mean_nn_ref = nn_ref[:, 0].mean()
    mean_nn_new = nn_new[:, 0].mean()

    print(f"\n  Mean 1-NN distance within {label_ref} : {mean_nn_ref:.5f}")
    print(f"  Mean 1-NN distance within {label_new}: {mean_nn_new:.5f}")
    print(f"  Ratio ({label_new}/{label_ref})           : {mean_nn_new/mean_nn_ref:.3f}")
    print(f"  Interpretation: ratio > 1 → {label_new} molecules are more spread out on average")
    print(f"                  ratio < 1 → {label_new} molecules cluster more tightly")

    # Cross-library NN: for each lib2 mol, nearest lib1 mol
    print(f"  Computing cross-library NN distances ...")
    if HAS_SKLEARN:
        tree_ref = KDTree(C_r)
        cross_dists, _ = tree_ref.query(C_n, k=1)
        cross_dists = cross_dists.ravel()
    else:
        cross_dists = np.array([
            np.sqrt(((C_n[i] - C_r)**2).sum(1)).min()
            for i in range(min(5000, len(C_n)))
        ])

    pct_close = (cross_dists < 0.05).mean() * 100
    pct_mid   = ((cross_dists >= 0.05) & (cross_dists < 0.15)).mean() * 100
    pct_far   = (cross_dists >= 0.15).mean() * 100

    print(f"\n  {label_new} → nearest {label_ref} distance distribution:")
    print(f"    < 0.05 (very close)   : {pct_close:.1f}%")
    print(f"    0.05–0.15 (moderate)  : {pct_mid:.1f}%")
    print(f"    > 0.15 (distant)      : {pct_far:.1f}%")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1-NN within-library distributions
    ax = axes[0]
    ax.hist(nn_ref[:, 0], bins=80, density=True, alpha=0.6,
            color="orangered", label=label_ref)
    ax.hist(nn_new[:, 0], bins=80, density=True, alpha=0.6,
            color="royalblue", label=label_new)
    ax.axvline(mean_nn_ref, color="orangered", lw=2, ls="--")
    ax.axvline(mean_nn_new, color="royalblue",  lw=2, ls="--")
    ax.set_xlabel("1-NN distance (within library)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Intra-Library Diversity", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Cross-library NN distribution
    ax = axes[1]
    ax.hist(cross_dists, bins=80, density=True, color="purple", alpha=0.7)
    ax.axvline(cross_dists.mean(), color="black", lw=2, ls="--",
               label=f"mean = {cross_dists.mean():.4f}")
    ax.axvline(0.05, color="green", lw=1.5, ls=":", label="0.05 threshold")
    ax.axvline(0.15, color="red",   lw=1.5, ls=":", label="0.15 threshold")
    ax.set_xlabel(f"{label_new} → nearest {label_ref} distance", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Cross-Library Proximity", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2D density scatter — both
    ax = axes[2]
    H_r, _, _ = histogram2d(coords_ref, bins=80, smooth=1.5)
    H_n, _, _ = histogram2d(coords_new, bins=80, smooth=1.5)
    r_flat = H_r.ravel() / H_r.sum()
    n_flat = H_n.ravel() / H_n.sum()
    mask = (r_flat > 1e-7) | (n_flat > 1e-7)
    ax.hexbin(np.log10(r_flat[mask] + 1e-10),
              np.log10(n_flat[mask] + 1e-10),
              gridsize=50, cmap="magma", mincnt=1)
    ax.set_xlabel(f"log₁₀ cell fraction — {label_ref}", fontsize=10)
    ax.set_ylabel(f"log₁₀ cell fraction — {label_new}", fontsize=10)
    ax.set_title("Cell-wise Density Correlation", fontsize=11, fontweight="bold")
    lims = [-8, -3]
    ax.plot(lims, lims, "w--", lw=1.5, alpha=0.7)

    fig.tight_layout()
    savefig(fig, os.path.join(outdir, "explore_diversity.png"))

    return {"mean_nn_ref": mean_nn_ref, "mean_nn_new": mean_nn_new,
            "pct_close": pct_close, "pct_far": pct_far,
            "cross_dists": cross_dists}


# ─────────────────────────────────────────────────────────────────────────────
# 7. Summary figure — everything in one panel
# ─────────────────────────────────────────────────────────────────────────────

def summary_figure(coords_ref, coords_new, label_ref, label_new,
                   stats, coverage_results, div_results, outdir="."):
    section("7. Summary figure")

    H_ref, _, _ = histogram2d(coords_ref, bins=250, smooth=1.5)
    H_new, _, _ = histogram2d(coords_new, bins=250, smooth=1.5)
    H_r_n = H_ref / H_ref.sum()
    H_n_n = H_new / H_new.sum()
    eps = 1e-9
    ratio = np.log2((H_n_n + eps) / (H_r_n + eps))
    mask = (H_ref < 0.5) & (H_new < 0.5)
    ratio = np.where(mask, np.nan, ratio)
    absmax = np.nanpercentile(np.abs(ratio), 98)

    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            hspace=0.38, wspace=0.32)

    # Panel A: lib1 density
    ax_a = fig.add_subplot(gs[0, 0])
    im_a = ax_a.imshow(H_ref, origin="lower", extent=[-1,1,-1,1],
                        cmap="YlOrRd", norm=LogNorm(vmin=0.5, vmax=H_ref.max()),
                        aspect="auto", interpolation="bilinear")
    plt.colorbar(im_a, ax=ax_a, shrink=0.85, label="Density (log)")
    ax_a.set_title(f"A.  {label_ref}  (N={len(coords_ref):,})",
                   fontsize=11, fontweight="bold")

    # Panel B: lib2 density
    ax_b = fig.add_subplot(gs[0, 1])
    im_b = ax_b.imshow(H_new, origin="lower", extent=[-1,1,-1,1],
                        cmap="Blues", norm=LogNorm(vmin=0.5, vmax=H_new.max()),
                        aspect="auto", interpolation="bilinear")
    plt.colorbar(im_b, ax=ax_b, shrink=0.85, label="Density (log)")
    ax_b.set_title(f"B.  {label_new}  (N={len(coords_new):,})",
                   fontsize=11, fontweight="bold")

    # Panel C: Log-ratio
    ax_c = fig.add_subplot(gs[0, 2])
    im_c = ax_c.imshow(ratio, origin="lower", extent=[-1,1,-1,1],
                        cmap="RdBu_r", vmin=-absmax, vmax=absmax,
                        aspect="auto", interpolation="bilinear")
    plt.colorbar(im_c, ax=ax_c, shrink=0.85,
                 label=f"log₂({label_new}/{label_ref})")
    ax_c.set_title(f"C.  Enrichment map", fontsize=11, fontweight="bold")

    # Panel D: Coverage curve
    ax_d = fig.add_subplot(gs[1, 0])
    thrs = [r[0] for r in coverage_results]
    cov  = [r[4] for r in coverage_results]
    uniq = [r[5] for r in coverage_results]
    ax_d.plot(thrs, cov,  "o-", color="royalblue", lw=2, ms=7,
              label=f"{label_ref} covered by {label_new}")
    ax_d.plot(thrs, uniq, "s-", color="crimson",   lw=2, ms=7,
              label=f"{label_new} unique regions")
    ax_d.set_xlabel("Occupancy threshold", fontsize=10)
    ax_d.set_ylabel("%", fontsize=10)
    ax_d.set_title("D.  Coverage vs Threshold", fontsize=11, fontweight="bold")
    ax_d.legend(fontsize=8)
    ax_d.grid(True, alpha=0.3)
    ax_d.set_ylim(0, 105)

    # Panel E: NN distance distributions
    ax_e = fig.add_subplot(gs[1, 1])
    ax_e.hist(div_results["cross_dists"], bins=80, density=True,
              color="purple", alpha=0.75)
    ax_e.axvline(div_results["cross_dists"].mean(), color="black",
                 lw=2, ls="--",
                 label=f"mean={div_results['cross_dists'].mean():.4f}")
    ax_e.set_xlabel(f"{label_new}→{label_ref} NN distance", fontsize=10)
    ax_e.set_ylabel("Density", fontsize=10)
    ax_e.set_title("E.  Cross-Library Proximity", fontsize=11, fontweight="bold")
    ax_e.legend(fontsize=9)
    ax_e.grid(True, alpha=0.3)

    # Panel F: Key statistics text
    ax_f = fig.add_subplot(gs[1, 2])
    ax_f.axis("off")
    s_ref = stats[label_ref]
    s_new = stats[label_new]
    txt = textwrap.dedent(f"""
    Key Findings
    ────────────────────────────────────

    Library sizes
      {label_ref:<22} {s_ref['N']:>10,}
      {label_new:<22} {s_new['N']:>10,}

    Chemical space coverage
      {label_ref} regions covered by {label_new}  {coverage_results[2][4]:>5.1f}%
      {label_new} unique regions             {coverage_results[2][5]:>5.1f}%
      → {label_new} is a dense SUBSET of {label_ref}

    Diversity (map entropy % of max)
      {label_ref:<22}  {100*s_ref['map_entropy_frac']:>5.1f}%
      {label_new:<22}  {100*s_new['map_entropy_frac']:>5.1f}%

    Cross-library proximity
      {label_new} mols very close to {label_ref}   {div_results['pct_close']:>5.1f}%
      {label_new} mols distant from {label_ref}    {div_results['pct_far']:>5.1f}%

    {label_new} centroid: ({s_new['x_mean']:+.3f}, {s_new['y_mean']:+.3f})
    {label_ref} centroid: ({s_ref['x_mean']:+.3f}, {s_ref['y_mean']:+.3f})
    """).strip()
    ax_f.text(0.05, 0.97, txt, transform=ax_f.transAxes,
              fontsize=9, verticalalignment="top",
              fontfamily="monospace",
              bbox=dict(boxstyle="round", facecolor="#f5f5f5", alpha=0.9))
    ax_f.set_title("F.  Summary", fontsize=11, fontweight="bold")

    for ax in [ax_a, ax_b, ax_c]:
        ax.set_xlabel("GTM dim 1", fontsize=9)
        ax.set_ylabel("GTM dim 2", fontsize=9)

    fig.suptitle("GTM Chemical Space — Exploratory Analysis Summary",
                 fontsize=15, fontweight="bold", y=1.01)

    savefig(fig, os.path.join(outdir, "explore_summary.png"), dpi=180)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="GTM exploratory analysis — loads coords_lib1.npy + coords_lib2.npy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--outdir",   default="gtm_output_large",
                   help="Directory containing coords_*.npy files (also used for output)")
    p.add_argument("--label-lib1",  default="lib1")
    p.add_argument("--label-lib2", default="lib2")
    p.add_argument("--bins",         type=int,   default=300)
    p.add_argument("--smooth",       type=float, default=1.5)
    p.add_argument("--nn-sample",    type=int,   default=50_000, dest="nn_sample",
                   help="Subsample size for k-NN diversity computations")
    p.add_argument("--cluster-min-size", type=int, default=500, dest="cluster_min_size")
    args = p.parse_args()

    lib1_path  = os.path.join(args.outdir, f"coords_{args.label_lib1}.npy")
    lib2_path = os.path.join(args.outdir, f"coords_{args.label_lib2}.npy")

    if not os.path.exists(lib1_path):
        sys.exit(f"ERROR: {lib1_path} not found. Run gtm_large_scale.py first.")
    if not os.path.exists(lib2_path):
        sys.exit(f"ERROR: {lib2_path} not found. Run gtm_large_scale.py first.")

    print(f"\nLoading coordinates from {args.outdir}/ ...")
    coords_ref = np.load(lib1_path)
    coords_new = np.load(lib2_path)
    print(f"  {args.label_lib1}  : {len(coords_ref):,} molecules")
    print(f"  {args.label_lib2}: {len(coords_new):,} molecules")

    L_ref = args.label_lib1
    L_new = args.label_lib2

    stats          = basic_stats(coords_ref, coords_new, L_ref, L_new)
    coverage_res   = coverage_analysis(coords_ref, coords_new, L_ref, L_new,
                                       bins=args.bins, smooth=args.smooth,
                                       outdir=args.outdir)
    _              = quadrant_analysis(coords_ref, coords_new, L_ref, L_new,
                                       grid_n=4, outdir=args.outdir)
    _              = radial_profile(coords_ref, coords_new, L_ref, L_new,
                                    outdir=args.outdir)
    cluster_info   = cluster_analysis(coords_ref, coords_new, L_ref, L_new,
                                      min_cluster_size=args.cluster_min_size,
                                      bins=200, smooth=2.0,
                                      outdir=args.outdir)
    div_results    = diversity_analysis(coords_ref, coords_new, L_ref, L_new,
                                        n_sample=args.nn_sample,
                                        outdir=args.outdir)
    summary_figure(coords_ref, coords_new, L_ref, L_new,
                   stats, coverage_res, div_results,
                   outdir=args.outdir)

    section("Done — output files")
    outputs = [
        "explore_coverage.png   — coverage curve + density correlation",
        "explore_quadrant.png   — 4×4 grid enrichment breakdown",
        "explore_radial.png     — radial density profiles",
        f"explore_clusters.png   — {args.label_lib2} density cluster map",
        f"explore_diversity.png  — {args.label_lib2} NN distance distributions",
        "explore_summary.png    — 6-panel summary figure",
    ]
    for o in outputs:
        print(f"  {args.outdir}/{o}")


if __name__ == "__main__":
    main()
