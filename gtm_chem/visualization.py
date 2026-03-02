"""
GTM visualization utilities.

All plot functions return a (fig, axes) tuple so callers can further
customise or save figures.  The 'landscape' is always the background;
molecules are overlaid as scatter points.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ── colour maps ──────────────────────────────────────────────────────────────
CMAP_DENSITY  = "YlOrRd"
CMAP_ACTIVITY = "RdYlGn"
CMAP_DIVERGE  = "coolwarm"


def _scatter_coords(ax, coords, c=None, cmap=None, label=None,
                    marker="o", s=18, alpha=0.7, zorder=5,
                    edgecolors="none", norm=None, vmin=None, vmax=None):
    """Thin wrapper around ax.scatter returning the PathCollection."""
    kw = dict(s=s, alpha=alpha, zorder=zorder,
              edgecolors=edgecolors, marker=marker, label=label)
    if c is not None:
        kw.update(c=c, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
    return ax.scatter(coords[:, 0], coords[:, 1], **kw)


def _add_colorbar(ax, sc, label=""):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.08)
    cb = plt.colorbar(sc, cax=cax)
    cb.set_label(label, fontsize=9)
    return cb


def plot_landscape(
    model,
    X_train,
    coords_train=None,
    title="GTM Chemical Space",
    point_color="steelblue",
    figsize=(8, 7),
    show=True,
    save_path=None,
):
    """
    Plot the density landscape of the trained GTM with training data overlay.

    Parameters
    ----------
    model        : fitted GTM instance.
    X_train      : (N, D) training fingerprints (used to compute landscape).
    coords_train : (N, 2) precomputed latent coords; recomputed if None.
    title        : figure title.
    point_color  : colour for the scatter dots.
    """
    land = model.landscape(X_train)   # (grid_size, grid_size)
    if coords_train is None:
        coords_train, _ = model.transform(X_train)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        land,
        origin="lower",
        extent=[-1, 1, -1, 1],
        cmap=CMAP_DENSITY,
        aspect="auto",
        interpolation="bilinear",
    )
    _add_colorbar(ax, im, "Mean log-likelihood")

    ax.scatter(coords_train[:, 0], coords_train[:, 1],
               c=point_color, s=14, alpha=0.55, zorder=5,
               edgecolors="none", label="Training set")

    ax.set_xlabel("GTM dimension 1", fontsize=11)
    ax.set_ylabel("GTM dimension 2", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Figure saved → {save_path}")
    if show:
        plt.show()
    return fig, ax


def plot_activity(
    model,
    X,
    values,
    coords=None,
    title="Activity / Property Map",
    value_label="Activity",
    cmap=CMAP_ACTIVITY,
    figsize=(8, 7),
    vmin=None,
    vmax=None,
    show=True,
    save_path=None,
):
    """
    Colour the GTM landscape by a continuous molecular property.

    Parameters
    ----------
    model  : fitted GTM.
    X      : (N, D) fingerprints.
    values : (N,) array of property values (pIC50, MW, logP, …).
    coords : (N, 2) precomputed coords; recomputed if None.
    """
    act_map = model.node_activity(X, values)   # (grid_size, grid_size)
    if coords is None:
        coords, _ = model.transform(X)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        act_map,
        origin="lower",
        extent=[-1, 1, -1, 1],
        cmap=cmap,
        aspect="auto",
        interpolation="bilinear",
        vmin=vmin,
        vmax=vmax,
    )
    _add_colorbar(ax, im, value_label)

    sc = _scatter_coords(ax, coords, c=values, cmap=cmap,
                         vmin=vmin, vmax=vmax, s=18, alpha=0.8)

    ax.set_xlabel("GTM dimension 1", fontsize=11)
    ax.set_ylabel("GTM dimension 2", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def plot_overlay(
    model,
    X_ref,
    X_new,
    coords_ref=None,
    coords_new=None,
    labels_ref="Reference",
    labels_new="New ligands",
    title="GTM — Reference vs New Molecules",
    figsize=(8, 7),
    show=True,
    save_path=None,
):
    """
    Overlay new molecules on the reference GTM map.

    KEY property: the landscape is computed from X_ref only.
    New molecules are projected via the FIXED model (E-step only)
    without disturbing reference positions — demonstrating GTM invariance.

    Parameters
    ----------
    model      : fitted GTM (trained on X_ref).
    X_ref      : (N_ref, D) reference fingerprints.
    X_new      : (N_new, D) new molecule fingerprints.
    coords_ref : precomputed reference coords.
    coords_new : precomputed new molecule coords.
    """
    land = model.landscape(X_ref)
    if coords_ref is None:
        coords_ref, _ = model.transform(X_ref)
    if coords_new is None:
        coords_new, _ = model.transform(X_new)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        land,
        origin="lower",
        extent=[-1, 1, -1, 1],
        cmap="Blues",
        aspect="auto",
        interpolation="bilinear",
        alpha=0.6,
    )
    _add_colorbar(ax, im, "Reference density")

    ax.scatter(coords_ref[:, 0], coords_ref[:, 1],
               c="royalblue", s=14, alpha=0.5, zorder=4,
               edgecolors="none", label=labels_ref)

    ax.scatter(coords_new[:, 0], coords_new[:, 1],
               c="crimson", s=45, alpha=0.85, zorder=6,
               edgecolors="white", linewidths=0.5,
               marker="*", label=labels_new)

    ax.set_xlabel("GTM dimension 1", fontsize=11)
    ax.set_ylabel("GTM dimension 2", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def plot_multi_class(
    model,
    X_ref,
    coords_list,
    class_labels,
    colors=None,
    markers=None,
    title="GTM — Multiple Classes",
    figsize=(8, 7),
    show=True,
    save_path=None,
):
    """
    Overlay multiple distinct molecular sets (e.g., actives vs decoys vs ligands).

    Parameters
    ----------
    model       : fitted GTM (trained on X_ref).
    X_ref       : (N_ref, D) reference set for landscape.
    coords_list : list of (N_i, 2) coordinate arrays.
    class_labels: list of str — legend labels.
    """
    assert len(coords_list) == len(class_labels)
    land = model.landscape(X_ref)

    default_colors  = ["royalblue", "crimson", "forestgreen",
                       "darkorange", "purple", "saddlebrown"]
    default_markers = ["o", "*", "^", "s", "D", "P"]
    if colors is None:
        colors = default_colors
    if markers is None:
        markers = default_markers

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(land, origin="lower", extent=[-1, 1, -1, 1],
              cmap="Greys", aspect="auto", interpolation="bilinear",
              alpha=0.5, zorder=1)

    for i, (coords, lbl) in enumerate(zip(coords_list, class_labels)):
        col = colors[i % len(colors)]
        mrk = markers[i % len(markers)]
        sz = 50 if mrk in ("*", "D", "P") else 22
        ax.scatter(coords[:, 0], coords[:, 1],
                   c=col, s=sz, alpha=0.75, zorder=5 + i,
                   edgecolors="white", linewidths=0.4,
                   marker=mrk, label=lbl)

    ax.set_xlabel("GTM dimension 1", fontsize=11)
    ax.set_ylabel("GTM dimension 2", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def plot_uncertainty(
    model,
    X,
    coords=None,
    title="GTM — Projection Uncertainty",
    figsize=(8, 7),
    show=True,
    save_path=None,
):
    """
    Colour molecules by responsibility entropy — how diffuse the projection is.
    High entropy = molecule sits between multiple grid regions (uncertain).
    """
    entropy = model.uncertainty(X)
    if coords is None:
        coords, _ = model.transform(X)

    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(coords[:, 0], coords[:, 1],
                    c=entropy, cmap="plasma",
                    s=20, alpha=0.8, zorder=5, edgecolors="none")
    _add_colorbar(ax, sc, "Entropy (nats)")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("GTM dimension 1", fontsize=11)
    ax.set_ylabel("GTM dimension 2", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def plot_training_curve(model, figsize=(7, 4), show=True, save_path=None):
    """Plot EM log-likelihood convergence curve."""
    hist = model.log_likelihood_history_
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(np.arange(1, len(hist) + 1), hist, color="steelblue", lw=2)
    ax.set_xlabel("EM iteration", fontsize=11)
    ax.set_ylabel("Mean log-likelihood", fontsize=11)
    ax.set_title("GTM Training Convergence", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax
