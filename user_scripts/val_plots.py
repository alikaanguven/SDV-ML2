import os
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import neptune
import ROOT


# --- Small, focused helpers ---------------------------------------------------

TMP_DIR = os.path.expandvars("/tmp/$USER/neptune")


def _ensure_tmp_dir() -> str:
    os.makedirs(TMP_DIR, exist_ok=True)
    return TMP_DIR


def _save_fig_to_tmp(neptune_run, name: str) -> str:
    """
    Save the current matplotlib figure to /tmp/$USER and log to Neptune.
    Returns the absolute file path.

    """
    _ensure_tmp_dir()
    save_path = os.path.join(TMP_DIR, f"{name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    neptune_run[name].append(neptune.types.File(save_path))
    plt.close()
    return save_path


# Vectorized Asimov significance from ROOT (same behavior as your code)
asimov_Z = np.vectorize(
    lambda s, b, db: ROOT.RooStats.AsimovSignificance(float(s), float(b), float(db)),
    otypes=[float],
)


# --- Core utilities -----------------------------------------------------------

def integral2d(
    counts: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    x_lo: float | None = None,
    x_up: float | None = None,
    y_lo: float | None = None,
    y_up: float | None = None,
) -> float:
    """
    Inclusive sum over the rectangle [x_lo, x_up] × [y_lo, y_up].

    counts : shape (nx, ny)
    x_edges, y_edges : histogram bin edges (length nx+1, ny+1)
    """
    nx, ny = counts.shape

    if x_lo is None: x_lo = x_edges[0]
    if x_up is None: x_up = x_edges[-1]
    if y_lo is None: y_lo = y_edges[0]
    if y_up is None: y_up = y_edges[-1]

    i0 = np.searchsorted(x_edges, x_lo, side="left")
    i1 = np.searchsorted(x_edges, x_up, side="right") - 1
    j0 = np.searchsorted(y_edges, y_lo, side="left")
    j1 = np.searchsorted(y_edges, y_up, side="right") - 1

    # empty/out-of-range checks
    if i0 > i1 or j0 > j1 or i1 < 0 or j1 < 0 or i0 >= nx or j0 >= ny:
        return float("nan")

    # clip to valid index range
    i0 = max(i0, 0); i1 = min(i1, nx - 1)
    j0 = max(j0, 0); j1 = min(j1, ny - 1)

    return float(counts[i0:i1 + 1, j0:j1 + 1].sum())


def scan_Z(
    sig_counts: np.ndarray,
    bkg_counts: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    x_thresh: np.ndarray,
    y_thresh: np.ndarray,
    *,
    rel_unc: float = 0.20,
    min_bkg: float = 0.50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Scan Asimov significance over threshold grid (x ≥ tx, y ≥ ty).
    Returns (Z, S_sum, B_sum) with shapes (len(x_thresh), len(y_thresh)).
    """
    eps = 1e-3  # to guard against S=0 numerics

    x_thresh = np.asarray(x_thresh, float)
    y_thresh = np.asarray(y_thresh, float)

    Z    = np.zeros((len(x_thresh), len(y_thresh)), dtype=float)
    Ssum = np.zeros_like(Z)
    Bsum = np.zeros_like(Z)

    for i, tx in enumerate(x_thresh):
        for j, ty in enumerate(y_thresh):
            s_yx = integral2d(sig_counts, x_edges, y_edges, tx, None, ty, None)
            b_yx = integral2d(bkg_counts, x_edges, y_edges, tx, None, ty, None)

            S = float(s_yx)
            B = float(b_yx)

            B_eff = max(B, min_bkg)
            S_eff = max(S, eps)

            Z[i, j]  = asimov_Z(S_eff, B_eff, B_eff * rel_unc)
            Ssum[i, j] = S
            Bsum[i, j] = B

    return Z, Ssum, Bsum


# --- Plotting functions (names kept) -----------------------------------------

def plot_hist1(
    p: np.ndarray,
    match: np.ndarray,
    savename: str,
    run,                  # Neptune run
    binwidth: float = 0.02,
):
    """
    1D histogram of scores for matched vs non-matched vertices.

    p      : scores/probabilities in [0, 1]
    match  : boolean mask (True = matched)
    """
    bins = np.arange(-0.01, 1.01, binwidth)

    vals_matched, edges_matched = np.histogram(p[match], bins=bins)
    centers_matched = 0.5 * (edges_matched[:-1] + edges_matched[1:])

    vals_non, edges_non = np.histogram(p[~match], bins=bins)
    centers_non = 0.5 * (edges_non[:-1] + edges_non[1:])

    plt.figure(figsize=(10, 8))
    plt.errorbar(
        centers_matched, vals_matched, np.sqrt(np.clip(vals_matched, 0, None)),
        fmt="o", capthick=1, capsize=3, markersize=3, color="indigo", label="matched"
    )
    plt.errorbar(
        centers_non, vals_non, np.sqrt(np.clip(vals_non, 0, None)),
        fmt="o", capthick=1, capsize=3, markersize=3, color="red", label="non-matched"
    )

    y_min = 1
    y_max = 2 * (vals_matched.sum() + vals_non.sum())

    plt.ylim(y_min, y_max)
    plt.yscale("log")
    plt.title("Validation Dataset")
    plt.xlabel(r"$S_{1}^{\mathrm{NN}}$", loc="right")
    plt.ylabel("vtx Counts",           loc="top")
    plt.legend()

    _save_fig_to_tmp(run, savename)

# --- ABCD non-closure (background) -------------------------------------------

def scan_nonclosure(
    bkg_counts: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    x_thresh: np.ndarray,
    y_thresh: np.ndarray,
    *,
    numeric_eps: float = 0.0,
):
    """
    Compute non-closure grid for background:
        nonclosure = | 1 - (NB * NC) / (NA * ND) |

    Regions (thresholds tx, ty):
      A: x>=tx, y>=ty
      B: x< tx, y>=ty
      C: x>=tx, y< ty
      D: x< tx, y< ty

    Returns:
      noncl : (len(x_thresh), len(y_thresh)) array
      NA, NB, NC, ND : same-shape arrays of region integrals
    """
    x_thresh = np.asarray(x_thresh, float)
    y_thresh = np.asarray(y_thresh, float)

    noncl = np.full((len(x_thresh), len(y_thresh)), np.nan, dtype=float)
    NA    = np.zeros_like(noncl)
    NB    = np.zeros_like(noncl)
    NC    = np.zeros_like(noncl)
    ND    = np.zeros_like(noncl)

    # get an eps smaller than threshold bin width
    # so that we don't count a bin twice
    x_thr_w = x_thresh[1] - x_thresh[0]
    y_thr_w = y_thresh[1] - y_thresh[0]
    eps = min(x_thr_w, y_thr_w) / 1e2

    for i, tx in enumerate(x_thresh):
        for j, ty in enumerate(y_thresh):
            a = integral2d(bkg_counts, x_edges, y_edges, tx, None, ty, None)
            b = integral2d(bkg_counts, x_edges, y_edges, None, tx-eps,  ty, None)
            c = integral2d(bkg_counts, x_edges, y_edges, tx, None, None, ty-eps)
            d = integral2d(bkg_counts, x_edges, y_edges, None, tx-eps,  None, ty-eps)

            NA[i, j] = a
            NB[i, j] = b
            NC[i, j] = c
            ND[i, j] = d

            denom = a * d
            numer = b * c

            if numeric_eps > 0.0:
                denom = max(denom, numeric_eps)

            if denom > 0.0:
                noncl[i, j] = abs(1.0 - (numer / denom))
            else:
                noncl[i, j] = np.nan

    return noncl, NA, NB, NC, ND


def plot_bkg_nonclosure(
    bkg_counts: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    x_thresh: np.ndarray, # Threshold grids
    y_thresh: np.ndarray, # Threshold grids
    savename: str,
    run,                  # Neptune run
    numeric_eps: float = 0.0,
):
    """
    Build bkg 2D histogram, scan thresholds, and plot the non-closure heatmap:
        |1 - (NB*NC)/(NA*ND)|
    """

    noncl, NA, NB, NC, ND = scan_nonclosure(
        bkg_counts, x_edges, y_edges, x_thresh, y_thresh, numeric_eps=numeric_eps
    )

    # Plot heatmap
    plt.figure()
    pcm = plt.pcolormesh(x_thresh, y_thresh, noncl.T, shading="auto")
    cbar = plt.colorbar(pcm)
    plt.clim(0-1e-2, 1+1e-2)
    cbar.set_label(r"ABCD non-closure  $|1 - (N_B N_C)/(N_A N_D)|$")

    # Optional contours at common tolerances
    finite_vals = noncl[np.isfinite(noncl)]
    if finite_vals.size:
        levels = [0.02, 0.05, 0.10, 0.20]
        levels = [lv for lv in levels if lv <= np.nanmax(finite_vals)]
        if levels:
            cont = plt.contour(x_thresh, y_thresh, noncl.T, levels=levels, colors="white", linewidths=1.0)
            plt.clabel(cont, fontsize="smaller")

        # Mark best-closure (minimum non-closure)
        ij_min = np.nanargmin(noncl)
        i_min, j_min = np.unravel_index(ij_min, noncl.shape)
        plt.scatter([x_thresh[i_min]], [y_thresh[j_min]],
                    s=60, marker="*", edgecolor="w", facecolor="none", label="min non-closure")
        plt.legend(loc='upper left')

    plt.xlabel(r"$S_{1}^{\mathrm{NN}}$")
    plt.ylabel(r"$S_{2}^{\mathrm{NN}}$")
    plt.tight_layout()
    _save_fig_to_tmp(run, f"{savename}_nonclosure")
    # END



def plot_hist2(
    p1: np.ndarray,
    p2: np.ndarray,
    match: np.ndarray,
    savename: str,
    run,                  # Neptune run
    binwidth: float = 0.02,
    cmin: float = 1e-3,
):
    """
    2D density plots for (p1, p2) for matched and non-matched,
    plus an Asimov-Z scan over (x_thresh, y_thresh).
    """
    _ensure_tmp_dir()

    # Bin edges that include 1.0 exactly
    x_bins = np.arange(0.0, 1.0 + binwidth, binwidth)
    y_bins = np.arange(0.0, 1.0 + binwidth, binwidth)

    # --- Signal (matched) -----------------------------------------------------
    plt.figure()
    sig_hist = plt.hist2d(
        p1[match], p2[match],
        bins=[x_bins, y_bins],
        density=True,
        cmin=cmin,              # hide very sparse bins
        norm=LogNorm(),         # temporary; refine vmin/vmax below
    )
    sig_counts, x_edges, y_edges, sig_ax = sig_hist
    sig_counts = np.nan_to_num(sig_counts, nan=0.0, posinf=0.0, neginf=0.0)

    positive = sig_counts[sig_counts > 0]
    if positive.size > 0:
        vmin = max(cmin, positive.min())
        vmax = positive.max()
        plt.colorbar(sig_ax, label="sig vtx density")
        sig_ax.set_norm(LogNorm(vmin=vmin, vmax=vmax))
    else:
        plt.colorbar(sig_ax, label="sig vtx density")

    plt.xlabel(r"$S_{1}^{\mathrm{NN}}$")
    plt.ylabel(r"$S_{2}^{\mathrm{NN}}$")
    plt.tight_layout()
    _save_fig_to_tmp(run, f"{savename}_sig")

    # --- Background (non-matched) --------------------------------------------
    plt.figure()
    bkg_hist = plt.hist2d(
        p1[~match], p2[~match],
        bins=[x_bins, y_bins],
        density=True,
        cmin=cmin,
        norm=LogNorm(),         # temporary; refine vmin/vmax below
    )
    bkg_counts, _, _, bkg_ax = bkg_hist
    bkg_counts = np.nan_to_num(bkg_counts, nan=0.0, posinf=0.0, neginf=0.0)

    positive = bkg_counts[bkg_counts > 0]
    if positive.size > 0:
        vmin = max(cmin, positive.min())
        vmax = positive.max()
        plt.colorbar(bkg_ax, label="bkg vtx density")
        bkg_ax.set_norm(LogNorm(vmin=vmin, vmax=vmax))
    else:
        plt.colorbar(bkg_ax, label="bkg vtx density")

    plt.xlabel(r"$S_{1}^{\mathrm{NN}}$")
    plt.ylabel(r"$S_{2}^{\mathrm{NN}}$")
    plt.tight_layout()
    _save_fig_to_tmp(run, f"{savename}_bkg")

    # --- Z scan over thresholds ----------------------------------------------
    x_thresh = np.linspace(0.0, 1.0, 41)
    y_thresh = np.linspace(0.0, 1.0, 41)

    Z, _, _ = scan_Z(
        sig_counts, bkg_counts, x_edges, y_edges, x_thresh, y_thresh,
        rel_unc=0.20, min_bkg=0.50
    )

    Z_max = np.nanmax(Z)
    ix_best, iy_best = np.unravel_index(np.nanargmax(Z), Z.shape)

    plt.figure()
    # For 1D threshold arrays, pcolormesh expects (Ny, Nx) data when passing centers; Z.T is correct
    pcm = plt.pcolormesh(x_thresh, y_thresh, Z.T, shading="auto")
    cont = plt.contour(x_thresh, y_thresh, Z.T,
                       levels=np.linspace(1.0, Z_max, 5)[1:-1],
                       colors="white", linewidths=1.0)

    plt.scatter([x_thresh[ix_best]], [y_thresh[iy_best]],
                s=60, marker="*", edgecolor="k", facecolor="none", label="max significance")
    plt.legend(loc='upper left')

    cbar = plt.colorbar(pcm)
    cbar.set_label("Asimov Significance")
    plt.clabel(cont, fontsize="smaller")
    plt.xlabel(r"$S_{1}^{\mathrm{NN}}$")
    plt.ylabel(r"$S_{2}^{\mathrm{NN}}$")
    plt.tight_layout()
    _save_fig_to_tmp(run, f"{savename}_Z")

    noncl, NA, NB, NC, ND = scan_nonclosure(bkg_counts, x_edges, y_edges, x_thresh, y_thresh)
    plot_bkg_nonclosure(bkg_counts, x_edges, y_edges, x_thresh, y_thresh, savename, run)
