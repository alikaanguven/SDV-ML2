import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import os
import neptune



def plot_hist1(p, match, PROJECT_DIR, savename, run, binwidth=0.02):
    """
    p:      Probability (e.g. sigmoid(logit))
    match:  vtx match result (bool)
    """
    bins  = np.arange(-0.01,1.01,binwidth)

    binVals1, binEdges1 = np.histogram(p[match], bins)
    binCenters1 = (binEdges1[:-1] + binEdges1[1:]) / 2

    binVals0, binEdges0 = np.histogram(p[~match], bins)
    binCenters0 = (binEdges0[:-1] + binEdges0[1:]) / 2
    
    plt.figure(figsize=(10,8))
    plt.errorbar(binCenters1, binVals1, np.sqrt(binVals1), fmt='o', capthick=1, capsize=3, color='indigo', markersize=3, label="matched")
    plt.errorbar(binCenters0, binVals0, np.sqrt(binVals0), fmt='o', capthick=1, capsize=3, color='red',    markersize=3, label="non-matched")

    y_min = 1
    y_max = 2*(sum(binVals1)+sum(binVals0))

    plt.ylim(y_min, y_max)
    plt.yscale('log')
    plt.title('Validation Dataset')
    plt.xlabel(r'$S_{1}^{\mathrm{NN}}$', loc='right')               # LaTeX-style axis label
    plt.ylabel('vtx Counts',             loc='top')
    plt.legend()
    fig_savepath = os.path.join(PROJECT_DIR, f'training/tmp/{savename}.png')
    plt.savefig(fig_savepath)
    
    run[savename].append(neptune.types.File(fig_savepath))






def plot_hist2(p1, p2, match, PROJECT_DIR, savename, run, binwidth=0.02, cmin=1e-3):
    """
    p1, p2: Probability (e.g. sigmoid(logit))
    """

    # bins that include 1.0
    xbins = np.arange(0, 1, binwidth)
    ybins = np.arange(0, 1, binwidth)

    plt.figure()
    h = plt.hist2d(
        p1[match], p2[match],
        bins=[xbins, ybins],
        range=((0, 1), (0, 1)),
        density=True,
        cmin=cmin,                         # hide very-sparse bins
        norm=LogNorm()                     # temporary; will override vmin/vmax below
    )

    # Determine vmin/vmax from shown (nonzero) data
    H = h[0]
    positive = H[H > 0]

    vmin = max(cmin, positive.min())
    vmax = positive.max()

    # pass ticks & formatter when creating the colorbar
    plt.colorbar(h[3], label='normalised vtx density')
    h[3].set_norm(LogNorm(vmin=vmin, vmax=vmax))

    plt.xlabel(r'$S_{1}^{\mathrm{NN}}$')
    plt.ylabel(r'$S_{2}^{\mathrm{NN}}$')
    plt.tight_layout()
    fig_savepath = os.path.join(PROJECT_DIR, f'training/tmp/{savename}_sig.png')
    plt.savefig(fig_savepath)
    run[f'{savename}_sig'].append(neptune.types.File(fig_savepath))

    plt.figure()
    h = plt.hist2d(
        p1[~match], p2[~match],
        bins=[xbins, ybins],
        range=((0, 1), (0, 1)),
        density=True,
        cmin=cmin,                         # hide very-sparse bins
        norm=LogNorm()                     # temporary; will override vmin/vmax below
    )

    # Determine vmin/vmax from shown (nonzero) data
    H = h[0]
    positive = H[H > 0]

    vmin = max(cmin, positive.min())
    vmax = positive.max()

    # pass ticks & formatter when creating the colorbar
    plt.colorbar(h[3], label='normalised vtx density')
    h[3].set_norm(LogNorm(vmin=vmin, vmax=vmax))

    plt.xlabel(r'$S_{1}^{\mathrm{NN}}$')
    plt.ylabel(r'$S_{2}^{\mathrm{NN}}$')
    plt.tight_layout()
    fig_savepath = os.path.join(PROJECT_DIR, f'training/tmp/{savename}_bkg.png')
    plt.savefig(fig_savepath)
    run[f'{savename}_bkg'].append(neptune.types.File(fig_savepath))