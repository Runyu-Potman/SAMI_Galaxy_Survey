import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os

def plot_mass_from_fits(fits_path, ax=None, output_plot_path=None, figtype='.png'):
    """
    Recreate the cumulative mass plot from a FITS file on a given Axes.

    Parameters
    ----------
    fits_path : str
        Path to the FITS file.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None, a new figure is created.
    output_plot_path : str, optional
        Where to save the figure. Ignored if ax is provided (saving must be done manually).
    figtype : str, optional
        File extension (used only if output_plot_path is None and ax is None).

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        If ax is None, returns the Figure; otherwise returns None.
    """
    # Read data
    with fits.open(fits_path) as hdul:
        data = hdul[1].data
        R_arcsec = data['R_arcsec']
        R_pc = data['R_pc']
        total_best = data['total_best']
        total_min = data['total_min']
        total_max = data['total_max']
        stellar_best = data['stellar_best']
        stellar_min = data['stellar_min']
        stellar_max = data['stellar_max']
        dm_best = data['dm_best']
        dm_min = data['dm_min']
        dm_max = data['dm_max']

    has_dm = np.any(dm_best > 0)

    # Derive arctpc from R_pc / R_arcsec
    idx = R_arcsec > 0.01
    arctpc = np.median(R_pc[idx] / R_arcsec[idx])

    # Rmax_arcs from maximum R_arcsec (which is Rmax_arcs*1.2 in original)
    Rmax_arcs = np.max(R_arcsec)

    # Y-axis upper limit
    max_total_max = np.max(total_max)
    maxmass = (int(max_total_max / 1e10) + 1) * 1e10

    # Create figure/axes if needed
    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1)
        own_figure = True
    else:
        fig = None
        own_figure = False
