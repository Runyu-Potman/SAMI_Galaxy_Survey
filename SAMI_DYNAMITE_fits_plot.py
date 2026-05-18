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

    # Plot settings
    xrange = np.array([0.1, Rmax_arcs])
    yrange = np.array([1e6, maxmass])

    ax.set_xlim(xrange)
    ax.set_ylim(yrange)
    ax.set_xlabel(r'$R$ [arcsec]', fontsize=9)
    ax.set_ylabel(r'Enclosed Mass [M$_{\odot}$]', fontsize=9)
    ax.tick_params(labelsize=8)

    # Twin axis for kpc (only works if ax is the original axes)
    ax2 = ax.twiny()
    ax2.set_xlim(xrange * arctpc / 1000.0)
    ax2.set_xlabel(r'$r$ [kpc]', fontsize=9)
    ax2.tick_params(labelsize=8)

    # Plot lines
    ax.plot(R_arcsec, total_best, '-', color='k', linewidth=2.0, label='Total')
    ax.fill_between(R_arcsec, total_min, total_max, facecolor='k', alpha=0.1)

    ax.plot(R_arcsec, stellar_best, '-', color='r', linewidth=2.0, label='Mass-follows-Light')
    ax.fill_between(R_arcsec, stellar_min, stellar_max, facecolor='r', alpha=0.1)

    if has_dm:
        ax.plot(R_arcsec, dm_best, '-', color='b', linewidth=2.0, label='Dark Matter')
        ax.fill_between(R_arcsec, dm_min, dm_max, facecolor='b', alpha=0.1)

    ax.legend(loc='upper left', fontsize=8)

    # Optional saving (only if we created the figure)
    if own_figure and output_plot_path is None:
        dirname = os.path.dirname(fits_path) if os.path.dirname(fits_path) else '.'
        output_plot_path = os.path.join(dirname, 'enclosedmassm_from_fits' + figtype)
    if own_figure and output_plot_path:
        fig.savefig(output_plot_path)
        print(f"Plot saved to {output_plot_path}")

    if own_figure:
        return fig
    else:
        return None