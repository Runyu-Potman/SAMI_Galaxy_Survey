import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


def plot_mass_from_fits(fits_path, output_plot_path=None, figtype='.png'):
    """
    Recreate the cumulative mass plot from a FITS file saved by mass_plot().

    Parameters
    ----------
    fits_path : str
        Path to the FITS file (e.g., 'enclosed_mass_profiles.fits').
    output_plot_path : str, optional
        Where to save the figure. If None, saves as 'enclosedmassm_from_fits.png'
        in the same directory as the FITS file.
    figtype : str, optional
        File extension (e.g., '.png', '.pdf'). Default '.png'.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
    """
    # Read the FITS data
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

    # Determine if dark matter is present (non-zero best)
    has_dm = np.any(dm_best > 0)

    # Derive arctpc from R_pc / R_arcsec (avoid division by zero at very small R)
    idx = R_arcsec > 0.01
    arctpc = np.median(R_pc[idx] / R_arcsec[idx])

    # Derive Rmax_arcs as the maximum radius used in the original plot
    # The original plot xlim goes to Rmax_arcs, but the stored R_arcsec goes
    # up to Rmax_arcs*1.2. We use the stored maximum as the plot limit,
    # which will be slightly larger than the original limit.
    # To match exactly, you would need to store Rmax_arcs in the FITS header.
    # For now we use the maximum R_arcsec, but if you prefer the original
    # behaviour, you can set Rmax_arcs = Rmax_arcs_from_header.
    Rmax_arcs = np.max(R_arcsec)  # This is actually Rmax_arcs*1.2
    # If you stored Rmax_arcs in header, use: Rmax_arcs = hdul[0].header['RMAX_ARCS']

    # Compute y-axis upper limit exactly as in mass_plot
    max_total_max = np.max(total_max)
    maxmass = (int(max_total_max / 1e10) + 1) * 1e10

    # Create figure (same size 5x5 inches)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)

    xrange = np.array([0.1, Rmax_arcs])
    yrange = np.array([1e6, maxmass])

    ax.set_xlim(xrange)
    ax.set_ylim(yrange)
    ax.set_xlabel(r'$R$ [arcsec]', fontsize=9)
    ax.set_ylabel(r'Enclosed Mass [M$_{\odot}$]', fontsize=9)
    ax.tick_params(labelsize=8)

    # Twin axis for kpc
    ax2 = ax.twiny()
    ax2.set_xlim(xrange * arctpc / 1000.0)
    ax2.set_xlabel(r'$r$ [kpc]', fontsize=9)
    ax2.tick_params(labelsize=8)

    # Plot total mass (black)
    ax.plot(R_arcsec, total_best, '-', color='k', linewidth=2.0, label='Total')
    ax.fill_between(R_arcsec, total_min, total_max, facecolor='k', alpha=0.1)

    # Plot stellar mass (red)
    ax.plot(R_arcsec, stellar_best, '-', color='r', linewidth=2.0, label='Mass-follows-Light')
    ax.fill_between(R_arcsec, stellar_min, stellar_max, facecolor='r', alpha=0.1)

    # Plot dark matter (blue) if present
    if has_dm:
        ax.plot(R_arcsec, dm_best, '-', color='b', linewidth=2.0, label='Dark Matter')
        ax.fill_between(R_arcsec, dm_min, dm_max, facecolor='b', alpha=0.1)

    ax.legend(loc='upper left', fontsize=8)
    plt.tight_layout()

    # Save the figure
    if output_plot_path is None:
        import os
        dirname = os.path.dirname(fits_path)
        if not dirname:
            dirname = '.'
        output_plot_path = os.path.join(dirname, 'enclosedmassm_from_fits' + figtype)
    fig.savefig(output_plot_path)
    print(f"Plot saved to {output_plot_path}")

    return fig

