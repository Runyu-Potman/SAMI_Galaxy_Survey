import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

def reproduce_mass_plot(fits_filename, ax = None, name = None, r_kdc = None, extrap_start = None, output_plot = None, label = False):
    '''
    Reproduce the cumulative mass plot from the FITS file saved by mass_plot().

    Parameters:
    - fits_filename: str. Path to the FITS file (e.g., 'enclosed_mass_profiles.fits').
    - ax: matplotlib.axes.Axes.
    - name: str. Name of the galaxy.
    - r_kdc: a vertical dotted line to represent the radius of the kinematically distinct component.
    - extrap_start: start radius for the shaded region representing the extrapolated zone.
    - output_plot: str, optional. Output filename for the plot.
    - label: bool, optional. Label the plot.

    Returns:
    - fig: matplotlib.figure.Figure
        The generated figure.
    '''

    # Read the FITS file (table extension is index 1).
    with fits.open(fits_filename) as hdul:
        hdu = hdul[1]  # BinTableHDU
        data = hdu.data
        header = hdu.header

    # Extract data columns
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

    # Get header values for plot limits
    arctpc = header['ARCTPC']
    Rmax_arcs = header['RMAXARC']

    # Determine if dark matter exists (all dm_best zero -> no DM)
    has_dm = np.any(dm_best != 0)

    # Compute y-axis upper limit (same as original: round to next 10^10)
    max_mass = np.max(total_max)
    ymax = (int(max_mass / 1e10) + 1) * 1e10

    # Define x range for axis limits (same as original)
    xrange = np.array([0.1, Rmax_arcs])
    yrange = np.array([1e6, ymax])

    # Create the plot
    if ax is None:
       fig, ax = plt.subplots(figsize = (5, 5))
       created_fig = True
    else:
       fig = ax.figure
       created_fig = False

    ax.set_xlim(xrange)
    ax.set_ylim(yrange)
    ax.set_xlabel(r'$R$ (arcsec)', fontsize = 10)
    if name is not None:
       ax.set_ylabel(f'Galaxy {name}\nEnclosed Mass (M$_{{\odot}}$)', fontsize = 10)
    else:
       ax.set_ylabel(r'Enclosed Mass (M$_{\odot}$)', fontsize = 10)

    # position of the y axis offset test.
    offset_text = ax.yaxis.get_offset_text()
    offset_text.set_position((-0.02, 0.95))

    # x and y axis tick settings.
    ax.tick_params(labelsize = 9, direction = 'in', top = True)
    # make the tick to have the highest zorder.
    ax.set_axisbelow(False)

    # Twin axis for kpc.
    ax2 = ax.twiny()
    ax2.set_xlim(xrange * arctpc / 1000.0)
    ax2.set_xlabel(r'$r$ (kpc)', fontsize = 10)
    ax2.tick_params(labelsize = 9, direction = 'in')
    ax2.set_axisbelow(False)

    # Plot total mass (black).
    if label is True:
       ax.plot(R_arcsec, total_best, '-', color = 'k', linewidth = 2.0, label = 'Total')
    else:
       ax.plot(R_arcsec, total_best, '-', color = 'k', linewidth = 2.0)

    ax.fill_between(R_arcsec, total_min, total_max, facecolor = 'k', alpha = 0.1)

    # Plot stellar (red) and dark matter (blue) if DM exists
    if has_dm:
        ax.plot(R_arcsec, stellar_best, '-', color='r', linewidth=2.0, label='Mass-follows-Light')
        ax.fill_between(R_arcsec, stellar_min, stellar_max, facecolor='r', alpha=0.1)

        ax.plot(R_arcsec, dm_best, '-', color='b', linewidth=2.0, label='Dark Matter')
        ax.fill_between(R_arcsec, dm_min, dm_max, facecolor='b', alpha=0.1)

    ax.legend(loc='upper left', fontsize=8)
    plt.tight_layout()

    if output_plot:
        plt.savefig(output_plot)
        print(f"Plot saved to {output_plot}")

    return fig