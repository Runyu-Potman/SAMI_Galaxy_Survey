import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import matplotlib.patheffects as pe
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
    - fig: matplotlib.figure.Figure.
    '''

    # Read the FITS file (table extension is index 1).
    with fits.open(fits_filename) as hdul:
        hdu = hdul[1]  # BinTableHDU
        data = hdu.data
        header = hdu.header

    # Extract data columns
    R_arcsec = data['R_arcsec']
    #R_pc = data['R_pc']
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
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins = 4))
    ax.set_xlabel(r'Radius (arcsec)', fontsize = 15)
    if name is not None:
       ax.set_ylabel(f'Galaxy {name}\nEnclosed Mass ($M_{{\odot}}$)', fontsize = 15)
    else:
       ax.set_ylabel(r'Enclosed Mass ($M_{\odot}$)', fontsize = 15)

    # position of the y axis offset text.
    offset_text = ax.yaxis.get_offset_text()
    offset_text.set_position((-0.012, 0.95))
    offset_text.set_fontsize(15)

    # x and y axis tick settings.
    ax.tick_params(labelsize = 15, direction = 'in', top = True)
    # make the tick to have the highest zorder.
    ax.set_axisbelow(False)

    # Twin axis for kpc.
    ax2 = ax.twiny()
    ax2.set_xlim(xrange * arctpc / 1000.0)
    ax2.xaxis.set_major_locator(plt.MaxNLocator(nbins = 4))
    ax2.set_xlabel(r'$r$ (kpc)', fontsize = 15)
    ax2.tick_params(labelsize = 15, direction = 'in')
    ax2.set_axisbelow(False)

    # Plot total mass (black).
    if label is True:
       ax.plot(R_arcsec, total_best, '-', color = 'k', linewidth = 2.0, label = 'Total')
    else:
       ax.plot(R_arcsec, total_best, '-', color = 'k', linewidth = 2.0)

    ax.fill_between(R_arcsec, total_min, total_max, facecolor = 'k', alpha = 0.1)

    # Plot stellar (red) and dark matter (blue) if DM exists
    if has_dm:
        if label is True:
           ax.plot(R_arcsec, stellar_best, '-', color = 'r', linewidth = 2.0, label = 'Mass-follows-Light')
        else:
           ax.plot(R_arcsec, stellar_best, '-', color = 'r', linewidth = 2.0)

        ax.fill_between(R_arcsec, stellar_min, stellar_max, facecolor = 'r', alpha = 0.1)

        if label is True:
           ax.plot(R_arcsec, dm_best, '-', color = 'b', linewidth = 2.0, label = 'Dark Matter')
        else:
           ax.plot(R_arcsec, dm_best, '-', color = 'b', linewidth = 2.0)

        ax.fill_between(R_arcsec, dm_min, dm_max, facecolor = 'b', alpha = 0.1)

    if label is True:
       ax.legend(loc = 'upper left', bbox_to_anchor = (0.01, 0.99), fontsize = 12)

    # a dashed line to represent the radius of the kinematically distinct component.
    if r_kdc is not None:
        ax.axvline(r_kdc, color = 'orange', linestyle = 'dotted', linewidth = 2.0)

    # the shaded region to represent the extrapolated region.
    if extrap_start is not None:
        # get current x_limits.
        xlim = ax.get_xlim()
        # shade from extrap_start to the right limit.
        ax.axvspan(extrap_start, xlim[1], facecolor = 'none', hatch = '\\\\',
                   edgecolor = 'gray', alpha = 0.3)

    if output_plot and created_fig:
        plt.savefig(output_plot)

    idx_kdc = np.argmin(np.abs(R_arcsec - r_kdc))
    print(f'Galaxy {name}: Enclosed stellar mass at r = {r_kdc} arcsec: {stellar_best[idx_kdc]:.3e} M_sun')

    idx_max = np.argmin(np.abs(R_arcsec - Rmax_arcs))
    print(f'Galaxy {name}: Enclosed stellar mass at r = {Rmax_arcs} arcsec: {stellar_best[idx_max]:.3e} M_sun')

    print(f'Galaxy {name}: kdc to total stellar mass ratio:', stellar_best[idx_kdc] / stellar_best[idx_max])

    return fig
#------------------------------------------------------------------------------------
def reproduce_orbit_plot(fits_file, ax = None, cbar = True, name = None, r_kdc = None, text = False):
    '''
    Reproduce the orbit density plot from the FITS file saved by orbit_plot().

    Parameters:
    - fits_file : str. Path to the FITS file.
    - ax : matplotlib.axes.Axes.
    - cbar : bool, optional. If True, add a horizontal colorbar at the top of the axis (default True).
             For a new figure, colorbar is always added regardless of this flag.
    - name: galaxy name added in the y axis label.
    - r_kdc: add a vertical dotted line to represent the radius of the kinematically distinct component.
    - text: if True, add text about each orbit type.

    Returns:
    - fig : matplotlib.figure.Figure.
    '''

    with fits.open(fits_file) as hdul:
         data = hdul[0].data # already R.T (transposed)
         hdr = hdul[0].header

    extent = [hdr['EX0'], hdr['EX1'], hdr['EY0'], hdr['EY1']]
    vmin, vmax = hdr['VMIN'], hdr['VMAX']
    interp = hdr.get('INTERP', 'spline16')
    n_ocut = hdr['OCUTN']
    ocut = [hdr[f'OCUT{i}'] for i in range(1, n_ocut + 1)]

    # Create axes if needed.
    if ax is None:
        fig, ax = plt.subplots(figsize = (6, 5))
        created_fig = True
        add_cbar = True
    else:
        fig = ax.figure
        created_fig = False
        add_cbar = cbar

    # Display the density map.
    im = ax.imshow(data, origin = 'lower', extent = extent, cmap = 'terrain_r',
                   interpolation = interp, vmin = vmin, vmax = vmax, aspect = 'auto')

    ax.tick_params(direction = 'in', labelsize = 15)
    ax.set_xlabel('Radius (arcsec)', fontsize = 15)
    if name is not None:
        ax.set_ylabel(f'Galaxy {name}\nCircularity $\lambda_{{z}}$', fontsize = 15)
    else:
        ax.set_ylabel(r'Circularity $\lambda_{z}$', fontsize = 15)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])

    # Add horizontal colorbar at the top of the axis.
    if add_cbar:
        cb = fig.colorbar(im, ax = ax, orientation = 'horizontal', location = 'top',
                          pad = 0.03, aspect = 30, shrink = 1.0)
        cb.ax.tick_params(direction = 'in', labelsize = 15)
        cb.set_label('Relative Orbit Density', labelpad = 5, fontsize = 15)

    # Draw dashed lines for ocut values.
    for cut in ocut:
        ax.axhline(cut, color = 'black', linestyle = '--', linewidth = 1, xmin = 0, xmax = 1)

    if text:
        boundaries = [1.0, ocut[0], ocut[1], ocut[2], ocut[3], -1.0]
        labels = ['Cold', 'Warm', 'Hot', 'CR-warm', 'CR-cold']
        x_text = extent[0] + 0.05 * (extent[1] - extent[0])
        for i in range(len(labels)):
            y_center = (boundaries[i] + boundaries[i + 1]) / 2
            txt = ax.text(x_text, y_center, labels[i], ha = 'left', va = 'center',
                          fontsize = 12, color = 'black')
            # Add white outline.
            txt.set_path_effects([pe.withStroke(linewidth = 3, foreground = 'white')])

    # a dotted line to represent the radius of the kinematically distinct component.
    if r_kdc is not None:
       ax.axvline(r_kdc, color = 'gray', linestyle = 'dotted', linewidth = 1.5)

    if created_fig:
        plt.tight_layout()

    return fig

#--------------------------------------------------------------------------------
def plot_kinematic_maps_from_fits(fits_path, number_gh=4):
    '''
    Recreate the 3‑row kinematic maps plot from a previously saved FITS file.

    Parameters:
    - fits_path : str. Path to the FITS file produced by `_plot_kinematic_maps_gaussherm`.
    - number_gh : int. Number of Gauss‑Hermite moments (h3, h4, ...). Must match the saved data.

    Returns:
    - fig : matplotlib.figure.Figure.
    '''
    with fits.open(fits_path) as hdul:
        # geometry:
        dx = hdul[0].header['DX']
        angle_deg = hdul[0].header['ANGLE']
        xs = hdul['x_coords'].data # 1D array of unique x positions
        ys = hdul['y_coords'].data # 1D array of unique y positions

        # display_pixels expects 1D arrays of every pixel centre.
        X, Y = np.meshgrid(xs, ys)
        x_flat = X.ravel()
        y_flat = Y.ravel()

        # helper to flatten a 2D map.
        def get_flat(extname):
            return hdul[extname].data.ravel()

        # column layout.
        n_col = number_gh + 1 # sb, vel, sig, h3, h4, ...
        gh_indices = list(range(3, number_gh + 1))

        # figure geometry (same as original).
        left_margin = 27 * 0.04
        right_margin = 27 * 0.03
        col_width = (27 - left_margin - right_margin) / 5
        fig_width = left_margin + col_width * n_col + right_margin
        text_x = 0.015 * 27 / fig_width

        fig = plt.figure(figsize = (fig_width, 12))
        kwtext = dict(size = 20, ha = 'center', va = 'center', rotation = 90.)
        fig.text(text_x, 0.83, 'Data', **kwtext)
        fig.text(text_x, 0.53, 'Model', **kwtext)
        fig.text(text_x, 0.2, 'Residual', **kwtext)

        fig.subplots_adjust(hspace = 0.01, wspace = 0.3,
                            left = left_margin/fig_width,
                            bottom = 0.05, top = 0.99,
                            right = 1 - right_margin/fig_width)

        # colormaps.
        map1 = cmr.get_sub_cmap('twilight_shifted', 0.05, 0.6)
        map2 = cmr.get_sub_cmap('twilight_shifted', 0.05, 0.95)

        kw_display_pixels1 = dict(pixelsize = dx, angle = angle_deg,
                                  colorbar = True, nticks = 7, cmap = map1)
        kw_display_pixels = dict(pixelsize = dx, angle = angle_deg,
                                 colorbar = True, nticks = 7, cmap = map2)

        # DATA row.
        # surface brightness.
        ax = plt.subplot(3, n_col, 1)
        c = get_flat('data_sb')
        display_pixels.display_pixels(x_flat, y_flat, c,
                                      vmin = np.nanmin(c), vmax = np.nanmax(c),
                                      **kw_display_pixels1)
        ax.set_title('surface brightness (log)', fontsize=20, pad=20)

        # velocity
        ax = plt.subplot(3, n_col, 2)
        c = get_flat('data_vel')
        vmax = np.nanmax(np.abs(c))
        display_pixels.display_pixels(x_flat, y_flat, c,
                                      vmin=-vmax, vmax=vmax,
                                      **kw_display_pixels)
        ax.set_title('velocity', fontsize=20, pad=20)

        # sigma
        ax = plt.subplot(3, n_col, 3)
        c = get_flat('data_sig')
        smin, smax = np.nanmin(c), np.nanmax(c)
        display_pixels.display_pixels(x_flat, y_flat, c,
                                      vmin=smin, vmax=smax,
                                      **kw_display_pixels1)
        ax.set_title('velocity dispersion', fontsize=20, pad=20)

        # GH moments
        for idx, i in enumerate(gh_indices):
            ax = plt.subplot(3, n_col, 4 + idx)
            c = get_flat(f'data_h{i}')
            display_pixels.display_pixels(x_flat, y_flat, c,
                                          vmin=-0.15, vmax=0.15,
                                          **kw_display_pixels)
            ax.set_title(f'$h_{{{i}}}$ moment', fontsize=20, pad=20)

        # ----- MODEL row -----
        plt.subplot(3, n_col, n_col + 1)
        c = get_flat('model_sb')
        display_pixels.display_pixels(x_flat, y_flat, c,
                                      vmin=np.nanmin(c), vmax=np.nanmax(c),
                                      **kw_display_pixels1)

        plt.subplot(3, n_col, n_col + 2)
        c = get_flat('model_vel')
        display_pixels.display_pixels(x_flat, y_flat, c,
                                      vmin=-vmax, vmax=vmax,
                                      **kw_display_pixels)

        plt.subplot(3, n_col, n_col + 3)
        c = get_flat('model_sig')
        display_pixels.display_pixels(x_flat, y_flat, c,
                                      vmin=smin, vmax=smax,
                                      **kw_display_pixels1)

        for idx, i in enumerate(gh_indices):
            plt.subplot(3, n_col, n_col + 4 + idx)
            c = get_flat(f'model_h{i}')
            display_pixels.display_pixels(x_flat, y_flat, c,
                                          vmin=-0.15, vmax=0.15,
                                          **kw_display_pixels)

        # ----- RESIDUAL row -----
        plt.subplot(3, n_col, 2*n_col + 1)
        c = get_flat('residual_sb')
        display_pixels.display_pixels(x_flat, y_flat, c,
                                      vmin=-0.05, vmax=0.05,
                                      **kw_display_pixels)

        plt.subplot(3, n_col, 2*n_col + 2)
        c = get_flat('residual_vel')
        display_pixels.display_pixels(x_flat, y_flat, c,
                                      vmin=-10, vmax=10,
                                      **kw_display_pixels)

        plt.subplot(3, n_col, 2*n_col + 3)
        c = get_flat('residual_sig')
        display_pixels.display_pixels(x_flat, y_flat, c,
                                      vmin=-10, vmax=10,
                                      **kw_display_pixels)

        for idx, i in enumerate(gh_indices):
            plt.subplot(3, n_col, 2*n_col + 4 + idx)
            c = get_flat(f'residual_h{i}')
            display_pixels.display_pixels(x_flat, y_flat, c,
                                          vmin=-10, vmax=10,
                                          **kw_display_pixels)

    return fig

#---------------------------------------------------------------------------------
if __name__ == '__main__':
    '''
    fits_filename_7969 = '7969/dynamite/dynamite_fits/enclosed_mass_profiles.fits'
    fits_filename_143287 = '143287/dynamite/dynamite_fits/enclosed_mass_profiles.fits'
    fits_filename_227266 = '227266/dynamite/dynamite_fits/enclosed_mass_profiles.fits'
    fits_filename_230776 = '230776/dynamite/dynamite_fits/enclosed_mass_profiles.fits'
    fits_filename_300787 = '300787/dynamite/dynamite_fits/enclosed_mass_profiles.fits'
    fits_filename_9239900248 = '9239900248/dynamite/dynamite_fits/enclosed_mass_profiles.fits'

    fig, ax = plt.subplots(2, 3, figsize=(12, 8))

    reproduce_mass_plot(fits_filename_7969, ax=ax[0, 0], name='7969', r_kdc=1.7, extrap_start=3.99, label=True)
    reproduce_mass_plot(fits_filename_143287, ax=ax[0, 1], name='143287', r_kdc=2.9, extrap_start=5.83)
    reproduce_mass_plot(fits_filename_227266, ax=ax[0, 2], name='227266', r_kdc=3.0, extrap_start=7.5)
    reproduce_mass_plot(fits_filename_230776, ax=ax[1, 0], name='230776', r_kdc=4.6, extrap_start=7.5)
    reproduce_mass_plot(fits_filename_300787, ax=ax[1, 1], name='300787', r_kdc=2.5, extrap_start=5.37)
    reproduce_mass_plot(fits_filename_9239900248, ax=ax[1, 2], name='9239900248', r_kdc=3.4, extrap_start=5.78)

    plt.tight_layout()

    plt.savefig('final\enclosed_mass_profiles.png', dpi=300, bbox_inches='tight')

    plt.show()
    '''

    fits_orbit_7969 = '7969/dynamite/dynamite_fits/orbit_density.fits'
    fits_orbit_143287 = '143287/dynamite/dynamite_fits/orbit_density.fits'
    fits_orbit_227266 = '227266/dynamite/dynamite_fits/orbit_density.fits'
    fits_orbit_230776 = '230776/dynamite/dynamite_fits/orbit_density.fits'
    fits_orbit_300787 = '300787/dynamite/dynamite_fits/orbit_density.fits'
    fits_orbit_9239900248 = '9239900248/dynamite/dynamite_fits/orbit_density.fits'

    fig, ax = plt.subplots(2, 3, figsize = (12, 8))
    reproduce_orbit_plot(fits_orbit_7969, ax = ax[0, 0], name = '7969', r_kdc = 1.7, text = True)
    reproduce_orbit_plot(fits_orbit_143287, ax = ax[0, 1], name = '143287', r_kdc = 2.9)
    reproduce_orbit_plot(fits_orbit_227266, ax = ax[0, 2], name = '227266', r_kdc = 3.0)
    reproduce_orbit_plot(fits_orbit_230776, ax = ax[1, 0], name = '230776', r_kdc = 4.6)
    reproduce_orbit_plot(fits_orbit_300787, ax = ax[1, 1], name = '300787', r_kdc = 2.5)
    reproduce_orbit_plot(fits_orbit_9239900248, ax = ax[1, 2], name = '9239900248', r_kdc = 3.4)

    plt.tight_layout()
    plt.savefig('final\orbit_density.png', dpi = 300, bbox_inches = 'tight')
    plt.show()