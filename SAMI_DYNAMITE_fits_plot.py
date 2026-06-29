import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import matplotlib.patheffects as pe
import cmasher as cmr
from plotbin import display_pixels

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
def plot_kinematic_maps_from_fits_grid(fits_paths, number_gh = 4, labels = None, figsize = (25, 20)):
    '''
    Plot 6 galaxies in a 3x2 grid of galaxy blocks (N = 5):

        Galaxy 1 (3×N maps) | Galaxy 2 (3×N maps)
        Galaxy 3 (3×N maps) | Galaxy 4 (3×N maps)
        Galaxy 5 (3×N maps) | Galaxy 6 (3×N maps)

    where each galaxy block contains:
        Data
        Model
        Residual

    Parameters:
    - fits_paths: Exactly 6 FITS files, one per galaxy.
    - number_gh: Highest GH moment number included.
    - labels: Optional labels for the 6 galaxies.
    '''

    if len(fits_paths) != 6:
        raise ValueError('Please provide exactly 6 FITS files in fits_paths.')

    if labels is None:
        labels = [f'Galaxy {i+1}' for i in range(6)]
    if len(labels) != 6:
        raise ValueError('Please provide exactly 6 labels, or leave labels=None.')

    n_col = number_gh + 1
    gh_indices = list(range(3, number_gh + 1))

    # Figure size tuned for 3x2 blocks.
    fig = plt.figure(figsize = figsize)

    # Outer layout: 3 rows x 2 columns of galaxy blocks.
    outer = fig.add_gridspec(
        3, 2,
        left=0.092,
        right=0.99,
        bottom=0.033,
        top=0.939,
        hspace=0.18,
        wspace=0.08
    )

    # Colormaps.
    map1 = cmr.get_sub_cmap('twilight_shifted', 0.05, 0.6)
    map2 = cmr.get_sub_cmap('twilight_shifted', 0.05, 0.95)

    def add_cbar(im, ax, bottom_label, top_label):
        cb = fig.colorbar(im, ax = ax, pad = 0, fraction = 0.1668, aspect = 5)
        cb.set_ticks([])
        cb.ax.text(0.5, 0.05, bottom_label, transform = cb.ax.transAxes,
                   ha = 'center', va = 'bottom', fontsize = 8, color = 'white', fontweight = 'bold',
                   path_effects = [pe.withStroke(linewidth = 2.5, foreground = 'black')])
        cb.ax.text(0.5, 0.95, top_label, transform = cb.ax.transAxes,
                   ha = 'center', va = 'top', fontsize = 8, color = 'white', fontweight = 'bold',
                   path_effects = [pe.withStroke(linewidth = 2.5, foreground = 'black')])
        return cb

    def format_axis(ax):
        ax.minorticks_on()
        ax.tick_params(direction = 'in', which = 'major', axis = 'both', length = 4, width = 1)
        # add minor ticks (shorter, no labels).
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(direction = 'in', which = 'minor', axis = 'both', length = 2, width = 1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_xticks([-10, -5, 0, 5, 10])
        ax.set_yticks([-10, -5, 0, 5, 10])
        ax.set_aspect('equal', adjustable = 'box')

    def draw_block(fits_path, block_spec, block_label, add_row_labels = False):
        '''
        Draw one galaxy block inside a 3 x n_col sub-grid.
        '''
        inner = block_spec.subgridspec(3, n_col, hspace = 0.0, wspace = 0.0)

        with fits.open(fits_path) as hdul:
            dx = hdul[0].header['DX']
            angle_deg = hdul[0].header['ANGLE']
            xs = hdul['x_coords'].data
            ys = hdul['y_coords'].data

            X, Y = np.meshgrid(xs, ys)
            x_flat = X.ravel()
            y_flat = Y.ravel()

            def get_flat(extname):
                return hdul[extname].data.ravel()

            kw_display_pixels1 = dict(
                pixelsize=dx,
                angle=angle_deg,
                colorbar=False,
                nticks=7,
                cmap=map1
            )
            kw_display_pixels = dict(
                pixelsize=dx,
                angle=angle_deg,
                colorbar=False,
                nticks=7,
                cmap=map2
            )

            # Global labels for this galaxy
            vel_data = get_flat('data_vel')
            vel_model = get_flat('model_vel')
            vel_abs = max(np.nanmax(np.abs(vel_data)), np.nanmax(np.abs(vel_model)))
            vel_val = int(round(1.0 * vel_abs))
            vel_up_label = f'{vel_val}'
            vel_low_label = f'{-vel_val}'

            sig_data = get_flat('data_sig')
            sig_model = get_flat('model_sig')
            sig_min = min(np.nanmin(sig_data), np.nanmin(sig_model))
            sig_max = max(np.nanmax(sig_data), np.nanmax(sig_model))
            sig_bottom_val = int(round(1.0 * sig_min))
            sig_top_val = int(round(1.0 * sig_max))
            sig_low_label = f'{sig_bottom_val}'
            sig_up_label = f'{sig_top_val}'

            row_first_axes = [None, None, None]

            # ---------- DATA row ----------
            ax = fig.add_subplot(inner[0, 0])
            row_first_axes[0] = ax
            c = get_flat('data_sb')
            vmin, vmax = np.nanmin(c), np.nanmax(c)
            im = display_pixels.display_pixels(
                x_flat, y_flat, c,
                vmin = vmin, vmax = vmax,
                **kw_display_pixels1
            )
            format_axis(ax)
            add_cbar(im, ax, int(round(vmin)), int(round(vmax)))
            ax.set_title(r'$\log_{10}(\mu)$', fontsize = 20, pad = 10)

            ax = fig.add_subplot(inner[0, 1])
            c = get_flat('data_vel')
            im = display_pixels.display_pixels(
                x_flat, y_flat, c,
                vmin = -vel_abs, vmax = vel_abs,
                **kw_display_pixels
            )
            format_axis(ax)
            add_cbar(im, ax, vel_low_label, vel_up_label)
            ax.set_title(r'$V_{\bigstar}$ (km/s)', fontsize = 20, pad = 10)

            ax = fig.add_subplot(inner[0, 2])
            c = get_flat('data_sig')
            im = display_pixels.display_pixels(
                x_flat, y_flat, c,
                vmin = sig_min, vmax = sig_max,
                **kw_display_pixels1
            )
            format_axis(ax)
            add_cbar(im, ax, sig_low_label, sig_up_label)
            ax.set_title(r'$\sigma_{\bigstar}$ (km/s)', fontsize = 20, pad = 10)

            for idx, i in enumerate(gh_indices):
                ax = fig.add_subplot(inner[0, 3 + idx])
                c = get_flat(f'data_h{i}')
                im = display_pixels.display_pixels(
                    x_flat, y_flat, c,
                    vmin = -0.15, vmax = 0.15,
                    **kw_display_pixels
                )
                format_axis(ax)
                add_cbar(im, ax, '-0.15', '0.15')
                ax.set_title(f'$h_{{{i}}}$', fontsize = 20, pad = 10)

            # ---------- MODEL row ----------
            ax = fig.add_subplot(inner[1, 0])
            row_first_axes[1] = ax
            c = get_flat('model_sb')
            vmin, vmax = np.nanmin(c), np.nanmax(c)
            im = display_pixels.display_pixels(
                x_flat, y_flat, c,
                vmin = vmin, vmax = vmax,
                **kw_display_pixels1
            )
            format_axis(ax)
            add_cbar(im, ax, int(round(vmin)), int(round(vmax)))

            ax = fig.add_subplot(inner[1, 1])
            c = get_flat('model_vel')
            im = display_pixels.display_pixels(
                x_flat, y_flat, c,
                vmin = -vel_abs, vmax = vel_abs,
                **kw_display_pixels
            )
            format_axis(ax)
            add_cbar(im, ax, vel_low_label, vel_up_label)

            ax = fig.add_subplot(inner[1, 2])
            c = get_flat('model_sig')
            im = display_pixels.display_pixels(
                x_flat, y_flat, c,
                vmin = sig_min, vmax = sig_max,
                **kw_display_pixels1
            )
            format_axis(ax)
            add_cbar(im, ax, sig_low_label, sig_up_label)

            for idx, i in enumerate(gh_indices):
                ax = fig.add_subplot(inner[1, 3 + idx])
                c = get_flat(f'model_h{i}')
                im = display_pixels.display_pixels(
                    x_flat, y_flat, c,
                    vmin = -0.15, vmax = 0.15,
                    **kw_display_pixels
                )
                format_axis(ax)
                add_cbar(im, ax, '-0.15', '0.15')

            # ---------- RESIDUAL row ----------
            ax = fig.add_subplot(inner[2, 0])
            row_first_axes[2] = ax
            c = get_flat('residual_sb')
            im = display_pixels.display_pixels(
                x_flat, y_flat, c,
                vmin=-0.05, vmax=0.05,
                **kw_display_pixels
            )
            format_axis(ax)
            add_cbar(im, ax, '-0.1', '0.1')

            ax = fig.add_subplot(inner[2, 1])
            c = get_flat('residual_vel')
            im = display_pixels.display_pixels(
                x_flat, y_flat, c,
                vmin=-10, vmax=10,
                **kw_display_pixels
            )
            format_axis(ax)
            add_cbar(im, ax, '-10', '10')

            ax = fig.add_subplot(inner[2, 2])
            c = get_flat('residual_sig')
            im = display_pixels.display_pixels(
                x_flat, y_flat, c,
                vmin=-10, vmax=10,
                **kw_display_pixels
            )
            format_axis(ax)
            add_cbar(im, ax, '-10', '10')

            for idx, i in enumerate(gh_indices):
                ax = fig.add_subplot(inner[2, 3 + idx])
                c = get_flat(f'residual_h{i}')
                im = display_pixels.display_pixels(
                    x_flat, y_flat, c,
                    vmin=-10, vmax=10,
                    **kw_display_pixels
                )
                format_axis(ax)
                add_cbar(im, ax, '-10', '10')

        # Add galaxy title above this block
        block_bbox = block_spec.get_position(fig)
        fig.text(
            0.5 * (block_bbox.x0 + block_bbox.x1),
            min(0.995, block_bbox.y1 + 0.03),
            block_label,
            ha='center',
            va='bottom',
            fontsize=20,
            fontweight='bold'
        )

        # Add Data / Model / Residual labels only for the left column blocks
        # so they serve the whole row of galaxies.
        if add_row_labels:
            row_names = ['Data', 'Model', 'Residual']
            x_text = max(0.015, block_bbox.x0 - 0.02)

            for name, ax0 in zip(row_names, row_first_axes):
                bbox = ax0.get_position()
                y = 0.5 * (bbox.y0 + bbox.y1)
                fig.text(
                    x_text,
                    y,
                    name,
                    size=20,
                    ha='center',
                    va='center',
                    rotation=90.
                )

    # Draw the 6 galaxy blocks
    for i, (fits_path, label) in enumerate(zip(fits_paths, labels)):
        r = i // 2
        c = i % 2
        draw_block(
            fits_path=fits_path,
            block_spec=outer[r, c],
            block_label=label,
            add_row_labels=(c == 0)   # labels only on the left block of each row
        )

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