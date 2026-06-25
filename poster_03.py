import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import matplotlib.patheffects as pe

def plot_6x2_velocity_grid(fits_paths, labels=None, cmap='RdBu_r', figsize=(19, 6)):

    n_gal = len(fits_paths)

    if labels is None:
        labels = [f'Galaxy {i+1}' for i in range(n_gal)]

    if len(labels) != n_gal:
        raise ValueError("labels must have the same length as fits_paths")

    data_maps = []
    model_maps = []
    extents = []
    vlims = []

    for fp in fits_paths:

        with fits.open(fp) as hdul:

            data_vel = hdul['data_vel'].data.astype(float)
            model_vel = hdul['model_vel'].data.astype(float)

            data_maps.append(data_vel)
            model_maps.append(model_vel)

            if 'x_coords' in hdul and 'y_coords' in hdul:

                x_coords = hdul['x_coords'].data.astype(float)
                y_coords = hdul['y_coords'].data.astype(float)

                if len(x_coords) > 1:
                    dx = np.median(np.diff(np.sort(x_coords)))
                else:
                    dx = 1.0

                if len(y_coords) > 1:
                    dy = np.median(np.diff(np.sort(y_coords)))
                else:
                    dy = dx

                extent = [
                    np.min(x_coords) - dx/2,
                    np.max(x_coords) + dx/2,
                    np.min(y_coords) - dy/2,
                    np.max(y_coords) + dy/2
                ]

            else:
                extent = None

            extents.append(extent)

            vmax_i = np.nanmax(np.abs(data_vel))

            if not np.isfinite(vmax_i) or vmax_i == 0:
                vmax_i = 1.0

            vlims.append(vmax_i)

    fig, axs = plt.subplots(2, n_gal, figsize=figsize)

    if n_gal == 1:
        axs = np.array(axs).reshape(2, 1)

    def style_map_axis(ax, show_xlabel=False):

        ax.set_xlim(-12.5, 12.5)
        ax.set_ylim(-12.5, 12.5)

        ax.set_xticks([-10, -5, 0, 5, 10])
        ax.set_yticks([-10, -5, 0, 5, 10])

        ax.tick_params(
            which='both',
            direction='in',
            length=5,
            width=1,
            bottom=True,
            top=False,
            left=True,
            right=False,
            labelleft=True,  # <-- SHOW y tick values always
            labelbottom=show_xlabel  # <-- only bottom row shows x labels
        )

    def style_colorbar(cbar, vmax_val):

        cbar.set_ticks([])
        cbar.ax.tick_params(
            length=0,
            labelleft=False,
            labelright=False
        )

        cbar.outline.set_linewidth(1.0)

        vmax_int = int(np.rint(0.98 * vmax_val))
        vmin_int = -vmax_int

        cbar.ax.text(
            0.5,
            0.9,
            f'{vmax_int:d}',
            transform=cbar.ax.transAxes,
            ha='center',
            va='top',
            fontsize=7,
            color='white',
            fontweight='bold',
            path_effects=[
                pe.withStroke(
                    linewidth=2.5,
                    foreground='black'
                )
            ]
        )

        cbar.ax.text(
            0.5,
            0.1,
            f'{vmin_int:d}',
            transform=cbar.ax.transAxes,
            ha='center',
            va='bottom',
            fontsize=7,
            color='white',
            fontweight='bold',
            path_effects=[
                pe.withStroke(
                    linewidth=2.5,
                    foreground='black'
                )
            ]
        )

    for i in range(n_gal):

        extent = extents[i]
        vmax = vlims[i]

        vmax_plot = 0.98 * vmax
        vmin_plot = -vmax_plot

        im_data = axs[0, i].imshow(
            data_maps[i],
            origin='lower',
            cmap=cmap,
            vmin=vmin_plot,
            vmax=vmax_plot,
            aspect='equal',
            extent=extent
        )

        axs[0, i].set_title(labels[i], fontsize=12)
        style_map_axis(axs[0, i])

        im_model = axs[1, i].imshow(
            model_maps[i],
            origin='lower',
            cmap=cmap,
            vmin=vmin_plot,
            vmax=vmax_plot,
            aspect='equal',
            extent=extent
        )

        style_map_axis(axs[1, i], show_xlabel=True)

        axs[1, i].set_xlabel('arcsec', fontsize=12)

        if i == 0:
            axs[0, i].set_ylabel('Data', fontsize=12)
            axs[1, i].set_ylabel('Model', fontsize=12)

        cbar1 = fig.colorbar(
            im_data,
            ax=axs[0, i],
            fraction=0.1,
            pad=0,
            aspect=9
        )

        style_colorbar(cbar1, vmax)

        cbar2 = fig.colorbar(
            im_model,
            ax=axs[1, i],
            fraction=0.1,
            pad=0,
            aspect=9
        )

        style_colorbar(cbar2, vmax)

    # Same aspect ratio for all panels
    for row in range(2):
        for col in range(n_gal):
            axs[row, col].set_box_aspect(1)

    fig.suptitle('Stellar Velocity Maps (km/s) of Six Galaxies Hosting Kinematically Distinct Components', fontsize=15)

    return fig

#------------------------------------------------------------------------
fits_paths = [
        '7969/dynamite/dynamite_fits/kinematics.fits', '143287/dynamite/dynamite_fits/kinematics.fits',
        '227266/dynamite/dynamite_fits/kinematics.fits', '230776/dynamite/dynamite_fits/kinematics.fits',
        '300787/dynamite/dynamite_fits/kinematics.fits', '9239900248/dynamite/dynamite_fits/kinematics.fits'
    ]

labels = ["Galaxy 7969", "Galaxy 143287", "Galaxy 227266", "Galaxy 230776", "Galaxy 300787", "Galaxy 9239900248"]

fig = plot_6x2_velocity_grid(fits_paths, labels=labels)
plt.tight_layout(h_pad=0, w_pad=0)
plt.savefig('final/poster_all.png', dpi=300, bbox_inches='tight')
plt.show()
'''

#########################################################################################

def plot_6x2_velocity_grid(fits_paths, labels=None, cmap='RdBu_r', figsize=(19, 3.5)):
    """
    Plot velocity data maps with model maps as corner insets (bottom‑right).
    Each main panel shows the data; the inset shows the model.
    Insets have no colour bar, no labels, no tick values, but show ticks.
    """
    n_gal = len(fits_paths)

    if labels is None:
        labels = [f'Galaxy {i+1}' for i in range(n_gal)]

    if len(labels) != n_gal:
        raise ValueError("labels must have the same length as fits_paths")

    data_maps = []
    model_maps = []
    extents = []
    vlims = []

    for fp in fits_paths:
        with fits.open(fp) as hdul:
            data_vel = hdul['data_vel'].data.astype(float)
            model_vel = hdul['model_vel'].data.astype(float)
            data_maps.append(data_vel)
            model_maps.append(model_vel)

            if 'x_coords' in hdul and 'y_coords' in hdul:
                x_coords = hdul['x_coords'].data.astype(float)
                y_coords = hdul['y_coords'].data.astype(float)
                if len(x_coords) > 1:
                    dx = np.median(np.diff(np.sort(x_coords)))
                else:
                    dx = 1.0
                if len(y_coords) > 1:
                    dy = np.median(np.diff(np.sort(y_coords)))
                else:
                    dy = dx
                extent = [
                    np.min(x_coords) - dx/2,
                    np.max(x_coords) + dx/2,
                    np.min(y_coords) - dy/2,
                    np.max(y_coords) + dy/2
                ]
            else:
                extent = None
            extents.append(extent)

            vmax_i = np.nanmax(np.abs(data_vel))
            if not np.isfinite(vmax_i) or vmax_i == 0:
                vmax_i = 1.0
            vlims.append(vmax_i)

    # Create a single row of subplots
    fig, axs = plt.subplots(1, n_gal, figsize=figsize)
    if n_gal == 1:
        axs = np.array([axs])

    def style_main_axis(ax, show_xlabel=True):
        """Style for the main data axes."""
        ax.set_xlim(-12.5, 12.5)
        ax.set_ylim(-12.5, 12.5)
        ax.set_xticks([-10, -5, 0, 5, 10])
        ax.set_yticks([-10, -5, 0, 5, 10])
        ax.tick_params(
            which='both',
            direction='in',
            length=5,
            width=1,
            bottom=True,
            top=False,
            left=True,
            right=False,
            labelleft=True,          # show y tick values
            labelbottom=show_xlabel  # show x tick values (True for bottom row)
        )
        if show_xlabel:
            ax.set_xlabel('arcsec', fontsize=12)

    def style_inset_axis(ax):
        """Style for the corner inset: no labels, no tick values, but show ticks."""
        ax.set_xlim(-8.5, 8.5)
        ax.set_ylim(-8.5, 8.5)
        ax.set_xticks([-5, 0, 5])
        ax.set_yticks([-5, 0, 5])
        ax.tick_params(
            which='both',
            direction='in',
            length=3,                # slightly smaller ticks for inset
            width=0.8,
            bottom=True,
            top=False,
            left=True,
            right=False,
            labelleft=False,         # no tick values
            labelbottom=False        # no tick values
        )
        # Remove axis labels
        ax.set_xlabel('')
        ax.set_ylabel('')

    def style_colorbar(cbar, vmax_val):
        """Add min/max values as text on the colour bar (no ticks)."""
        cbar.set_ticks([])
        cbar.ax.tick_params(length=0, labelleft=False, labelright=False)
        cbar.outline.set_linewidth(1.0)

        vmax_int = int(np.rint(0.98 * vmax_val))
        vmin_int = -vmax_int

        cbar.ax.text(
            0.5, 0.98, f'{vmax_int:d}',
            transform=cbar.ax.transAxes,
            ha='center', va='top',
            fontsize=7, color='white', fontweight='bold',
            path_effects=[pe.withStroke(linewidth=2.5, foreground='black')]
        )
        cbar.ax.text(
            0.5, 0.02, f'{vmin_int:d}',
            transform=cbar.ax.transAxes,
            ha='center', va='bottom',
            fontsize=7, color='white', fontweight='bold',
            path_effects=[pe.withStroke(linewidth=2.5, foreground='black')]
        )

        cbar.ax.text(
            0.5, 0.5, 'km/s',
            transform=cbar.ax.transAxes,
            ha='center', va='center',
            fontsize=7, color='white', fontweight='bold',
            path_effects=[pe.withStroke(linewidth=2.5, foreground='black')]
        )

    for i in range(n_gal):
        extent = extents[i]
        vmax = vlims[i]
        vmax_plot = 0.98 * vmax
        vmin_plot = -vmax_plot

        # ---- Main data plot ----
        im_data = axs[i].imshow(
            data_maps[i],
            origin='lower',
            cmap=cmap,
            vmin=vmin_plot,
            vmax=vmax_plot,
            aspect='equal',
            extent=extent
        )
        axs[i].set_title(labels[i], fontsize=12)
        style_main_axis(axs[i], show_xlabel=True)   # only row, so show x labels
        axs[i].text(
            0.5, 0.95, 'Data',
            transform=axs[i].transAxes,
            ha='center', va='top',
            fontsize=12, color='white', fontweight='bold',
            path_effects=[pe.withStroke(linewidth=2.5, foreground='black')]
        )

        # ---- Inset model plot (bottom‑right corner) ----
        # Position: lower‑left corner at (0.70, 0.05) with size 0.25 x 0.25 (axes fraction)
        inset_ax = axs[i].inset_axes([0.65, 0.035, 0.35, 0.35])
        im_model = inset_ax.imshow(
            model_maps[i],
            origin='lower',
            cmap=cmap,
            vmin=vmin_plot,
            vmax=vmax_plot,
            aspect='equal',
            extent=extent
        )
        style_inset_axis(inset_ax)

        inset_ax.text(
            0.5, 0.95, 'Model',
            transform=inset_ax.transAxes,
            ha='center', va='top',
            fontsize=7, color='white', fontweight='bold',
            path_effects=[pe.withStroke(linewidth=2.0, foreground='black')]
        )

        # Optionally add a border or background to make inset stand out
        for spine in inset_ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.0)

        # ---- Colour bar for the main data ----
        cbar = fig.colorbar(
            im_data,
            ax=axs[i],
            fraction=0.1,
            pad=0,
            aspect=9
        )
        style_colorbar(cbar, vmax)

        # Force equal aspect for the main axis
        axs[i].set_box_aspect(1)

    fig.suptitle(
        'Stellar Velocity Maps of Six Galaxies Hosting Kinematically Distinct Components',
        fontsize=15
    )
    '''
    # add psf.
    add_psf(ax=axs[0], psffwhm=2.108)
    add_psf(ax=axs[1], psffwhm=2.144)

    add_psf(ax=axs[2], psffwhm=1.561)
    add_psf(ax=axs[3], psffwhm=2.250)

    add_psf(ax=axs[4], psffwhm=1.941)
    add_psf(ax=axs[5], psffwhm=2.118)

    return fig

# ------------------------------------------------------------------------
fits_paths = [
    '227266/dynamite/dynamite_fits/kinematics.fits',
    '230776/dynamite/dynamite_fits/kinematics.fits',

    '7969/dynamite/dynamite_fits/kinematics.fits',
    '143287/dynamite/dynamite_fits/kinematics.fits',

    '300787/dynamite/dynamite_fits/kinematics.fits',
    '9239900248/dynamite/dynamite_fits/kinematics.fits'
]

labels = [
    "Galaxy 7969", "Galaxy 143287", "Galaxy 227266",
    "Galaxy 230776", "Galaxy 300787", "Galaxy 9239900248"
]

fig = plot_6x2_velocity_grid(fits_paths, labels=labels, figsize=(19, 3.5))
plt.tight_layout(h_pad=0, w_pad=0)
plt.savefig('final/poster_all.png', dpi=300, bbox_inches='tight')
plt.show()











