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
            0.03,
            f'{vmin_int:d}',
            transform=cbar.ax.transAxes,
            ha='center',
            va='bottom',
            fontsize=8,
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

        axs[0, i].set_title(labels[i], fontsize=16)
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

        style_map_axis(axs[1, i])

        if i == 0:
            axs[0, i].set_ylabel('data vel', fontsize=16)
            axs[1, i].set_ylabel('model vel', fontsize=16)

        cbar1 = fig.colorbar(
            im_data,
            ax=axs[0, i],
            fraction=0.106,
            pad=0,
            aspect=9
        )

        style_colorbar(cbar1, vmax)

        cbar2 = fig.colorbar(
            im_model,
            ax=axs[1, i],
            fraction=0.106,
            pad=0,
            aspect=9
        )

        style_colorbar(cbar2, vmax)

    # Same aspect ratio for all panels
    for row in range(2):
        for col in range(n_gal):
            axs[row, col].set_box_aspect(1)

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











