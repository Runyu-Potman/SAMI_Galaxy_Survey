import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

def plot_6x2_velocity_grid(fits_paths, labels=None, cmap='RdBu_r', figsize=(24, 8)):
    """
    Plot a 2 x N grid from saved FITS files:
    row 1 = data velocity
    row 2 = model velocity

    For each galaxy (each column):
    - the data velocity map has its own symmetric range [-vmax, vmax]
    - the model velocity map uses the same range as the data map in that column
    - each subplot gets its own colorbar
    - x/y limits are fixed to [-8.5, 8.5]
    - ticks are at [-5, 0, 5]
    - tick direction is 'in'
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

    # Read all maps first
    for fp in fits_paths:
        with fits.open(fp) as hdul:
            data_vel = hdul['data_vel'].data.astype(float)
            model_vel = hdul['model_vel'].data.astype(float)

            data_maps.append(data_vel)
            model_maps.append(model_vel)

            # Use saved coordinates if available
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
                    np.min(x_coords) - dx / 2,
                    np.max(x_coords) + dx / 2,
                    np.min(y_coords) - dy / 2,
                    np.max(y_coords) + dy / 2
                ]
            else:
                extent = None

            extents.append(extent)

            # Per-galaxy scale from the DATA map only
            vmax_i = np.nanmax(np.abs(data_vel))
            if not np.isfinite(vmax_i) or vmax_i == 0:
                vmax_i = 1.0
            vlims.append((-vmax_i, vmax_i))

    fig, axs = plt.subplots(2, n_gal, figsize=figsize, constrained_layout=True)

    if n_gal == 1:
        axs = np.array(axs).reshape(2, 1)

    for i in range(n_gal):
        extent = extents[i]
        vmin, vmax = vlims[i]

        # Top row: data velocity
        im_data = axs[0, i].imshow(
            data_maps[i],
            origin='lower',
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect='equal',
            extent=extent
        )
        axs[0, i].set_title(labels[i], fontsize=12)
        axs[0, i].set_xlim(-8.5, 8.5)
        axs[0, i].set_ylim(-8.5, 8.5)
        axs[0, i].set_xticks([-5, 0, 5])
        axs[0, i].set_yticks([-5, 0, 5])
        axs[0, i].tick_params(direction='in')

        # Bottom row: model velocity, same scale as data in this column
        im_model = axs[1, i].imshow(
            model_maps[i],
            origin='lower',
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect='equal',
            extent=extent
        )
        axs[1, i].set_xlim(-8.5, 8.5)
        axs[1, i].set_ylim(-8.5, 8.5)
        axs[1, i].set_xticks([-5, 0, 5])
        axs[1, i].set_yticks([-5, 0, 5])
        axs[1, i].tick_params(direction='in')

        if i == 0:
            axs[0, i].set_ylabel('data vel', fontsize=12)
            axs[1, i].set_ylabel('model vel', fontsize=12)

        # Each subplot gets its own colorbar
        cbar1 = fig.colorbar(im_data, ax=axs[0, i], fraction=0.046, pad=0)
        cbar1.set_label('velocity')

        cbar2 = fig.colorbar(im_model, ax=axs[1, i], fraction=0.046, pad=0)
        cbar2.set_label('velocity')

    return fig

#------------------------------------------------------------------------
fits_paths = [
        '7969/dynamite/dynamite_fits/kinematics.fits', '143287/dynamite/dynamite_fits/kinematics.fits',
        '227266/dynamite/dynamite_fits/kinematics.fits', '230776/dynamite/dynamite_fits/kinematics.fits',
        '300787/dynamite/dynamite_fits/kinematics.fits', '9239900248/dynamite/dynamite_fits/kinematics.fits'
    ]

labels = ["Galaxy 7969", "Galaxy 143287", "Galaxy 227266", "Galaxy 230776", "Galaxy 300787", "Galaxy 9239900248"]

fig = plot_6x2_velocity_grid(fits_paths, labels=labels)
plt.show()











