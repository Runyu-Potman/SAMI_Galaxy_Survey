import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from SAMI_stellar_velocity_quality_cut_functions import quality_cut_stellar_velocity_map_csv, quality_cut_gaseous_velocity_map_csv
import matplotlib.patches as patches
from PIL import Image
#---------------------------------------------------------------
def plot_vel_or_sig(csv_path, cmap = 'jet', cbar_label = 'km/s', value_type = 'vel',
                    show_colorbar = True, fontsize = 8, bar_fraction = 0.046, bar_pad = 0.04, label_pad = 0.5, ax = None,
                    vmin = None, vmax = None, title = None, csv_path_uncut = None,
                    background_alpha = 0.3, PAs = None, line_length = 5, plot_psf = False, psffwhm = 2.10812):
    """
    Plot velocity or sigma map from CSV file on a given matplotlib axis.

    Parameters:
    - csv_path: str, path to the CSV file (must include 'x_arcsec', 'y_arcsec', 'vel', 'sig')
    - ax: matplotlib.axes.Axes object. If None, creates a new figure.
    - value_type: 'vel' or 'sig' — determines which column to plot
    - cmap: colormap for imshow
    - vmin, vmax: color scale limits
    - title: plot title
    - cbar_label: label for colorbar
    - show_colorbar: whether to display colorbar
    - background_alpha: alpha transparency for background layer.
    - PAs: list of position angles (in degree, counterclockwise from North, 0 to 360).
    - line_length: length of the line to plot, in arcsec.

    Returns:
    - ax: the axis used for plotting
    """
    # read the csv file.
    quality_cut_map = pd.read_csv(csv_path)

    # extract unique x and y values
    x_values = np.unique(quality_cut_map['x_arcsec'])
    y_values = np.unique(quality_cut_map['y_arcsec'])

    grid = np.full((len(y_values), len(x_values)), np.nan)

    for index, row in quality_cut_map.iterrows():
        x_grid = np.where(x_values == row['x_arcsec'])[0][0]
        y_grid = np.where(y_values == row['y_arcsec'])[0][0]
        grid[y_grid, x_grid] = row[value_type]

    if csv_path_uncut:
        uncut_map = pd.read_csv(csv_path_uncut)
        bg_grid = np.full_like(grid, np.nan)
        for index, row in uncut_map.iterrows():
            x_grid = np.where(x_values == row['x_arcsec'])[0][0]
            y_grid = np.where(y_values == row['y_arcsec'])[0][0]
            bg_grid[y_grid, x_grid] = row[value_type]

    # set up plot.
    if ax is None:
        fig, ax = plt.subplots()

    if vmin is None:
        vmin = np.nanmin(grid)
    if vmax is None:
        vmax = np.nanmax(grid)

    if csv_path_uncut:
        ax.imshow(bg_grid, origin = 'lower', aspect = 'equal',
                  cmap = 'Greys_r', alpha = background_alpha,
                  extent = [x_values.min(), x_values.max(), y_values.min(), y_values.max()]
                  )

    im = ax.imshow(
        grid, origin = 'lower', aspect = 'equal',
        cmap = cmap, vmin = vmin, vmax = vmax,
        extent = [x_values.min(), x_values.max(), y_values.min(), y_values.max()]
    )

    # tick setting.
    # set fixed axis limits (-12.5 to 12.5 arcsec).
    ax.set_xlim([-12.5, 12.5])
    ax.set_ylim([-12.5, 12.5])

    # set ticks every 5 arcsec, centered on 0.
    tick_locs = np.arange(-10, 11, 5)
    ax.set_xticks(tick_locs)
    ax.set_yticks(tick_locs)

    ax.set_xlabel('Offset [arcsec]', fontsize = fontsize, labelpad = label_pad)
    ax.set_ylabel('Offset [arcsec]', fontsize = fontsize, labelpad = label_pad)

    # colorbar.
    if show_colorbar:
        cbar = plt.colorbar(im, ax = ax, fraction = bar_fraction, pad = bar_pad)
        if cbar_label:
            cbar.set_label(cbar_label, fontsize = fontsize, labelpad = label_pad)

    if title:
        ax.set_title(title)

    # make these (major) ticks longer.
    ax.tick_params(axis = 'both', which = 'major', length = 4, width = 1)

    # add minor ticks (shorter, no labels).
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis = 'both', which = 'minor', length = 4, width = 1)

    return ax

#----------------------------------------------------------------------














#----------------------------------------------------------------

# 12 CSV file paths, ordered as: vel1, sig1, vel2, sig2, ..., vel6, sig6
csv_vel_sig_files = [
    "gal1_vel.csv", "gal1_sig.csv",
    "gal2_vel.csv", "gal2_sig.csv",
    "gal3_vel.csv", "gal3_sig.csv",
    "gal4_vel.csv", "gal4_sig.csv",
    "gal5_vel.csv", "gal5_sig.csv",
    "gal6_vel.csv", "gal6_sig.csv"
]

# Set up the 3x4 figure
fig, axs = plt.subplots(3, 4, figsize=(18, 12))

# Row 0
plot_velocity_or_sigma_map_from_csv(csv_vel_sig_files[0], ax=axs[0, 0], value_type='vel', title='Galaxy 1 Velocity', cbar_label='km/s')
plot_velocity_or_sigma_map_from_csv(csv_vel_sig_files[1], ax=axs[0, 1], value_type='sig', title='Galaxy 1 Sigma', cbar_label='km/s')
plot_velocity_or_sigma_map_from_csv(csv_vel_sig_files[2], ax=axs[0, 2], value_type='vel', title='Galaxy 2 Velocity', cbar_label='km/s')
plot_velocity_or_sigma_map_from_csv(csv_vel_sig_files[3], ax=axs[0, 3], value_type='sig', title='Galaxy 2 Sigma', cbar_label='km/s')

# Row 1
plot_velocity_or_sigma_map_from_csv(csv_vel_sig_files[4], ax=axs[1, 0], value_type='vel', title='Galaxy 3 Velocity', cbar_label='km/s')
plot_velocity_or_sigma_map_from_csv(csv_vel_sig_files[5], ax=axs[1, 1], value_type='sig', title='Galaxy 3 Sigma', cbar_label='km/s')
plot_velocity_or_sigma_map_from_csv(csv_vel_sig_files[6], ax=axs[1, 2], value_type='vel', title='Galaxy 4 Velocity', cbar_label='km/s')
plot_velocity_or_sigma_map_from_csv(csv_vel_sig_files[7], ax=axs[1, 3], value_type='sig', title='Galaxy 4 Sigma', cbar_label='km/s')

# Row 2
plot_velocity_or_sigma_map_from_csv(csv_vel_sig_files[8], ax=axs[2, 0], value_type='vel', title='Galaxy 5 Velocity', cbar_label='km/s')
plot_velocity_or_sigma_map_from_csv(csv_vel_sig_files[9], ax=axs[2, 1], value_type='sig', title='Galaxy 5 Sigma', cbar_label='km/s')
plot_velocity_or_sigma_map_from_csv(csv_vel_sig_files[10], ax=axs[2, 2], value_type='vel', title='Galaxy 6 Velocity', cbar_label='km/s')
plot_velocity_or_sigma_map_from_csv(csv_vel_sig_files[11], ax=axs[2, 3], value_type='sig', title='Galaxy 6 Sigma', cbar_label='km/s')

# Final layout adjustment and display
plt.tight_layout()
plt.show()
























#---------------------------------------------------------------------------














