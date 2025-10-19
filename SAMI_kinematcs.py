import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from SAMI_stellar_velocity_quality_cut_functions import quality_cut_stellar_velocity_map_csv, quality_cut_gaseous_velocity_map_csv
import matplotlib.patches as patches
from PIL import Image
#---------------------------------------------------------------
def plot_vel_or_sig(csv_path, cmap = 'jet', cbar_label = 'km/s', value_type = 'vel',
                    show_colorbar = True, fontsize = 10, bar_fraction = 0.046, bar_pad = 0.04, label_pad = 0.85,
                    ax = None, vmin = None, vmax = None, title = None, csv_path_uncut = None,
                    background_alpha = 0.3, PAs = None, line_length = 12.5, plot_psf = False, psffwhm = 2, text_ul = None):
    """
    Plot velocity or sigma map from CSV file on a given matplotlib axis.

    Parameters:
    - csv_path: str, path to the CSV file (must include 'x_arcsec', 'y_arcsec', 'vel', 'sig').
    - cmap: colormap for imshow.
    - cbar_label: label for colorbar.
    - value_type: 'vel' or 'sig' — determines which column to plot.
    - show_colorbar: whether to display colorbar.
    - fontsize: fontsize for x, y, and colorbar label.
    - bar_fraction: bar fraction for colorbar.
    - bar_pad: bar padding for colorbar.
    - label_pad: x and y label padding.
    - ax: matplotlib.axes.Axes object. If None, creates a new figure.
    - vmin: colorbar min value.
    - vmax: colorbar max value.
    - title: add title.
    - csv_path_uncut: path to CSV file without applying quality cuts.
    - background_alpha: alpha transparency for background layer.
    - PAs: list of position angles (in degree, counterclockwise from North, 0 to 360).
    - line_length: length of the line to plot, in arcsec.
    - plot_psf: whether to plot the PSF or not.
    - psffwhm: psf value.
    - text_ul: text to use for PAs.

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
            x_grid = np.argmin(np.abs(x_values - row['x_arcsec']))
            y_grid = np.argmin(np.abs(y_values - row['y_arcsec']))
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

    ax.set_xlabel('Offset (arcsec)', fontsize = fontsize, labelpad = 8)
    ax.set_ylabel('Offset (arcsec)', fontsize = fontsize, labelpad = label_pad)

    # colorbar.
    if show_colorbar:
        cbar = plt.colorbar(im, ax = ax, fraction = bar_fraction, pad = bar_pad)
        cbar.ax.yaxis.set_tick_params(length = 4, width = 1, direction = 'in')
        cbar.ax.tick_params(labelsize = fontsize)
        if cbar_label:
            cbar.set_label(cbar_label, fontsize = fontsize, labelpad = label_pad)

    if title:
        ax.set_title(title)

    # make these (major) ticks longer.
    ax.tick_params(axis = 'both', which = 'major', length = 4, width = 1, direction = 'in')

    # add minor ticks (shorter, no labels).
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis = 'both', which = 'minor', length = 2, width = 1, direction = 'in')

    # if PAs is provided, plot multiple lines.
    if PAs is not None:
        for PA in PAs:
            # convert pa from degrees to radians.
            PA_rad = np.deg2rad(PA + 90)

            # calculate the line coordinates based on the position angle.
            x0, y0 = 0, 0
            x1 = x0 + line_length * np.cos(PA_rad)
            y1 = y0 + line_length * np.sin(PA_rad)

            x2 = x0 - line_length * np.cos(PA_rad)
            y2 = y0 - line_length * np.sin(PA_rad)

            ax.plot([x1, x2], [y1, y2], color = 'black', lw = 2)

    if plot_psf:
        radius = psffwhm / 2
        circle = patches.Circle(
            (-10, -10), # position of the circle in arcsec
            radius,
            edgecolor = 'black', # color of the circle's border
            facecolor = 'none', # no fill color
            linewidth = 1.5, # thickness of the circle's edge
            linestyle = '-'
        )

        ax.add_patch(circle)

    # add optional upper-left corner text.
    if text_ul:
        ax.text(0.05, 0.95, text_ul, transform = ax.transAxes, fontsize = fontsize,
                horizontalalignment = 'left', verticalalignment = 'top')


    return ax

#---------------------------------------------------------------------------------------
def add_circle(ax, img, linewidth = 1.5):
    '''
    Add a circle in the e.g. SDSS optical image to show e.g. the field of view of SAMI.

    Parameters:
    - ax: plot position, e.g., axs[0, 0].
    - img: SDSS optical image path.
    - linewidth: line width of the circle.

    Returns:
    - None

    '''
    scale = 0.4  # arcsec/pixel
    radius = 15 / 2  # arcsec
    radius_pix = radius / scale  # pixel

    width, height = img.size
    center_x = width / 2
    center_y = height / 2

    circle = patches.Circle((center_x, center_y), radius_pix,
                            edgecolor = 'white', facecolor = 'none',
                            linestyle = '--', linewidth = linewidth)
    ax.add_patch(circle)








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














