import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from SAMI_stellar_velocity_quality_cut_functions import quality_cut_stellar_velocity_map_csv
import matplotlib.patches as patches
from PIL import Image
import cmasher as cmr
#------------------------------------------------------------------
def plot_vel_or_sig(csv_path, cmap = 'RdBu_r', cbar_label = 'km/s', value_type = 'vel',
                    show_colorbar = True, fontsize = 10, bar_fraction = 0.0467, bar_pad = 0.02, label_pad = 0.85,
                    ax = None, vmin = None, vmax = None, title = None, csv_path_uncut = None,
                    background_alpha = 0.3, PAs = None, line_length = 12.5, line_styles = None,
                    plot_psf = False, psffwhm = 2, text_ul = None):
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
            cbar.set_label(cbar_label, fontsize = fontsize, labelpad = 4)

    if title is not None:
        ax.set_title(title, fontsize = fontsize)

    # make these (major) ticks longer.
    ax.tick_params(axis = 'both', which = 'major', length = 4, width = 1, direction = 'in')

    # add minor ticks (shorter, no labels).
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis = 'both', which = 'minor', length = 2, width = 1, direction = 'in')

    # if PAs is provided, plot multiple lines.
    if PAs is not None:
        # allow line_length to be either a scalar (applies to all PAs) or an iterable with same length as PAs.
        if np.isscalar(line_length):
            lengths = [float(line_length)] * len(PAs)
        else:
            lengths = list(line_length)
            if len(lengths) != len(PAs):
                raise ValueError('If line_length is iterable it must have the same length as PAs.')

        # if user gives multiple styles, use them; otherwise default to solid
        if line_styles is None:
            line_styles = ['-'] * len(PAs)
        elif isinstance(line_styles, str):
            line_styles = [line_styles] * len(PAs)
        elif len(line_styles) != len(PAs):
            raise ValueError('line_styles must be a string or a list with the same length as PAs.')

        for PA, ll, ls in zip(PAs, lengths, line_styles):
            # convert pa from degrees to radians.
            PA_rad = np.deg2rad(PA + 90)

            # calculate the line coordinates based on the position angle.
            x0, y0 = 0, 0
            x1 = x0 + ll * np.cos(PA_rad)
            y1 = y0 + ll * np.sin(PA_rad)

            x2 = x0 - ll * np.cos(PA_rad)
            y2 = y0 - ll * np.sin(PA_rad)

            ax.plot([x1, x2], [y1, y2], color = 'lightgreen', lw = 2, linestyle = ls)

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
def add_circle(ax, img, radius = 7.5, linewidth = 1, scale = 0.4, label = False, tick_num = 11, E_bar = 5):
    '''
    Add a circle in the e.g. SDSS optical image to show e.g. the field of view of SAMI.
    Optionally, add direction, scale bar label: '|-|', and border ticks to the image. Note that the input
    optical image cut should be a square image and the width (w) and height of the image should
    satisfy: w = scale_bar (in arcsec) / pixel_scale (0.4''/pixel by default) * (tick_num - 1) for best visualization.

    Parameters:
    - ax: plot position, e.g., axs[0, 0].
    - img: SDSS optical jpg image path.
    - radius: radius of the circle in arcsec. Default is 7.5'' for SAMI FoV.
    - linewidth: line width of the circle and the scale bar.
    - scale: pixel scale of the sdss optical image (0.4''/pixel by default).
    - label: whether to add other features or not (scale label, compass, and ticks).
    - tick_num: number of ticks to add to the image. Default is 11.
    - E_bar: sometimes the label 'E' can not align perfectly with the horizontal line, so we adjust manually.

    Returns:
    - None

    '''

    # radius of the circle in pixel scale.
    radius_pix = radius / scale

    # the shape of the optical image in pixel scale.
    width, height = img.size

    if width != height:
        raise ValueError('Image must be square with the same width and height.')

    center_x = width / 2
    center_y = height / 2

    # add a circle showing FoV.
    circle = patches.Circle((center_x, center_y), radius_pix,
                            edgecolor = 'white', facecolor = 'none',
                            linestyle = '--', linewidth = linewidth)
    ax.add_patch(circle)

    # optionally, add a label showing the scale of the pixel size, border tick labels showing the extent, and a compass showing the direction.
    if label:
        # choose scale automatically.
        bar_pix = width * (1 / (tick_num - 1))
        bar_arcsec = bar_pix * scale

        # round to nice number (10,20,50...).
        bar_arcsec = round(bar_arcsec / 10) * 10
        bar_pix = bar_arcsec / scale

        # the place to put the scale bar.
        x0 = width * (1 / (tick_num - 1))
        y0 = height * (1 / (tick_num - 1))

        # the half-length of the vertical cap for the scale bar.
        cap = y0 * (1 / (tick_num - 1))

        # horizontal bar.
        ax.plot([x0, x0 + bar_pix], [y0, y0], color = 'white', lw = linewidth)

        # vertical caps  |-|.
        ax.plot([x0, x0], [y0 - cap, y0 + cap], color = 'white', lw = linewidth)
        ax.plot([x0 + bar_pix, x0 + bar_pix], [y0 - cap, y0 + cap], color = 'white', lw = linewidth)

        # text label.
        ax.text(x0 + bar_pix/1.8, y0,
                f'{int(bar_arcsec)}"', color = 'white',
                ha = 'center', va = 'bottom', fontsize = 10)

        # add tick labels (assume the input image is a square).
        ticks = np.linspace(0, width - 1, tick_num, dtype = int)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        ax.tick_params(axis = 'both', which = 'both', direction = 'in',
                       color = 'white', length = 4, width = 1,
                       top = True, right = True,
                       labelbottom = False, labeltop = False,
                       labelleft = False, labelright = False)

        # ensure square aspect so ticks look symmetric for square images.
        ax.set_aspect('equal', adjustable = 'box')

        # add compass.
        ax.plot([center_x, center_x], [center_y - bar_pix, center_y - bar_pix * 3],
                color = 'white', lw = linewidth, solid_capstyle = 'butt')
        ax.text(center_x, center_y - bar_pix * 3.2, 'N', color = 'white',
                ha = 'center', va = 'bottom', fontsize = 10)

        ax.plot([center_x - bar_pix * 3, center_x - bar_pix], [center_y, center_y],
                color = 'white', lw = linewidth, solid_capstyle = 'butt')
        ax.text(center_x - bar_pix * 3.2, center_y + E_bar, 'E', color = 'white',
                ha = 'right', va = 'center', fontsize = 10)

#-------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # 7969 stellar kinematic files.
    star_vel_7969 = '7969/kinematic/7969_A_stellar-velocity_default_two-moment.fits'
    star_sig_7969 = '7969/kinematic/7969_A_stellar-velocity-dispersion_default_two-moment.fits'
    star_output_file_7969 = '7969/kinematic/7969_quality_cut_stellar_velocity_map.csv'

    # 7969 stellar quality cut.
    quality_cut_stellar_velocity_map_csv(star_vel_7969, star_sig_7969, star_output_file_7969, pixel_to_arc = True)
    # -----------------------------------------------------------------------------------------
    # 143287 stellar kinematic files.
    star_vel_143287 = '143287/kinematic/143287_A_stellar-velocity_default_two-moment.fits'
    star_sig_143287 = '143287/kinematic/143287_A_stellar-velocity-dispersion_default_two-moment.fits'
    star_output_file_143287 = '143287/kinematic/143287_quality_cut_stellar_velocity_map.csv'

    # 143287 stellar quality cut.
    quality_cut_stellar_velocity_map_csv(star_vel_143287, star_sig_143287, star_output_file_143287, pixel_to_arc = True)
    #-----------------------------------------------------------------------------------------
    # 227266 stellar kinematic files.
    star_vel_227266 = '227266/kinematic/227266_A_stellar-velocity_default_two-moment.fits'
    star_sig_227266 = '227266/kinematic/227266_A_stellar-velocity-dispersion_default_two-moment.fits'
    star_output_file_227266 = '227266/kinematic/227266_quality_cut_stellar_velocity_map.csv'

    # 227266 stellar quality cut.
    quality_cut_stellar_velocity_map_csv(star_vel_227266, star_sig_227266, star_output_file_227266, pixel_to_arc = True)
    # -------------------------------------------------------------------------------------------
    # 230776 stellar kinematic files.
    star_vel_230776 = '230776/kinematic/230776_A_stellar-velocity_default_two-moment.fits'
    star_sig_230776 = '230776/kinematic/230776_A_stellar-velocity-dispersion_default_two-moment.fits'
    star_output_file_230776 = '230776/kinematic/230776_quality_cut_stellar_velocity_map.csv'

    # 230776 stellar quality cut.
    quality_cut_stellar_velocity_map_csv(star_vel_230776, star_sig_230776, star_output_file_230776, pixel_to_arc = True)
    # ------------------------------------------------------------------------------------------------
    # 9239900248 stellar kinematic files.
    star_vel_9239900248 = '9239900248/kinematic/9239900248_A_stellar-velocity_default_two-moment.fits'
    star_sig_9239900248 = '9239900248/kinematic/9239900248_A_stellar-velocity-dispersion_default_two-moment.fits'
    star_output_file_9239900248 = '9239900248/kinematic/9239900248_quality_cut_stellar_velocity_map.csv'

    # 9239900248 stellar quality cut.
    quality_cut_stellar_velocity_map_csv(star_vel_9239900248, star_sig_9239900248, star_output_file_9239900248, pixel_to_arc = True)
    #-------------------------------------------------------------------------------------------------
    # the total plot.
    fig, axs = plt.subplots(5, 3, figsize=(10, 15))
    # introduce a new colormap consistent with DYNAMITE.
    vel_cmap = cmr.get_sub_cmap('twilight_shifted', 0.05, 0.95)
    vel_cmap_7969 = cmr.get_sub_cmap('twilight_shifted', 0, 0.85)
    vel_cmap_230776 = cmr.get_sub_cmap('twilight_shifted', 0, 0.70)

    sig_cmap = cmr.get_sub_cmap('twilight_shifted', 0.05, 0.6)
    # ------------------------------------------------------------------------------------------
    # 7969 optical.
    optical_7969 = Image.open('7969/optical/7969_optical_image.jpg')
    axs[0, 0].imshow(optical_7969)
    axs[0, 0].set_ylabel('Galaxy 7969', fontsize = 10, labelpad = 0.85)
    add_circle(axs[0, 0], optical_7969, linewidth = 1, label = True, E_bar = 1.6)
    axs[0, 0].set_title('SDSS Optical Image', fontsize = 10)

    # 7969 kinematics.
    plot_vel_or_sig(csv_path = star_output_file_7969, value_type = 'vel', ax = axs[0, 1], cmap = vel_cmap_7969, cbar_label= 'Velocity (km/s)', plot_psf = True, fontsize = 10, psffwhm = 1.561, vmin = -80, vmax = 80)
    plot_vel_or_sig(csv_path = star_output_file_7969, value_type = 'sig', ax = axs[0, 2], cmap = sig_cmap, cbar_label = 'Velocity Dispersion (km/s)', plot_psf = True, fontsize = 10, psffwhm = 1.561, vmin = 130, vmax = 290)
    axs[0, 1].set_title('Stellar Velocity', fontsize = 10)
    axs[0, 2].set_title('Stellar Velocity Dispersion', fontsize = 10)

    # ------------------------------------------------------------------------------------------
    # 143287 optical.
    optical_143287 = Image.open('143287/optical/143287_optical_image.jpg')
    axs[1, 0].imshow(optical_143287)
    axs[1, 0].set_ylabel('Galaxy 143287', fontsize = 10, labelpad = 0.85)
    add_circle(axs[1, 0], optical_143287, linewidth = 1, label = True, E_bar = 1.5)

    # 143287 kinematics.
    plot_vel_or_sig(csv_path = star_output_file_143287, value_type = 'vel', ax = axs[1, 1], cmap = vel_cmap_143287, cbar_label = 'Velocity (km/s)', plot_psf = True, fontsize = 10, psffwhm = 2.250)
    plot_vel_or_sig(csv_path = star_output_file_143287, value_type = 'sig', ax = axs[1, 2], cmap = sig_cmap_143287, cbar_label = 'Velocity Dispersion (km/s)', plot_psf = True, fontsize = 10, psffwhm = 2.250, vmin = 55, vmax = 165)
    #--------------------------------------------------------------------------------------------
    # 227266 optical.
    optical_227266 = Image.open('227266/optical/227266_optical_image.jpg')
    axs[2, 0].imshow(optical_227266)
    axs[2, 0].set_ylabel('Galaxy 227266', fontsize = 10, labelpad = 0.85)
    add_circle(axs[2, 0], optical_227266, linewidth = 1, label = True, E_bar = 3.3)

    # 227266 kinematics.
    plot_vel_or_sig(csv_path = star_output_file_227266, value_type = 'vel', ax = axs[2, 1], cmap = vel_cmap, cbar_label = 'Velocity (km/s)', plot_psf = True, fontsize = 10, psffwhm = 2.108, vmin = -55, vmax = 55)
    plot_vel_or_sig(csv_path = star_output_file_227266, value_type = 'sig', ax = axs[2, 2], cmap = sig_cmap, cbar_label = 'Velocity Dispersion (km/s)', plot_psf = True, fontsize = 10, psffwhm = 2.108)
    # ------------------------------------------------------------------------------------------
    # 230776 optical.
    optical_230776 = Image.open('230776/optical/230776_optical_image.jpg')
    axs[3, 0].imshow(optical_230776)
    axs[3, 0].set_ylabel('Galaxy 230776', fontsize = 10, labelpad = 0.85)
    add_circle(axs[3, 0], optical_230776, linewidth = 1, label = True, E_bar = 6.5)

    # 230776 kinematics.
    plot_vel_or_sig(csv_path = star_output_file_230776, value_type = 'vel', ax = axs[3, 1], cmap = vel_cmap_230776, cbar_label = 'Velocity (km/s)', plot_psf = True, fontsize = 10, psffwhm = 2.144, vmin = -38, vmax = 38)
    plot_vel_or_sig(csv_path = star_output_file_230776, value_type = 'sig', ax = axs[3, 2], cmap = sig_cmap, cbar_label = 'Velocity Dispersion (km/s)', plot_psf = True, fontsize = 10, psffwhm = 2.144)
    # --------------------------------------------------------------------------------------------
    # same axis ratio.
    axs[0, 0].set_box_aspect(1)
    axs[0, 1].set_box_aspect(1)
    axs[0, 2].set_box_aspect(1)
    axs[1, 0].set_box_aspect(1)
    axs[1, 1].set_box_aspect(1)
    axs[1, 2].set_box_aspect(1)
    axs[2, 0].set_box_aspect(1)
    axs[2, 1].set_box_aspect(1)
    axs[2, 2].set_box_aspect(1)
    axs[3, 0].set_box_aspect(1)
    axs[3, 1].set_box_aspect(1)
    axs[3, 2].set_box_aspect(1)
    axs[4, 0].set_box_aspect(1)
    axs[4, 1].set_box_aspect(1)
    axs[4, 2].set_box_aspect(1)
    #---------------------------------------------------
    plt.tight_layout(h_pad = 0.85, w_pad = 0.85)
    plt.savefig('final/stellar_kinematic_image.png', dpi = 1000, bbox_inches = 'tight')
    plt.show()