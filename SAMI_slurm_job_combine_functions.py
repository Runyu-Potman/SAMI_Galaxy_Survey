import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from matplotlib.ticker import AutoMinorLocator
import matplotlib.patches as patches

def slurm_job_combine(base_dir, center_x = 25, center_y = 25):
    '''
    combine the slurm jobs, do preparation for plotting.

    Parameters:
    - base_dir: slurm jon result directory.
    - center_x: center of galaxy.
    - center_y: center of galaxy.

    Returns:
    - age_full: full age map.
    - metal_full: full metal map.
    - age_array: age array for gradient.
    - metal_array: metal array for gradient.
    - r_all: radius in gradient plot relative to galaxy center.
    '''

    # initialize the full maps for mean and standard deviation.
    age_map = np.full((50, 50), np.nan)
    metal_map = np.full_like(age_map, np.nan)

    age_std_map = np.full_like(age_map, np.nan)
    metal_std_map = np.full_like(age_map, np.nan)

    # loop over the tasks (job_id: 0 - 24).
    for job_id in range(25):
        # load the partial maps (mean and standard deviation).
        age_part = np.load(f'{base_dir}/output/age_map_part_{job_id}.npy')
        metal_part = np.load(f'{base_dir}/output/metal_map_part_{job_id}.npy')

        age_std_part = np.load(f'{base_dir}/output/age_std_part_{job_id}.npy')
        metal_std_part = np.load(f'{base_dir}/output/metal_std_part_{job_id}.npy')

        # calculate the slice indices for inserting the data.
        start_y = job_id * 2
        end_y = start_y + age_part.shape[0]

        # insert the partial maps into full maps.
        age_map[start_y:end_y, :] = age_part
        metal_map[start_y:end_y, :] = metal_part

        age_std_map[start_y:end_y, :] = age_std_part
        metal_std_map[start_y:end_y, :] = metal_std_part

    age_full = age_map.copy()
    metal_full = metal_map.copy()

    np.save(f'{base_dir}/age_map_full.npy', age_map)
    np.save(f'{base_dir}/metal_map_full.npy', metal_map)
    np.save(f'{base_dir}/age_std_map_full.npy', age_std_map)
    np.save(f'{base_dir}/metal_std_map_full.npy', metal_std_map)

    # begin making gradient plots.
    x_bar = np.load(f'{base_dir}/x_bar.npy')
    y_bar = np.load(f'{base_dir}/y_bar.npy')

    # shift the center back.
    x_bar = x_bar + center_x
    y_bar = y_bar + center_y

    # find the nearest pixel.
    x_bar = np.rint(x_bar).astype(int)
    y_bar = np.rint(y_bar).astype(int)

    r_all = np.sqrt((x_bar - center_x)** 2 + (y_bar - center_y) ** 2) * 0.5

    age_map = age_map[y_bar, x_bar]
    metal_map = metal_map[y_bar, x_bar]
    age_std_map = age_std_map[y_bar, x_bar]
    metal_std_map = metal_std_map[y_bar, x_bar]

    # mask invalid data.
    valid = ~np.isnan(age_map) & ~np.isnan(metal_map) & ~np.isnan(age_std_map) & ~np.isnan(metal_std_map)
    r_all = r_all[valid]
    age_map = age_map[valid]
    metal_map = metal_map[valid]

    print('data point in the gradient plot:', np.sum(valid))

    age_array = age_map.copy()
    metal_array = metal_map.copy()

    return age_full, metal_full, age_array, metal_array, r_all
#-----------------------------------------------------------------------------------
def plot_age_and_Z(axs_x, age_full, metal_full, r_all, age_array, metal_array,
                   label_pad = 0.85, bar_fraction = 0.0485, bar_pad = 0.02, fontsize = 10,
                   r_dash = None, vmin_age = None, vmax_age = None, vmin_z = None, vmax_z = None,
                   name = None, title = False, cmap_1_2 = 'RdYlBu_r', cmap_3_4 = 'viridis'):
    '''
    Plot the spatially resolved age and metal maps and gradient plots for three galaxies.

    Paramters:
    - axs_x: column of plot.
    - age_full: full age map from slurm_job_combine function.
    - metal_full: full metal map from slurm_job_combine function.
    - r_all: radius in gradient plot relative to galaxy center from slurm_job_combine function.
    - age_array: age array for gradient from slurm_job_combine function.
    - metal_array: metal array from slurm_job_combine function.
    - label_pad: label pad.
    - bar_fraction: bar fraction for colorbar.
    - bar_pad: bar pad for colorbar.
    - fontsize: font size.
    - r_dash: a vertical dash line indicating the radius of the KDC.
    - vmin_age: age map vmin.
    - vmax_age: age map vmax.
    - vmin_z: metal map vmin.
    - vmax_z: metal map vmax.
    - name: galaxy name.
    - title: plot title.
    - cmap_1_2: color map for spatially resolved maps.
    - cmap_3_4: color map for gradient plots.

    Returns:
    - None.
    '''

    if vmin_age is None:
        vmin_age = np.nanmin(10 ** (age_full - 9))
    if vmax_age is None:
        vmax_age = np.nanmax(10 ** (age_full - 9))

    # age map (the first column).
    im = axs[axs_x, 0].imshow(10 ** (age_full - 9), origin = 'lower', aspect = 'equal',
                          cmap = cmap_1_2, extent = [-12.5, 12.5, -12.5, 12.5],
                          vmin = vmin_age, vmax = vmax_age)

    # set ticks (first column).
    axs[axs_x, 0].set_xlim([-12.5, 12.5])
    axs[axs_x, 0].set_ylim([-12.5, 12.5])

    tick_locs = np.arange(-10, 11, 5)
    axs[axs_x, 0].set_xticks(tick_locs)
    axs[axs_x, 0].set_yticks(tick_locs)

    # set labels (first column).
    axs[axs_x, 0].set_xlabel('Offset (arcsec)', fontsize = fontsize, labelpad = 8)
    axs[axs_x, 0].set_ylabel(f'{name}\nOffset (arcsec)', fontsize = fontsize, labelpad = label_pad)

    # set color bar (first column).
    cbar = plt.colorbar(im, ax = axs[axs_x, 0], fraction = bar_fraction, pad = bar_pad)
    cbar.ax.yaxis.set_tick_params(direction = 'in')
    cbar.set_label('Age(Gyr)', fontsize = fontsize, labelpad = 4)

    # set titles (first column).
    if title:
        axs[axs_x, 0].set_title('Resolved Stellar Age', fontsize = fontsize)

    # make these (major) ticks longer.
    axs[axs_x, 0].tick_params(axis = 'both', which = 'major', length = 4, width = 1, direction = 'in')

    # add minor ticks (shorter, no labels).
    axs[axs_x, 0].xaxis.set_minor_locator(AutoMinorLocator())
    axs[axs_x, 0].yaxis.set_minor_locator(AutoMinorLocator())
    axs[axs_x, 0].tick_params(axis = 'both', which = 'minor', length = 2, width = 1, direction = 'in')

    if vmin_z is None:
        vmin_z = np.nanmin(metal_full)
    if vmax_z is None:
        vmax_z = np.nanmax(metal_full)

    # [M/H] map (second column).
    im = axs[axs_x, 1].imshow(metal_full, origin = 'lower', aspect = 'equal',
                          cmap = cmap_1_2, extent = [-12.5, 12.5, -12.5, 12.5], vmin = vmin_z, vmax = vmax_z)

    # set ticks.
    axs[axs_x, 1].set_xlim([-12.5, 12.5])
    axs[axs_x, 1].set_ylim([-12.5, 12.5])

    tick_locs = np.arange(-10, 11, 5)
    axs[axs_x, 1].set_xticks(tick_locs)
    axs[axs_x, 1].set_yticks(tick_locs)

    # set labels.
    axs[axs_x, 1].set_xlabel('Offset (arcsec)', fontsize = fontsize, labelpad = 8)
    axs[axs_x, 1].set_ylabel('Offset (arcsec)', fontsize = fontsize, labelpad = label_pad)

    # set color bar.
    cbar = plt.colorbar(im, ax = axs[axs_x, 1], fraction = bar_fraction, pad = bar_pad)
    cbar.ax.yaxis.set_tick_params(direction = 'in')
    cbar.set_label('[M/H]', fontsize = fontsize, labelpad = 4)

    if title:
        # set titles.
        axs[axs_x, 1].set_title('Resolved Stellar Metallicity', fontsize = fontsize)

    # make these (major) ticks longer.
    axs[axs_x, 1].tick_params(axis = 'both', which = 'major', length = 4, width = 1, direction = 'in')

    # add minor ticks (shorter, no labels).
    axs[axs_x, 1].xaxis.set_minor_locator(AutoMinorLocator())
    axs[axs_x, 1].yaxis.set_minor_locator(AutoMinorLocator())
    axs[axs_x, 1].tick_params(axis = 'both', which = 'minor', length = 2, width = 1, direction = 'in')

    # age gradient (third column).
    im = axs[axs_x, 2].scatter(r_all, 10 ** (age_array - 9), c = r_all, cmap = cmap_3_4, s = 10, alpha = 0.7)

    # set ticks.
    axs[axs_x, 2].set_xlim([0, 8.5])
    axs[axs_x, 2].set_ylim([0, 14])

    axs[axs_x, 2].set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8])
    axs[axs_x, 2].set_yticks([0, 2, 4, 6, 8, 10, 12, 14])

    # set labels.
    axs[axs_x, 2].set_xlabel('Radius (arcsec)', fontsize = fontsize, labelpad = 8)
    axs[axs_x, 2].set_ylabel('Age (Gyr)', fontsize = fontsize, labelpad = label_pad)

    if r_dash is not None:
        axs[axs_x, 2].axvline(r_dash, color = 'gray', linestyle = '--', linewidth = 1)

    if title:
        # set titles.
        axs[axs_x, 2].set_title('Stellar Age Gradient', fontsize = fontsize)

    # make these (major) ticks longer.
    axs[axs_x, 2].tick_params(axis = 'both', which = 'major', length = 4, width = 1, direction = 'in')

    # add minor ticks (shorter, no labels).
    axs[axs_x, 2].xaxis.set_minor_locator(AutoMinorLocator())
    axs[axs_x, 2].yaxis.set_minor_locator(AutoMinorLocator())
    axs[axs_x, 2].tick_params(axis = 'both', which = 'minor', length = 2, width = 1, direction = 'in')

    # metal gradient (fourth column).
    im = axs[axs_x, 3].scatter(r_all, metal_array, c = r_all, cmap = cmap_3_4, s = 10, alpha = 0.7)

    # set ticks.
    axs[axs_x, 3].set_xlim([0, 8.5])
    axs[axs_x, 3].set_ylim([-1.5, 0.5])

    axs[axs_x, 3].set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8])
    axs[axs_x, 3].set_yticks([-1.5, -1, -0.5, 0, 0.5])

    # set labels.
    axs[axs_x, 3].set_xlabel('Radius (arcsec)', fontsize = fontsize, labelpad = 8)
    axs[axs_x, 3].set_ylabel('[M/H]', fontsize = fontsize, labelpad = label_pad)

    if r_dash is not None:
        axs[axs_x, 3].axvline(r_dash, color = 'gray', linestyle = '--', linewidth = 1)

    if title:
        # set titles.
        axs[axs_x, 3].set_title('Stellar Metallicity Gradient', fontsize = fontsize)

    # make these (major) ticks longer.
    axs[axs_x, 3].tick_params(axis = 'both', which = 'major', length = 4, width = 1, direction = 'in')

    # add minor ticks (shorter, no labels).
    axs[axs_x, 3].xaxis.set_minor_locator(AutoMinorLocator())
    axs[axs_x, 3].yaxis.set_minor_locator(AutoMinorLocator())
    axs[axs_x, 3].tick_params(axis = 'both', which = 'minor', length = 2, width = 1, direction = 'in')
#-------------------------------------------------------------------------
def add_psf(ax, psffwhm):
    '''
    Add a psf circle in the left corner.

    Parameters:
    - ax: matplotlib axes.
    - psffwhm: psf_fwhm in arcsec.

    Returns:
    - None

    '''

    # radius in arcsec.
    radius = psffwhm / 2

    # add a circle showing PSF.
    circle = patches.Circle(
        (-10, -10),  # position of the circle in arcsec
        radius,
        edgecolor = 'black',  # color of the circle's border
        facecolor = 'none',  # no fill color
        linewidth = 1.5,  # thickness of the circle's edge
        linestyle = '-'
    )

    ax.add_patch(circle)
#-------------------------------------------------------------------------------------