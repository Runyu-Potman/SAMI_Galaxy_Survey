import numpy as np
import matplotlib.pyplot as plt
from SAMI_kinematcs import plot_vel_or_sig
import cmasher as cmr
from matplotlib.colors import ListedColormap, BoundaryNorm
from astropy.io import fits
from matplotlib.ticker import AutoMinorLocator
import matplotlib.patches as patches
import matplotlib.patheffects as pe
################################################################################
def plot_nii_spatial(ax, Ha_fits_path, Hb_fits_path, OIII_fits_path, NII_fits_path,
                     threshold, scale=0.5, fontsize=12, cbar_pad=0.02,
                     bar_fraction=0.0467, labelpad_x=8, labelpad_y=0.85, bpt_AGN = False):
    """
    Load the Hα, Hβ, [OIII] and [NII] maps, apply quality cuts,
    compute the [NII]-BPT classification, and plot the spatial classification
    map on the given Axes.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axis on which to draw the spatial map.
    Ha_fits_path, Hb_fits_path, OIII_fits_path, NII_fits_path : str
        Paths to the FITS files (each should contain primary map in [0] and error in [1]).
    threshold : float
        SNR threshold for quality cut.
    scale : float, optional
        Pixel scale in arcsec/pixel. Default 0.5.
    fontsize : int, optional
        Font size for labels. Default 10.
    cbar_pad, bar_fraction : float, optional
        Colorbar padding and fraction. Default 0.02, 0.0466.
    labelpad_x, labelpad_y : float, optional
        Label padding. Default 8, 0.85.

    Returns:
    --------
    classification : numpy.ndarray (2D, int)
        Classification array: 0=SF, 1=Composite, 2=AGN.
    x_coords, y_coords : numpy.ndarray
        Coordinate arrays in arcsec (same shape as classification).
    """

    # -------------------- Load maps --------------------
    with fits.open(Ha_fits_path) as Ha:
        Ha_map = Ha[0].data[0, :, :]   # total component
        Ha_err = Ha[1].data[0, :, :]
    with fits.open(Hb_fits_path) as Hb:
        Hb_map = Hb[0].data
        Hb_err = Hb[1].data
    with fits.open(OIII_fits_path) as OIII:
        OIII_map = OIII[0].data
        OIII_err = OIII[1].data
    with fits.open(NII_fits_path) as NII:
        NII_map = NII[0].data
        NII_err = NII[1].data

    # -------------------- Mask invalid and negative --------------------
    Ha_map = np.ma.masked_invalid(Ha_map)
    Ha_err = np.ma.masked_invalid(Ha_err)
    Hb_map = np.ma.masked_invalid(Hb_map)
    Hb_err = np.ma.masked_invalid(Hb_err)
    OIII_map = np.ma.masked_invalid(OIII_map)
    OIII_err = np.ma.masked_invalid(OIII_err)
    NII_map = np.ma.masked_invalid(NII_map)
    NII_err = np.ma.masked_invalid(NII_err)

    # Negative flux
    Ha_map = np.ma.masked_where(Ha_map < 0, Ha_map)
    Hb_map = np.ma.masked_where(Hb_map < 0, Hb_map)
    OIII_map = np.ma.masked_where(OIII_map < 0, OIII_map)
    NII_map = np.ma.masked_where(NII_map < 0, NII_map)

    # Non-positive errors
    Ha_err = np.ma.masked_where(Ha_err <= 0, Ha_err)
    Hb_err = np.ma.masked_where(Hb_err <= 0, Hb_err)
    OIII_err = np.ma.masked_where(OIII_err <= 0, OIII_err)
    NII_err = np.ma.masked_where(NII_err <= 0, NII_err)

    # -------------------- SNR and threshold --------------------
    Ha_SNR = Ha_map / Ha_err
    Hb_SNR = Hb_map / Hb_err
    OIII_SNR = OIII_map / OIII_err
    NII_SNR = NII_map / NII_err

    Ha_map = np.ma.masked_where(Ha_SNR <= threshold, Ha_map)
    Hb_map = np.ma.masked_where(Hb_SNR <= threshold, Hb_map)
    OIII_map = np.ma.masked_where(OIII_SNR <= threshold, OIII_map)
    NII_map = np.ma.masked_where(NII_SNR <= threshold, NII_map)

    # -------------------- Combined mask for NII BPT --------------------
    combined_mask = np.ma.getmask(Ha_map)
    combined_mask = np.ma.mask_or(combined_mask, np.ma.getmask(Ha_err))
    combined_mask = np.ma.mask_or(combined_mask, np.ma.getmask(Hb_map))
    combined_mask = np.ma.mask_or(combined_mask, np.ma.getmask(Hb_err))
    combined_mask = np.ma.mask_or(combined_mask, np.ma.getmask(OIII_map))
    combined_mask = np.ma.mask_or(combined_mask, np.ma.getmask(OIII_err))
    combined_mask = np.ma.mask_or(combined_mask, np.ma.getmask(NII_map))
    combined_mask = np.ma.mask_or(combined_mask, np.ma.getmask(NII_err))

    Ha_masked = np.ma.masked_array(Ha_map, mask=combined_mask)
    Hb_masked = np.ma.masked_array(Hb_map, mask=combined_mask)
    OIII_masked = np.ma.masked_array(OIII_map, mask=combined_mask)
    NII_masked = np.ma.masked_array(NII_map, mask=combined_mask)

    # -------------------- Compute log ratios --------------------
    # Use masked arrays directly (they keep the original shape)
    log_NII_Ha = np.log10(NII_masked / Ha_masked)
    log_OIII_Hb = np.log10(OIII_masked / Hb_masked)

    # -------------------- Define BPT boundaries --------------------
    def boundary_1(x, clip=False):
        if clip:
            x = np.clip(x, -1.274, 0.469)
        return (0.61 / (x - 0.47)) + 1.19   # Kewley et al. 2006

    def boundary_2(x, clip=False):
        if clip:
            x = np.clip(x, -1.274, 0.05)
        return (0.61 / (x - 0.05)) + 1.3    # Kewley et al. 2006

    # -------------------- Classify pixels --------------------
    classification = np.zeros_like(log_OIII_Hb, dtype=int)   # 0 = SF (will not be coloured)
    sf = (log_OIII_Hb < boundary_1(log_NII_Ha, clip=True)) & (log_OIII_Hb < boundary_2(log_NII_Ha, clip=True))
    comp = (log_OIII_Hb >= boundary_2(log_NII_Ha, clip=True)) & (log_OIII_Hb < boundary_1(log_NII_Ha, clip=True))
    agn = (log_OIII_Hb >= boundary_1(log_NII_Ha, clip=True))

    classification[sf] = 0
    classification[comp] = 1
    classification[agn] = 2

    # Mask out invalid regions (where any map was masked)
    classification = np.ma.masked_where(combined_mask, classification)

    # -------------------- Build coordinate arrays (arcsec) --------------------
    center_x, center_y = 24.5, 24.5   # fixed as in the original
    y_coords = (np.arange(Ha_masked.shape[0]) - center_y) * scale
    x_coords = (np.arange(Ha_masked.shape[1]) - center_x) * scale

    # -------------------- Plot on the given axis --------------------
    if bpt_AGN:
        cmap = ListedColormap(['salmon', 'purple'])  # Comp=1, AGN=2 (SF=0 not shown)
        bounds = [1, 2, 3]  # only colours for Comp and AGN
    else:
        cmap = ListedColormap(['gray', 'salmon'])   # Comp=1, AGN=2 (SF=0 not shown)
        bounds = [0, 1, 2]                            # only colours for Comp and AGN

    norm = BoundaryNorm(bounds, cmap.N)

    im = ax.imshow(classification, origin='lower', cmap=cmap, norm=norm,
                   extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]])

    ax.set_xlim([-12.5, 12.5])
    ax.set_ylim([-12.5, 12.5])
    ax.set_xticks(np.arange(-10, 11, 5))
    ax.set_yticks(np.arange(-10, 11, 5))
    ax.set_xlabel('Offset (arcsec)', fontsize=fontsize, labelpad=labelpad_x)
    ax.set_ylabel('Offset (arcsec)', fontsize=fontsize, labelpad=labelpad_y)
    ax.set_title('Resolved [NII]-BPT', fontsize=fontsize)

    # Ticks
    ax.tick_params(axis='both', which='major', length=4, width=1, direction='in')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis='both', which='minor', length=2, width=1, direction='in')

    # Colorbar (attached to this axis)
    cbar = plt.gcf().colorbar(im, ax=ax, boundaries=bounds, ticks=[],
                               pad=cbar_pad, fraction=bar_fraction)
    cbar.ax.tick_params(axis='y', which='both', length=0)
    cbar.ax.text(1.6, 1.5, 'Comp', va='center', ha='left', rotation=90,
                 fontsize=fontsize, color='black')
    if bpt_AGN:
        cbar.ax.text(1.6, 2.5, 'AGN', va='center', ha='left', rotation=90,
                     fontsize=fontsize, color='black')
    else:
        cbar.ax.text(1.6, 0.5, 'SF', va='center', ha='left', rotation=90,
                 fontsize=fontsize, color='black')

    return classification, x_coords, y_coords

#####################################################################################
def slurm_job_combine(base_dir, center_x = 24.5, center_y = 24.5):
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
    - age_std_array: standard deviation of age array.
    - metal_std_array: standard deviation of metal array.
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

    # this is for making the spatially resolved maps.
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

    # radius in gradient plot relative to the galaxy center.
    # transfer from pixel scale into arcsec scale.
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
    age_std_map = age_std_map[valid]
    metal_std_map = metal_std_map[valid]

    print('data point in the gradient plot:', np.sum(valid))

    age_array = age_map.copy()
    metal_array = metal_map.copy()
    age_std_array = age_std_map.copy()
    metal_std_array = metal_std_map.copy()

    return age_full, metal_full, age_array, metal_array, age_std_array, metal_std_array, r_all
#-----------------------------------------------------------------------------------
def plot_age_and_Z(axs_x, age_full, metal_full, r_all, age_array, metal_array, age_std_array, metal_std_array,
                   label_pad = 0.85, bar_fraction = 0.0467, bar_pad = 0.02, fontsize = 12,
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

    # colorbar setting.
    if vmin_age is None:
        vmin_age = np.nanmin(10 ** (age_full - 9))
    if vmax_age is None:
        vmax_age = np.nanmax(10 ** (age_full - 9))

    # age map (the first column).
    im = axs[axs_x, 2].imshow(10 ** (age_full - 9), origin = 'lower', aspect = 'equal',
                          cmap = cmap_1_2, extent = [-12.5, 12.5, -12.5, 12.5],
                          vmin = vmin_age, vmax = vmax_age)

    # set ticks (first column).
    axs[axs_x, 2].set_xlim([-12.5, 12.5])
    axs[axs_x, 2].set_ylim([-12.5, 12.5])

    tick_locs = np.arange(-10, 11, 5)
    axs[axs_x, 2].set_xticks(tick_locs)
    axs[axs_x, 2].set_yticks(tick_locs)

    # set labels (first column).
    axs[axs_x, 2].set_xlabel('Offset (arcsec)', fontsize = fontsize, labelpad = 8)
    axs[axs_x, 2].set_ylabel('Offset (arcsec)', fontsize = fontsize, labelpad = label_pad)

    # set color bar (first column).
    cbar = plt.colorbar(im, ax = axs[axs_x, 2], fraction = bar_fraction, pad = bar_pad)
    cbar.ax.yaxis.set_tick_params(direction = 'in')
    cbar.set_label('Age (Gyr)', fontsize = fontsize, labelpad = 4)

    # set titles (first column).
    if title:
        axs[axs_x, 2].set_title('Resolved Stellar Age', fontsize = fontsize)

    # make these (major) ticks longer.
    axs[axs_x, 2].tick_params(axis = 'both', which = 'major', length = 4, width = 1, direction = 'in')

    # add minor ticks (shorter, no labels).
    axs[axs_x, 2].xaxis.set_minor_locator(AutoMinorLocator())
    axs[axs_x, 2].yaxis.set_minor_locator(AutoMinorLocator())
    axs[axs_x, 2].tick_params(axis = 'both', which = 'minor', length = 2, width = 1, direction = 'in')

    # color bar setting.
    if vmin_z is None:
        vmin_z = np.nanmin(metal_full)
    if vmax_z is None:
        vmax_z = np.nanmax(metal_full)
    '''
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
    '''
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
#####################################################################################
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
    ax.set_xlabel(r'Radius (arcsec)', fontsize = 12, labelpad = 4)
    if name is not None:
       ax.set_ylabel(f'Enclosed Mass ($M_{{\odot}}$)', fontsize = 12, labelpad = 0.85)
    else:
       ax.set_ylabel(r'Enclosed Mass ($M_{\odot}$)', fontsize = 12, labelpad = 0.85)

    # position of the y axis offset text.
    offset_text = ax.yaxis.get_offset_text()
    offset_text.set_position((-0.012, 0.95))
    offset_text.set_fontsize(12)

    # x and y axis tick settings.
    ax.tick_params(labelsize = 10, direction = 'in', top = False)
    # make the tick to have the highest zorder.
    ax.set_axisbelow(False)

    '''
    # Twin axis for kpc.
    ax2 = ax.twiny()
    ax2.set_xlim(xrange * arctpc / 1000.0)
    ax2.xaxis.set_major_locator(plt.MaxNLocator(nbins = 4))
    ax2.set_xlabel(r'$r$ (kpc)', fontsize = 12)
    ax2.tick_params(labelsize = 10, direction = 'in')
    ax2.set_axisbelow(False)
    '''

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
       ax.legend(loc = 'upper left', bbox_to_anchor = (0.01, 0.99), fontsize = 8)

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
#####################################################################################
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

    ax.tick_params(direction = 'in', labelsize = 10)
    ax.set_xlabel('Radius (arcsec)', fontsize = 12, labelpad = 8)
    if name is not None:
        ax.set_ylabel(f'Circularity $\lambda_{{z}}$', fontsize = 12, labelpad = 0.85)
    else:
        ax.set_ylabel(r'Circularity $\lambda_{z}$', fontsize = 12, labelpad = 0.85)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])

    # Add horizontal colorbar at the top of the axis.
    if add_cbar:
        cb = fig.colorbar(im, ax = ax,
                          pad = 0.02, fraction = 0.0467)
        cb.ax.tick_params(direction = 'in', labelsize = 10)
        cb.set_label('Relative Orbit Density', labelpad = 5, fontsize = 12)

    # Draw dashed lines for ocut values.
    for cut in ocut:
        ax.axhline(cut, color = 'black', linestyle = '--', linewidth = 1, xmin = 0, xmax = 1)

    if text:
        boundaries = [1.0, ocut[0], ocut[1], ocut[2], ocut[3], -1.0]
        labels = ['Cold', 'Warm', 'Hot', 'CR-warm', 'CR-cold']
        x_text = extent[0] + 0.68 * (extent[1] - extent[0])
        for i in range(len(labels)):
            y_center = (boundaries[i] + boundaries[i + 1]) / 2
            txt = ax.text(x_text, y_center, labels[i], ha = 'left', va = 'center',
                          fontsize = 9, color = 'black')
            # Add white outline.
            txt.set_path_effects([pe.withStroke(linewidth = 3, foreground = 'white')])

    # a dotted line to represent the radius of the kinematically distinct component.
    if r_kdc is not None:
       ax.axvline(r_kdc, color = 'orange', linestyle = 'dotted', linewidth = 2.0)

    if created_fig:
        plt.tight_layout()

    return fig

######################################################################################
# 300787 gas kinematic output.
gas_output_file_300787_kinematics = '300787/kinematic/300787_quality_cut_gas_velocity_map.csv'

# 300787 stellar kinematic output.
star_output_file_300787_kinematics = '300787/kinematic/300787_quality_cut_stellar_velocity_map.csv'

# --------------------------------------------------------------------------
fig, axs = plt.subplots(2, 3, figsize=(10, 6))
fig.suptitle('Galaxy 300787 Which Hosts a Counter-rotating Disk', fontsize = 18)

# colormap consistent with DYNAMITE.
vel_cmap = cmr.get_sub_cmap('twilight_shifted', 0.05, 0.95)
sig_cmap = cmr.get_sub_cmap('twilight_shifted', 0.05, 0.6)

# directly copied.
vel_cmap_300787_star = cmr.get_sub_cmap('twilight_shifted', 0.1, 1.0)

# 300787 plotting.

plot_vel_or_sig(csv_path=gas_output_file_300787_kinematics, cmap=vel_cmap, cbar_label=r'$V_\mathrm{gas}$ (km/s)',
                value_type='vel', ax=axs[0, 1], PAs=[127.5], line_length=10, plot_psf=True, psffwhm=1.941)


# directly copied from stellar kinematic.
plot_vel_or_sig(csv_path=star_output_file_300787_kinematics, value_type='vel', ax=axs[0, 0], cmap=vel_cmap_300787_star,
                cbar_label=r'$V_{\bigstar}$ (km/s)',
                plot_psf=True, fontsize=12, psffwhm=1.941, vmin=-130, PAs=[149.12, -53.95], line_length=[2, 7.5])

plot_vel_or_sig(csv_path = star_output_file_300787_kinematics, value_type = 'sig', ax = axs[1, 0], cmap = sig_cmap,
                cbar_label = r'$\sigma_{\bigstar}$ (km/s)', plot_psf = True, fontsize = 12, psffwhm = 1.941)

axs[0, 0].set_title('Stellar Velocity', fontsize=10)
axs[0, 1].set_title('Stellar Velocity Dispersion', fontsize=10)
axs[0, 2].set_title('Ionized Gas Velocity', fontsize=10)

# 300787 gas pa.
axs[0, 2].text(-1, 7.8, r'$\mathrm{PA}_\mathrm{\,gas}$=$127.5^\circ$', color='black', fontsize=10, ha='left')

# 300787 stellar pa.
axs[0, 0].text(-1.5, -3.8, r'$\mathrm{PA}_\bigstar$=$149.12^\circ$', color='black', fontsize=10, ha='left')
axs[0, 0].text(-2, 5.5, r'$\mathrm{PA}_\bigstar$=$-53.95^\circ$', color='black', fontsize=10, ha='left')

##########################################################################################
Ha_fits_path = '300787/emission_line/300787_A_Halpha_default_recom-comp.fits'
Hb_fits_path = '300787/emission_line/300787_A_Hbeta_default_recom-comp.fits'
OIII_fits_path = '300787/emission_line/300787_A_OIII5007_default_recom-comp.fits'
NII_fits_path = '300787/emission_line/300787_A_NII6583_default_recom-comp.fits'
classification, x_coords, y_coords = plot_nii_spatial(
    ax = axs[0, 3],
    Ha_fits_path=Ha_fits_path,
    Hb_fits_path=Hb_fits_path,
    OIII_fits_path=OIII_fits_path,
    NII_fits_path=NII_fits_path,
    threshold=5.0,
    scale=0.5
)

#########################################################
base_dir = '300787/age_z'
age_300787, metal_300787, age_300787_array, metal_300787_array, age_300787_std, metal_300787_std, r_all_300787 = slurm_job_combine(
        base_dir)

plot_age_and_Z(axs_x = 1, age_full = age_300787, metal_full = metal_300787, r_all = r_all_300787,
                   age_array = age_300787_array,
                   metal_array = metal_300787_array, age_std_array = age_300787_std, metal_std_array = metal_300787_std,
                   r_dash = 2.5, title = True
                   )

############################################################
fits_filename_300787 = '300787/dynamite/dynamite_fits/enclosed_mass_profiles.fits'
reproduce_mass_plot(fits_filename_300787, ax=axs[1, 2], name='300787', r_kdc=2.5, extrap_start=5.37, label=True)

axs[1, 2].set_title('Mass Profile', fontsize=10)

fits_orbit_300787 = '300787/dynamite/dynamite_fits/orbit_density.fits'
reproduce_orbit_plot(fits_orbit_300787, ax = axs[1, 3], name = '300787', r_kdc = 2.5, text = True)
axs[1, 3].set_title('Orbit Distribution', fontsize=10)

#############################################################
add_psf(ax = axs[0, 3], psffwhm = 1.941)
add_psf(ax = axs[1, 0], psffwhm = 1.941)
add_psf(ax = axs[1, 1], psffwhm = 1.941)
# ---------------------------------------------------------------------------
# same axis ratio.
axs[0, 0].set_box_aspect(1)
axs[0, 1].set_box_aspect(1)
axs[0, 2].set_box_aspect(1)
axs[0, 3].set_box_aspect(1)

axs[1, 0].set_box_aspect(1)
axs[1, 1].set_box_aspect(1)
axs[1, 2].set_box_aspect(1)
axs[1, 3].set_box_aspect(1)
#-------------------------------------------------------------------------
plt.tight_layout(h_pad=0, w_pad=0)
plt.savefig('final/poster_300787.png', dpi=300, bbox_inches='tight')
plt.show()