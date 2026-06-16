import numpy as np
import matplotlib.pyplot as plt
from SAMI_kinematcs import plot_vel_or_sig
import cmasher as cmr
from matplotlib.colors import ListedColormap, BoundaryNorm
from astropy.io import fits
from matplotlib.ticker import AutoMinorLocator
import matplotlib.patches as patches

################################################################################
def plot_nii_spatial(ax, Ha_fits_path, Hb_fits_path, OIII_fits_path, NII_fits_path,
                     threshold, scale=0.5, fontsize=10, cbar_pad=0.02,
                     bar_fraction=0.0467, labelpad_x=8, labelpad_y=0.85):
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
    cmap = ListedColormap(['salmon', 'purple'])   # Comp=1, AGN=2 (SF=0 not shown)
    bounds = [1, 2, 3]                            # only colours for Comp and AGN
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
    cbar.ax.text(1.6, 2.5, 'AGN', va='center', ha='left', rotation=90,
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










galaxy = '300787'
# ----------------------------------------------------------------------------
# 227266 gas kinematic output.
gas_output_file_227266_kinematics = '227266/kinematic/227266_quality_cut_gas_velocity_map.csv'

# 227266 stellar kinematic output.
star_output_file_227266_kinematics = '227266/kinematic/227266_quality_cut_stellar_velocity_map.csv'

# 300787 gas kinematic output.
gas_output_file_300787_kinematics = '300787/kinematic/300787_quality_cut_gas_velocity_map.csv'

# 300787 stellar kinematic output.
star_output_file_300787_kinematics = '300787/kinematic/300787_quality_cut_stellar_velocity_map.csv'

# --------------------------------------------------------------------------
fig, axs = plt.subplots(2, 4, figsize=(40 / 3, 6))

# colormap consistent with DYNAMITE.
vel_cmap = cmr.get_sub_cmap('twilight_shifted', 0.05, 0.95)
sig_cmap = cmr.get_sub_cmap('twilight_shifted', 0.05, 0.6)

# directly copied.
vel_cmap_300787_star = cmr.get_sub_cmap('twilight_shifted', 0.1, 1.0)
'''
# 227266 plotting.
# we should mask the particular spaxel!

plot_vel_or_sig(csv_path=gas_output_file_227266_kinematics, cmap=vel_cmap, cbar_label=r'$V_\mathrm{gas}$ (km/s)',
                value_type='vel', ax=axs[1, 1], PAs=[17], line_length=9, plot_psf=True, psffwhm=2.108)

# directly copied from stellar kinematics.
plot_vel_or_sig(csv_path=star_output_file_227266_kinematics, value_type='vel', ax=axs[1, 3], cmap=vel_cmap,
                cbar_label=r'$V_{\bigstar}$ (km/s)',
                plot_psf=True, fontsize=10, psffwhm=2.108, vmin=-55, vmax=55, PAs=[171.75, 34.53], line_length=[3, 10],
                pa_center_x=0.75, pa_center_y=0.75)
'''
# 300787 plotting.

plot_vel_or_sig(csv_path=gas_output_file_300787_kinematics, cmap=vel_cmap, cbar_label=r'$V_\mathrm{gas}$ (km/s)',
                value_type='vel', ax=axs[0, 2], PAs=[127.5], line_length=10, plot_psf=True, psffwhm=1.941)

# directly copied from stellar kinematic.
plot_vel_or_sig(csv_path=star_output_file_300787_kinematics, value_type='vel', ax=axs[0, 0], cmap=vel_cmap_300787_star,
                cbar_label=r'$V_{\bigstar}$ (km/s)',
                plot_psf=True, fontsize=10, psffwhm=1.941, vmin=-130, PAs=[149.12, -53.95], line_length=[2, 7.5])

plot_vel_or_sig(csv_path = star_output_file_300787_kinematics, value_type = 'sig', ax = axs[0, 1], cmap = sig_cmap,
                cbar_label = r'$\sigma_{\bigstar}$ (km/s)', plot_psf = True, fontsize = 10, psffwhm = 1.941)


axs[0, 2].set_title('Gas Velocity', fontsize=10)

axs[0, 0].set_title('Stellar Velocity', fontsize=10)
axs[0, 1].set_title('Stellar Velocity Dispersion', fontsize=10)



'''
axs[1, 1].text(-4, 9.4, r'$\mathrm{PA}_\mathrm{\,gas}$=$-163^\circ$', color='black', fontsize=10, ha='left')
axs[1, 3].text(-11, -4.4, r'$\mathrm{PA}_\bigstar$=$171.75^\circ$', color='black', fontsize=10, ha='left')
axs[1, 3].text(-6.4, 9.8, r'$\mathrm{PA}_\bigstar$=$34.53^\circ$', color='black', fontsize=10, ha='left')
'''


# 300787 gas pa.
axs[0, 2].text(-1, 7.8, r'$\mathrm{PA}_\mathrm{\,gas}$=$127.5^\circ$', color='black', fontsize=10, ha='left')

# 300787 stellar pa.
axs[0, 0].text(-1.5, -3.8, r'$\mathrm{PA}_\bigstar$=$149.12^\circ$', color='black', fontsize=10, ha='left')
axs[0, 0].text(-2, 5.5, r'$\mathrm{PA}_\bigstar$=$-53.95^\circ$', color='black', fontsize=10, ha='left')



##########################################################################################
'''
Ha_fits_path = '227266/emission_line/227266_A_Halpha_default_recom-comp.fits'
Hb_fits_path = '227266/emission_line/227266_A_Hbeta_default_recom-comp.fits'
OIII_fits_path = '227266/emission_line/227266_A_OIII5007_default_recom-comp.fits'
NII_fits_path = '227266/emission_line/227266_A_NII6583_default_recom-comp.fits'
classification, x_coords, y_coords = plot_nii_spatial(
    ax = axs[0, 3],
    Ha_fits_path=Ha_fits_path,
    Hb_fits_path=Hb_fits_path,
    OIII_fits_path=OIII_fits_path,
    NII_fits_path=NII_fits_path,
    threshold=5.0,
    scale=0.5
)

'''
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

plt.tight_layout(h_pad=-8, w_pad=0.6)
plt.savefig('final/poster.png', dpi=300, bbox_inches='tight')
plt.show()