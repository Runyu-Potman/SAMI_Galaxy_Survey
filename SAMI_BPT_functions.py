'''
This script can be used to create the Baldwin, Phillips & Terlevich empirical diagnostic diagrams
for SAMI galaxies using the optical line ratios.
'''

import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from photutils.aperture import CircularAperture, aperture_photometry
from SAMI_kinematcs import plot_vel_or_sig
from SAMI_stellar_velocity_quality_cut_functions import quality_cut_gaseous_velocity_map_csv
from matplotlib.ticker import AutoMinorLocator
from pafit.fit_kinematic_pa import fit_kinematic_pa
#---------------------------------------------------------------------------------------------------------------------
def gas_distribution(gas_fits_path, output_file = None, threshold = None, dust_correction = False, dust_fits = None,
                     csv = True, x_center = 25, y_center = 25, scale = 0.5, log_flux = False):
    '''
    This function do the preparation for creating the gas distribution (flux intensity) plot for the SAMI galaxies by
    doing the quality cut (e.g., S/N <= 5) and correcting for dust using the dust correction maps provided by SAMI.
    The assumed pixel scale is 0.5 arcsec/pixel in the output csv file.

    Parameters:
    - gas_fits_path: gas fits file path.
    - output_file: output csv file path when setting csv = True.
    - threshold: threshold for quality cut, spaxels <= threshold will be excluded.
    - dust_correction: whether to apply dust correction or not, if dust_correction = True, then the dust correction fits
                       file should be provided (SAMI provides dust correction maps).
    - dust_fits: dust fits file path when setting dust_correction = True.
    - csv: whether to output csv file, the csv file consists of x_arcsec, y_arcsec, gas flux value, and gas flux error value.
    - x_center: x center of gas map in pixel.
    - y_center: y center of gas map in pixel.
    - scale: pixel scale (e.g., 0.5 arcsec/pixel for SAMI galaxy survey).
    - log_flux: whether to log flux values or not (for plotting).

    Returns:
    - gas_data: gas flux (after quality cut and/or dust corrected) data.
    - gas_err_data: error in gas flux (after quality cut and/or dust corrected).
    - mask: mask map.

    '''

    # load the optical emission line maps (primary map[0] and error map [1]).
    with fits.open(gas_fits_path) as gas_map:
        gas_data = gas_map[0].data
        gas_err_data = gas_map[1].data

    # extract the total component (0) of Ha (50*50*4 -> 50*50) when using the recommended component.
    # for the 1-component case, [0, :, :] is the same as [1, :, :].
    gas_data = gas_data[0, :, :]
    gas_err_data = gas_err_data[0, :, :]

    # mask NaN values.
    gas_data = np.ma.masked_invalid(gas_data)
    gas_err_data = np.ma.masked_invalid(gas_err_data)

    # mask negative flux and error.
    gas_data = np.ma.masked_where(gas_data < 0, gas_data)
    gas_err_data = np.ma.masked_where(gas_err_data <= 0, gas_err_data)

    # calculate the signal-to-noise ratio (SNR) for each emission line.
    '''
    be noticed that any pixel where err_map is masked will result in the corresponding 
    ratio also being masked, even if primary_map at that pixel is valid.
    '''
    if threshold is not None:

        gas_SNR = gas_data / gas_err_data

        print(f'Gas_SNR: min = {np.min(gas_SNR)}, max = {np.max(gas_SNR)}.')

        # mask data points where SNR is below a specific threshold.
        gas_data = np.ma.masked_where(gas_SNR <= threshold, gas_data)
        gas_err_data = np.ma.masked_where(gas_SNR <= threshold, gas_err_data)

    if dust_correction and dust_fits is None:
        raise ValueError('Please provide the dust fits file when setting dust correction = True.')

    if dust_correction and dust_fits is not None:
        with fits.open(dust_fits) as dust:
            dust_data = dust[0].data

        # mask invalid spaxels in the dust map.
        dust_data = np.ma.masked_invalid(dust_data)

        # correct for dust and get the intrinsic flux value.
        gas_data = gas_data * dust_data
        gas_err_data = gas_err_data * dust_data

    if log_flux:
        gas_data = np.ma.masked_where(gas_data == 0, gas_data)
        gas_data = np.log10(gas_data)

    # get the mask for return.
    mask = np.ma.getmask(gas_data)

    if csv and output_file is None:
        raise ValueError('Please provide the output file path when setting csv = True.')

    if csv and output_file is not None:
        # prepare the csv data for plotting.
        ny, nx = gas_data.shape

        data_to_save = []

        for i in range(ny):
            for j in range(nx):
                if dust_correction:
                    if (not gas_data.mask[i, j] and not gas_err_data.mask[i, j] and not dust_data.mask[i, j]):
                        x_arcsec = (j - x_center) * scale
                        y_arcsec = (i - y_center) * scale

                        print(f'{x_arcsec}, {y_arcsec}, {gas_data[i, j]}, {gas_err_data[i, j]}')
                        data_to_save.append((x_arcsec, y_arcsec, gas_data[i, j], gas_err_data[i, j]))

                else:
                    if (not gas_data.mask[i, j] and not gas_err_data.mask[i, j]):
                        x_arcsec = (j - x_center) * scale
                        y_arcsec = (i - y_center) * scale

                        print(f'{x_arcsec}, {y_arcsec}, {gas_data[i, j]}, {gas_err_data[i, j]}')
                        data_to_save.append((x_arcsec, y_arcsec, gas_data[i, j], gas_err_data[i, j]))

        with open(output_file, 'w') as f:
            f.write('x_arcsec,y_arcsec,gas,gas_err\n')
            for entry in data_to_save:
                f.write(f'{entry[0]},{entry[1]},{entry[2]},{entry[3]}\n')

    return gas_data, gas_err_data, mask

#-----------------------------------------------------------------------------------------------------------------
def bpt_plot(Ha_map_clean, ax, log_x, log_y, center_x = 25, center_y = 25, scale = 0.5):
    '''
    Function to plot the BPT with data points color coded by the distance to the galaxy center.

    Parameters:
    - Ha_map_clean: The cleaned Halpha map for extracting shape and mask.
    - ax: The axes on which the plot will be drawn.
    - log_x: log(emission/Halpha).
    - log_y: log(OIII/Hbeta)
    - center_x: Center of the galaxy.
    - center_y: Center of the galaxy.

    Returns:
    - sc: The BPT plot.
    - center_x: Center of the galaxy.
    - center_y: Center of the galaxy.
    - scale: Pixel scale.

    '''
    y, x = np.meshgrid(np.arange(Ha_map_clean.shape[0]), np.arange(Ha_map_clean.shape[1]))
    x = x[~Ha_map_clean.mask]
    y = y[~Ha_map_clean.mask]

    # here we show the distance in arcsec.
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2) * scale

    # normalize distance.
    #normalized_distance = (distance - np.min(distance)) / (np.max(distance) - np.min(distance))

    # plot the BPT diagram.
    sc = ax.scatter(log_x, log_y, c = distance, cmap = plt.cm.coolwarm,
                    s = 8, alpha = 1) # s: marker size

    return sc, center_x, center_y

#-------------------------------------------------------------------------------------
def bpt(
        Ha_fits_path, Hb_fits_path, OIII_fits_path, OI_fits_path,
        SII_6716_fits_path, SII_6731_fits_path, NII_fits_path,
        threshold, fontsize = 10, cbar_pad = 0.02, bar_fraction = 0.0468,
        labelpad_x = 8, labelpad_y = 0.85, labelpad_cbar = 4, scale = 0.5
):
    '''
    Do the quality cut on emission line maps and make the BPT diagram.

    Parameters:
    - Ha_fits_path: Path to the Halpha fits file.
    - Hb_fits_path: Path to the Hbeta fits file.
    - OIII_fits_path: Path to the OIII fits file.
    - OI_fits_path: Path to the OI fits file.
    - SII_6716_fits_path: Path to the SII 6716 fits file.
    - SII_6731_fits_path: Path to the SII 6731 fits file.
    - NII_fits_path: Path to the NII fits file.
    - threshold: Threshold for applying the quality cut on emission line maps.
    - fontsize: Font size for labels.
    - cbar_pad: Pad for the colorbar.
    - bar_fraction: Bar fraction for the colorbar.
    - labelpad_x: Pad for the labels.
    - labelpad_y: Pad for the labels.
    - labelpad_cbar: Pad for the colorbar labels.
    - scale: Pixel scale.

    Returns:
    - None

    '''

    # load the optical emission line maps (primary map[0] and error map [1]) for each line.
    with fits.open(Ha_fits_path) as Ha:
        Ha_map = Ha[0].data
        Ha_err = Ha[1].data

    with fits.open(Hb_fits_path) as Hb:
        Hb_map = Hb[0].data
        Hb_err = Hb[1].data

    with fits.open(OIII_fits_path) as OIII:
        OIII_map = OIII[0].data
        OIII_err = OIII[1].data

    with fits.open(OI_fits_path) as OI:
        OI_map = OI[0].data
        OI_err = OI[1].data

    with fits.open(SII_6716_fits_path) as SII_6716:
        SII_6716_map = SII_6716[0].data
        SII_6716_err = SII_6716[1].data

    with fits.open(SII_6731_fits_path) as SII_6731:
        SII_6731_map = SII_6731[0].data
        SII_6731_err = SII_6731[1].data

    with fits.open(NII_fits_path) as NII:
        NII_map = NII[0].data
        NII_err = NII[1].data

    # extract the total component (0) of Ha (50*50*4 -> 50*50).
    Ha_map = Ha_map[0, :, :]
    Ha_err = Ha_err[0, :, :]

    # mask NaN values in all 14 maps.
    Ha_map = np.ma.masked_invalid(Ha_map)
    Ha_err = np.ma.masked_invalid(Ha_err)

    Hb_map = np.ma.masked_invalid(Hb_map)
    Hb_err = np.ma.masked_invalid(Hb_err)

    OIII_map = np.ma.masked_invalid(OIII_map)
    OIII_err = np.ma.masked_invalid(OIII_err)

    OI_map = np.ma.masked_invalid(OI_map)
    OI_err = np.ma.masked_invalid(OI_err)

    SII_6716_map = np.ma.masked_invalid(SII_6716_map)
    SII_6716_err = np.ma.masked_invalid(SII_6716_err)

    SII_6731_map = np.ma.masked_invalid(SII_6731_map)
    SII_6731_err = np.ma.masked_invalid(SII_6731_err)

    NII_map = np.ma.masked_invalid(NII_map)
    NII_err = np.ma.masked_invalid(NII_err)

    # mask negative flux and error spaxels in all 14 maps.
    Ha_map = np.ma.masked_where(Ha_map < 0, Ha_map)
    Hb_map = np.ma.masked_where(Hb_map < 0, Hb_map)
    OIII_map = np.ma.masked_where(OIII_map < 0, OIII_map)
    OI_map = np.ma.masked_where(OI_map < 0, OI_map)
    SII_6716_map = np.ma.masked_where(SII_6716_map < 0, SII_6716_map)
    SII_6731_map = np.ma.masked_where(SII_6731_map < 0, SII_6731_map)
    NII_map = np.ma.masked_where(NII_map < 0, NII_map)

    Ha_err = np.ma.masked_where(Ha_err <= 0, Ha_err)
    Hb_err = np.ma.masked_where(Hb_err <= 0, Hb_err)
    OIII_err = np.ma.masked_where(OIII_err <= 0, OIII_err)
    OI_err = np.ma.masked_where(OI_err <= 0, OI_err)
    SII_6716_err = np.ma.masked_where(SII_6716_err <= 0, SII_6716_err)
    SII_6731_err = np.ma.masked_where(SII_6731_err <= 0, SII_6731_err)
    NII_err = np.ma.masked_where(NII_err <= 0, NII_err)

    # calculate the signal-to-noise ratio (SNR) for each emission line.
    '''
    be noticed that any pixel where err_map is masked will result in the corresponding 
    ratio also being masked, even if primary_map at that pixel is valid.
    '''

    Ha_SNR = Ha_map / Ha_err
    Hb_SNR = Hb_map / Hb_err
    OIII_SNR = OIII_map / OIII_err
    OI_SNR = OI_map / OI_err
    SII_6716_SNR = SII_6716_map / SII_6716_err
    SII_6731_SNR = SII_6731_map / SII_6731_err
    NII_SNR = NII_map / NII_err

    print(f'Ha_SNR: min = {np.min(Ha_SNR)}, max = {np.max(Ha_SNR)}.')
    print(f'Hb_SNR: min = {np.min(Hb_SNR)}, max = {np.max(Hb_SNR)}.')
    print(f'OIII_SNR: min = {np.min(OIII_SNR)}, max = {np.max(OIII_SNR)}.')
    print(f'OI_SNR: min = {np.min(OI_SNR)}, max = {np.max(OI_SNR)}.')
    print(f'SII_6716_SNR: min = {np.min(SII_6716_SNR)}, max = {np.max(SII_6716_SNR)}.')
    print(f'SII_6731_SNR: min = {np.min(SII_6731_SNR)}, max = {np.max(SII_6731_SNR)}.')
    print(f'NII_SNR: min = {np.min(NII_SNR)}, max = {np.max(NII_SNR)}.')

    # mask data points where SNR is below a specific threshold.
    Ha_map = np.ma.masked_where(Ha_SNR <= threshold, Ha_map)
    Hb_map = np.ma.masked_where(Hb_SNR <= threshold, Hb_map)
    OIII_map = np.ma.masked_where(OIII_SNR <= threshold, OIII_map)
    OI_map = np.ma.masked_where(OI_SNR <= threshold, OI_map)
    SII_6716_map = np.ma.masked_where(SII_6716_SNR <= threshold, SII_6716_map)
    SII_6731_map = np.ma.masked_where(SII_6731_SNR <= threshold, SII_6731_map)
    NII_map = np.ma.masked_where(NII_SNR <= threshold, NII_map)

    '''
    For the first BPT plot ([NII]/Ha verse [OIII]/Hb), if a data point is invalid (NaN or SNR < threshold)
    in any of the emission line maps, it should be excluded from all maps.

    Note that when creating the first BPT plot, the mask from the second and the third plots should not be
    included, similar for the second plot and the third plot. Therefore, we have different combined_mask 
    for different plots. 
    '''

    # combined_mask_NII is the combined mask for the first BPT plot.
    combined_mask_NII = np.ma.getmask(Ha_map)
    combined_mask_NII = np.ma.mask_or(combined_mask_NII, np.ma.getmask(Ha_err))

    combined_mask_NII = np.ma.mask_or(combined_mask_NII, np.ma.getmask(Hb_map))
    combined_mask_NII = np.ma.mask_or(combined_mask_NII, np.ma.getmask(Hb_err))

    combined_mask_NII = np.ma.mask_or(combined_mask_NII, np.ma.getmask(OIII_map))
    combined_mask_NII = np.ma.mask_or(combined_mask_NII, np.ma.getmask(OIII_err))

    combined_mask_NII = np.ma.mask_or(combined_mask_NII, np.ma.getmask(NII_map))
    combined_mask_NII = np.ma.mask_or(combined_mask_NII, np.ma.getmask(NII_err))

    # combined_mask_SII is the combined mask for the second BPT plot.
    combined_mask_SII = np.ma.getmask(Ha_map)
    combined_mask_SII = np.ma.mask_or(combined_mask_SII, np.ma.getmask(Ha_err))

    combined_mask_SII = np.ma.mask_or(combined_mask_SII, np.ma.getmask(Hb_map))
    combined_mask_SII = np.ma.mask_or(combined_mask_SII, np.ma.getmask(Hb_err))

    combined_mask_SII = np.ma.mask_or(combined_mask_SII, np.ma.getmask(OIII_map))
    combined_mask_SII = np.ma.mask_or(combined_mask_SII, np.ma.getmask(OIII_err))

    combined_mask_SII = np.ma.mask_or(combined_mask_SII, np.ma.getmask(SII_6716_map))
    combined_mask_SII = np.ma.mask_or(combined_mask_SII, np.ma.getmask(SII_6716_err))

    combined_mask_SII = np.ma.mask_or(combined_mask_SII, np.ma.getmask(SII_6731_map))
    combined_mask_SII = np.ma.mask_or(combined_mask_SII, np.ma.getmask(SII_6731_err))

    # combined_mask_OI is the combined mask for the third BPT plot.
    combined_mask_OI = np.ma.getmask(Ha_map)
    combined_mask_OI = np.ma.mask_or(combined_mask_OI, np.ma.getmask(Ha_err))

    combined_mask_OI = np.ma.mask_or(combined_mask_OI, np.ma.getmask(Hb_map))
    combined_mask_OI = np.ma.mask_or(combined_mask_OI, np.ma.getmask(Hb_err))

    combined_mask_OI = np.ma.mask_or(combined_mask_OI, np.ma.getmask(OIII_map))
    combined_mask_OI = np.ma.mask_or(combined_mask_OI, np.ma.getmask(OIII_err))

    combined_mask_OI = np.ma.mask_or(combined_mask_OI, np.ma.getmask(OI_map))
    combined_mask_OI = np.ma.mask_or(combined_mask_OI, np.ma.getmask(OI_err))

    # apply the combined mask across all maps to ensure consistency.
    # the first BPT plot.
    Ha_map_NII = np.ma.masked_array(Ha_map, mask = combined_mask_NII)
    Hb_map_NII = np.ma.masked_array(Hb_map, mask = combined_mask_NII)
    OIII_map_NII = np.ma.masked_array(OIII_map, mask = combined_mask_NII)
    NII_map_NII = np.ma.masked_array(NII_map, mask = combined_mask_NII)

    # the second BPT plot.
    Ha_map_SII = np.ma.masked_array(Ha_map, mask = combined_mask_SII)
    Hb_map_SII = np.ma.masked_array(Hb_map, mask = combined_mask_SII)
    OIII_map_SII = np.ma.masked_array(OIII_map, mask = combined_mask_SII)
    SII_6716_map_SII = np.ma.masked_array(SII_6716_map, mask = combined_mask_SII)
    SII_6731_map_SII = np.ma.masked_array(SII_6731_map, mask = combined_mask_SII)

    # the third BPT plot.
    Ha_map_OI = np.ma.masked_array(Ha_map, mask = combined_mask_OI)
    Hb_map_OI = np.ma.masked_array(Hb_map, mask = combined_mask_OI)
    OIII_map_OI = np.ma.masked_array(OIII_map, mask = combined_mask_OI)
    OI_map_OI = np.ma.masked_array(OI_map, mask = combined_mask_OI)

    '''
    After applying the combined_mask to individual maps, these maps themselves will still retain the 
    same shape as before the mask was applied (50*50). However, the values in the maps that correspond
    to the masked regions will become 'masked', meaning they won't be considered in subsequent calculations.

    Now we extract the valid data points for the calculation of log ratios (we need an array of data). 
    Be careful that after the valid data points are extracted, the shape will no longer be 50*50. 
    Therefore, when creating the spatial map, these valid arrays should not be used. 
    '''

    # valid data array for the first BPT plot.
    valid_Ha_map_NII = Ha_map_NII[~Ha_map_NII.mask]
    valid_Hb_map_NII = Hb_map_NII[~Hb_map_NII.mask]
    valid_OIII_map_NII = OIII_map_NII[~OIII_map_NII.mask]
    valid_NII_map_NII = NII_map_NII[~NII_map_NII.mask]

    # valid data array for the second BPT plot.
    valid_Ha_map_SII = Ha_map_SII[~Ha_map_SII.mask]
    valid_Hb_map_SII = Hb_map_SII[~Hb_map_SII.mask]
    valid_OIII_map_SII = OIII_map_SII[~OIII_map_SII.mask]
    valid_SII_6716_map_SII = SII_6716_map_SII[~SII_6716_map_SII.mask]
    valid_SII_6731_map_SII = SII_6731_map_SII[~SII_6731_map_SII.mask]

    # valid data array for the third BPT plot.
    valid_Ha_map_OI = Ha_map_OI[~Ha_map_OI.mask]
    valid_Hb_map_OI = Hb_map_OI[~Hb_map_OI.mask]
    valid_OIII_map_OI = OIII_map_OI[~OIII_map_OI.mask]
    valid_OI_map_OI = OI_map_OI[~OI_map_OI.mask]

    # calculate the ratio: log([NII]/Ha) vs log([OIII]/Hb) for the first BPT plot.
    log_NII_Ha = np.log10(valid_NII_map_NII / valid_Ha_map_NII)
    log_OIII_Hb_NII = np.log10(valid_OIII_map_NII / valid_Hb_map_NII)

    # calculate the ratio: log([SII]/Ha) vs log([OIII]/Hb) for the second BPT plot.
    log_SII_Ha = np.log10((valid_SII_6716_map_SII + valid_SII_6731_map_SII) /
                          valid_Ha_map_SII)
    log_OIII_Hb_SII = np.log10(valid_OIII_map_SII / valid_Hb_map_SII)

    # calculate the ratio: log([OI]/Ha) vs log([OIII]/Hb) for the third BPT plot.
    log_OI_Ha = np.log10(valid_OI_map_OI / valid_Ha_map_OI)
    log_OIII_Hb_OI = np.log10(valid_OIII_map_OI / valid_Hb_map_OI)

    # begin making plots.
    # the first row would be three spatial maps, the second row would be three BPT plots.
    fig, axs = plt.subplots(2, 2, figsize = (20/3, 6))

    # three BPT plots in the second row.
    #########################################################################
    # first BPT plot (NII).
    # boundary for different classification region.
    def boundary_1(x, clip = False):
        if clip:
            x = np.clip(x, -1.274, 0.469)
        return (0.61 / (x - 0.47)) + 1.19 # Kewley et al. 2006

    def boundary_2(x, clip = False):
        if clip:
            x = np.clip(x, -1.274, 0.05)
        return (0.61 / (x - 0.05)) + 1.3 # Kewley et al. 2006

    sc, center_x, center_y = bpt_plot(Ha_map_clean = Ha_map_NII, ax = axs[1, 0],
             log_x = log_NII_Ha, log_y = log_OIII_Hb_NII)

    # the boundary range for plotting.
    boundary_1_range = np.linspace(-1.4, 0.234, 1000)
    boundary_2_range = np.linspace(-1.274, -0.176, 1000)

    # the boundary values for plotting.
    boundary_1_values = boundary_1(boundary_1_range)
    boundary_2_values = boundary_2(boundary_2_range)

    # plot two boundaries.
    axs[1, 0].plot(boundary_1_range, boundary_1_values, color = 'black', linestyle = '-', linewidth = 1)
    axs[1, 0].plot(boundary_2_range, boundary_2_values, color = 'black', linestyle = '--', linewidth = 1)

    # add labels for each region, ha = 'center' centers the text horizontally at the specified x position.
    axs[1, 0].text(-0.8, -0.8, 'SF', color = 'grey', fontsize = fontsize, ha = 'center')
    axs[1, 0].text(0.0, -1, 'Comp', color = 'salmon', rotation = 90, fontsize = fontsize, ha = 'center')
    axs[1, 0].text(0.0, 0.8, 'AGN', color = 'purple', fontsize = fontsize, ha = 'center')

    # set axis labels, title, and limits.
    axs[1, 0].set_xlabel(r'log$_{10}$([NII]/H$\alpha$)', fontsize = fontsize, labelpad = labelpad_x)
    axs[1, 0].set_ylabel(r'log$_{10}$([OIII]/H$\beta$)', fontsize = fontsize, labelpad = labelpad_y)
    axs[1, 0].set_title(r'[NII]-BPT', fontsize = fontsize)

    axs[1, 0].set_xlim(-1.4, 0.6)
    axs[1, 0].set_ylim(-1.4, 1.4)

    # set x and y ticks.
    axs[1, 0].set_xticks([-1.0, -0.5, 0.0, 0.5])
    axs[1, 0].set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])

    # add major ticks.
    axs[1, 0].tick_params(axis = 'both', which = 'major', length = 4, width = 1, direction = 'in')

    # add minor ticks.
    axs[1, 0].xaxis.set_minor_locator(AutoMinorLocator())
    axs[1, 0].yaxis.set_minor_locator(AutoMinorLocator())
    axs[1, 0].tick_params(axis = 'both', which = 'minor', length = 2, width = 1, direction = 'in')

    # add color bar, showing the distance in arcsec.
    cbar = fig.colorbar(sc, ax = axs[1, 0], fraction = 0.0466, pad = cbar_pad)
    cbar.set_label('Radius (arcsec)', fontsize = fontsize, labelpad = labelpad_cbar)
    cbar.ax.yaxis.set_tick_params(direction = 'in')

    ####################################################################
    # second BPT plot (SII).
    # boundary for different classification region.
    def boundary_3(x, clip = False):
        if clip:
            x = np.clip(x, -0.313, np.inf)
        return 1.89 * x + 0.76 # Kewley et al. 2006

    def boundary_4(x, clip = False):
        if clip:
            x = np.clip(x, -np.inf, 0.319)
        return (0.72 / (x - 0.32)) + 1.30 # Kewley et al. 2006

    sc, center_x, center_y = bpt_plot(Ha_map_clean = Ha_map_SII, ax = axs[1, 1],
             log_x = log_SII_Ha, log_y = log_OIII_Hb_SII)

    # the boundary range for plotting.
    boundary_3_range = np.linspace(-0.313, 0.338, 1000)
    boundary_4_range = np.linspace(-1.4, 0.0533, 1000)

    # the boundary values for plotting.
    boundary_3_values = boundary_3(boundary_3_range)
    boundary_4_values = boundary_4(boundary_4_range)

    # plot two boundaries.
    axs[1, 1].plot(boundary_3_range, boundary_3_values, color = 'black', linestyle = '--', linewidth = 1)
    axs[1, 1].plot(boundary_4_range, boundary_4_values, color = 'black', linestyle = '-', linewidth = 1)

    # add labels for each region, ha = 'center' centers the text horizontally at the specified x position.
    axs[1, 1].text(-0.8, -0.8, 'SF', color = 'grey', fontsize = fontsize, ha = 'center')
    axs[1, 1].text(0.3, -0.8, 'LINER', color = 'lightblue', fontsize = fontsize, ha = 'center')
    axs[1, 1].text(-0.35, 0.8, 'AGN', color = 'purple', fontsize = fontsize, ha = 'center')

    # set axis labels, title, and limits.
    axs[1, 1].set_xlabel(r'log$_{10}$([SII]/H$\alpha$)', fontsize = fontsize, labelpad = labelpad_x)
    axs[1, 1].set_ylabel(r'log$_{10}$([OIII]/H$\beta$)', fontsize = fontsize, labelpad = labelpad_y)
    axs[1, 1].set_title(r'[SII]-BPT', fontsize = fontsize)

    axs[1, 1].set_xlim(-1.4, 0.6)
    axs[1, 1].set_ylim(-1.4, 1.4)

    # set x and y ticks.
    axs[1, 1].set_xticks([-1.0, -0.5, 0.0, 0.5])
    axs[1, 1].set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])

    # add major ticks.
    axs[1, 1].tick_params(axis = 'both', which = 'major', length = 4, width = 1, direction = 'in')

    # add minor ticks.
    axs[1, 1].xaxis.set_minor_locator(AutoMinorLocator())
    axs[1, 1].yaxis.set_minor_locator(AutoMinorLocator())
    axs[1, 1].tick_params(axis = 'both', which = 'minor', length = 2, width = 1, direction = 'in')

    cbar = fig.colorbar(sc, ax = axs[1, 1], pad = cbar_pad, fraction = 0.0466)
    cbar.set_label('Radius (arcsec)', fontsize = fontsize, labelpad = labelpad_cbar)
    cbar.ax.yaxis.set_tick_params(direction = 'in')

    #################################################################################
    '''
    # third BPT plot (OI).
    # boundary for different classification region.
    def boundary_5(x, clip = False):
        if clip:
            x = np.clip(x, -1.13, np.inf)
        return 1.18 * x + 1.30 # Kewley et al. 2006

    def boundary_6(x, clip = False):
        if clip:
            x = np.clip(x, -np.inf, -0.591)
        return (0.73 / (x + 0.59)) + 1.33 # Kewley et al. 2006

    sc, center_x, center_y = bpt_plot(Ha_map_clean = Ha_map_OI, ax = axs[1, 2],
             log_x = log_OI_Ha, log_y = log_OIII_Hb_OI)

    # the boundary range for plotting.
    boundary_5_range = np.linspace(-1.13, -0.254, 1000)
    boundary_6_range = np.linspace(-2.0, -0.902, 1000)

    # the boundary values for plotting.
    boundary_5_values = boundary_5(boundary_5_range)
    boundary_6_values = boundary_6(boundary_6_range)

    # plot two boundaries.
    axs[1, 2].plot(boundary_5_range, boundary_5_values, color = 'black', linestyle = '--', linewidth = 1)
    axs[1, 2].plot(boundary_6_range, boundary_6_values, color = 'black', linestyle = '-', linewidth = 1)

    # add labels for each region, ha = 'center' centers the text horizontally at the specified x position.
    axs[1, 2].text(-1.5, -0.55, 'SF', color = 'grey', fontsize = fontsize, ha = 'center')
    axs[1, 2].text(-0.5, -0.55, 'LINER', color = 'lightblue', fontsize = fontsize, ha = 'center')
    axs[1, 2].text(-1, 0.55, 'AGN', color = 'purple', fontsize = fontsize, ha = 'center')

    # set axis labels, title, and limits.
    axs[1, 2].set_xlabel(r'log([OI]/H$\alpha$)', fontsize = fontsize, labelpad = labelpad_x)
    axs[1, 2].set_ylabel(r'log([OIII]/H$\beta$)', fontsize = fontsize, labelpad = labelpad_y)

    axs[1, 2].set_xlim(-2.0, 0.0)
    axs[1, 2].set_ylim(-1.0, 1.0)

    # set x and y ticks.
    axs[1, 2].set_xticks([-2.0, -1.5, -1.0, -0.5, 0.0])
    axs[1, 2].set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])

    # add major ticks.
    axs[1, 2].tick_params(axis = 'both', which = 'major', length = 4, width = 1, direction = 'in')

    # add minor ticks.
    axs[1, 2].xaxis.set_minor_locator(AutoMinorLocator())
    axs[1, 2].yaxis.set_minor_locator(AutoMinorLocator())
    axs[1, 2].tick_params(axis = 'both', which = 'minor', length = 2, width = 1, direction = 'in')

    cbar = fig.colorbar(sc, ax = axs[1, 2], pad = cbar_pad)
    cbar.set_label('Radius (arcsec)', fontsize = fontsize, labelpad = labelpad_cbar)
    cbar.ax.yaxis.set_tick_params(direction = 'in')
    '''

    #########################################################################
    # three spatial plots in the first row.
    ########################################################################
    # transfer from pixel size to arcsec.
    y = (np.arange(Ha_map.shape[0]) - center_y) * scale
    x = (np.arange(Ha_map.shape[1]) - center_x) * scale

    #######################################################################
    # first spatial plot (SF: grey, Comp: salmon, AGN: purple).
    cmap = ListedColormap(['gray', 'salmon', 'purple'])
    log_y = np.log10(OIII_map_NII / Hb_map_NII)
    log_x = np.log10(NII_map_NII / Ha_map_NII)

    # SF: 0, Comp: 1, AGN: 2.
    classification = np.zeros_like(log_y, dtype = int)

    sf = (log_y < boundary_1(log_x, clip = True)) & (log_y < boundary_2(log_x, clip = True))
    classification[sf] = 0

    comp = (log_y >= boundary_2(log_x, clip = True)) & (log_y < boundary_1(log_x, clip = True))
    classification[comp] = 1

    agn = (log_y >= boundary_1(log_x, clip = True))
    classification[agn] = 2

    bounds = [0, 1, 2, 3]
    norm = BoundaryNorm(bounds, cmap.N)

    im = axs[0, 0].imshow(classification, origin = 'lower', cmap = cmap, norm = norm,
                     extent = [x[0], x[-1], y[0], y[-1]])

    axs[0, 0].set_xlim([-12.5, 12.5])
    axs[0, 0].set_ylim([-12.5, 12.5])

    axs[0, 0].set_xticks(np.arange(-10, 11, 5))
    axs[0, 0].set_yticks(np.arange(-10, 11, 5))

    axs[0, 0].set_xlabel('Offset (arcsec)', fontsize = fontsize, labelpad = labelpad_x)
    axs[0, 0].set_ylabel('Offset (arcsec)', fontsize = fontsize, labelpad = labelpad_y)
    axs[0, 0].set_title(r'Resolved [NII]-BPT', fontsize = fontsize)

    # add major ticks.
    axs[0, 0].tick_params(axis = 'both', which = 'major', length = 4, width = 1, direction = 'in')
    # add minor ticks.
    axs[0, 0].xaxis.set_minor_locator(AutoMinorLocator())
    axs[0, 0].yaxis.set_minor_locator(AutoMinorLocator())
    axs[0, 0].tick_params(axis = 'both', which = 'minor', length = 2, width = 1, direction = 'in')

    # add colorbar.
    cbar = fig.colorbar(im, ax = axs[0, 0], boundaries = bounds, ticks = [], pad = cbar_pad, fraction = bar_fraction)

    # remove tick lines and set no ticks.
    cbar.ax.tick_params(axis = 'y', which = 'both', length = 0)

    cbar.ax.text(1.2, 0.5, 'SF', va = 'center', ha = 'left', rotation = 90, fontsize = fontsize, color = 'black')
    cbar.ax.text(1.2, 1.5, 'Comp', va = 'center', ha = 'left', rotation = 90, fontsize = fontsize, color = 'black')
    cbar.ax.text(1.2, 2.5, 'AGN', va = 'center', ha = 'left', rotation = 90, fontsize = fontsize, color = 'black')

    ###############################################################################
    # second spatial plot (SF: grey, LINER: lightblue, AGN: purple).
    cmap = ListedColormap(['grey', 'lightblue', 'purple'])
    log_y = np.log10(OIII_map_SII / Hb_map_SII)
    log_x = np.log10((SII_6716_map_SII + SII_6731_map_SII) / Ha_map_SII)

    # SF: 0, LINER: 1, AGN: 2.
    classification = np.zeros_like(log_y, dtype = int)

    sf = (log_y < boundary_4(log_x, clip = True))
    classification[sf] = 0

    liner = (log_y <= boundary_3(log_x, clip = True)) & (log_y >= boundary_4(log_x, clip = True))
    classification[liner] = 1

    agn = (log_y > boundary_3(log_x, clip = True)) & (log_y >= boundary_4(log_x, clip = True))
    classification[agn] = 2

    bounds = [0, 1, 2, 3]
    norm = BoundaryNorm(bounds, cmap.N)

    im = axs[0, 1].imshow(classification, origin = 'lower', cmap = cmap, norm = norm,
                     extent = [x[0], x[-1], y[0], y[-1]])

    axs[0, 1].set_xlim([-12.5, 12.5])
    axs[0, 1].set_ylim([-12.5, 12.5])

    axs[0, 1].set_xticks(np.arange(-10, 11, 5))
    axs[0, 1].set_yticks(np.arange(-10, 11, 5))

    axs[0, 1].set_xlabel('Offset (arcsec)', fontsize = fontsize, labelpad = labelpad_x)
    axs[0, 1].set_ylabel('Offset (arcsec)', fontsize = fontsize, labelpad = labelpad_y)
    axs[0, 1].set_title(r'Resolved [SII]-BPT', fontsize = fontsize)

    # add major ticks.
    axs[0, 1].tick_params(axis = 'both', which = 'major', length = 4, width = 1, direction = 'in')
    # add minor ticks.
    axs[0, 1].xaxis.set_minor_locator(AutoMinorLocator())
    axs[0, 1].yaxis.set_minor_locator(AutoMinorLocator())
    axs[0, 1].tick_params(axis = 'both', which = 'minor', length = 2, width = 1, direction = 'in')

    # add colorbar.

    cbar = fig.colorbar(im, ax = axs[0, 1], boundaries = bounds, ticks = [], pad = cbar_pad, fraction = bar_fraction)

    # remove tick lines and set no ticks.
    cbar.ax.tick_params(axis = 'y', which = 'both', length = 0)

    cbar.ax.text(1.5, 0.5, 'SF', va = 'center', ha = 'left', rotation = 90, fontsize = fontsize, color = 'black')
    cbar.ax.text(1.5, 1.5, 'LINER', va = 'center', ha = 'left', rotation = 90, fontsize = fontsize, color = 'black')
    cbar.ax.text(1.5, 2.5, 'AGN', va = 'center', ha = 'left', rotation = 90, fontsize = fontsize, color = 'black')

    #############################################################################
    '''
    # third spatial plot (SF: grey, LINER: lightblue, AGN: purple).
    cmap = ListedColormap(['grey', 'lightblue', 'purple'])
    log_y = np.log10(OIII_map_OI / Hb_map_OI)
    log_x = np.log10(OI_map_OI / Ha_map_OI)

    # SF: 0, LINER: 1, AGN: 2.
    classification = np.zeros_like(log_y, dtype = int)

    sf = (log_y < boundary_6(log_x, clip = True))
    classification[sf] = 0

    liner = (log_y <= boundary_5(log_x, clip = True)) & (log_y >= boundary_6(log_x, clip = True))
    classification[liner] = 1

    agn = (log_y > boundary_5(log_x, clip = True)) & (log_y >= boundary_6(log_x, clip = True))
    classification[agn] = 2

    bounds = [0, 1, 2, 3]
    norm = BoundaryNorm(bounds, cmap.N)

    im = axs[0, 2].imshow(classification, origin = 'lower', cmap = cmap, norm = norm,
                          extent = [x[0], x[-1], y[0], y[-1]])

    axs[0, 2].set_xlim([-12.5, 12.5])
    axs[0, 2].set_ylim([-12.5, 12.5])

    axs[0, 2].set_xticks(np.arange(-10, 11, 5))
    axs[0, 2].set_yticks(np.arange(-10, 11, 5))

    axs[0, 2].set_xlabel('Offset [arcsec]', fontsize = fontsize, labelpad = labelpad_x)
    axs[0, 2].set_ylabel('Offset [arcsec]', fontsize = fontsize, labelpad = labelpad_y)
    axs[0, 2].set_title(r'[OIII]/H$\beta$ vs [OI]/H$\alpha$', fontsize = fontsize)

    # add major ticks.
    axs[0, 2].tick_params(axis = 'both', which = 'major', length = 4, width = 1, direction = 'in')
    # add minor ticks.
    axs[0, 2].xaxis.set_minor_locator(AutoMinorLocator())
    axs[0, 2].yaxis.set_minor_locator(AutoMinorLocator())
    axs[0, 2].tick_params(axis = 'both', which = 'minor', length = 2, width = 1, direction = 'in')

    # add colorbar.

    cbar = fig.colorbar(im, ax = axs[0, 2], boundaries = bounds, ticks = [], pad = cbar_pad)

    # remove tick lines and set no ticks.
    cbar.ax.tick_params(axis = 'y', which = 'both', length = 0)

    cbar.ax.text(1.2, 0.5, 'SF', va = 'center', ha = 'left', rotation = 90, fontsize = fontsize, color = 'black')
    cbar.ax.text(1.2, 1.5, 'LINER', va = 'center', ha = 'left', rotation = 90, fontsize = fontsize, color = 'black')
    cbar.ax.text(1.2, 2.5, 'AGN', va = 'center', ha = 'left', rotation = 90, fontsize = fontsize, color = 'black')
    '''

#----------------------------------------------------------




