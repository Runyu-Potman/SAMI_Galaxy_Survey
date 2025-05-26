'''
This script can be used to create the Baldwin, Phillips & Terlevich empirical diagnostic diagrams
for SAMI galaxies using the optical line ratios.
'''

import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from SAMI_kinematcs import plot_vel_or_sig
from SAMI_stellar_velocity_quality_cut_functions import quality_cut_gaseous_velocity_map_csv

def gas_distribution(gas_fits_path, output_file, threshold = None):
    # load the optical emission line maps (primary map[0] and error map [1]) for each line.
    gas_map = fits.open(gas_fits_path)
    gas_data = gas_map[0].data
    gas_err_data = gas_map[1].data

    # extract the total component (0) of Ha (50*50*4 -> 50*50).
    # for the 1-component case, [0, :, :] is the same as [1, :, :].
    gas_data = gas_data[0, :, :]
    gas_err_data = gas_err_data[0, :, :]

    # mask NaN values in all 14 maps.
    gas_data = np.ma.masked_invalid(gas_data)
    gas_err_data = np.ma.masked_invalid(gas_err_data)

    # calculate the signal-to-noise ratio (SNR) for each emission line.
    '''
    be noticed that any pixel where err_map is masked will result in the corresponding 
    ratio also being masked, even if primary_map at that pixel is valid.
    '''
    if threshold is not None:
        gas_err_data = np.ma.masked_where(gas_err_data == 0, gas_err_data)

        gas_SNR = gas_data / gas_err_data

        print(f'Gas_SNR: min = {np.min(gas_SNR)}, max = {np.max(gas_SNR)}.')

        # mask data points where SNR is below a specific threshold.
        gas_data = np.ma.masked_where(gas_SNR <= threshold, gas_data)
        gas_err_data = np.ma.masked_where(gas_SNR <= threshold, gas_err_data)

    # prepare the csv data for plotting.
    ny, nx = gas_data.shape

    data_to_save = []

    for i in range(ny):
        for j in range(nx):
            if (not gas_data.mask[i, j] and not gas_err_data.mask[i, j]):

                x_arcsec = (j - 24) * 0.5
                y_arcsec = (i - 24) * 0.5

                print(f'{x_arcsec}, {y_arcsec}, {gas_data[i, j]}, {gas_err_data[i, j]}')
                data_to_save.append((x_arcsec, y_arcsec, gas_data[i, j], gas_err_data[i, j]))

    with open(output_file, 'w') as f:
        f.write('x_arcsec,y_arcsec,gas,gas_err\n')
        for entry in data_to_save:
            f.write(f'{entry[0]}, {entry[1]}, {entry[2]}, {entry[3]}\n')

    # close the fits files after use.
    gas_map.close()

#-----------------------------------------------------------------------------------------------------------------
def bpt(
        Ha_fits_path, Hb_fits_path, OIII_fits_path, OI_fits_path,
        SII_6716_fits_path, SII_6731_fits_path, NII_fits_path,
        threshold
):
    '''
    Apply the quality cut criteria and make plot with x_axis and y_axis in pixel unit,
    note that the center of the galaxy is located at around (25, 25).

    Parameters:
    - vel_fits_path: str, path to the stellar velocity fits file.
    - sig_fits_path: str, path to the stellar velocity dispersion fits file.
    - vmin: int, minimum value for the color bar.
    - vmax: int, maximum value for the color bar.

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

    OI = fits.open(OI_fits_path)
    OI_map = OI[0].data
    OI_err = OI[1].data

    SII_6716 = fits.open(SII_6716_fits_path)
    SII_6716_map = SII_6716[0].data
    SII_6716_err = SII_6716[1].data

    SII_6731 = fits.open(SII_6731_fits_path)
    SII_6731_map = SII_6731[0].data
    SII_6731_err = SII_6731[1].data

    NII = fits.open(NII_fits_path)
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
    Ha_map = np.ma.masked_where(Ha_SNR < threshold, Ha_map)
    Hb_map = np.ma.masked_where(Hb_SNR < threshold, Hb_map)
    OIII_map = np.ma.masked_where(OIII_SNR < threshold, OIII_map)
    OI_map = np.ma.masked_where(OI_SNR < threshold, OI_map)
    SII_6716_map = np.ma.masked_where(SII_6716_SNR < threshold, SII_6716_map)
    SII_6731_map = np.ma.masked_where(SII_6731_SNR < threshold, SII_6731_map)
    NII_map = np.ma.masked_where(NII_SNR < threshold, NII_map)

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
































































