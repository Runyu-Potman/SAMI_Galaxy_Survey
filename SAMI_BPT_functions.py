'''
This script can be used to create the Baldwin, Phillips & Terlevich empirical diagnostic diagrams
for SAMI galaxies using the optical line ratios.
'''

import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def gas_distribution(Hα_fits_path, threshold):
    # load the optical emission line maps (primary map[0] and error map [1]) for each line.
    Hα = fits.open(Hα_fits_path)
    Hα_map = Hα[0].data
    Hα_err = Hα[1].data

    # extract the total component (0) of Hα (50*50*4 -> 50*50).
    Hα_map = Hα_map[0, :, :]
    Hα_err = Hα_err[0, :, :]

    print(
        f'Shape of Hα_map after extraction: {Hα_map.shape}, shape of Hα_err after extraction: {Hα_err.shape}.'
    )

    # mask NaN values in all 14 maps.
    Hα_map = np.ma.masked_invalid(Hα_map)
    Hα_err = np.ma.masked_invalid(Hα_err)

    # calculate the signal-to-noise ratio (SNR) for each emission line.
    '''
    be noticed that any pixel where err_map is masked will result in the corresponding 
    ratio also being masked, even if primary_map at that pixel is valid.
    '''

    Hα_SNR = Hα_map / Hα_err

    print(f'Hα_SNR: min = {np.min(Hα_SNR)}, max = {np.max(Hα_SNR)}.')

    # mask data points where SNR is below a specific threshold.
    Hα_map = np.ma.masked_where(Hα_SNR < threshold, Hα_map)

    # combined_mask_NII is the combined mask for the first BPT plot.
    combined_mask = np.ma.getmask(Hα_map)
    combined_mask = np.ma.mask_or(combined_mask, np.ma.getmask(Hα_err))

    Hα_map = np.ma.masked_array(Hα_map, mask=combined_mask)

    plt.figure(figsize = (10, 10))
    plt.imshow(Hα_map, origin = 'lower', cmap = 'viridis')
    plt.colorbar(label = 'flux intensity')
    plt.xlabel('pixel')
    plt.ylabel('pixel')
    plt.title('gas distribution map')
    plt.show()
#-----------------------------------------------------------------------------------------------------------------
Hα_fits_path = '230776_A_Halpha_adaptive_recom-comp.fits'
threshold = 3

gas_distribution(Hα_fits_path, threshold)













#-------------------------------------------------------------------------------------------------------------------

def bpt(
        Hα_fits_path, Hβ_fits_path, OIII_fits_path, OI_fits_path,
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
    Hα = fits.open(Hα_fits_path)
    Hα_map = Hα[0].data
    Hα_err = Hα[1].data

    Hβ = fits.open(Hβ_fits_path)
    Hβ_map = Hβ[0].data
    Hβ_err = Hβ[1].data

    OIII = fits.open(OIII_fits_path)
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

    # extract the total component (0) of Hα (50*50*4 -> 50*50).
    Hα_map = Hα_map[0, :, :]
    Hα_err = Hα_err[0, :, :]

    print(
        f'Shape of Hα_map after extraction: {Hα_map.shape}, shape of Hα_err after extraction: {Hα_err.shape}.'
    )

    # mask NaN values in all 14 maps.
    Hα_map = np.ma.masked_invalid(Hα_map)
    Hα_err = np.ma.masked_invalid(Hα_err)

    Hβ_map = np.ma.masked_invalid(Hβ_map)
    Hβ_err = np.ma.masked_invalid(Hβ_err)

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

    Hα_SNR = Hα_map / Hα_err
    Hβ_SNR = Hβ_map / Hβ_err
    OIII_SNR = OIII_map / OIII_err
    OI_SNR = OI_map / OI_err
    SII_6716_SNR = SII_6716_map / SII_6716_err
    SII_6731_SNR = SII_6731_map / SII_6731_err
    NII_SNR = NII_map / NII_err

    print(f'Hα_SNR: min = {np.min(Hα_SNR)}, max = {np.max(Hα_SNR)}.')
    print(f'Hβ_SNR: min = {np.min(Hβ_SNR)}, max = {np.max(Hβ_SNR)}.')
    print(f'OIII_SNR: min = {np.min(OIII_SNR)}, max = {np.max(OIII_SNR)}.')
    print(f'OI_SNR: min = {np.min(OI_SNR)}, max = {np.max(OI_SNR)}.')
    print(f'SII_6716_SNR: min = {np.min(SII_6716_SNR)}, max = {np.max(SII_6716_SNR)}.')
    print(f'SII_6731_SNR: min = {np.min(SII_6731_SNR)}, max = {np.max(SII_6731_SNR)}.')
    print(f'NII_SNR: min = {np.min(NII_SNR)}, max = {np.max(NII_SNR)}.')

    # mask data points where SNR is below a specific threshold.
    Hα_map = np.ma.masked_where(Hα_SNR < threshold, Hα_map)
    Hβ_map = np.ma.masked_where(Hβ_SNR < threshold, Hβ_map)
    OIII_map = np.ma.masked_where(OIII_SNR < threshold, OIII_map)
    OI_map = np.ma.masked_where(OI_SNR < threshold, OI_map)
    SII_6716_map = np.ma.masked_where(SII_6716_SNR < threshold, SII_6716_map)
    SII_6731_map = np.ma.masked_where(SII_6731_SNR < threshold, SII_6731_map)
    NII_map = np.ma.masked_where(NII_SNR < threshold, NII_map)

    '''
    For the first BPT plot ([NII]/Hα verse [OIII]/Hβ), if a data point is invalid (NaN or SNR < threshold)
    in any of the emission line maps, it should be excluded from all maps.

    Note that when creating the first BPT plot, the mask from the second and the third plots should not be
    included, similar for the second plot and the third plot. Therefore, we have different combined_mask 
    for different plots. 
    '''

    # combined_mask_NII is the combined mask for the first BPT plot.
    combined_mask_NII = np.ma.getmask(Hα_map)
    combined_mask_NII = np.ma.mask_or(combined_mask_NII, np.ma.getmask(Hα_err))

    combined_mask_NII = np.ma.mask_or(combined_mask_NII, np.ma.getmask(Hβ_map))
    combined_mask_NII = np.ma.mask_or(combined_mask_NII, np.ma.getmask(Hβ_err))

    combined_mask_NII = np.ma.mask_or(combined_mask_NII, np.ma.getmask(OIII_map))
    combined_mask_NII = np.ma.mask_or(combined_mask_NII, np.ma.getmask(OIII_err))

    combined_mask_NII = np.ma.mask_or(combined_mask_NII, np.ma.getmask(NII_map))
    combined_mask_NII = np.ma.mask_or(combined_mask_NII, np.ma.getmask(NII_err))

    # combined_mask_SII is the combined mask for the second BPT plot.
    combined_mask_SII = np.ma.getmask(Hα_map)
    combined_mask_SII = np.ma.mask_or(combined_mask_SII, np.ma.getmask(Hα_err))

    combined_mask_SII = np.ma.mask_or(combined_mask_SII, np.ma.getmask(Hβ_map))
    combined_mask_SII = np.ma.mask_or(combined_mask_SII, np.ma.getmask(Hβ_err))

    combined_mask_SII = np.ma.mask_or(combined_mask_SII, np.ma.getmask(OIII_map))
    combined_mask_SII = np.ma.mask_or(combined_mask_SII, np.ma.getmask(OIII_err))

    combined_mask_SII = np.ma.mask_or(combined_mask_SII, np.ma.getmask(SII_6716_map))
    combined_mask_SII = np.ma.mask_or(combined_mask_SII, np.ma.getmask(SII_6716_err))

    combined_mask_SII = np.ma.mask_or(combined_mask_SII, np.ma.getmask(SII_6731_map))
    combined_mask_SII = np.ma.mask_or(combined_mask_SII, np.ma.getmask(SII_6731_err))

    # combined_mask_OI is the combined mask for the third BPT plot.
    combined_mask_OI = np.ma.getmask(Hα_map)
    combined_mask_OI = np.ma.mask_or(combined_mask_OI, np.ma.getmask(Hα_err))

    combined_mask_OI = np.ma.mask_or(combined_mask_OI, np.ma.getmask(Hβ_map))
    combined_mask_OI = np.ma.mask_or(combined_mask_OI, np.ma.getmask(Hβ_err))

    combined_mask_OI = np.ma.mask_or(combined_mask_OI, np.ma.getmask(OIII_map))
    combined_mask_OI = np.ma.mask_or(combined_mask_OI, np.ma.getmask(OIII_err))

    combined_mask_OI = np.ma.mask_or(combined_mask_OI, np.ma.getmask(OI_map))
    combined_mask_OI = np.ma.mask_or(combined_mask_OI, np.ma.getmask(OI_err))

    # apply the combined mask across all maps to ensure consistency.
    # the first BPT plot.
    Hα_map_NII = np.ma.masked_array(Hα_map, mask=combined_mask_NII)
    Hβ_map_NII = np.ma.masked_array(Hβ_map, mask=combined_mask_NII)
    OIII_map_NII = np.ma.masked_array(OIII_map, mask=combined_mask_NII)
    NII_map_NII = np.ma.masked_array(NII_map, mask=combined_mask_NII)

    # the second BPT plot.
    Hα_map_SII = np.ma.masked_array(Hα_map, mask=combined_mask_SII)
    Hβ_map_SII = np.ma.masked_array(Hβ_map, mask=combined_mask_SII)
    OIII_map_SII = np.ma.masked_array(OIII_map, mask=combined_mask_SII)
    SII_6716_map_SII = np.ma.masked_array(SII_6716_map, mask=combined_mask_SII)
    SII_6731_map_SII = np.ma.masked_array(SII_6731_map, mask=combined_mask_SII)

    # the third BPT plot.
    Hα_map_OI = np.ma.masked_array(Hα_map, mask=combined_mask_OI)
    Hβ_map_OI = np.ma.masked_array(Hβ_map, mask=combined_mask_OI)
    OIII_map_OI = np.ma.masked_array(OIII_map, mask=combined_mask_OI)
    OI_map_OI = np.ma.masked_array(OI_map, mask=combined_mask_OI)

    '''
    After applying the combined_mask to individual maps, these maps themselves will still retain the 
    same shape as before the mask was applied (50*50). However, the values in the maps that correspond
    to the masked regions will become 'masked', meaning they won't be considered in subsequent calculations.

    Now we extract the valid data points for the calculation of log ratios (we need an array of data). 
    Be careful that after the valid data points are extracted, the shape will no longer be 50*50. 
    Therefore, when creating the spatial map, these valid maps should not be used. 
    '''

    # valid data array for the first BPT plot.
    valid_Hα_map_NII = Hα_map_NII[~Hα_map_NII.mask]
    valid_Hβ_map_NII = Hβ_map_NII[~Hβ_map_NII.mask]
    valid_OIII_map_NII = OIII_map_NII[~OIII_map_NII.mask]
    valid_NII_map_NII = NII_map_NII[~NII_map_NII.mask]

    # valid data array for the second BPT plot.
    valid_Hα_map_SII = Hα_map_SII[~Hα_map_SII.mask]
    valid_Hβ_map_SII = Hβ_map_SII[~Hβ_map_SII.mask]
    valid_OIII_map_SII = OIII_map_SII[~OIII_map_SII.mask]
    valid_SII_6716_map_SII = SII_6716_map_SII[~SII_6716_map_SII.mask]
    valid_SII_6731_map_SII = SII_6731_map_SII[~SII_6731_map_SII.mask]

    # valid data array for the third BPT plot.
    valid_Hα_map_OI = Hα_map_OI[~Hα_map_OI.mask]
    valid_Hβ_map_OI = Hβ_map_OI[~Hβ_map_OI.mask]
    valid_OIII_map_OI = OIII_map_OI[~OIII_map_OI.mask]
    valid_OI_map_OI = OI_map_OI[~OI_map_OI.mask]

    # calculate the ratio: log([NII]/Hα) vs log([OIII]/Hβ) for the first BPT plot.
    log_NII_Hα = np.log10(valid_NII_map_NII / valid_Hα_map_NII)
    log_OIII_Hβ_NII = np.log10(valid_OIII_map_NII / valid_Hβ_map_NII)

    # calculate the ratio: log([SII]/Hα) vs log([OIII]/Hβ) for the second BPT plot.
    log_SII_Hα = np.log10((valid_SII_6716_map_SII + valid_SII_6731_map_SII) /
                          valid_Hα_map_SII)
    log_OIII_Hβ_SII = np.log10(valid_OIII_map_SII / valid_Hβ_map_SII)

    # calculate the ratio: log([OI]/Hα) vs log([OIII]/Hβ) for the third BPT plot.
    log_OI_Hα = np.log10(valid_OI_map_OI / valid_Hα_map_OI)
    log_OIII_Hβ_OI = np.log10(valid_OIII_map_OI / valid_Hβ_map_OI)

    # begin making plots.
    # the first row would be three spatial maps, the second row would be three BPT plots.
































































