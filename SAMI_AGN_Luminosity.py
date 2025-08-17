'''
version_01: 08/13/2025
'''
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from photutils.aperture import CircularAperture, aperture_photometry
from astropy.cosmology import FlatLambdaCDM
#---------------------------------------------------------------
def plot(data):
    plt.figure(figsize = (10, 8))

    plt.imshow(data, origin = 'lower', aspect = 'auto',
               cmap = 'jet')
    plt.xlabel('SPAXEL')
    plt.ylabel('SPAXEL')

    plt.show()

#-----------------------------------------------------------------
def agn_luminosity(OIII_fits_path, Ha_fits_path, Hb_fits_path, threshold, psf_fwhm, redshift,
                   scale = 0.5, xc = 25, yc = 25, factor = 10**(-16), H0 = 73, om0 = 0.3,
                   dust_correction = False, Bassani = False, dust_fits_path = None, Bolometric = False):
    '''
    Caculate the AGN luminosity using [OIII].

    Parameter:
    - OIII_fits_path: str, path to the OIII fits file.
    - Ha_fits_path: str, path to the Ha fits file.
    - Hb_fits_path: str, path to the Hb fits file.
    - threshold: exclude spaxels in all emission maps where S/N < threshold.
    - psf_fwhm: float, FWHM of the PSF in arcseconds.
    - redshift: redshift value to calculate luminosity distance.
    - scale: pixel scale (e.g., 0.5 arcsec/pixel for SAMI).
    - xc: galaxy center in pixel.
    - yc: galaxy center in pixel.
    - factor: flux factor (e.g., for SAMI, the unit in the emission line maps is 10^(-16) erg/s/cm^2/pixel).
    - H0: Hubble constant at redshift = 0 in km/s/Mpc.
    - om0: Omega matter: density of non-relativistic matter in units of the critical density at redshift = 0.
    - dust_correction: bool, whether to apply dust correction.
    - Bassani: bool, if dust_correction is true, one can choose whether to apply Bassani dust correction.
    - dust_fits_path: if dust_correction is true and Bassani dust correction is not applied, then one can
                      choose to provide the path to the dust fits file (e.g., products given by SAMI).
    - Bolometric: bool, whether to transform the dust corrected luminosity to Bolometric luminosity.

    Returns:

    '''

    # load the optical emission line maps (primary map[0] and error map [1]) for each line.
    #  using [OIII] to estimate AGN luminosity.
    with fits.open(OIII_fits_path) as OIII:
        OIII_map = OIII[0].data
        OIII_err = OIII[1].data

    # Ha and Hb would be used to calculate the Balmer decrement.
    with fits.open(Ha_fits_path) as Ha:
        Ha_map = Ha[0].data
        Ha_err = Ha[1].data

    with fits.open(Hb_fits_path) as Hb:
        Hb_map = Hb[0].data
        Hb_err = Hb[1].data

    # extract the total component (0) of Ha (50*50*4 -> 50*50).
    Ha_map = Ha_map[0, :, :]
    Ha_err = Ha_err[0, :, :]

    # mask NaN values in all 6 maps.
    OIII_map = np.ma.masked_invalid(OIII_map)
    OIII_err = np.ma.masked_invalid(OIII_err)

    Ha_map = np.ma.masked_invalid(Ha_map)
    Ha_err = np.ma.masked_invalid(Ha_err)

    Hb_map = np.ma.masked_invalid(Hb_map)
    Hb_err = np.ma.masked_invalid(Hb_err)

    # calculate the signal-to-noise ratio (SNR) for each emission line.
    OIII_SNR = OIII_map / OIII_err
    Ha_SNR = Ha_map / Ha_err
    Hb_SNR = Hb_map / Hb_err

    print(f'OIII_SNR: min = {np.min(OIII_SNR)}, max = {np.max(OIII_SNR)}.')
    print(f'Ha_SNR: min = {np.min(Ha_SNR)}, max = {np.max(Ha_SNR)}.')
    print(f'Hb_SNR: min = {np.min(Hb_SNR)}, max = {np.max(Hb_SNR)}.')

    # mask data points where SNR is below a specific threshold.
    OIII_map = np.ma.masked_where(OIII_SNR < threshold, OIII_map)
    Ha_map = np.ma.masked_where(Ha_SNR < threshold, Ha_map)
    Hb_map = np.ma.masked_where(Hb_SNR < threshold, Hb_map)

    # if a spaxel is invalid in any map, it would be excluded entirely.
    combined_mask = np.ma.getmask(OIII_map)
    combined_mask = np.ma.mask_or(combined_mask, np.ma.getmask(OIII_err))

    combined_mask = np.ma.mask_or(combined_mask, np.ma.getmask(Ha_map))
    combined_mask = np.ma.mask_or(combined_mask, np.ma.getmask(Ha_err))

    combined_mask = np.ma.mask_or(combined_mask, np.ma.getmask(Hb_map))
    combined_mask = np.ma.mask_or(combined_mask, np.ma.getmask(Hb_err))

    # apply the combined mask.
    OIII_map = np.ma.masked_array(OIII_map, mask = combined_mask)
    OIII_err = np.ma.masked_array(OIII_err, mask = combined_mask)

    Ha_map = np.ma.masked_array(Ha_map, mask = combined_mask)
    Ha_err = np.ma.masked_array(Ha_err, mask = combined_mask)

    Hb_map = np.ma.masked_array(Hb_map, mask = combined_mask)
    Hb_err = np.ma.masked_array(Hb_err, mask = combined_mask)

    # the next step is to transfer the flux into luminosity.
    # first of all, calculate the total flux within an aperture with radius being 2 * PSF_FWHM.
    radius = 2 * psf_fwhm # in arcsec scale.
    radius = radius / scale # in pixel scale.

    # calculate the total flux within the radius.
    aperture = CircularAperture((xc, yc), radius)
    phot_table = aperture_photometry(OIII_map, aperture, error = OIII_err)
    flux_agn = phot_table['aperture_sum'][0] * factor
    print(f'Nuclear [OIII] flux: {flux_agn:} erg/s/cm^2')

    # try another method:
    y, x = np.mgrid[:OIII_map.shape[0], :OIII_map.shape[1]]
    mask = np.sqrt((x - xc)**2 + (y - yc)**2) <= radius
    flux_agn_02 = np.sum(OIII_map[mask]) * factor

    print(f'directly using mask instead of using photutils: {flux_agn_02} erg/s/cm^2')

    # the assumed cosmology.
    cosmo = FlatLambdaCDM(H0 = H0, Om0 = om0)

    # The luminosity distance in Mpc at the redshift.
    # This is the distance to use when converting between the bolometric flux and its bolometric luminosity.
    # 1 Mpc = 3.08568*10^24 cm
    DL_cm = cosmo.luminosity_distance(redshift).to('cm').value

    # the AGN luminosity.
    L_agn = 4 * np.pi * DL_cm**2 * flux_agn

    print(f'the AGN luminosity: {L_agn:} erg/s.')

#------------------------------------------------------------------------