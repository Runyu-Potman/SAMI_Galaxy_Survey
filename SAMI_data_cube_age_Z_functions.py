import numpy as np
from astropy.io import fits
from ppxf.ppxf_util import log_rebin
import matplotlib.pyplot as plt
from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util
import ppxf.sps_util as lib
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
#from SAMI_data_cube_quality_cut_functions import data_cube_clean_percentage, data_cube_clean_snr

from pathlib import Path
from urllib import request
from SAMI_ppxf_util_functions import emission_lines
from SAMI_data_cube_vorbin_functions import nan_safe_gaussian_filter1d
#--------------------------------------------------------------------------------------------
'''
Seven functions are defined in this script:
1. plot_spectrum()
2. bootstrap_residuals()
3. safe_log_rebin()
4. ppxf_pre_spectrum()
5. ppxf_pre_data_cube()
6. ppxf_age_z()
7. decide_mdegree()

After using the functions in the code: SAMI_data_cube_quality_cut_functions.py, we could get a 
cleaned data cube. On the one hand, we could generate co-added spectrum based on the cleaned data 
cube, and making use of ppxf_pre_spectrum() and ppxf_age_z() to estimate the age and metallicity 
for this spectrum. On the other hand, we could make spatially resolved (pixel by pixel) maps for 
age and metallicity by using ppxf_pre_data_cube() and ppxf_age_z() starting from the cleaned data
cube. The final derived age and metallicity are obtained by using the bootstrapping method. The 
decide_mdegree is used to derive the optimal mdegree value. However, in this script we just assume
mdegree = 10 (a commonly used value when it comes to the stellar population estimation for SAMI).

Note that all functions defined in this script assume low-redshift by default. Please set the keyword
high_redshift to be true when dealing with high redshift galaxies. In high redshift situation, we do not 
do the de-redshifting, otherwise, the de-redshifted wavelength range is likely to become smaller than the 
lower limit of the templates (e.g., MILES SSP: 3540.5 Å), also, if we do fwhm_gal = fwhm_blue / (1 + redshift), 
the fwhm_gal is likely to become smaller than the fwhm of the templates (e,g., MILES SSP: 2.51 Å). 
  
version_01: 03/04/2025: initial version.
version_02: 08/04/2025: new functions added.
version_03: 17/04/2025: add high-redshift situation.
version_04: 20/04/2025: handle NaN and inf in the input flux by replacing with a large value before log_rebin
            by using the newly defined function: safe_log_rebin.
version_05: 13/06/2025: new high redshift method with convolve and wave_clip keyword is added, do not use the high_redshift keyword anymore.
'''
#-------------------------------------------------------------------------------
def plot_spectrum(wavelength, spectrum):
    '''
    plot the spectrum before doing log-rebin.

    Parameters:
    - wavelength: rest-frame (or observed-frame) wavelength in Å.
    - spectrum: flux with the unit proportional to erg/s/cm**2/angstrom.

    Returns:
    - None
    '''

    plt.figure(figsize = (15, 10))
    plt.plot(wavelength, spectrum)
    plt.xlabel('wavelength (Å)')
    plt.ylabel('flux (10**(-16) erg/s/cm**2/angstrom/pixel)')
    plt.title('spectrum before log-rebin')
    plt.show()

#--------------------------------------------------------------------------------
def spectrum_template_convolve(flux, redshift, cdelt3, fwhm_blue = 2.65, fwhm_miles = 2.51):
    fwhm_miles_obs = fwhm_miles * (1 + redshift)
    if fwhm_miles_obs > fwhm_blue:
        fwhm_conv = np.sqrt(fwhm_miles_obs**2 - fwhm_blue**2)
        sig_conv = fwhm_conv / (2 * np.sqrt(2 * np.log(2)))
        sig_conv = sig_conv / cdelt3

        flux_conv = nan_safe_gaussian_filter1d(flux, sig_conv)
    else:
        raise ValueError('No need to do the convolution!')

    return flux_conv

#---------------------------------------------------------------------------------
def bootstrap_residuals(model, resid, wild = True):
    '''
    https://en.wikipedia.org/wiki/Bootstrapping_(statistics) # Resampling_residuals
    https://en.wikipedia.org/wiki/Bootstrapping_(statistics) # Wild_bootstrap

    Davidson & Flachaire (2008) eq.(12) gives the recommended form
    of the wild bootstrapping probability used here.

    https://doi.org/10.1016/j.jeconom.2008.08.003

    Parameters:
    - spec: model (e.g. best fitting spectrum).
    - res: residuals (best_fit - observed).
    - wild: use wild bootstrap to allow for variable errors.

    Returns:
    - model + eps: new model with bootstrapped residuals.
    '''

    if wild:    # Wild Bootstrapping: generates -resid or resid with prob=1/2
        eps = resid*(2*np.random.randint(2, size = resid.size) - 1)

    else:       # Standard Bootstrapping: random selection with repetition
        eps = np.random.choice(resid, size = resid.size)

    return model + eps

#--------------------------------------------------------------------------------------------
def safe_log_rebin(wave, flux, nan_fill = 1e10):
    '''
    Run log_rebin safely by handling NaNs or inf in the input flux spectrum. Invalid values
    are replaced by a large number nan_fill. These large values can be excluded using the
    parameter goodpixels while using pPXF.

    Note that flux = False by default.

    Parameters:
    - wave: rest-frame (or observed-frame) wavelength in Å.
    - flux: flux with the unit proportional to erg/s/cm**2/angstrom.
    - nan_fill: fill invalid values (NaN or inf) with a large number nan_fill.

    Returns:
    - specNew: log-rebinned co-added spectrum.
    - ln_lam: wavelength range in log(e) scale.
    - velscale: velocity scale.
    '''

    flux_clean = np.copy(flux)
    flux_clean[~np.isfinite(flux_clean)] = nan_fill

    specNew, ln_lam, velscale = log_rebin(wave, flux_clean, flux = False)

    return specNew, ln_lam, velscale

#---------------------------------------------------------------------------------------------
def ppxf_pre_spectrum(cube_fits, spectrum_fits, high_redshift = False, save_fits = False,
                      convolve = False, wave_clip = False, miles_low = 3540.5, miles_high = 7409.6):
    '''
    Log-rebin co-added spectrum generated from data cube, do preparations for pPXF. The data cube
    will be used to construct wavelength and extract redshift. When the redshift is high, set the
    high_redshift parameter to be true.

    Parameters:
    - cube_fits: str, path to the data cube fits file.
    - spectrum_fits: str, path to the co-added spectrum fits file.
    - high_redshift: boolean, if True then high redshift situation will be considered.
    - save_fits: boolean, if True then save the initial spectrum and the log-rebinned spectrum.

    Returns:
    - goodpixels_nan: good pixels which will be fitted in pPXF.
    - specNew: log-rebinned co-added spectrum.
    - ln_lam: wavelength range in log(e) scale.
    - velscale: velocity scale.
    - redshift: redshift extracted from data cube.
    '''

    # open the data cube to construct the wavelength and extract redshift value.
    with fits.open(cube_fits) as blue_hdul:
        blue_header = blue_hdul[0].header

    # CRVAL3: coordinate value at reference point (Å).
    # NAXIS3: 2048 wavelength slices.
    # CRPIX3: 1024, pixel coordinate of reference point.
    # CDELT3: coordinate increment at reference point (Å).
    # np.arange(blue_header['NAXIS3']): generates an array of pixel indices from 0 to (NAXIS3 - 1).
    # np.arange(blue_header['NAXIS3']) - blue_header['CRPIX3']: the offset of each pixel from the reference.
    # * blue_header['CDELT3']: converts the pixel offset to a wavelength offset.
    # + blue_header['CRVAL3']: adds the starting wavelength to the calculated offset to get the actual wavelength for each pixel.
    # note that CRPIX is 1-based but np.arange() is 0-based, so + 1.
    blue_wavelength = blue_header['CRVAL3'] + (np.arange(blue_header['NAXIS3']) - blue_header['CRPIX3'] + 1) * blue_header['CDELT3']

    # de-redshifting to the rest frame.
    redshift = blue_header['Z_SPEC']
    cdelt3 = blue_header['CDELT3']
    rest_wavelength = blue_wavelength / (1 + redshift)

    # spectrum generated from the data cube.
    with fits.open(spectrum_fits) as spec_hdul:
        data_cube_spectrum = spec_hdul[0].data

    if convolve:
        data_cube_spectrum = spectrum_template_convolve(flux=data_cube_spectrum, redshift=redshift, cdelt3=cdelt3)

    if wave_clip:
        wave_mask = (rest_wavelength >= miles_low + 34) & (rest_wavelength <= miles_high)
        rest_wavelength = rest_wavelength[wave_mask]
        data_cube_spectrum = data_cube_spectrum[wave_mask]

    if high_redshift:
        # perform log_rebin with wavelength in observed frame (flux = False for flux density).
        specNew, ln_lam, velscale = safe_log_rebin(blue_wavelength, data_cube_spectrum)
    else:
        # perform log_rebin with wavelength in rest frame (flux = False for flux density).
        specNew, ln_lam, velscale = safe_log_rebin(rest_wavelength, data_cube_spectrum)

    # the goodpixels_nan will be used to normalize the galaxy and do the pPXF fitting.
    goodpixels_nan = np.where(specNew < 1e5)[0]

    if save_fits:
        # extension 1 [1] is the spectrum before log_rebin.
        hdu_initial_flux = fits.ImageHDU(data_cube_spectrum, name = 'INITIAL_FLUX')
        # the start of x-axis (index 1).
        hdu_initial_flux.header['CRVAL1'] = 1
        # the increment is 1 (for index).
        hdu_initial_flux.header['CDELT1'] = 1
        hdu_initial_flux.header['COMMENT'] = 'spectrum before log_rebin'

        # extension 2 [2] is the spectrum after log_rebin, which will be used to do the pPXF fitting.
        hdu_specNew = fits.ImageHDU(specNew, name = 'LOG_REBIN_FLUX')
        # the start of x-axis (index 1).
        hdu_specNew.header['CRVAL1'] = 1
        # the increment is 1 (for index).
        hdu_specNew.header['CDELT1'] = 1
        hdu_specNew.header['COMMENT'] = 'Log-rebinned spectrum prepared for pPXF'

        hdul = fits.HDUList([fits.PrimaryHDU(), hdu_initial_flux, hdu_specNew])
        hdul.writeto('prepared_spectrum_for_pPXF.fits', overwrite = True)

    return goodpixels_nan, specNew, ln_lam, velscale, redshift

#-----------------------------------------------------------------------------------
def ppxf_pre_data_cube(
        blue_spectrum, blue_cube_fits, red_spectrum = None, red_cube_fits = None, high_redshift = False, plot = False,
        convolve = False, wave_clip = False, miles_low = 3540.5, miles_high = 7409.6, fwhm_blue = 2.65, fwhm_red = 1.61):
    '''
    Log-rebin spectrum in the data cube, do preparations for pPXF. The data cube will be used to construct wavelength
    and extract redshift. When the redshift is high, set the high_redshift parameter to be true. If red spectrum and
    red_cube_fits are provided, the red spectrum will be convolved to match the resolution of the blue spectrum following
    the method described in Sande et al. 2017, the process would be: convolution -> interpolation -> combination -> log-rebin.

    Parameters:
    - blue_spectrum: spectrum extracted from blue data cube.
    - blue_cube_fits: str, path to the blue data cube fits file.
    - red_spectrum: spectrum extracted from red data cube.
    - red_cube_fits: str, path to the red data cube fits file.
    - high_redshift: boolean, if True then the high redshift situation will be considered.
    - plot: boolean, if True then the plot will be displayed.

    Returns:
    - goodpixels_nan: good pixels which will be fitted in pPXF.
    - specNew: log-rebinned spectrum.
    - ln_lam: wavelength range in log(e) scale.
    - velscale: velocity scale in km/s.
    - redshift: redshift extracted from blue data cube.
    '''

    # open the blue data cube to construct the blue wavelength and extract redshift value.
    with fits.open(blue_cube_fits) as blue_hdul:
        blue_header = blue_hdul[0].header
        redshift = blue_header['Z_SPEC']
        cdelt3 = blue_header['CDELT3']

    # note that CRPIX is 1-based but np.arange() is 0-based, so + 1.
    blue_wavelength = blue_header['CRVAL3'] + (np.arange(blue_header['NAXIS3']) - blue_header['CRPIX3'] + 1) * blue_header['CDELT3']

    if (red_spectrum is not None and red_cube_fits is None) or (red_spectrum is None and red_cube_fits is not None):
        raise ValueError('red_spectrum and red_cube_fits must be provided together.')

    if red_cube_fits is not None and red_spectrum is not None:
        with fits.open(red_cube_fits) as red_hdul:
            red_header = red_hdul[0].header

        # note that CRPIX is 1-based but np.arange() is 0-based, so + 1.
        red_wavelength = red_header['CRVAL3'] + (np.arange(red_header['NAXIS3']) - red_header['CRPIX3'] + 1) * red_header['CDELT3']

        # do the convolution to match the resolution of the red to the resolution of the blue.
        fwhm_conv = np.sqrt(fwhm_blue**2 - fwhm_red**2)
        sig_conv = fwhm_conv / (2 * np.sqrt(2 * np.log(2)))
        sig_conv = sig_conv / red_header['CDELT3'] # transfer to pixel scale.
        red_flux = gaussian_filter1d(red_spectrum, sig_conv)

        # do the interpolation.
        cdelt3 = blue_header['CDELT3']
        red_wave_interp = np.arange(red_wavelength[0], red_wavelength[-1] + cdelt3, cdelt3)

        interp_func = interp1d(red_wavelength, red_flux, kind = 'linear', bounds_error = False, fill_value = np.nan)
        red_flux_interp = interp_func(red_wave_interp)

        # introduce a gap between the blue wavelength range and the red wavelength range.
        # set the flux value in this gap region to be NaN, which could be excluded by using the goodpixel keyword.
        gap_wavelength = np.arange(blue_wavelength[-1] + cdelt3, red_wave_interp[0], cdelt3)
        gap_flux = np.full_like(gap_wavelength, np.nan)

        combined_wavelength = np.concatenate([blue_wavelength, gap_wavelength, red_wave_interp])
        combined_flux = np.concatenate([blue_spectrum, gap_flux, red_flux_interp])

        rest_wavelength = combined_wavelength / (1 + redshift)

        if high_redshift:
            # do the log-rebin with the wavelength in observed frame.
            specNew, ln_lam, velscale = safe_log_rebin(combined_wavelength, combined_flux)

            if plot:
                plot_spectrum(combined_wavelength, combined_flux)
        else:
            # do the log-rebin with the wavelength in rest frame.
            specNew, ln_lam, velscale = safe_log_rebin(rest_wavelength, combined_flux)

            if plot:
                plot_spectrum(rest_wavelength, combined_flux)

    else:
        rest_wavelength = blue_wavelength / (1 + redshift)

        if convolve:
            blue_spectrum = spectrum_template_convolve(flux = blue_spectrum, redshift = redshift, cdelt3 = cdelt3)

        if wave_clip:
            wave_mask = (rest_wavelength >= miles_low + 34) & (rest_wavelength <= miles_high)
            rest_wavelength = rest_wavelength[wave_mask]
            blue_spectrum = blue_spectrum[wave_mask]

        if high_redshift:
            specNew, ln_lam, velscale = safe_log_rebin(blue_wavelength, blue_spectrum)

            if plot:
                plot_spectrum(blue_wavelength, blue_spectrum)

        else:
            specNew, ln_lam, velscale = safe_log_rebin(rest_wavelength, blue_spectrum)

            if plot:
                plot_spectrum(rest_wavelength, blue_spectrum)

    # the goodpixels_nan will be used to normalize the galaxy and do the pPXF fitting.
    goodpixels_nan = np.where(specNew < 1e5)[0]

    return goodpixels_nan, specNew, ln_lam, velscale, redshift

#-----------------------------------------------------------------------------------

def ppxf_age_z(specNew, goodpixels_nan, ln_lam, noise_value, redshift, filename, velscale, start,
               nrand, optimal_regul = None, find_regul = False, high_redshift = False,
               buffer = None, plot = False, fwhm_blue = 2.65):
    '''
    calculate mean age and [M/Z] with ppxf using mdegree = 10, regul, constant noise value, bootstrapping and the MILES ssp model.
    Only two moments (vel, sig) are fitted.

    Parameters:
    - specNew: log-rebinned spectrum.
    - goodpixels_nan: good pixels which will be fitted in pPXF.
    - ln_lam: wavelength range in log(e) scale.
    - noise_value: assume a constant noise (reduced Chi2 ~ 1 without regularization).
    - redshift: redshift extracted from blue data cube.
    - filename: ssp model used to fit the input spectrum.
    - velscale: velocity scale in km/s.
    - start: initial start guessing value for three components ([[vel, sig], [vel, sig], [vel, sig]]).
    - nrand: do bootstrapping for nrand times.
    - optimal_regul: optimal regul value (e.g., 100).
    - find_regul: boolean, if True then the optimal regul value will be calculated and used.
    - high_redshift: boolean, if true then the high redshift situation will be considered.
    - buffer: adjust this value to ensure that lam_gal is within the wavelength range of the template.
    - plot: boolean, if True then the plot will be displayed.

    Returns:
    - age_mean: mean log age.
    - metallicity_mean: mean [M/H].
    - age_std: standard deviation of log age.
    - metallicity_std: standard deviation of [M/H].
    '''

    if high_redshift and buffer is None:
        raise ValueError('When high_redshift is True, buffer must be provided.')

    # normalize spectrum to avoid numerical issues.
    # note that this normalization factor should be multiplied back when calculating the ew for lick indices
    # and when generating the noise spectrum after ppxf fitting.
    galaxy = specNew.copy()
    galaxy = galaxy / np.median(galaxy[goodpixels_nan])

    '''
    In the ppxf_example_population code, they applied a conversion (lg --> ln), which is not needed here. 
    The log_rebin has already done the job.
    '''

    ln_lam_gal = ln_lam.copy()
    lam_gal = np.exp(ln_lam_gal)

    # the noise level is chosen to give Chi^2/DOF = 1 without regularization.
    noise = np.full_like(galaxy, noise_value)

    '''
    If the galaxy is at a significant redshift (z >= 0.03), one would need to apply a
    large velocity shift in pPXF to match the template to the galaxy spectrum. This would
    require a large initial value for the velocity (v >= 1e4 km/s) in the input parameter
    START = [v, sig]. An alternative solution consists of bringing the galaxy spectrum 
    roughly to the rest-frame wavelength, before calling pPXF. In practice there is no need
    to modify the spectrum before the usual LOG_REBIN, given that a redshift corresponds to
    a linear shift of the log-rebinned spectrum. One just needs to compute the wavelength 
    range in the rest-frame and adjust the instrumental resolution of the galaxy observations.
    '''

    if high_redshift:
        fwhm_gal = fwhm_blue

    else:
        fwhm_gal = fwhm_blue / (1 + redshift)

    # normalize the templates to mean = 1 within the FWHM (wavelength range) of the V-band (5000 Å ~ 6000 Å).
    # in this way the weights returned by pPXF and mean values are light-weighted quantities.
    sps = lib.sps_lib(filename, velscale, fwhm_gal, norm_range = [5000, 5500])

    # reshape the stellar templates into a 2-dim array with each spectrum as a column.
    # save the original array dimensions, which are needed to specify the regularization dimensions.
    reg_dim = sps.templates.shape[1:]
    stars_templates = sps.templates.reshape(sps.templates.shape[0], -1)

    # set up the gas emission lines templates.
    # estimated wavelength fitted range in the rest frame.
    if high_redshift:
        # lam_gal is in observed frame.
        lam_range_gal = np.array([np.min(lam_gal), np.max(lam_gal)]) / (1 + redshift)

    else:
        # lam_gal is already in rest frame.
        lam_range_gal = np.array([np.min(lam_gal), np.max(lam_gal)])

    # construct a set of Gaussian emission line templates.
    # here we modified the emission_line functions to only fit Hgamma, Hbeta, Halpha, [NII], [SII], [OI] and [OIII].
    # see SAMI_ppxf_util_functions for more details.
    gas_templates, gas_names, line_wave = emission_lines(sps.ln_lam_temp, lam_range_gal,
                                                              fwhm_gal, tie_balmer = 1)

    # combine the stellar and gaseous templates into a single array of templates.
    # during the pPXF fit they will be assigned a different kinematic component value.
    templates = np.column_stack([stars_templates, gas_templates])

    #############################################################################################################
    # important!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # the find_regul parameter will not be used.
    # here we normalize the templates and use optimal_regul = 100.
    templates = templates / np.median(templates)
    #############################################################################################################

    # consider two gas components, one for the Balmer and another for the forbidden lines.
    n_temps = stars_templates.shape[1]
    #############################################################################################################
    # important!
    # because we use tie_balmer, so here n_balmer would just be 1
    n_balmer = 1
    n_others = len(gas_names) - n_balmer
    #############################################################################################################

    # assign component = 0 to the stellar templates.
    # component = 1 to the Balmer gas emission lines templates.
    # component = 2 to the other gas emission lines.
    component = [0] * n_temps + [1] * n_balmer + [2] * n_others
    # gas_component = True for gas templates.
    gas_component = np.array(component) > 0

    # fit two moments (v, sig) moments = 2 for the stars and for the two gas kinematic components.
    moments = [2, 2, 2]

    if high_redshift:

        # we use the buffer to make sure the goodpixels are safely inside the template wavelength coverage.
        lam_mask = (lam_gal > sps.lam_temp[0]) & (lam_gal < sps.lam_temp[-1] - buffer)

        galaxy = galaxy[lam_mask]
        lam_gal = lam_gal[lam_mask]
        noise = noise[lam_mask]

        goodpixels_nan = np.where(galaxy < 1e5)[0]

    '''
    In the ppxf_example_population_gas_sdss.py, the author mentioned that: to avoid affecting the line strength
    of the spectral features, the additive polynomials are excluded (degree = -1), and only the multiplicative ones
    are used. This is only recommended for population, not for kinematic extraction, where additive polynomials are 
    always recommended.
    '''

    # decide_mdegree()

    print('Begin performing unregularized fit with mdegree = 10...')

    # the first unreg_pPXF fit to have a rough estimation of vel, sig and noise.
    pp_unreg = ppxf(templates = templates, galaxy = galaxy, noise = noise, velscale = velscale, start = start,
                    moments = moments, degree = -1, mdegree = 10, lam = lam_gal, lam_temp = sps.lam_temp,
                    goodpixels = goodpixels_nan, component = component, gas_component = gas_component,
                    gas_names = gas_names, reddening = 0, gas_reddening = 0)

    # rescale noise to achieve reduced Chi2 ~ 1.
    noise_rescaled = noise * np.sqrt(pp_unreg.chi2)

    start = pp_unreg.sol.copy()

    # the second unreg_pPXF fit based on better constrained vel, sig and noise.
    pp_unreg = ppxf(templates = templates, galaxy = galaxy, noise = noise_rescaled, velscale = velscale, start = start,
                    moments = moments, degree = -1, mdegree = 10, lam = lam_gal, lam_temp = sps.lam_temp,
                    goodpixels = goodpixels_nan, component = component, gas_component = gas_component,
                    gas_names = gas_names, reddening = 0, gas_reddening = 0)

    # better constrained start value.
    start = pp_unreg.sol.copy()

    # the third unreg_pPXF fit involving clean keyword to identify bad pixels.
    pp_unreg = ppxf(templates = templates, galaxy = galaxy, noise = noise_rescaled, velscale = velscale, start = start,
                    moments = moments, degree = -1, mdegree = 10, lam = lam_gal, lam_temp = sps.lam_temp,
                    goodpixels = goodpixels_nan, component = component, gas_component = gas_component,
                    gas_names = gas_names, reddening = 0, gas_reddening = 0, clean = True)

    # newly defined goodpixels.
    goodpixels_nan = pp_unreg.goodpixels.copy()

    # better constrained start value.
    start = pp_unreg.sol.copy()

    # the forth unreg_pPXF fit with newly defined goodpixels.
    pp_unreg = ppxf(templates = templates, galaxy = galaxy, noise = noise_rescaled, velscale = velscale, start = start,
                    moments = moments, degree = -1, mdegree = 10, lam = lam_gal, lam_temp = sps.lam_temp,
                    goodpixels = goodpixels_nan, component = component, gas_component = gas_component,
                    gas_names = gas_names, reddening = 0, gas_reddening = 0)

    # newly defined noise based on the newly defined goodpixels.
    noise_rescaled = noise_rescaled * np.sqrt(pp_unreg.chi2)

    # better constrained start value.
    start = pp_unreg.sol.copy()

    # final unreg_pPXF fit with newly rescaled noise based on newly defined goodpixels.
    pp_unreg = ppxf(templates = templates, galaxy = galaxy, noise = noise_rescaled, velscale = velscale, start = start,
                    moments = moments, degree = -1, mdegree = 10, lam = lam_gal, lam_temp = sps.lam_temp,
                    goodpixels = goodpixels_nan, component = component, gas_component = gas_component,
                    gas_names = gas_names, reddening = 0, gas_reddening = 0)

    start = pp_unreg.sol.copy()

    print(f'Unregularized fits finished, final reduced Chi2: {pp_unreg.chi2:.3f}')

    if optimal_regul is None and not find_regul:
        raise ValueError('Please provide a regul value or set find_regul to True.')

    # find the optimal regul value.
    # make the fit to be regularized because this suppresses the noise makes it more representative of the underlying galaxy spectrum.
    if optimal_regul is None and find_regul:

        # starting regularization guess value.
        regul = 10
        # multiplicative step size.
        regul_step = 1.5

        iteration = 0
        max_iter = 50  # maximum number of iterations
        best_regul = None

        # degrees of freedom.
        dof = goodpixels_nan.size

        while iteration < max_iter:
            print(f'Iteration {iteration + 1}: Testing regul = {regul}...')

            # perform regularized fit.
            pp_reg = ppxf(templates = templates, galaxy = galaxy, noise = noise_rescaled, velscale = velscale, start = start,
                          moments = moments, degree = -1, mdegree = 10, lam = lam_gal, lam_temp = sps.lam_temp,
                          goodpixels = goodpixels_nan, regul = regul, reg_dim = reg_dim, component = component, gas_component = gas_component,
                          gas_names = gas_names, reddening = 0, gas_reddening = 0, quiet = True)

            # chi2 with regularization and with rescaled noise spectrum.
            chi2_reg = pp_reg.chi2 * dof
            # compute delta chi2
            delta_chi2 = chi2_reg - dof
            delta_chi2_target = np.sqrt(2 * dof)
            print(f'ΔChi^2: {delta_chi2:.3f}, target ΔChi^2: {delta_chi2_target:.3f}')

            # check if the target ΔChi2 is achieved.
            if abs(delta_chi2 - delta_chi2_target) < 0.05 * delta_chi2_target:
                best_regul = regul
                break

            # adjust regul.
            if delta_chi2 < delta_chi2_target:
                regul *= regul_step  # increase regul.
            else:
                regul /= regul_step  # decrease regul.
                regul_step = np.sqrt(regul_step)  # make smaller adjustments

            iteration += 1

        if best_regul is None:
            print('Failed to converge to an optimal regul within max iterations.')
        else:
            print(f'Optimal regul found = {best_regul}')
            print(f'Final reduced Chi^2 = {pp_reg.chi2:.3f}')
            optimal_regul = best_regul

    # the pPXF fit with optimal_regul, rescaled noise and mdegree.
    pp = ppxf(templates = templates, galaxy = galaxy, noise = noise_rescaled, velscale = velscale, start = start,
              moments = moments, degree = -1, mdegree = 10, lam = lam_gal, lam_temp = sps.lam_temp,
              goodpixels = goodpixels_nan, regul = optimal_regul, reg_dim = reg_dim, component = component, gas_component = gas_component,
              gas_names = gas_names, reddening = 0, gas_reddening = 0)

    plt.figure(figsize = (15, 5))
    pp.plot()
    plt.title('pPXF fit with rescaled noise, regularization, mdegree and miles ssp')

    weights = pp.weights[~gas_component]  # exclude weights of the gas templates
    weights = weights.reshape(reg_dim) / weights.sum()  # normalized

    sps.mean_age_metal(weights)

    plt.figure(figsize = (9, 3))
    sps.plot(weights)

    plt.show()

    # start bootstrapping.
    # note that regul will not be included while mdegree will still be included.
    bestfit = pp.bestfit.copy()
    resid = galaxy[goodpixels_nan] - bestfit[goodpixels_nan]

    plt.figure(figsize = (15, 5))
    plt.plot(resid)
    plt.title('residual')
    plt.show()

    start = pp.sol.copy()

    # do not include regularization when doing the bootstrapping.
    np.random.seed(123)  # for reproducible results

    ages = []
    metallicities = []

    print('Start bootstrapping...')

    for j in range(nrand):

        # reconstruct full galaxy spectrum and bootstrapping the goodpixels regions only.
        galaxy_boot = bestfit.copy()
        galaxy_boot[goodpixels_nan] = bootstrap_residuals(bestfit[goodpixels_nan], resid)

        pp_boot = ppxf(templates = templates, galaxy = galaxy_boot, noise = noise_rescaled, velscale = velscale, start = start,
                       moments = moments, degree = -1, mdegree = 10, lam = lam_gal, lam_temp = sps.lam_temp,
                       goodpixels = goodpixels_nan, component = component, gas_component = gas_component,
                       gas_names = gas_names, reddening = 0, gas_reddening = 0, quiet = True)

        weights = pp_boot.weights[~gas_component]
        weights = weights.reshape(reg_dim) / weights.sum()

        mean_lg_age, mean_metal = sps.mean_age_metal(weights)
        ages.append(mean_lg_age)
        metallicities.append(mean_metal)

    ages = np.array(ages)
    age_mean = np.mean(ages)
    age_std = np.std(ages)

    metallicities = np.array(metallicities)
    metallicity_mean = np.mean(metallicities)
    metallicity_std = np.std(metallicities)

    print(f'lg_age: {age_mean:.3f} ± {age_std:.3f}')
    print(f'[M/H]: {metallicity_mean:.3f} ± {metallicity_std:.3f}')

    if plot:
        # visualization.
        # kde = True argument adds a smooth line representing the kernel density estimate.
        sns.set(style = 'whitegrid')

        plt.figure(figsize=(10, 8))

        # plot distribution of lg_age.
        plt.subplot(1, 2, 1)
        sns.histplot(ages, kde = True, color = 'skyblue', bins = 20)
        plt.title(f'Distribution of lg_age\nmean: {age_mean:.3f}, std: {age_std:.3f}')
        plt.xlabel('lg_age')

        # plot distribution of metallicity.
        plt.subplot(1, 2, 2)
        sns.histplot(metallicities, kde = True, color = 'orange', bins = 20)
        plt.title(f'Distribution of [M/H]\nmean: {metallicity_mean:.3f}, std: {metallicity_std:.3f}')
        plt.xlabel('[M/H]')

        plt.tight_layout()
        plt.show()

    return age_mean, metallicity_mean, age_std, metallicity_std

#----------------------------------------------------------------------------------------
'''
def decide_mdegree():

    plt.figure(figsize = (25, 15))

    # loop over mdegree values from 1 to 20.
    for mdegree in range(1, 21):
        # the first pPXF fit without regularization.
        pp_unreg = ppxf(templates = templates, galaxy = galaxy, noise = noise, velscale = velscale, start = start, 
                        moments = moments, degree = -1, mdegree = mdegree, lam = lam_gal, lam_temp = sps.lam_temp,
                        goodpixels = goodpixels_nan, component = component, gas_component = gas_component, 
                        gas_names = gas_names, reddening = 0, gas_reddening = 0)

        residual = galaxy - pp_unreg.bestfit
        offset = mdegree / 8
        plt.plot(residual[goodpixels_nan] + offset, label = f'mdegree = {mdegree}', alpha = 0.7)
        plt.axhline(y = offset, color = 'k', linestyle = '--', linewidth = 0.5)

    plt.xlabel('wavelength index')
    plt.ylabel('residuals')
    plt.legend(loc = 'upper right')
    plt.show()
'''
#---------------------------------------------------------------------------------------------
if __name__ == '__main__':

    # ----------------------------------------------------------------------------------
    # define constants.
    fwhm_blue = 2.65  # Å
    fwhm_red = 1.61  # Å
    c = 299792.458  # speed of light (km/s).

    # ------------------------------------------------------------------------------------
    '''
    # clean the blue and red data cube.
    sn_threshold = 10
    emission_free_range_blue = (4600, 4800)
    emission_free_range_red = (6800, 7000)
    wavelength_slice_index = 1024

    blue_fits_path = 'CATID_A_cube_blue.fits'
    blue_cleaned_data_cube = data_cube_clean_snr(blue_fits_path, sn_threshold, emission_free_range_blue, wavelength_slice_index)

    red_fits_path = 'CATID_A_cube_red.fits'
    red_cleaned_data_cube = data_cube_clean_snr(red_fits_path, sn_threshold, emission_free_range_red, wavelength_slice_index)
    '''
    #-----------------------------------------------------------------------------------
    # here we directly use the adaptively binned data cube.
    blue_fits_path = 'CATID_A_adaptive_blue.fits'
    red_fits_path = 'CATID_A_adaptive_red.fits'

    with fits.open(blue_fits_path) as hdul:
        blue_cube = hdul[0].data
        blue_cleaned_data_cube = np.ma.masked_invalid(blue_cube)

    with fits.open(red_fits_path) as hdul:
        red_cube = hdul[0].data
        red_cleaned_data_cube = np.ma.masked_invalid(red_cube)

    #-----------------------------------------------------------------------------------
    noise_value = 0.022

    # here we use the miles ssp model (see the code miles_ssp.py for more details).
    filename = 'miles_ssp_models_ch_padova.npz'

    start = [[0., 200.], [0., 200.], [0., 200.]]
    nrand = 100

    # -----------------------------------------------------------------------------------
    # process the entire 50*50 spatial grid.
    # initialize empty arrays to store the age and metallicity maps.
    age_map = np.zeros((50, 50))
    metallicity_map = np.zeros((50, 50))

    # loop through each spatial pixel in the 50*50 grid.
    for x in range(50):
        for y in range(50):

            # if a pixel is invalid in the blue cube or in the red cube, skip this pixel.
            if blue_cleaned_data_cube.mask[:, x, y].all() or red_cleaned_data_cube.mask[:, x, y].all():
                continue

            # extract the blue spectrum and corresponding red spectrum for each pixel.
            blue_spectrum = blue_cleaned_data_cube[:, x, y]
            red_spectrum = red_cleaned_data_cube[:, x, y]

            # set masked regions to be NaN.
            blue_spectrum = blue_spectrum.filled(np.nan)
            red_spectrum = red_spectrum.filled(np.nan)

            # around NaN values, sometimes there will be some very small values.
            # for each NaN value (e.g., at index a), set the adjacent values (a-1 and a+1) to NaN as well.
            for spec in [blue_spectrum, red_spectrum]:
                nan_idx = np.isnan(spec)
                for idx in range(1, len(spec) - 1):
                    if nan_idx[idx]:
                        spec[idx - 1] = np.nan
                        spec[idx + 1] = np.nan

            # combine the blue and red spectrum and do the log-rebin.
            goodpixels_nan, specNew, ln_lam, velscale, redshift = ppxf_pre_data_cube(
                blue_spectrum, blue_fits_path, red_spectrum, red_fits_path, high_redshift = False, plot = True
            )

            #vel = c * np.log(1 + redshift)
            #start = [[vel, 200.], [vel, 200.], [vel, 200.]]

            age_mean, metallicity_mean, age_std, metallicity_std = ppxf_age_z(
                specNew = specNew, goodpixels_nan = goodpixels_nan, ln_lam = ln_lam,
                noise_value = noise_value, redshift = redshift, filename = filename,
                velscale = velscale, start = start, nrand = nrand, optimal_regul = 100,
                find_regul = False, high_redshift = False, plot = True)

            age_map[x, y] = age_mean
            metallicity_map[x, y] = metallicity_mean

    plt.figure(figsize = (12, 8))

    plt.subplot(1, 2, 1)
    plt.imshow(age_map, cmap = 'jet', origin = 'lower')
    plt.title('log age map')
    plt.colorbar(label = 'log age')
    plt.title('log age map with adaptively binning for CATID')

    plt.subplot(1, 2, 2)
    plt.imshow(metallicity_map, cmap = 'jet', origin = 'lower')
    plt.colorbar(label = '[M/H]')
    plt.title('[M/H] map with adaptively binning for CATID')
    plt.tight_layout()

    plt.show()
    # ------------------------------------------------------------------------------------