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
#--------------------------------------------------------------------------------------------
'''
Six functions are defined in this script:
1. plot_spectrum()
2. bootstrap_residuals()
3. ppxf_pre_spectrum()
4. ppxf_pre_data_cube()
5. ppxf_age_z()
6. decide_mdegree()

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
    plt.xlabel('rest-frame wavelength (Å)')
    plt.ylabel('flux')
    plt.title('spectrum before log-rebin')
    plt.show()

#--------------------------------------------------------------------------------
def bootstrap_residuals(model, resid, wild = True):
    """
    https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Resampling_residuals
    https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Wild_bootstrap

    Davidson & Flachaire (2008) eq.(12) gives the recommended form
    of the wild bootstrapping probability used here.

    https://doi.org/10.1016/j.jeconom.2008.08.003

    :param spec: model (e.g. best fitting spectrum)
    :param res: residuals (best_fit - observed)
    :param wild: use wild bootstrap to allow for variable errors
    :return: new model with bootstrapped residuals

    """
    if wild:    # Wild Bootstrapping: generates -resid or resid with prob=1/2
        eps = resid*(2*np.random.randint(2, size = resid.size) - 1)
    else:       # Standard Bootstrapping: random selection with repetition
        eps = np.random.choice(resid, size = resid.size)

    return model + eps

#---------------------------------------------------------------------------------------------
def ppxf_pre_spectrum(cube_fits, spectrum_fits):
    '''
    Log-rebin co-added spectrum generated from data cube, do preparations for pPXF. The data cube
    will be used to construct wavelength and extract redshift.

    Parameters:
    - cube_fits: str, path to the data cube fits file.
    - spectrum_fits: str, path to the co-added spectrum fits file.

    Returns:
    - goodpixels_nan: goodpixels which will be fitted in pPXF.
    - SpecNew: log-rebinned co-added spectrum.
    - ln_lam: log-resclaed wavelength range.
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
    blue_wavelength = blue_header['CRVAL3'] + (np.arange(blue_header['NAXIS3']) - blue_header['CRPIX3']) * blue_header['CDELT3']

    # de-redshifting to the rest frame.
    redshift = blue_header['Z_SPEC']
    rest_wavelength = blue_wavelength / (1 + redshift)

    # spectrum generated from the data cube.
    data_cube_spectrum = fits.open(spectrum_fits)
    data_cube_spectrum = data_cube_spectrum[0].data

    # perform log_rebin (flux = False for flux density).
    specNew, ln_lam, velscale = log_rebin(rest_wavelength, data_cube_spectrum, flux = False)

    # identify NaN values in the log_rebinned flux, replace NaN values with a large number.
    nan_mask_flux = np.isnan(specNew)
    specNew[nan_mask_flux] = 1e10

    # the goodpixels_nan will be used to normalize the galaxy and do the pPXF fitting.
    goodpixels_nan = np.where(specNew < 1e5)[0]

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
def ppxf_pre_data_cube(spectrum_blue, blue_cube_fits, spectrum_red = None, red_cube_fits = None, plot = False):
    # open the blue data cube to construct the blue wavelength and extract redshift value.
    with fits.open(blue_cube_fits) as blue_hdul:
        blue_header = blue_hdul[0].header
        redshift = blue_header['Z_SPEC']

    blue_wavelength = blue_header['CRVAL3'] + (np.arange(blue_header['NAXIS3']) - blue_header['CRPIX3']) * blue_header['CDELT3']

    if (spectrum_red is not None and red_cube_fits is None) or (spectrum_red is None and red_cube_fits is not None):
        raise ValueError('spectrum_red and red_cube_fits must be provided together.')

    if red_cube_fits is not None and spectrum_red is not None:
        with fits.open(red_cube_fits) as red_hdul:
            red_header = red_hdul[0].header

        red_wavelength = red_header['CRVAL3'] + (np.arange(red_header['NAXIS3']) - red_header['CRPIX3']) * red_header['CDELT3']

        # do the convolution to match the resolution of red to the resolution of the blue.
        fwhm_conv = np.sqrt(fwhm_blue**2 - fwhm_red**2)
        sig_conv = fwhm_conv / (2 * np.sqrt(2 * np.log(2)))
        sig_conv = sig_conv / red_header['CDELT3']
        red_flux = gaussian_filter1d(spectrum_red, sig_conv)

        # introduce a gap between the blue wavelength range and the red wavelength range.
        # set the flux value in this gap region to be NaN, which could be excluded by using the goodpixel keyword.
        # use a smaller of the two CDELT3s (better spectral resolution).
        cdelt3 = min(abs(blue_header['CDELT3']), abs(red_header['CDELT3']))
        gap_wavelength = np.arange(blue_wavelength[-1] + cdelt3, red_wavelength[0], cdelt3)
        gap_flux = np.full_like(gap_wavelength, np.nan)

        combined_wavelength = np.concatenate([blue_wavelength, gap_wavelength, red_wavelength])
        combined_flux = np.concatenate([spectrum_blue, gap_flux, red_flux])

        rest_wavelength = combined_wavelength / (1 + redshift)

        specNew, ln_lam, velscale = log_rebin(rest_wavelength, combined_flux, flux = False)

        if plot:
            plot_spectrum(rest_wavelength, combined_flux)

    else:
        rest_wavelength = blue_wavelength / (1 + redshift)

        specNew, ln_lam, velscale = log_rebin(rest_wavelength, spectrum_blue, flux = False)

        if plot:
            plot_spectrum(rest_wavelength, spectrum_blue)

    # identify NaN values in the log_rebinned flux, replace NaN values with a large number.
    nan_mask_flux = np.isnan(specNew)
    specNew[nan_mask_flux] = 1e10

    # the goodpixels_nan will be used to normalize the galaxy and do the pPXF fitting.
    goodpixels_nan = np.where(specNew < 1e5)[0]

    return goodpixels_nan, specNew, ln_lam, velscale, redshift

#-----------------------------------------------------------------------------------

def ppxf_age_z(specNew, goodpixels_nan, ln_lam, noise_value, redshift, filename, velscale,
               start, nrand, optimal_regul = None, find_regul = False, plot = False):

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
    lam_range_gal = np.array([np.min(lam_gal), np.max(lam_gal)])

    # construct a set of Gaussian emission line templates.
    gas_templates, gas_names, line_wave = util.emission_lines(sps.ln_lam_temp, lam_range_gal,
                                                              fwhm_gal, tie_balmer = 1)

    # combine the stellar and gaseous templates into a single array of templates.
    # during the pPXF fit they will be assigned a different kinematic component value.
    templates = np.column_stack([stars_templates, gas_templates])

    # consider two gas components, one for the Balmer and another for the forbidden lines.
    n_temps = stars_templates.shape[1]
    n_forbidden = np.sum(["[" in a for a in gas_names])
    n_balmer = len(gas_names) - n_forbidden

    # assign component = 0 to the stellar templates.
    # component = 1 to the Balmer gas emission lines templates.
    # component = 2 to the gas forbidden lines.
    component = [0] * n_temps + [1] * n_balmer + [2] * n_forbidden
    # gas_component = True for gas templates.
    gas_component = np.array(component) > 0

    # fit two moments (v, sig) moments = 2 for the stars and for the two gas kinematic components.
    moments = [2, 2, 2]

    '''
    In the ppxf_example_population_gas_sdss.py, the author mentioned that: to avoid affecting the line strength
    of the spectral features, the additive polynomials are excluded (degree = -1), and only the multiplicative ones
    are used. This is only recommended for population, not for kinematic extraction, where additive polynomials are 
    always recommended.
    '''

    # decide_mdegree()

    print('Performing unregularized fit with mdegree = 10...')

    # the first pPXF fit without regularization.
    pp_unreg = ppxf(templates = templates, galaxy = galaxy, noise = noise, velscale = velscale, start = start,
                    moments = moments, degree = -1, mdegree = 10, lam = lam_gal, lam_temp = sps.lam_temp,
                    goodpixels = goodpixels_nan, component = component, gas_component = gas_component,
                    gas_names = gas_names, reddening = 0, gas_reddening = 0)

    # extract Chi2 of the first fit.
    reduced_chi2_unreg = pp_unreg.chi2
    print(f'Unregularized reduced Chi^2 with initial noise spectrum: {reduced_chi2_unreg:.3f}')

    # degrees of freedom.
    dof = goodpixels_nan.size

    # rescale noise to achieve reduced chi-squared ~ 1.
    noise_rescaled = noise * np.sqrt(reduced_chi2_unreg)

    start = pp_unreg.sol.copy()

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

    # the pPXF fit with optima_regul, rescaled noise and mdegree.
    pp = ppxf(templates = templates, galaxy = galaxy, noise = noise_rescaled, velscale = velscale, start = start,
              moments = moments, degree = -1, mdegree = 10, lam = lam_gal, lam_temp = sps.lam_temp,
              goodpixels = goodpixels_nan, regul = optimal_regul, reg_dim = reg_dim, component = component, gas_component = gas_component,
              gas_names = gas_names, reddening = 0, gas_reddening = 0)

    plt.figure(figsize = (15, 10))
    pp.plot()
    plt.title('pPXF fit with rescaled noise, regularization and mdegree')

    weights = pp.weights[~gas_component]  # exclude weights of the gas templates
    weights = weights.reshape(reg_dim) / weights.sum()  # normalized

    sps.mean_age_metal(weights)

    plt.figure(figsize = (9, 6))
    sps.plot(weights)

    plt.show()

    # start bootstrapping.
    # note that regul will not be included while mdegree will still be included.
    bestfit = pp.bestfit.copy()
    resid = galaxy - bestfit
    # previously we replace those bad pixels with extreme values.
    resid[resid > 1e5] = 0

    start = pp.sol.copy()

    # do not include regularization when doing the bootstrapping.
    np.random.seed(123)  # for reproducible results

    ages = []
    metallicities = []

    for j in range(nrand):

        galaxy_boot = bootstrap_residuals(bestfit, resid)

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
        # set up the plotting style.
        sns.set(style = 'whitegrid')

        plt.figure(figsize=(10, 8))

        # plot distribution of lg_age.
        plt.subplot(1, 2, 1)
        sns.histplot(ages, kde = True, color = 'skyblue', bins = 50)
        plt.title(f'Distribution of lg_age\nmean: {age_mean:.3f}, std: {age_std:.3f}')
        plt.xlabel('lg_age')

        # plot distribution of metallicity.
        plt.subplot(1, 2, 2)
        sns.histplot(metallicities, kde = True, color = 'orange', bins = 50)
        plt.title(f'Distribution of [M/H]\nmean: {metallicity_mean:.3f}, std: {metallicity_std:.3f}')
        plt.xlabel('[M/H]')

        plt.tight_layout()
        plt.show()

    return age_mean, metallicity_mean

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
    red_cleaned_data_cube = data_cube_clean_percentage(red_fits_path, percentage, wavelength_slice_index)

    # -----------------------------------------------------------------------------------
    # step 2: extract the blue spectrum and corresponding red spectrum for each pixel.
    # if the pixel is masked in blue cube or in the red cube, skip this pixel.
    # a testing example is given here (25, 25):
    x = 25
    y = 25

    blue_spectrum = blue_cleaned_data_cube[:, x, y]
    red_spectrum = red_cleaned_data_cube[:, x, y]

    blue_spectrum = blue_spectrum.filled(np.nan)
    red_spectrum = red_spectrum.filled(np.nan)

    # around NaN values, sometimes there will be some very small values.
    # for each NaN value (e.g., at index a), set the adjacent values (a-1 and a+1) to NaN as well.
    nan_indices = np.isnan(blue_spectrum)
    for idx in range(1, len(blue_spectrum) - 1):
        if nan_indices[idx]:
            blue_spectrum[idx - 1] = np.nan
            blue_spectrum[idx + 1] = np.nan

    nan_indices = np.isnan(red_spectrum)
    for idx in range(1, len(red_spectrum) - 1):
        if nan_indices[idx]:
            red_spectrum[idx - 1] = np.nan
            red_spectrum[idx + 1] = np.nan

    #-----------------------------------------------------------------------------------
    # step 3: combine blue and red spectrum and do the log-rebin.
    goodpixels_nan, specNew, ln_lam, velscale, redshift = ppxf_pre_data_cube(
        blue_spectrum, blue_fits_path, red_spectrum, red_fits_path, plot = True
    )







    #---------------------------------------------------------------------------------------
    # step 4: extimate age and metallicity.
    noise_value = 0.022

    # here we use the miles ssp model (see the code miles_ssp.py for more details).
    filename = 'miles_ssp_models_ch_padova.npz'
    start = [[0., 187.], [0., 187.], [0., 187.]]
    nrand = 200


    ppxf_age_z(specNew = specNew, goodpixels_nan = goodpixels_nan, ln_lam = ln_lam,
               noise_value = noise_value, redshift = redshift, filename = filename,
               velscale = velscale, start = start, nrand = nrand, find_regul = True,
               plot = True)
    # ------------------------------------------------------------------------------------