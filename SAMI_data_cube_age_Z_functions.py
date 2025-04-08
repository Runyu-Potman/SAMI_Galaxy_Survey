import numpy as np
from astropy.io import fits
from ppxf.ppxf_util import log_rebin
from pathlib import Path
from urllib import request
import matplotlib.pyplot as plt
from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util
import ppxf.sps_util as lib
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from SAMI_data_cube_quality_cut_functions import data_cube_clean_percentage, data_cube_clean_snr
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

Note that all functions defined in this script assume low-redshift . Please refer to
SAMI_data_cube_age_Z_high_redshift_functions.py for more information.
  
version_01: 03/04/2025
version_02: 08/04/2025: new functions added.
'''
#-------------------------------------------------------------------------------
def plot_spectrum(wavelength, spectrum):
    plt.figure(figsize = (10, 8))
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

    # the pPXF fit with optima_regul, rescaled noise and clean.
    pp = ppxf(templates = templates, galaxy = galaxy, noise = noise_rescaled, velscale = velscale, start = start,
              moments = moments, degree = -1, mdegree = 10, lam = lam_gal, lam_temp = sps.lam_temp,
              goodpixels = goodpixels_nan, regul = optimal_regul, reg_dim = reg_dim, component = component, gas_component = gas_component,
              gas_names = gas_names, reddening = 0, gas_reddening = 0, clean = True)

    plt.figure(figsize = (15, 10))
    pp.plot()
    plt.title('pPXF fit with rescaled noise, regularization, mdegree and clean')

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
    start = pp.sol.copy()

    # do not include regularization when doing the bootstrapping.
    np.random.seed(123)  # for reproducible results

ages = []
metallicities = []
ews = {index: [] for index in lick_indices_log.keys()}
dn4000s = []

for j in range(nrand):
    galaxy_boot = bootstrap_residuals(bestfit, resid)

    pp_boot = ppxf(templates = templates, galaxy = galaxy_boot, noise = noise, velscale = velscale,
              start = start, moments = moments, degree = -1, mdegree = 10, lam = lam_gal, lam_temp = sps.lam_temp,
              goodpixels = goodpixels_nan, component = component, gas_component = gas_component, gas_names = gas_names,
              reddening = 0, gas_reddening = 0, quiet = True)

    weights = pp_boot.weights[~gas_component]
    weights = weights.reshape(reg_dim) / weights.sum()

    mean_lg_age, mean_metal = sps.mean_age_metal(weights)
    ages.append(mean_lg_age)
    metallicities.append(mean_metal)

    pp_boot_stellar = pp_boot.bestfit - pp_boot.gas_bestfit

    for index in lick_indices_log.keys():
        ews[index].append(compute_atom_index(ln_lam_gal, pp_boot_stellar, index))

    blue_band = (3850, 3950)
    red_band = (4000, 4100)
    blue_flux = np.mean(pp_boot_stellar[(ln_lam_gal >= np.log(blue_band[0])) & (ln_lam_gal <= np.log(blue_band[1]))])
    red_flux = np.mean(pp_boot_stellar[(ln_lam_gal >= np.log(red_band[0])) & (ln_lam_gal <= np.log(red_band[1]))])
    dn4000 = red_flux / blue_flux
    dn4000s.append(dn4000)

ages = np.array(ages)
age_mean = np.mean(ages)
age_std = np.std(ages)

metallicities = np.array(metallicities)
metallicity_mean = np.mean(metallicities)
metallicity_std = np.std(metallicities)

dn4000s = np.array(dn4000s)
dn4000_mean = np.mean(dn4000s)
dn4000_std = np.std(dn4000s)

print(f'lg_age: {age_mean:.3f} ± {age_std:.3f}')
print(f'[M/H]: {metallicity_mean:.3f} ± {metallicity_std:.3f}')
print(f'Dn4000: {dn4000_mean:.3f} ± {dn4000_std:.3f}')

for index, ew_values in ews.items():
    ew_mean = np.mean(ew_values)
    ew_std = np.std(ew_values)
    print(f'{index} EW: {ew_mean:.3f} ± {ew_std:.3f}')

#------------------------------------------------------------------------
# visualization.
# kde = True argument adds a smooth line representing the kernel density estimate.
# set up the plotting style.
sns.set(style = 'whitegrid')

# create a figure for all the plots.
plt.figure(figsize = (16, 12))

# plot distribution of lg_age.
plt.subplot(2, 2, 1)
sns.histplot(ages, kde = True, color = 'skyblue', bins = 20)
plt.title(f'Distribution of lg_age\nMean: {age_mean:.3f}, Std: {age_std:.3f}')
plt.xlabel('lg_age')

# plot distribution of metallicity.
plt.subplot(2, 2, 2)
sns.histplot(metallicities, kde = True, color = 'orange', bins = 20)
plt.title(f'Distribution of [M/H]\nMean: {metallicity_mean:.3f}, Std: {metallicity_std:.3f}')
plt.xlabel('[M/H]')

# plot distribution of Dn4000.
plt.subplot(2, 2, 3)
sns.histplot(dn4000s, kde = True, color = 'green', bins = 20)
plt.title(f'Distribution of Dn4000\nMean: {dn4000_mean:.3f}, Std: {dn4000_std:.3f}')
plt.xlabel('Dn4000')

# plot distributions of ew for each lick index.
# If ew_values is a dictionary, loop over them.
plt.subplot(2, 2, 4)
for index, ew_values in ews.items():
    sns.histplot(ew_values, kde = True, label = f'{index} EW', bins = 20)
plt.title(f'Distribution of Equivalent Widths (EW)')
plt.xlabel('Equivalent width')

plt.tight_layout()
plt.show()



























