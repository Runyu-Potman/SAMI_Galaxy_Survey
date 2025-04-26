import numpy as np
from astropy.io import fits
from ppxf.ppxf_util import log_rebin
import matplotlib.pyplot as plt
from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util
import ppxf.sps_util as lib
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import simps
import warnings
from SAMI_data_cube_age_Z_functions import plot_spectrum, bootstrap_residuals, safe_log_rebin, ppxf_pre_data_cube
#-----------------------------------------------------------------------------
def compute_atom_index(ln_lam, flux, index_name):
    bands = lick_indices_log[index_name]
    feature_mask = (ln_lam >= bands['feature'][0]) & (ln_lam <= bands['feature'][1])
    blue_mask = (ln_lam >= bands['blue'][0]) & (ln_lam <= bands['blue'][1])
    red_mask = (ln_lam >= bands['red'][0]) & (ln_lam <= bands['red'][1])

    # calculate the continum by averaging the flux in the blue and red regions.
    blue_mean =np.mean(flux[blue_mask])
    red_mean =np.mean(flux[red_mask])

    # interpolate the continuum in the feature region.
    blue_wave = np.mean(ln_lam[blue_mask])
    red_wave = np.mean(ln_lam[red_mask])

    # interpolate between the blue and red regions to get the continuum.
    cont_interp = np.interp(ln_lam[feature_mask], [blue_wave, red_wave], [blue_mean, red_mean])

    if feature_mask.any():
        ew = simps((1 - flux[feature_mask] / cont_interp), np.exp(ln_lam[feature_mask]))
    else:
        ew = np.nan

    return ew
#-------------------------------------------------------------------------------------
def ppxf_ew_Dn4000():
    galaxy = specNew.copy()

    # normalize the spectrum to avoid numerical issues.
    # in principle, this normalization factor should be multiplied back when extracting lick indices.
    galaxy = galaxy / np.median(galaxy[goodpixels_nan])

    '''
    In the ppxf_example_population code, they applied a conversion (lg --> ln), which is not needed here. 
    The log_rebin has already done the job.
    '''

    ln_lam_gal = ln_lam.copy()
    lam_gal = np.exp(ln_lam_gal)

    # choose the noise to give Chi2/DOF = 1 without regularization (regul = 0).
    # assume a uniform noise spectrum such that pPXF weights all pixels equally, without masking out any wavelength regions.
    noise = np.full_like(galaxy, noise_value)

    if high_redshift:
        fwhm_gal = fwhm_blue

    else:
        fwhm_gal = fwhm_blue / (1 + redshift)

    # lam_range: a two-elements vector specifying the wavelength range in Angstroms for which to extract the stellar templates.
    lam_range_temp = [3000, 8000]

    sps = lib.sps_lib(filename, velscale, fwhm_gal, lam_range = lam_range_temp)

    #goodpixels = util.determine_goodpixels(np.log(lam_gal), lam_range_temp)
    #combined_goodpixels = np.intersect1d(goodpixels_nan, goodpixels)

    # the first pPXF fit with uniform noise spectrum.
    pp_01 = ppxf(templates = sps.templates, galaxy = galaxy, noise = noise, velscale = velscale,
                 start = start, goodpixels = goodpixels_nan, moments = 2, degree = 10, mdegree = -1,
                 lam = lam_gal, lam_temp = sps.lam_temp)

    if plot:
        pp_01.plot()
        plt.title('pPXF initial fit with MILES stellar library')
        plt.show()

    noise_rescaled = noise * np.sqrt(pp_01.chi2)

    start = [pp_01.sol[0], pp_01.sol[1]]

    pp_01 = ppxf(templates = sps.templates, galaxy = galaxy, noise = noise_rescaled, velscale = velscale,
                 start = start, goodpixels = goodpixels_nan, moments = 2, degree = 10, mdegree = -1,
                 lam = lam_gal, lam_temp = sps.lam_temp)

    start = [pp_01.sol[0], pp_01.sol[1]]

    # in the second fit, the rescaled noise is utilized, and the clean keyword is set.
    pp_02 = ppxf(templates = sps.templates, galaxy = galaxy, noise = noise_rescaled, velscale = velscale,
                 start = start, goodpixels = goodpixels_nan, moments = 2, degree = 10, mdegree = -1,
                 lam = lam_gal, lam_temp = sps.lam_temp, clean = True)

    if plot:
        pp_02.plot()
        plt.title('pPXF second fit with MILES stellar library')
        plt.show()

    # identify new bad pixels.
    new_bad_pixels = np.setdiff1d(goodpixels_nan, pp_02.goodpixels)
    # ensure sorted order.
    new_bad_pixels = np.sort(new_bad_pixels)

    # a list including the new identified bad pixels and their expanded regions.
    expanded_new_bad_pixels = []

    # traverse through contiguous bad pixel regions.
    # loop through all bad pixels.
    i = 0
    while i < len(new_bad_pixels):
        # detect a contiguous bad pixel region.
        # first pixel in the region.
        begin = new_bad_pixels[i]
        # end keeps track of the last pixel in the bad pixel region.
        end = begin

        # expand the region until we find a gap.
        # check if the next pixel (new_bad_pixels[i + 1]) is exactly 1 step ahead of the current pixel.
        # find contiguous bad pixel region.
        while i + 1 < len(new_bad_pixels) and new_bad_pixels[i + 1] == new_bad_pixels[i] + 1:
            # extend the region.
            end = new_bad_pixels[i + 1]
            # move to the next pixel.
            i += 1

        # region_width: total number of pixels in this bad region.
        region_width = end - begin + 1

        # compute expansion width: 25% of region width, minium 1 pixel.
        if region_width == 1:
            expand = 1
        elif region_width <= 4:
            expand = 1
        elif region_width <= 8:
            expand = 2
        elif region_width <= 12:
            expand = 3
        elif region_width <= 16:
            expand = 4
        elif region_width <= 20:
            expand = 5
        elif region_width <= 24:
            expand = 6
        elif region_width <= 28:
            expand = 7
        elif region_width <= 32:
            expand = 8
        elif region_width <= 36:
            expand = 9
        elif region_width <= 40:
            expand = 10
        else:
            expand = max(10, int(0.25 * region_width))
            warnings.warn(f"Too large region width {region_width} encountered!")

        # expand begin and end of the region.
        # max(): ensures we don't go below pixel index 0.
        # min(): ensures we don't exceed the last valid pixel index.
        expanded_new_bad_pixels.extend(range(max(0, begin - expand), min(len(galaxy) - 1, end + expand + 1)))

        # move to the next region.
        i += 1

    # combine unique new and new expanded bad pixels.
    expanded_new_bad_pixels = np.union1d(new_bad_pixels, expanded_new_bad_pixels)
    expanded_new_bad_pixels = np.unique(expanded_new_bad_pixels)

    # combine initial bad pixels, new and new expanded bad pixels.
    bad_pixels_initial = np.setdiff1d(np.arange(len(galaxy)), goodpixels_nan)
    all_bad_pixels = np.union1d(bad_pixels_initial, expanded_new_bad_pixels)

    # final good pixels.
    goodpixels_final = np.setdiff1d(np.arange(len(galaxy)), all_bad_pixels)

    if not quiet:
        print(f'Number of initial bad pixels: {len(bad_pixels_initial)}')
        print(f'Number of new bad pixels identified using clean: {len(new_bad_pixels)}')
        print(f'Number of new and new expanded bad pixels: {len(expanded_new_bad_pixels)}')
        print(f'Number of all bad pixels: {len(all_bad_pixels)}')
        print(f'Number of original good pixels: {len(goodpixels_nan)}')
        print(f'Number of final good pixels for third fit: {len(goodpixels_final)}')

    start = [pp_02.sol[0], pp_02.sol[1]]


























































lick_indices = {
    'HdeltaA': {'feature': (4083.50, 4122.25), 'blue': (4041.60, 4079.75), 'red': (4128.50, 4161.00)}}

# because galaxy is being log_rebinned, so we transfer the defined wavelength range for each lick indices from linear space into log space.
lick_indices_log = {
    index: {
        'feature': np.sort(np.log(np.array(bands['feature']))),
        'blue': np.sort(np.log(np.array(bands['blue']))),
        'red': np.sort(np.log(np.array(bands['red'])))
    }
    for index, bands in lick_indices.items()
}



# start bootstrapping.
bestfit = pp.bestfit.copy()
resid = galaxy - bestfit
start = pp.sol.copy()

# do not include regularization when doing the bootstrapping.
np.random.seed(123) # for reproducible results
nrand = 100

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

