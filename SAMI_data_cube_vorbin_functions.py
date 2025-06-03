import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from SAMI_data_cube_quality_cut_functions import data_cube_clean_snr

#-------------------------------------------------------------------------------
def nan_safe_gaussian_filter1d (data, sigma):
    # identify NaNs.
    nan_mask = np.isnan(data)

    # replace NaNs with 0 temporarily, but keep track of them.
    data_filled = np.nan_to_num(data, nan = 0)

    # make a weight array: 1 where data is valid, 0 where it's NaN.
    weights = (~nan_mask).astype(float)

    # convolve both the filled data and the weights.
    smoothed_data = gaussian_filter1d(data_filled, sigma = sigma)
    smoothed_weights = gaussian_filter1d(weights, sigma = sigma)

    # avoid division by zero by adding a small epsilon to the denominator.
    epsilon = 1e-10
    smoothed_weights = np.where(smoothed_weights == 0, epsilon, smoothed_weights)

    # normalize the result by the weights.
    result = smoothed_data / smoothed_weights

    # where there was no valid data, set output to NaN.
    result[smoothed_weights == epsilon] = np.nan

    return result

#------------------------------------------------------------------------------
def vorbin_pre_cube_combine(blue_cube_fits, red_cube_fits, output_filename):
    # open the blue data cube.
    with fits.open(blue_cube_fits) as blue_hdul:
        blue_flux = blue_hdul[0].data
        blue_var = blue_hdul[1].data
        blue_header = blue_hdul[0].header

    # open the red data cube.
    with fits.open(red_cube_fits) as red_hdul:
        red_flux = red_hdul[0].data
        red_var = red_hdul[1].data
        red_header = red_hdul[0].header

    # data cube shape for later use.
    blue_flux_shape_wave = blue_flux.shape[0]
    blue_flux_shape_y = blue_flux.shape[1]
    blue_flux_shape_x = blue_flux.shape[2]
    # debug.
    print('blue data cube shape:', blue_flux_shape_wave, blue_flux_shape_y, blue_flux_shape_x)

    # mask invalid values in blue cube.
    blue_cleaned_flux_cube = np.ma.masked_invalid(blue_flux)
    blue_cleaned_var_cube = np.ma.masked_invalid(blue_var)

    # mask invalid values in red cube.
    red_cleaned_flux_cube = np.ma.masked_invalid(red_flux)
    red_cleaned_var_cube = np.ma.masked_invalid(red_var)

    # construct blue and red wavelengths.
    # note that CRPIX is 1-based but np.arange() is 0-based, so + 1.
    blue_wavelength = blue_header['CRVAL3'] + (np.arange(blue_header['NAXIS3']) - blue_header['CRPIX3'] + 1) * blue_header['CDELT3']
    red_wavelength = red_header['CRVAL3'] + (np.arange(red_header['NAXIS3']) - red_header['CRPIX3'] + 1) * red_header['CDELT3']
    # debug.
    print('blue wavelength range:', blue_wavelength[0], blue_wavelength[-1])
    print('red wavelength range:', red_wavelength[0], red_wavelength[-1])

    # do the convolution to match the resolution of the red to the resolution of the blue.
    fwhm_conv = np.sqrt(fwhm_blue ** 2 - fwhm_red ** 2)
    sig_conv = fwhm_conv / (2 * np.sqrt(2 * np.log(2)))
    sig_conv = sig_conv / red_header['CDELT3']  # transfer to pixel scale.

    # do the interpolation on red wavelength based on the increment of blue wavelength.
    cdelt3 = blue_header['CDELT3']
    red_wave_interp = np.arange(red_wavelength[0], red_wavelength[-1] + cdelt3, cdelt3)

    # introduce a gap between the blue wavelength range and the red wavelength range.
    # set the flux value in this gap region to be NaN, which could be excluded by using the goodpixel keyword.
    gap_wavelength = np.arange(blue_wavelength[-1] + cdelt3, red_wave_interp[0], cdelt3)
    gap_flux = np.full_like(gap_wavelength, np.nan)
    gap_var = np.full_like(gap_wavelength, np.nan)

    # combine blue, gap and interpolated red wavelength.
    combined_wavelength = np.concatenate([blue_wavelength, gap_wavelength, red_wave_interp])

    # length of combined wavelength for fits file storing.
    nwave_combined = len(combined_wavelength)

    # output fits file cubes.
    shape = (nwave_combined, blue_flux_shape_y, blue_flux_shape_x)
    # extension one is combined flux.
    combined_flux_cube = np.full(shape, np.nan, dtype = np.float32)
    # extension two is combined variance.
    combined_var_cube = np.full(shape, np.nan, dtype = np.float32)

    # loop through each spatial pixel in the 50*50 grid.
    for x in range(blue_flux_shape_x):
        for y in range(blue_flux_shape_y):
            # skip if a spaxel is totally masked in the wavelength dimension.
            if (blue_cleaned_flux_cube.mask[:, y, x].all() or
                red_cleaned_flux_cube.mask[:, y, x].all() or
                blue_cleaned_var_cube.mask[:, y, x].all() or
                red_cleaned_var_cube.mask[:, y, x].all()):

                continue

            # extract the flux spectrum and variance spectrum.
            # fill the masked regions with NaN.
            blue_flux = blue_cleaned_flux_cube[:, y, x].filled(np.nan)
            red_flux = red_cleaned_flux_cube[:, y, x].filled(np.nan)
            blue_var = blue_cleaned_var_cube[:, y, x].filled(np.nan)
            red_var = red_cleaned_var_cube[:, y, x].filled(np.nan)

            # first step would be the convolution.
            red_flux = nan_safe_gaussian_filter1d(red_flux, sig_conv)
            red_noise = np.sqrt(red_var)
            red_noise = nan_safe_gaussian_filter1d(red_noise, sig_conv)
            red_var = red_noise**2

            # second step would be the interpolation.
            interp_func_flux = interp1d(red_wavelength, red_flux, kind = 'linear', bounds_error = False, fill_value = np.nan)
            interp_func_var = interp1d(red_wavelength, red_var, kind = 'linear', bounds_error = False, fill_value = np.nan)

            red_flux_interp = interp_func_flux(red_wave_interp)
            red_var_interp = interp_func_var(red_wave_interp)

            # third step would be the combination.
            combined_flux = np.concatenate([blue_flux, gap_flux, red_flux_interp])
            combined_var = np.concatenate([blue_var, gap_var, red_var_interp])

            combined_flux_cube[:, x, y] = combined_flux
            combined_var_cube[:, x, y] = combined_var

    hdu_flux = fits.PrimaryHDU(data = combined_flux_cube)
    hdu_flux.header['CTYPE3'] = 'WAVELENGTH'
    hdu_flux.header['CUNIT3'] = 'Angstrom'
    hdu_flux.header['CRVAL3'] = combined_wavelength[0]
    hdu_flux.header['CRPIX3'] = 1
    hdu_flux.header['CDELT3'] = cdelt3
    z = blue_header['Z_SPEC']
    hdu_flux.header['Z_SPEC'] = z

    hdu_var = fits.ImageHDU(data = combined_var_cube, name = 'VARIANCE')

    hdul = fits.HDUList([hdu_flux, hdu_var])
    hdul.writeto(output_filename, overwrite = True)

#--------------------------------------------------------------------------

if __name__ == "__main__":
    # define constants.
    fwhm_blue = 2.65  # Å
    fwhm_red = 1.61  # Å

    #------------------------------------------------------------------
    blue_cube_fits = 'CATID_A_cube_blue.fits'
    red_cube_fits = 'CATID_A_cube_red.fits'
    output_filename = 'CATID_A_cube_combined_pre_vorbin.fits'
    vorbin_pre_cube_combine(blue_cube_fits, red_cube_fits, output_filename)

    #-------------------------------------------------------------------
    fits_path = 'CATID_A_cube_combined_pre_vorbin.fits'
    #fits_path = 'binned_data_cube_s_n_20.fits'
    #fits_path = 'CATID_A_adaptive_blue.fits'

    output_filename = 'CATID_A_cube_combined_vorbin_20.fits'

    sn_threshold = 3
    target_sn = 20
    wavelength_slice_index = 1024
    cleaned_data_cube, binNum, x_gen, y_gen, x_bar, y_bar, sn, nPixels, scale = data_cube_clean_snr(
        fits_path = fits_path, sn_threshold = sn_threshold, wavelength_slice_index = wavelength_slice_index, output_filename = output_filename, vorbin = True, target_sn = target_sn)









