import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from SAMI_data_cube_quality_cut_functions import data_cube_clean_snr

#------------------------------------------------------------------------------
def vorbin_pre_cube_combine(blue_cube_fits, red_cube_fits, output_filename):
    with fits.open(blue_cube_fits) as blue_hdul:
        blue_flux = blue_hdul[0].data
        blue_var = blue_hdul[1].data
        blue_header = blue_hdul[0].header

    with fits.open(red_cube_fits) as red_hdul:
        red_flux = red_hdul[0].data
        red_var = red_hdul[1].data
        red_header = red_hdul[0].header

    blue_cleaned_flux_cube = np.ma.masked_invalid(blue_flux)
    blue_cleaned_var_cube = np.ma.masked_invalid(blue_var)

    red_cleaned_flux_cube = np.ma.masked_invalid(red_flux)
    red_cleaned_var_cube = np.ma.masked_invalid(red_var)

fits_path = '227266_A_cube_red.fits'
#fits_path = 'binned_data_cube_s_n_20.fits'
#fits_path = '227266_A_adaptive_blue.fits'
output_filename = '227266_A_cube_red_vorbin_20.fits'

sn_threshold = 3
target_sn = 20
wavelength_slice_index = 1024
cleaned_data_cube, binNum, x_gen, y_gen, x_bar, y_bar, sn, nPixels, scale = data_cube_clean_snr(
    fits_path = fits_path, sn_threshold = sn_threshold, wavelength_slice_index = wavelength_slice_index, output_filename = output_filename, vorbin = True, target_sn = target_sn)





