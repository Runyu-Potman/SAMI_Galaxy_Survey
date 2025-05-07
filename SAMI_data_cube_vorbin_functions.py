import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from SAMI_data_cube_quality_cut_functions import data_cube_clean_snr


fits_path = '227266_A_cube_red.fits'
#fits_path = 'binned_data_cube_s_n_20.fits'
#fits_path = '227266_A_adaptive_blue.fits'
output_filename = '227266_A_cube_red_vorbin_20.fits'

sn_threshold = 3
target_sn = 20
wavelength_slice_index = 1024
cleaned_data_cube, binNum, x_gen, y_gen, x_bar, y_bar, sn, nPixels, scale = data_cube_clean_snr(
    fits_path = fits_path, sn_threshold = sn_threshold, wavelength_slice_index = wavelength_slice_index, output_filename = output_filename, vorbin = True, target_sn = target_sn)





