from SAMI_data_cube_quality_cut_functions import data_cube_clean_snr, kdc_separation
from SAMI_stellar_velocity_quality_cut_functions import quality_cut_stellar_velocity_map
from SAMI_data_cube_age_Z_functions import ppxf_pre_spectrum, ppxf_age_z
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

#-------------------------------------------------------------------
def spectrum_age_Z(vel_fits_file, sig_fits_file, combined_data_cube, ellipticity, a_pixel, pa, output_spectrum,
                   sn_threshold = 3, x_center = 25, y_center = 25, vmin = None, vmax = None, kdc = True):

    # look into the stellar velocity map for reference when extracting KDC.
    combined_mask, cleaned_vel_data = quality_cut_stellar_velocity_map(
        vel_fits_file, sig_fits_file, vmin = vmin, vmax = vmax)

    # clean the cube (the blue has already combined with the red).
    cleaned_data_cube = data_cube_clean_snr(fits_path = combined_data_cube,
                                            sn_threshold = sn_threshold)

    # kdc extraction.
    b_a = 1 - ellipticity
    # the value based on Kinemetry.
    b = a_pixel * b_a

    ellipse_mask = kdc_separation(x_center = x_center, y_center = y_center,
                                  a = a_pixel, b = b, pa = pa, cleaned_vel_data = cleaned_vel_data,
                                  vmin = vmin, vmax = vmax)

    # extend to 3D.
    ellipse_mask = np.repeat(ellipse_mask[np.newaxis, :, :], cleaned_data_cube.shape[0], axis = 0)

    if kdc:
        # mask = ~ellipse_mask leads to a kdc spectrum.
        cleaned_data_cube = np.ma.masked_array(cleaned_data_cube, mask = ~ellipse_mask)
    else:
        # mask = ellipse_mask leads to a none-kdc spectrum.
        cleaned_data_cube = np.ma.masked_array(cleaned_data_cube, mask = ellipse_mask)

    # co-adding the spectra.
    coadded_spectrum = np.ma.mean(cleaned_data_cube, axis = (1, 2))

    # replace masked values with NaN for saving.
    spectrum_to_save = coadded_spectrum.filled(np.nan)

    hdu = fits.PrimaryHDU(spectrum_to_save)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(output_spectrum, overwrite = True)

    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(cleaned_data_cube.shape[0]), spectrum_to_save)
    plt.xlabel('wavelength index')
    plt.ylabel('co-added flux')
    plt.title('co-added spectrum')
    plt.show()

#---------------------------------------------------------------------------------




































