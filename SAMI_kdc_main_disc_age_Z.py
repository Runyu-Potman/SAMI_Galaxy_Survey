from SAMI_data_cube_quality_cut_functions import data_cube_clean_snr, kdc_separation
from SAMI_stellar_velocity_quality_cut_functions import quality_cut_stellar_velocity_map
from SAMI_data_cube_age_Z_functions import ppxf_pre_spectrum, ppxf_age_z
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

#-------------------------------------------------------------------
def spectrum_age_Z(vel_fits_file, sig_fits_file, combined_data_cube, ellipticity, a_pixel, pa, output_spectrum,
                   sn_threshold = 3, x_center = 25, y_center = 25, vmin = None, vmax = None, kdc = True):
    '''
    Extract the co-added spectrum for the kdc region or outside the kdc region. Do the cleaning before the co-adding
    process using the combined (blue and red) data cube.

    Parameters:
    - vel_fits_file: stellar velocity fits file, just for visualization.
    - sig_fits_file: stellar velocity dispersion fits file, just for visualization.
    - combined_data_cube: the combined (blue and red) data cube file.
    - ellipticity: extracted from the catalog.
    - a_pixel: kdc radius in pixel, given by kinemetry.
    - pa: position angle of the kdc or non-kdc region, starting from North, counterclockwise to the receding side.
    - output_spectrum: path to the output co-added spectrum file.
    - sn_threshold: s/n threshold cleaning for the combined data cube before co-adding.
    - x_center: x center of the galaxy.
    - y_center: y center of the galaxy.
    - vmin: colorbar.
    - vmax: colorbar.
    - kdc: if True, kdc region will be co-added, if False, non-kdc region will be co-added.

    Returns:
    - None.

    '''

    # look into the stellar velocity map for reference when extracting KDC.
    combined_mask, cleaned_vel_data = quality_cut_stellar_velocity_map(
        vel_fits_file, sig_fits_file, vmin = vmin, vmax = vmax)

    # clean the cube (the blue has already combined with the red).
    cleaned_data_cube = data_cube_clean_snr(fits_path = combined_data_cube,
                                            sn_threshold = sn_threshold)

    # kdc extraction.
    # the axis ratio b/a.
    b_a = 1 - ellipticity
    # b is also in pixel scale.
    b = a_pixel * b_a

    # the mask separating kdc and main disc.
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
'''
import ppxf.sps_util as lib
from pathlib import Path
from urllib import request

# emiles.
sps_name = 'emiles'

ppxf_dir = Path(lib.__file__).parent
basename = f"spectra_{sps_name}_9.0.npz"
filename = ppxf_dir / 'sps_models' / basename
if not filename.is_file():
    url = "https://raw.githubusercontent.com/micappe/ppxf_data/main/" + basename
    request.urlretrieve(url, filename)
'''
#--------------------------------------------------------------------------------
# define constant.
filename = 'miles_ssp_models_basti_ch_baseFe.npz'
start = [[0., 200.], [0., 200.], [0., 200.]]
nrand = 100
#--------------------------------------------------------------------------------
galaxy_name = 143287
noise_value = 0.025
#----------------------------------------------------------------------------------
vel_fits_file = f'{galaxy_name}/kinematic/{galaxy_name}_A_stellar-velocity_default_two-moment.fits'
sig_fits_file = f'{galaxy_name}/kinematic/{galaxy_name}_A_stellar-velocity-dispersion_default_two-moment.fits'
combined_data_cube = f'{galaxy_name}/age_z/{galaxy_name}_A_cube_combined_pre_vorbin.fits'
output_spectrum = f'{galaxy_name}/age_z/{galaxy_name}_co-added_spectrum_kdc.fits'

spectrum_age_Z(vel_fits_file = vel_fits_file, sig_fits_file = sig_fits_file, combined_data_cube = combined_data_cube,
               ellipticity = 0.368, a_pixel = 6.4, pa = 43.13, output_spectrum = output_spectrum, kdc = True)
#--------------------------------------------------------------------------------
spectrum_fits = output_spectrum
cube_fits = combined_data_cube




































