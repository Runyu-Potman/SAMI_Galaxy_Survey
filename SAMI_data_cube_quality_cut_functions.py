from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

#------------------------------------------------------------------------------------------
'''
First of all, we do the quality cut on the data cube. The size of the data 
cube is 2048*50*50 (a spatial dimension: 50*50 and a wavelength dimension: 2048).
The y axis in the wavelength dimension is flux with the unit: 
10**(-16) erg/s/cm**2/angstrom/pixel.

When doing the quality cut, there are two different approaches:
1. For each pixel in the spatial map, calculate the total flux (do the integration
along the wavelength dimension), find the central pixel which has the peak flux
value, exclude those pixels with a total flux value less than 1% of the peak.

2. Similarly, calculate the total flux and also the total noise. Then, we exclude those 
pixels with a S/N smaller than a specific threshold. In terms of the noise, we choose the
variance spectrum (will consider the covariance in a future version).

we want to select those pixels which belong to the kdc region, and get the coadded spectrum.
A function using ellipse mask to separate the kdc region is defined. Also the combined_mask, 
which is defined when doing the quality cut for the stellar velocity map, should also be included. 

version_01: 01/04/2025
'''
#-------------------------------------------------------------------------------------------
# extract the combined mask based on the quality cut on the stellar velocity map.
# those functions are stored in the directory: clone, which is within the ppxf directory.
from SAMI_stellar_velocity_quality_cut_functions import quality_cut_stellar_velocity_map
vel_fits_file = 'CATID_A_stellar-velocity_default_two-moment.fits'
sig_fits_path = 'CATID_A_stellar-velocity-dispersion_default_two-moment.fits'
vmin = -75
vmax = 75

combined_mask, cleaned_vel_data = quality_cut_stellar_velocity_map(vel_fits_file, sig_fits_path)
#--------------------------------------------------------------------------------------------
def data_cube_clean_percentage(fits_path, percentage, wavelength_slice_index, combined_mask = None):
    '''
    Clean the data cube by excluding pixels with a total flux value less than
    a given percentage of the peak flux.

    Parameters:
    - fits_path: str, path to the data cube fits file.
    - percentage: float, percentage of flux to do the cleaning, e.g., 0.01.
    - wavelength_slice_index: int, wavelength slice to visualize before clean and after cleaning.
    - combined_mask: 2D mask, the mask from other maps (e.g., stellar velocity maps).

    Returns:
    - cleaned_data_cube: masked array containing the cleaned data cube.
    '''

    # read the primary data in extension [0].
    data_cube = fits.open(fits_path)
    header = data_cube[0].header
    data_cube = data_cube[0].data
    print('Shape of data cube:', data_cube.shape)

    # extract the wavelength step (in Angstroms) from the header.
    delta_lambda = header['CDELT3']

    # mask all the invalid values across all dimensions.
    data_cube = np.ma.masked_invalid(data_cube)

    # axis = 0 corresponds to the wavelength dimension.
    # calculate the total flux by integration along the wavelength dimension.
    # after summation, total flux will have the shape 50*50.
    total_flux = np.sum(data_cube, axis = 0) * delta_lambda
    print('Shape of the total flux map:', total_flux.shape)

    plt.imshow(total_flux, cmap = 'jet')
    plt.colorbar()
    plt.xlabel('SPAXEL X')
    plt.ylabel('SPAXEL Y')
    plt.title(f'data cube being collapsed along the wavelength dimension')
    plt.show()

    # find the location of the peak flux in the total flux map.
    peak_location = np.unravel_index(np.argmax(total_flux), total_flux.shape)
    print('In the total flux map, the peak flux is located at:', peak_location)
    peak_value = total_flux[peak_location]
    print(f'peak total flux value: {peak_value} * 10**(-16) erg/s/cm**2.')

    # exclude those pixels with total flux value less than the threshold.
    threshold = percentage * peak_value
    mask = total_flux < threshold
    # broadcast the 2D mask to match the shape of the 3D data cube.
    mask = np.repeat(mask[np.newaxis, :, :], data_cube.shape[0], axis = 0)

    cleaned_data_cube = np.ma.masked_where(mask, data_cube)

    if combined_mask is not None:
        combined_mask = np.repeat(combined_mask[np.newaxis, :, :], cleaned_data_cube.shape[0], axis = 0)
        cleaned_data_cube = np.ma.masked_where(combined_mask, cleaned_data_cube)

    # the data cube before cleaning.
    plt.imshow(data_cube[wavelength_slice_index, :, :], cmap = 'jet')
    plt.colorbar()
    plt.xlabel('SPAXEL X')
    plt.ylabel('SPAXEL Y')
    plt.title(f'data cube before cleaning at the wavelength slice: {wavelength_slice_index}')
    plt.show()

    # the data cube after cleaning.
    plt.imshow(cleaned_data_cube[wavelength_slice_index, :, :], cmap = 'jet')
    plt.colorbar()
    plt.xlabel('SPAXEL X')
    plt.ylabel('SPAXEL Y')
    plt.title(f'data cube after cleaning at the wavelength slice: {wavelength_slice_index}')
    plt.show()

    return cleaned_data_cube

#------------------------------------------------------------------------------
def data_cube_clean_snr(fits_path, total_sn_threshold, wavelength_slice_index, per_wave_sn_threshold = None, combined_mask = None):
    '''
    Clean the data cube using a two-step S/N filtering approach:
    1. Compute  total S/N per pixel (integrated over wavelength) and exclude pixels with low total S/N ratio.
    2. For the remaining pixels, apply a per-wavelength S/N cut.

    Parameters:
    - fits_path: str, path to the data cube fits file.
    - total_sn_threshold: float, minimum total S/N required for a pixel to be included.
    - per_wave_sn_threshold: float, minimum S/N required per wavelength slice.
    - wavelength_slice_index: int, wavelength slice to visualize before clean and after cleaning.
    - combined_mask: 2D mask, the mask from other maps (e.g., stellar velocity maps).

    Returns:
    - cleaned_data_cube: masked array containing the cleaned data cube.
    '''

    # read the primary data in extension [0] and the variance in [1].
    with fits.open(fits_path) as hdul:
        data_cube = hdul[0].data # flux data
        var = hdul[1].data # variance data

    data_cube = np.ma.masked_invalid(data_cube)
    var = np.ma.masked_invalid(var)

    total_flux = np.sum(data_cube, axis = 0)
    total_noise = np.sqrt(np.sum(var, axis = 0))
    total_sn = np.where(total_noise > 0, total_flux / total_noise, 0)
    total_sn_mask = total_sn < total_sn_threshold

    # plot the total S/N map before masking.
    plt.imshow(total_sn, cmap = 'jet')
    plt.colorbar(label = 'total S/N')
    plt.xlabel('SPAXEL X')
    plt.ylabel('SPAXEL Y')
    plt.title('total S/N map before any masking')
    plt.show()

    total_sn_mask = np.repeat(total_sn_mask[np.newaxis, :, :], data_cube.shape[0], axis = 0)
    cleaned_data_cube = np.ma.masked_where(total_sn_mask, data_cube)

    if per_wave_sn_threshold is not None:
        # apply per_wavelength S/N mask.
        # compute per_wavelength S/N for the remaining pixels.
        noise = np.sqrt(var)
        per_sn = np.where(noise > 0, cleaned_data_cube / noise, 0)
        per_sn_mask = per_sn < per_wave_sn_threshold
        cleaned_data_cube = np.ma.masked_where(per_sn_mask, cleaned_data_cube)

    if combined_mask is not None:
        combined_mask = np.repeat(combined_mask[np.newaxis, :, :], cleaned_data_cube.shape[0], axis = 0)
        cleaned_data_cube = np.ma.masked_where(combined_mask, cleaned_data_cube)

    plt.imshow(data_cube[wavelength_slice_index, :, :], cmap = 'jet')
    plt.colorbar()
    plt.xlabel('SPAXEL X')
    plt.ylabel('SPAXEL Y')
    plt.title(f'data cube before cleaning at the wavelength slice: {wavelength_slice_index}')
    plt.show()

    plt.imshow(cleaned_data_cube[wavelength_slice_index, :, :], cmap = 'jet')
    plt.colorbar()
    plt.xlabel('SPAXEL X')
    plt.ylabel('SPAXEL Y')
    plt.title(f'data cube after cleaning at the wavelength slice: {wavelength_slice_index}')
    plt.show()

    return cleaned_data_cube

#-------------------------------------------------------------------------------
fits_path = 'CATID_A_cube_blue.fits'
#total_sn_threshold = 400
#per_wave_sn_threshold = 10
percentage = 0.1
wavelength_slice_index = 1024
combined_mask = combined_mask
#cleaned_data_cube = data_cube_clean_snr(fits_path, total_sn_threshold, wavelength_slice_index, combined_mask)
cleaned_data_cube = data_cube_clean_percentage(fits_path, percentage, wavelength_slice_index, combined_mask)
#----------------------------------------------------------------------------------------
# we separate the kdc region.
def kdc_separation(x_center, y_center, a, b, pa):
    '''
    Parameters:
    - x_center: center of the ellipse in x direction (in pixels).
    - y_center: center of the ellipse in y direction (in pixels).
    - a: semi-major axis of the ellipse (in pixels).
    - b: semi-minor axis of the ellipse (in pixels).
    - pa: position angle of the ellipse in degrees (measured from the x-axis).

    Returns:
    - ellipse_mask: a boolean mask for the ellipse region.
    '''

    y, x = np.indices((50, 50))
    pa = np.deg2rad(pa)
    x_rot = np.cos(pa) * (x - x_center) + np.sin(pa) * (y - y_center)
    y_rot = -np.sin(pa) * (x - x_center) + np.cos(pa) * (y - y_center)
    ellipse_mask = (x_rot**2 / a**2 + y_rot**2 / b**2) <= 1

    kdc = np.ma.masked_array(cleaned_vel_data, mask = ~ellipse_mask)

    plt.imshow(kdc, cmap = 'jet')
    plt.colorbar()
    plt.xlabel('SPAXEL X')
    plt.ylabel('SPAXEL Y')
    plt.title('kdc region')
    plt.show()

    not_kdc = np.ma.masked_array(cleaned_vel_data, mask = ellipse_mask)

    plt.imshow(not_kdc, cmap = 'jet')
    plt.colorbar()
    plt.xlabel('SPAXEL X')
    plt.ylabel('SPAXEL Y')
    plt.title('other region')
    plt.show()

    return ellipse_mask

x_center = 25
y_center = 25
a = 8
b = 8
pa = 90
ellipse_mask = kdc_separation(x_center, y_center, a, b, pa)
ellipse_mask = np.repeat(ellipse_mask[np.newaxis, :, :], cleaned_data_cube.shape[0], axis = 0)

# mask = ellipse_mask leads to a not-kdc spectrum, mask = ~ellipse_mask leads to a kdc spectrum.
cleaned_data_cube = np.ma.masked_array(cleaned_data_cube, mask = ellipse_mask)

# co-adding the spectra.
coadded_spectrum = np.ma.sum(cleaned_data_cube, axis = (1, 2))

# average the coadded_spectrum by the number of valid pixels inside (or outside) the ellipse.
#valid_pixels = np.ma.count(cleaned_data_cube, axis = (1, 2))
#print('valid pixels:', valid_pixels)
#coadded_spectrum = coadded_spectrum / valid_pixels

# replace masked values with NaN for saving.
spectrum_to_save = coadded_spectrum.filled(np.nan)

# around NaN values, sometimes there will be some very small values.
# for each NaN value (e.g., at index a), set the adjacent values (a-1 and a+1) to NaN as well.
nan_indices = np.isnan(spectrum_to_save)
for idx in range(1, len(spectrum_to_save) - 1):
    if nan_indices[idx]:
        spectrum_to_save[idx - 1] = np.nan
        spectrum_to_save[idx + 1] = np.nan

hdu = fits.PrimaryHDU(spectrum_to_save)
hdulist = fits.HDUList([hdu])
hdulist.writeto('coadded_spectrum_CATID.fits', overwrite = True)

plt.figure(figsize=(10, 6))
plt.plot(np.arange(cleaned_data_cube.shape[0]), spectrum_to_save)
plt.xlabel('wavelength index')
plt.ylabel('coadded flux')
plt.title('coadded spectrum')
plt.show()
