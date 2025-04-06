from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

#------------------------------------------------------------------------------------------
'''
First of all, we do the quality cut on the data cube. The size of the data cube 
is 2048*50*50 (two spatial dimensions: 50*50 and a wavelength dimension: 2048).
The y axis in the wavelength dimension is flux with the unit: 
10**(-16) erg/s/cm**2/angstrom/pixel.

When doing the quality cut, there are two different approaches:

1. For each pixel in the spatial map, calculate the total flux (do the integration
along the wavelength dimension), find the central pixel which has the peak flux
value, exclude those pixels with a total flux value less than a specific percentage
(e.g., 1%) of the peak total flux value.

2. Calculate the mean flux and also the mean noise at a specific emission-free
wavelength range. Then, we exclude those pixels with a S/N smaller than a specific threshold 
across whole wavelength range. In terms of the noise, we choose the square root of the variance 
spectrum (will consider the covariance in a future version).

Note that for both methods, we do the cleaning after collapsing the wavelength dimension
(we only focus on the total flux and the total noise). No need to do any cleaning at each
wavelength slice. A pixel that contributes little to the total flux across all wavelengths
might not be important for our analysis, even if it has a high flux at specific wavelengths.

In this script, we also show how to select those pixels which belong to the kdc region, and
how to get the co-added spectrum. A function using ellipse mask to isolate the kdc region 
is defined. Also the combined_mask, which is defined when doing the quality cut for the stellar 
velocity map, should also be included. 

version_01: 01/04/2025: initial version.
version_02: 06/04/2025: 
'''
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
# extract the combined mask based on the quality cut on the stellar velocity map.
from SAMI_stellar_velocity_quality_cut_functions import quality_cut_stellar_velocity_map
vel_fits_file = 'CATID_A_stellar-velocity_default_two-moment.fits'
sig_fits_path = 'CATID_A_stellar-velocity-dispersion_default_two-moment.fits'
vmin = -75
vmax = 75

combined_mask, cleaned_vel_data = quality_cut_stellar_velocity_map(vel_fits_file, sig_fits_path, vmin = vmin, vmax = vmax)
#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
def data_cube_clean_percentage(fits_path, percentage, wavelength_slice_index, combined_mask = None):
    '''
    Clean the data cube by excluding pixels with a total flux value less than
    a given percentage of the peak total flux value.

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
    # calculate the total flux by integrating along the wavelength dimension.
    # after collapsing, total flux will have the shape 50*50.
    total_flux = np.sum(data_cube, axis = 0) * delta_lambda
    print('Shape of the total flux map:', total_flux.shape)

    plt.imshow(total_flux, cmap = 'jet', origin = 'lower')
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

    # combined_mask is some mask from other maps.
    if combined_mask is not None:
        combined_mask = np.repeat(combined_mask[np.newaxis, :, :], cleaned_data_cube.shape[0], axis = 0)
        cleaned_data_cube = np.ma.masked_where(combined_mask, cleaned_data_cube)

    # the data cube before cleaning.
    plt.imshow(data_cube[wavelength_slice_index, :, :], cmap = 'jet', origin = 'lower')
    plt.colorbar()
    plt.xlabel('SPAXEL X')
    plt.ylabel('SPAXEL Y')
    plt.title(f'data cube before cleaning at the wavelength slice: {wavelength_slice_index}')
    plt.show()

    # the data cube after cleaning.
    plt.imshow(cleaned_data_cube[wavelength_slice_index, :, :], cmap = 'jet', origin = 'lower')
    plt.colorbar()
    plt.xlabel('SPAXEL X')
    plt.ylabel('SPAXEL Y')
    plt.title(f'data cube after cleaning at the wavelength slice: {wavelength_slice_index}')
    plt.show()

    return cleaned_data_cube

#------------------------------------------------------------------------------
def data_cube_clean_snr(fits_path, sn_threshold, emission_free_range, wavelength_slice_index, combined_mask = None):
    '''
    Clean the data cube by excluding pixels with low S/N (mean flux / mean noise) in a specific emission-free
    wavelength range, and exclude them across the whole cube.

    Parameters:
    - fits_path: str, path to the data cube fits file.
    - sn_threshold: float, minimum S/N required for a pixel to be included.
    - emission_free_range: tuple (min_wavelength, max_wavelength), wavelength range free of emission lines defined in rest-frame.
    - wavelength_slice_index: int, wavelength slice to visualize before cleaning and after cleaning.
    - combined_mask: 2D mask, the mask from other maps (e.g., stellar velocity maps).

    Returns:
    - cleaned_data_cube: masked array containing the cleaned data cube.
    '''

    # read the primary data in extension [0] and the variance in [1].
    with fits.open(fits_path) as hdul:
        data_cube = hdul[0].data # flux data
        var = hdul[1].data # variance data
        header = hdul[0].header

    data_cube = np.ma.masked_invalid(data_cube)
    var = np.ma.masked_invalid(var)

    wavelength = header['CRVAL3'] + (np.arange(header['NAXIS3']) - header['CRPIX3']) * header['CDELT3']
    redshift = header['Z_SPEC']
    rest_wave = wavelength / (1 + redshift)
    emission_free = (rest_wave >= emission_free_range[0]) & (rest_wave <= emission_free_range[1])

    mean_flux = np.mean(data_cube[emission_free, :, :], axis = 0)
    mean_noise = np.sqrt(np.mean(var[emission_free, :, :], axis = 0))
    sn = np.where(mean_noise > 0, mean_flux / mean_noise, 0)

    # plot the total S/N map before masking.
    plt.imshow(sn, cmap = 'jet', origin = 'lower')
    plt.colorbar(label = 'S/N within the emission-free wavelength range')
    plt.xlabel('SPAXEL X')
    plt.ylabel('SPAXEL Y')
    plt.title('S/N map before any masking')
    plt.show()

    sn_mask = sn < sn_threshold
    sn_mask = np.repeat(sn_mask[np.newaxis, :, :], data_cube.shape[0], axis = 0)
    cleaned_data_cube = np.ma.masked_where(sn_mask, data_cube)

    if combined_mask is not None:
        combined_mask = np.repeat(combined_mask[np.newaxis, :, :], cleaned_data_cube.shape[0], axis = 0)
        cleaned_data_cube = np.ma.masked_where(combined_mask, cleaned_data_cube)

    plt.imshow(data_cube[wavelength_slice_index, :, :], cmap = 'jet', origin = 'lower')
    plt.colorbar()
    plt.xlabel('SPAXEL X')
    plt.ylabel('SPAXEL Y')
    plt.title(f'data cube before cleaning at the wavelength slice: {wavelength_slice_index}')
    plt.show()

    plt.imshow(cleaned_data_cube[wavelength_slice_index, :, :], cmap = 'jet', origin = 'lower')
    plt.colorbar()
    plt.xlabel('SPAXEL X')
    plt.ylabel('SPAXEL Y')
    plt.title(f'data cube after cleaning at the wavelength slice: {wavelength_slice_index}')
    plt.show()

    return cleaned_data_cube

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
fits_path = 'CATID_A_cube_blue.fits'
sn_threshold = 10
percentage = 0.01
wavelength_slice_index = 1024
emission_free_range = (4600, 4800) # https://doi.org/10.1111/j.1365-2966.2011.20109.x
combined_mask = combined_mask
cleaned_data_cube = data_cube_clean_snr(fits_path, sn_threshold, emission_free_range, wavelength_slice_index, combined_mask)
#cleaned_data_cube = data_cube_clean_percentage(fits_path, percentage, wavelength_slice_index, combined_mask)
#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
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

    plt.imshow(kdc, cmap = 'jet', origin = 'lower', vmin = vmin, vmax = vmax)
    plt.colorbar()
    plt.xlabel('SPAXEL X')
    plt.ylabel('SPAXEL Y')
    plt.title('kdc region')
    plt.show()

    not_kdc = np.ma.masked_array(cleaned_vel_data, mask = ellipse_mask)

    plt.imshow(not_kdc, cmap = 'jet', origin = 'lower', vmin = vmin, vmax = vmax)
    plt.colorbar()
    plt.xlabel('SPAXEL X')
    plt.ylabel('SPAXEL Y')
    plt.title('other region')
    plt.show()

    return ellipse_mask

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
x_center = 25
y_center = 25
a = 8
b = 8
pa = 90
ellipse_mask = kdc_separation(x_center, y_center, a, b, pa)
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

ellipse_mask = np.repeat(ellipse_mask[np.newaxis, :, :], cleaned_data_cube.shape[0], axis = 0)

# mask = ellipse_mask leads to a not-kdc spectrum, mask = ~ellipse_mask leads to a kdc spectrum.
cleaned_data_cube = np.ma.masked_array(cleaned_data_cube, mask = ellipse_mask)

# co-adding the spectra.
coadded_spectrum = np.ma.sum(cleaned_data_cube, axis = (1, 2))

# average the co-added_spectrum by the number of valid pixels inside (or outside) the ellipse.
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
hdulist.writeto('co-added_spectrum_CATID.fits', overwrite = True)

plt.figure(figsize=(10, 6))
plt.plot(np.arange(cleaned_data_cube.shape[0]), spectrum_to_save)
plt.xlabel('wavelength index')
plt.ylabel('coadded flux')
plt.title('coadded spectrum')
plt.show()
