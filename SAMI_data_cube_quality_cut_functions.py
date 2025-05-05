from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from SAMI_stellar_velocity_quality_cut_functions import quality_cut_stellar_velocity_map
from vorbin.voronoi_2d_binning import voronoi_2d_binning

'''
Three functions are defined in this script:
1. data_cube_clean_percentage()
2. data_cube_clean_snr()
3. kdc_separation()

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
spectrum (will consider the covariance and the weight map in a future version).

Note that for both methods, there is no need to do any cleaning at each wavelength slice 
(e.g., for the first method, a pixel that contributes little to the total flux across all 
wavelengths might not be important for our analysis, even if it has a high flux at specific 
wavelengths).

In this script, we also show how to select those pixels which belong to the kdc region, and
how to get the co-added spectrum. A function using ellipse mask to isolate the kdc region 
is defined. Also the combined_mask, which is defined when doing the quality cut for the stellar 
velocity map, should also be included. 

version_01: 01/04/2025: initial version.
version_02: 06/04/2025: modification on the data_cube_clean_snr function, we use mean flux and
            mean noise to calculate the S/N in a specific emission-free wavelength range. 
'''
#---------------------------------------------------------------------------------------------------
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
def data_cube_clean_snr(fits_path, sn_threshold, emission_free_range, wavelength_slice_index, combined_mask = None, vorbin = False, target_sn = None):
    '''
    Clean the data cube by excluding pixels with low S/N (mean flux / mean noise) in a specific emission-free
    wavelength range, and exclude them across the whole cube. This function could also be used to do preparation
    for vorbin.

    Parameters:
    - fits_path: str, path to the data cube fits file.
    - sn_threshold: float, minimum S/N required for a pixel to be included.
    - emission_free_range: [(min_wavelength_01, max_wavelength_01), (min_wavelength_02, max_wavelength_02),...], wavelength range free of emission lines defined in rest-frame.
    - wavelength_slice_index: int, wavelength slice to visualize before cleaning and after cleaning.
    - combined_mask: 2D mask, the mask from other maps (e.g., stellar velocity maps).
    - vorbin: bool, whether to perform Voronoi binning.
    - target_sn: float, target S/N for Voronoi binning (required if vorbin is True).

    Returns:
    - cleaned_data_cube: masked array containing the cleaned data cube.
    - (optional if vorbin is True) Voronoi binning outputs.
    '''

    # read the primary flux data in extension [0] and the variance data in [1].
    with fits.open(fits_path) as hdul:
        flux_cube = hdul[0].data # flux data (2048*50*50)
        var_cube = hdul[1].data # variance data (2048*50*50)

    # mask invalid data.
    flux_cube = np.ma.masked_invalid(flux_cube)
    var_cube = np.ma.masked_invalid(var_cube)
    var_cube = np.ma.masked_where(var_cube <= 0, var_cube)

    # compute S/N cube (same shape as the flux and variance cube).
    sn_cube = flux_cube / np.sqrt(var_cube)

    # for each spaxel, compute the S/N at each wavelength slice (axis = 0).
    # the median S/N will be used to represent the S/N for each spaxel.
    sn = np.ma.median(sn_cube, axis = 0) # 50*50

    # for each spaxel, find the wavelength slice where the median S/N is obtained.
    # the flux and noise at this wavelength slice will be used to do vorbin.
    # the np.ma.argmin() function returns the index of the minimum value.
    sn_slice = np.ma.argmin(np.abs(sn_cube - sn), axis = 0)

    # plot the total S/N map before masking.
    plt.imshow(sn, cmap = 'jet', origin = 'lower')
    plt.colorbar(label = 'median S/N across whole wavelength ranges')
    plt.xlabel('SPAXEL X')
    plt.ylabel('SPAXEL Y')
    plt.title('S/N map before quality cut')
    plt.show()

    # mask spaxels with low S/N.
    sn_mask = sn < sn_threshold # 50*50
    sn_mask = np.broadcast_to(sn_mask, flux_cube.shape) # 2048*50*50
    cleaned_flux_cube = np.ma.masked_where(sn_mask, flux_cube)
    cleaned_var_cube = np.ma.masked_where(sn_mask, var_cube)

    if combined_mask is not None:
        combined_mask = np.broadcast_to(combined_mask, cleaned_flux_cube.shape)
        cleaned_flux_cube = np.ma.masked_where(combined_mask, cleaned_flux_cube)
        cleaned_var_cube = np.ma.masked_where(combined_mask, cleaned_var_cube)

    plt.imshow(flux_cube[wavelength_slice_index, :, :], cmap = 'jet', origin = 'lower')
    plt.colorbar(label = 'flux value')
    plt.xlabel('SPAXEL X')
    plt.ylabel('SPAXEL Y')
    plt.title(f'data cube before quality cut at the wavelength slice: {wavelength_slice_index}')
    plt.show()

    plt.imshow(cleaned_flux_cube[wavelength_slice_index, :, :], cmap = 'jet', origin = 'lower')
    plt.colorbar(label = 'flux value')
    plt.xlabel('SPAXEL X')
    plt.ylabel('SPAXEL Y')
    plt.title(f'data cube after quality cut at the wavelength slice: {wavelength_slice_index}')
    plt.show()

    if vorbin and target_sn is not None:
        # prepare x, y coordinates for the Voronoi binning.
        n_x, n_y = data_cube.shape[1], data_cube.shape[2]
        x = np.tile(np.arange(n_x), n_y)
        y = np.repeat(np.arange(n_y), n_x)

        signal = np.ma.median(cleaned_data_cube[emission_free, :, :], axis = 0)
        noise = np.sqrt(np.ma.median(cleaned_var[emission_free, :, :], axis = 0))

        # flatten the 2D signal and noise arrays to 1D for Voronoi binning.
        signal = signal.flatten()
        noise = noise.flatten()

        # mask out any NaN or invalid values in signal and noise.
        valid_mask = (np.isfinite(signal) & np.isfinite(noise) & (noise > 0))
        signal = signal[valid_mask]
        noise = noise[valid_mask]
        x = x[valid_mask]
        y = y[valid_mask]
        x = x - 24
        y = y - 24

        binNum, x_gen, y_gen, x_bar, y_bar, sn, nPixels, scale = voronoi_2d_binning(
            x, y, signal, noise, target_sn, plot = 1, quiet = 0
        )

        plt.show()
        return cleaned_data_cube, binNum, x_gen, y_gen, x_bar, y_bar, sn, nPixels, scale

    return cleaned_data_cube

#-----------------------------------------------------------------------------------------------
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

#----------------------------------------------------------------------------
if __name__ == '__main__':
    # extract the combined mask based on the quality cut on the stellar velocity map.
    vel_fits_file = '230776_A_stellar-velocity_default_two-moment.fits'
    sig_fits_path = '230776_A_stellar-velocity-dispersion_default_two-moment.fits'
    vmin = -75
    vmax = 75

    combined_mask, cleaned_vel_data = quality_cut_stellar_velocity_map(vel_fits_file, sig_fits_path,
                                                                       vmin=vmin, vmax=vmax)
    # -------------------------------------------------------------------------------
    fits_path = '230776_A_cube_blue.fits'
    sn_threshold = 10
    percentage = 0.01
    wavelength_slice_index = 1024
    emission_free_range = (4600, 4800)  # https://doi.org/10.1111/j.1365-2966.2011.20109.x
    combined_mask = combined_mask
    cleaned_data_cube = data_cube_clean_snr(fits_path, sn_threshold, emission_free_range, wavelength_slice_index,
                                            combined_mask)
    # cleaned_data_cube = data_cube_clean_percentage(fits_path, percentage, wavelength_slice_index, combined_mask)
    # -------------------------------------------------------------------------------
    x_center = 25
    y_center = 25
    a = 8
    b = 8
    pa = 90
    ellipse_mask = kdc_separation(x_center, y_center, a, b, pa)
    # ------------------------------------------------------------------------------
    ellipse_mask = np.repeat(ellipse_mask[np.newaxis, :, :], cleaned_data_cube.shape[0], axis=0)

    # mask = ellipse_mask leads to a not-kdc spectrum, mask = ~ellipse_mask leads to a kdc spectrum.
    cleaned_data_cube = np.ma.masked_array(cleaned_data_cube, mask=ellipse_mask)

    # co-adding the spectra.
    coadded_spectrum = np.ma.sum(cleaned_data_cube, axis=(1, 2))

    # average the co-added_spectrum by the number of valid pixels inside (or outside) the ellipse.
    # valid_pixels = np.ma.count(cleaned_data_cube, axis = (1, 2))
    # print('valid pixels:', valid_pixels)
    # coadded_spectrum = coadded_spectrum / valid_pixels

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
    hdulist.writeto('co-added_spectrum_CATID.fits', overwrite=True)

    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(cleaned_data_cube.shape[0]), spectrum_to_save)
    plt.xlabel('wavelength index')
    plt.ylabel('co-added flux')
    plt.title('co-added spectrum')
    plt.show()