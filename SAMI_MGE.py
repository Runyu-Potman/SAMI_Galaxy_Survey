from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from scipy.ndimage import rotate
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------------------------------------------
def image_cutout(fits_path, ra, dec, scale, cut_size, output_path, vmin = None, vmax = None):
    '''
    The function can be used to do preparation for SEXTRACTOR or MGE. Large cut_size to identify stars and estimate PSF
    with SEXTRACTOR, then switch to small cut_size to apply MGE. Note that the cutout image would be rotated to be in
    the traditional north-up orientation.

    Parameters:
    - fits_path: path to the input optical FITS file in pixel scale.
    - ra: right ascension of the target galaxy in degrees.
    - dec: declination of the target galaxy in degrees.
    - scale: scale of the input optical image (e.g., 0.396 arcsec/pixel for SDSS).
    - cut_size: cut size in arcsec. Large cut_size for identify stars in SEXTRACTOR, small cut_size for mge fitting.
    - output_path: path to output FITS file, the file is stored in the same unit as the input FITS file.
    - vmin: colorbar plotting.
    - vmax: colorbar plotting.

    Return:
    - None.
    '''

    # open the input optical image.
    with fits.open(fits_path) as hdu:
        data = hdu[0].data
        header = hdu[0].header

    # read the header information.
    wcs = WCS(header)

    # cut_size in pixel scale.
    cut_size_pixel = int(cut_size / scale)

    # locate the target galaxy based on RA and Dec.
    position = wcs.world_to_pixel_values(ra, dec)
    print(f'Position of the target galaxy (in pixel scale): {position}')

    # apply the cut.
    cutout = Cutout2D(data, position, (cut_size_pixel, cut_size_pixel), wcs = wcs)
    print(f'Initial image shape (pixel):{data.shape}, cutout shape (pixel):{cutout.data.shape}')

    # save the output fits file.
    cutout_hdu = fits.PrimaryHDU(data = cutout.data)
    hdul = fits.HDUList([cutout_hdu])
    hdul.writeto(output_path, overwrite = True)

    # visualization.
    plt.figure(figsize = (9, 8))
    plt.imshow(cutout.data, cmap = 'jet', origin = 'lower', vmin = vmin, vmax = vmax)
    plt.colorbar(label = 'Surface Brightness')
    plt.title('cutout image')
    plt.xlabel('RA (degrees)')
    plt.ylabel('Dec (degrees)')

    plt.show()
#-----------------------------------------------------------------------------------

#--------------------------------------------------------------------------------