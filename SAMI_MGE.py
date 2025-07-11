from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
import numpy as np
import mgefit as mge

#----------------------------------------------------------------------------
'''
version_01: 11/07/2025: initial function.
'''
#----------------------------------------------------------------------------------------------------------------
def image_cutout(fits_path, ra, dec, scale, cut_size, output_path, vmin = None, vmax = None):
    '''
    The function can be used to do preparation for SEXTRACTOR or MGE. Large cut_size to identify stars and estimate PSF
    with SEXTRACTOR, then switch to small cut_size to apply MGE.

    Parameters:
    - fits_path: path to the input optical FITS file in pixel scale.
    - ra: right ascension of the target galaxy in degrees.
    - dec: declination of the target galaxy in degrees.
    - scale: size of pixel in arcsec (e.g., 0.396 arcsec/pixel for SDSS).
    - cut_size: cut size in arcsec. Large cut_size for identify stars in SEXTRACTOR, small cut_size for mge fitting.
    - output_path: path to output FITS file, the file is stored in the same unit as the input FITS file.
    - vmin: colorbar plotting.
    - vmax: colorbar plotting.

    Return:
    - cutout.data: the data containing the brightness of all sources.
    '''

    # open the input optical image.
    with fits.open(fits_path) as hdu:
        data = hdu[0].data
        header = hdu[0].header

    # read the header information and extract wcs.
    wcs = WCS(header)

    # cut_size in pixel scale.
    cut_size_pixel = int(cut_size / scale)

    # locate the target galaxy based on RA and Dec.
    position = wcs.world_to_pixel_values(ra, dec)
    print(f'Position of the target galaxy (in pixel scale): {position}')

    # apply the cut.
    cutout = Cutout2D(data, position, (cut_size_pixel, cut_size_pixel), wcs = wcs)
    print(f'Initial image shape (pixel):{data.shape}, cutout shape (pixel):{cutout.data.shape}')

    '''
    # rotate the direction such that the image would be in the traditional north-up orientation.
    if 'SPA' in header:
        # position angle from the header.
        pa = header['SPA']
        print(f'Position angle (SPA) from header: {pa} degrees')

        # rotate the image to make north-up.
        rotated_data = rotate(cutout.data, 0, reshape = True)

    else:
        rotated_data = cutout.data

    '''

    # save the output fits file.
    cutout_hdu = fits.PrimaryHDU(data = cutout.data)
    hdul = fits.HDUList([cutout_hdu])
    hdul.writeto(output_path, overwrite = True)

    # visualization.
    plt.figure(figsize = (9, 8))
    plt.imshow(cutout.data, cmap = 'jet', origin = 'lower', vmin = vmin, vmax = vmax)
    plt.colorbar(label = 'surface brightness in nanomaggy')
    plt.title('cutout image')
    plt.xlabel('spaxel')
    plt.ylabel('spaxel')

    plt.show()

    return cutout.data
#-----------------------------------------------------------------------------------
def mask_map(fits_path, target_label):
    '''
    After using the sextractor, we obtain a segmentation map where each star or galaxy would
    have a unique label, and the background would have a 0 value. After applying this function,
    all the contamination sources will have a label of 1, and the background and the target galaxy
    will have a label of 0. The returned mask map can be used to exclude those contamination while
    using MGE.

    Parameters:
    - fits_path: path to the input segmentation fits file given by sextractor.
    - target_label: the label of the target galaxy given by sextractor.

    Returns:
    - mask_map: the map with contamination sources having a label of 1.
    '''
    with fits.open(fits_path) as hdu:
        data = hdu[0].data
        print(f'Unique label values in the data: {np.unique(data)}')

    # we set pixels corresponding to the target galaxy to have a value of 0 and contamination objects to have a value of 1.
    # note the background would have a 0 value in the output image of Sextractor.
    # after this process, we mask the contamination objects, which will have a value of 1, the target galaxy and the background will have a value of 0.
    mask_map = np.where((data != target_label) & (data != 0), 1, 0)
    mask_map = mask_map.astype(np.uint8)

    # visulization.
    plt.figure(figsize = (8, 8))
    plt.imshow(mask_map, cmap = 'jet', origin = 'lower')
    plt.title('mask map for mge')
    plt.xlabel('pixel')
    plt.ylabel('pixel')
    plt.show()

    return mask_map
#-----------------------------------------------------------------------------------
def apply_mge(cut_data, mask_map, level, minlevel, fwhm, skylev = 0, scale = 0.396, ngauss = 12):
    '''

    Parameters:
        cut_data: the cutout data given by image_cutout function.
        mask_map: the mask map given by mask_map function.
        level: level above which to select pixels to consider in the estimate of the galaxy parameters (in find_galaxy function).
        minlevel: The minimum `counts` level to include in the photometry. The measurement along one profile stops
                  when the `counts` first go below this level (in sectors_photometry function).
        fwhm: fwhm value in pixel scale estimated from the sextractor.
        skylev: the sky level in the unit of counts/pixel which will be subtracted from the input data.
        scale: pixel scale in arcsec (e.g., 0.396 arcsec/pixel for SDSS), this is *only* used for the scale
               on the plots. It has no influence on the output.
        ngauss: maximum number of Gaussians allowed in the MGE fit. Typical values are in
                the range ``10 -- 20`` when ``linear=False`` (default: ``ngauss=12``) and
                ``20**2 -- 40**2`` when ``linear=True`` (default: ``ngauss=30**2=900``).

    Returns:

    '''
    if cut_data.shape != mask_map.shape:
        raise ValueError('The shape of cut_data and mask_map must be the same!')

    # subtract sky.
    cut_data = cut_data - skylev

    # estimated PSF value.
    # a more accurate estimate would be using MGE and fit stars identified with SEXTRACTOR.
    # This method would be updated in a later version.
    sigmapsf = fwhm / 2.355

    # use the find_galaxy function.
    plt.clf()
    f = mge.find_galaxy(img = cut_data, level = level, plot = True)
    plt.pause(1)

    # where mask_image == 0 (valid regions), the boolean mask will be True.
    # where mask_image == 1 (invalid regions), the boolean mask will be False.
    # false values are masked and ignored in the photometry.
    target_mask = mask_map == 0

    # create a mask for the kdc region.
    # radius_pixel = 4.5 / scale
    # y, x = np.indices(img.shape)
    # kdc_mask = (x - f.xpeak)**2 + (y - f.ypeak)**2 >= radius_pixel**2

    # mask = kdc_mask & target_mask

    # perform galaxy photometry.
    plt.clf()
    s = mge.sectors_photometry(
        cut_data, f.eps, f.theta, f.xpeak, f.ypeak,
        minlevel = minlevel, mask = target_mask, plot = True)
    plt.pause(1)

    # do the MGE fit.
    plt.clf()
    m = mge.fit_sectors_regularized(s.radius, s.angle, s.counts, f.eps,
                        ngauss = ngauss, sigmapsf = sigmapsf,
                        scale = scale, plot = True, linear = False)
    plt.pause(1)

    # show contour plots of the results.
    plt.clf()
    mge.print_contours(cut_data, f.theta, f.xpeak, f.ypeak, m.sol, scale = scale,
                       sigmapsf = sigmapsf, mask = target_mask, minlevel = minlevel)

    plt.xlim(-50, 50)
    plt.ylim(-50, 50)

    plt.pause(1)

    # peak surface brightness.

#-----------------------------------------------------------------------------------
# usage example.
# step one: cutout the image and estimate PSF.
fits_path = 'sextractor_and_mge/NGC_6278/frame-r-NGC_6278.fits'
ra = 255.20968
dec = 23.01104
scale = 0.396
cut_size = 200
output_path = 'sextractor_and_mge/NGC_6278/NGC_6287_cut_image.fits'
cut_data = image_cutout(fits_path, ra, dec, scale, cut_size, output_path, vmin = 0, vmax = 10)
#--------------------------------------------------------------------------------