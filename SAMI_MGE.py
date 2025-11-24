from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
import numpy as np
import mgefit as mge
from matplotlib.ticker import AutoMinorLocator
#----------------------------------------------------------------------------
'''
version_01: 11/07/2025: initial function.
version_02: 13/07/2025: consider rotation. However, it seems the rotation of the r band image
            will not affect the final MGE results (counts, sigma and q).
version_03: 30/07/2025: convert the unit into Lsolar/pc2 in preparation for dynamical modeling. Note after using the 'test',
            we conclude that there is no need to set the 'calibration' keyword. Just starting directly from nanomaggy and 
            using Pogson magnitude to convert the unit into mag/arcsec^2.
version_04: 18/09/2025: allow twist in MGE/
'''
#----------------------------------------------------------------------------------------------------------------
def image_cutout(fits_path, ra, dec, scale, cut_size, output_path, vmin = None, vmax = None, rotation = False, calibration = False,
                 align_major = False, kin_pa = None, counterclock = True):
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
    - rotation: rotate the image to the traditional up north direction.
    - calibration: whether to calibrate the unit from nanomaggy to count or not, normally, no need to do that.
    - align_major: whether to align the major axis of the image with the horizontal level or not.
    - kin_pa: the kinematic position angle applied when using the align_major option, defined as staring from north,
              counterclockwise.

    Return:
    - rotated_data: the data containing the brightness of all sources.
    '''

    # open the input optical image.
    with fits.open(fits_path) as hdu:
        data = hdu[0].data # in the unit of nanomaggy.
        calib = hdu[1].data # the calibration factor in nanomaggy/count.
        header = hdu[0].header

    # expand the calibration vector to match the image shape.
    calib = np.tile(calib, (data.shape[0], 1))

    if calibration:
        # decalibrate the data from nanomaggy to count (in preparation for MGE).
        data = data / calib  # now the data is in count/pixel.

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

    # rotate the direction such that the image would be in the traditional north-up orientation.
    if rotation:
        if 'SPA' in header:
            # position angle from the header.
            pa = header['SPA']
            print(f'Position angle (SPA) from header: {pa} degrees')

            if align_major:
                if counterclock:
                    rotated_data = rotate(cutout.data, -(pa + (90 - kin_pa)), reshape=True)
                else:
                    rotated_data = rotate(cutout.data, -pa + kin_pa, reshape=True)

            else:
                # rotate the image to make north-up.
                rotated_data = rotate(cutout.data, -pa, reshape=True)

            # set those new regions (with 0 value) to be nan.
            rotated_data[rotated_data == 0] = np.nan

        else:
            rotated_data = cutout.data
    else:
        rotated_data = cutout.data

    # save the output fits file.
    cutout_hdu = fits.PrimaryHDU(data = rotated_data)
    hdul = fits.HDUList([cutout_hdu])
    hdul.writeto(output_path, overwrite = True)

    # visualization.
    plt.figure(figsize = (9, 8))
    plt.imshow(rotated_data, cmap = 'jet', origin = 'lower', vmin = vmin, vmax = vmax)
    plt.colorbar(label = 'surface brightness in counts/pixel')
    plt.title('cutout image')
    plt.xlabel('spaxel')
    plt.ylabel('spaxel')

    plt.show()

    if rotation:
        # convert nan values back to zero in preparation for MGE.
        rotated_data = np.nan_to_num(rotated_data, nan=0)

    return rotated_data
#-----------------------------------------------------------------------------------
def mask_map_create(fits_path, target_label):
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
def apply_mge(cut_data, level, minlevel, fwhm, Ar, skylev = 0, scale = 0.396, ngauss = 12, Msolar = 4.68, mask_map = None, twist = False, ax = None):
    '''

    Parameters:
    - cut_data: the cutout data given by image_cutout function.
    - level: level above which to select pixels to consider in the estimate of the galaxy parameters (in find_galaxy function).
    - minlevel: The minimum `counts` level to include in the photometry. The measurement along one profile stops
                  when the `counts` first go below this level (in sectors_photometry function).
    - fwhm: fwhm value in pixel scale (e.g., estimated from the sextractor or directly extracted from sdss catalog).
    - Ar: the extinction in magnitude for a given filter.
    - skylev: the sky level in the unit of counts/pixel which will be subtracted from the input data.
    - scale: pixel scale in arcsec (e.g., 0.396 arcsec/pixel for SDSS), this is *only* used for the scale
               on the plots. It has no influence on the output.
    - ngauss: maximum number of Gaussians allowed in the MGE fit. Typical values are in
                the range ``10 -- 20`` when ``linear=False`` (default: ``ngauss=12``) and
                ``20**2 -- 40**2`` when ``linear=True`` (default: ``ngauss=30**2=900``).
    - Msolar: absolute magnitude of the sun (e.g., 4.68 for SDSS r band).
    - mask_map: the mask map given by mask_map function after using sextractor.
    - twist: whether to use the twist version of the MGE.
    - ax: matplotlib axes object.
    - label_pad: pad of the y axis label.
    - tick_lim: x and y tick range max value.
    - loc_min: tick interval setting (tick starting value).
    - loc_max: tick interval setting (tick ending value).
    - loc_step: tick interval setting (tick step).
    - fontsize: label font size.
    - title: title of the plot.
    - Re_circle: whether to plot a circular region showing the effective radius.
    - Re: effective radius in arcsec.
    - plot_psf: whether to plot the PSF of the photometric image.
    - psf_label_x: the x position of the PSF label.
    - psf_label_y: the y position of the PSF label.
    - extra_minlevel: extral level to add in the photometric plot.
    - compass: whether to plot compass lable.
    - xc: x position of the compass label.
    - yc: y position of the compass label.
    - pa: position angle of the compass label.
    - length: length of the compass label.
    - N_extra: extra space in the lable 'N'.
    - E_extra: extra space in the lable 'E'.

    Returns:

    '''

    # if the mask map is given.
    if mask_map is not None:
        # check the shape of the input data and mask map.
        if cut_data.shape != mask_map.shape:
            raise ValueError('The shape of cut_data and mask_map must be the same!')

    # subtract sky.
    cut_data = cut_data - skylev

    # estimated PSF value.
    # if using sdss r band image, a more accurate estimate would be using MGE and fit stars identified with SEXTRACTOR.
    sigmapsf = fwhm / 2.355

    # use the find_galaxy function.
    tem = plt.figure()
    f = mge.find_galaxy(img = cut_data, level = level, plot = True)
    tem.show()
    plt.pause(1)

    if mask_map is not None:
        # where mask_image == 0 (valid regions), the boolean mask will be True.
        # where mask_image == 1 (invalid regions), the boolean mask will be False.
        # false values are masked and ignored in the photometry.
        target_mask = mask_map == 0

        # create a mask for the kdc region.
        # radius_pixel = 30
        # y, x = np.indices(cut_data.shape)
        # kdc_mask = (x - f.xpeak)**2 + (y - f.ypeak)**2 <= radius_pixel**2

        # target_mask = kdc_mask & target_mask
    else:
        target_mask = None

    # perform galaxy photometry.
    tem = plt.figure()
    if twist:
        s = mge.sectors_photometry_twist(
            cut_data, f.theta, f.xpeak, f.ypeak,
            minlevel=minlevel, mask=target_mask, plot=True)
    else:
        s = mge.sectors_photometry(
            cut_data, f.eps, f.theta, f.xpeak, f.ypeak,
            minlevel=minlevel, mask=target_mask, plot=True)

    tem.show()
    plt.pause(1)

    # do the MGE fit.
    tem = plt.figure()
    if twist:
        m = mge.fit_sectors_twist(s.radius, s.angle, s.counts, f.eps,
                                  ngauss=ngauss, sigmapsf=sigmapsf,
                                  scale=scale, plot=True)
    else:
        m = mge.fit_sectors_regularized(s.radius, s.angle, s.counts, f.eps,
                                        ngauss=ngauss, sigmapsf=sigmapsf,
                                        scale=scale, plot=True)

    tem.show()
    plt.pause(1)

    # set up plot.
    if ax is None:
        fig, ax = plt.subplots()

    # show contour plots of the results.
    # activate the axs.
    plt.sca(ax)

    if twist:
        mge.print_contours_twist(cut_data, f.theta, f.xpeak, f.ypeak, m.sol, scale=scale,
                                 sigmapsf=sigmapsf, mask=target_mask, minlevel=minlevel)

    else:
        mge_print_contours(cut_data, f.theta, f.xpeak, f.ypeak, m.sol, scale=scale,
                           sigmapsf=sigmapsf, mask=target_mask, minlevel=minlevel + extra_minlevel)

    # tick setting.
    ax.set_xlim([-tick_lim, tick_lim])
    ax.set_ylim([-tick_lim, tick_lim])

    tick_locs = np.arange(loc_min, loc_max, loc_step)
    ax.set_xticks(tick_locs)
    ax.set_yticks(tick_locs)

    # label settins.
    ax.set_xlabel('Offset(arcsec)', fontsize = fontsize, labelpad = 2)
    ax.set_ylabel('Offset(arcsec)', fontsize = fontsize, labelpad = label_pad)

    # make these (major) ticks longer.
    ax.tick_params(axis = 'both', which = 'major', length = 4, width = 1, direction = 'in')

    # add minor ticks (shorter, no labels).
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis = 'both', which = 'minor', length = 2, width = 1, direction = 'in')

    # set title.
    if title is not None:
        ax.set_title(title, fontsize = fontsize)

    # show the effective radius.
    if Re_circle and Re is not None:
        circle = patches.Circle((0, 0), Re, edgecolor = 'royalblue', facecolor = 'none',
                                linestyle = '--', linewidth = 2, zorder = 2)
        ax.add_patch(circle)

    if plot_psf and psf_label_x is not None and psf_label_y is not None:
        # half of the psf_FWHM in arcsec.
        psf_half = fwhm * scale / 2

        # PSF circle.
        circle = patches.Circle((psf_label_x, psf_label_y),
                                psf_half, edgecolor = 'black', facecolor = 'none', linewidth = 1.5,
                                linestyle = '-', zorder = 2)

        ax.add_patch(circle)

    # add a direction label.
    if compass:
        add_NE_compass(ax = ax, xc = xc, yc = yc, pa_deg = pa, length = length)

    # large zorder for axis
    for spine in ax.spines.values():
        spine.set_zorder(1000)

    ax.tick_params(zorder=1001)

    ax.set_axisbelow(False)

    # final results.
    if twist:
        total_counts, sigma, q_obs, pa = m.sol
    else:
        total_counts, sigma, q_obs = m.sol

    # the following steps are in preparation for the dynamical modeling, see readme_mge_fit_sectors.pdf for more details.
    # the total counts of each Gaussian can be converted into the corresponding peak surface brightness.
    # if calibration = True in the image_cutout function, then C0 would have the unit of counts/pixel.
    C0 = total_counts / (2 * np.pi * q_obs * sigma**2)

    # convert to the surface brightness in mag/arcsec^2 using the SDSS Pogson magnitude equation.
    # note that sdss provides calibrated quantities and no exposure time needs to enter.
    # note the extinction is not considered.
    u = 22.5 - 2.5 * np.log10(C0/scale**2) - Ar

    # convert to surface density in Lsolar/pc^2.
    I = (64800/np.pi)**2 * 10**(0.4*(Msolar - u))

    # sigma in arcsec, in preparation for dynamical modeling.
    sigma_arcsec = sigma * scale

    print('surface density in Lsolar/pc^2', I)
    print('sigma in arcsec', sigma_arcsec)
    if twist:
        print('Gaussian component twist:', pa)

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
# step two: prepare the mask map.
fits_path = 'sextractor_and_mge/NGC_6278/segmentation.fits'
target_label = 1
mask_map = mask_map(fits_path, target_label)
#---------------------------------------------------------------------------------
# third step: apply the MGE.
apply_mge(cut_data = cut_data, mask_map = mask_map, level = 0.1, minlevel = 0.1, fwhm = 2.30843, ngauss = 12)