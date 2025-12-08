from astropy.io import fits
from SAMI_MGE import image_cutout
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
#----------------------------------------------------------------
def nMgy_to_mag(flux):
    '''
    Convert nanomaggy into Pogson magnitude.

    Parameters:
    - flux: flux in the unit of nanomaggy.

    Returns:
    - mag: Pogson magnitude.

    '''

    mag = np.full_like(flux, np.nan, dtype = float)

    # good pixels
    good = (flux > 0) & np.isfinite(flux)

    # transfer from nanomaggy into Pogson magnitude.
    mag[good] = 22.5 - 2.5 * np.log10(flux[good])

    return mag
#----------------------------------------------------------------
def r_band_dust_correction(galaxy_name, target_label, ra, dec, xc, yc, q, kin_pa,
                           psf_g, psf_i, convolve = True, scale = 0.396, cut_size = 200,
                           counterclock = True, flux_frac = 0.01, max_iter = 10,
                           sigma_clip = 3, E_thresh = 0.1, logm_max = None):
    '''
    Apply the dust correction on the r band photometric data based on Scott 2013 paper.

    Parameters:
    - galaxy_name: SAMI CATID.
    - target_label: target galaxy label in the segmentation map from SEXTRACTOR.
    - ra: right ascension in degrees.
    - dec: declination in degrees.
    - xc: galaxy center in pixel.
    - yc: galaxy center in pixel.
    - q: axis ratio based on ellipticity.
    - kin_pa: additional position angle to align the photometric major axis to the horizontal level
              after aligning the north to the up direction.
    - psf_g: g band fwhm_psf in arcsec (the convolution in the code assumes psf_g > psf_i).
    - psf_i: i band fwhm_psf in arcsec (the convolution in the code assumes psf_g > psf_i).
    - convolve: whether or not to convolve the photometric i band data to match the resolution of g band data
                before constructing the g-i plot.
    - scale: pixel size (SDSS: 0.396 arcsec/pixel).
    - cut_size: cutsize in arcsec.
    - counterclock: whether to apply the additional PA counterclockwise or not, default is True.
    - flux_frac: quality cut, g, r, and i band pixels with flux value smaller than flux_frac * peak flux will be excluded.
    - max_iter: iteration number for the robust linear fitting.
    - sigma_clip: sigma clipping factor, default is 3 (pixels with the residual < 3 * sigma will be kept).
    - E_thresh: color excess threshold, pixel above this threshold in the color excess plot will be corrected for dust.
    - logm_max: only correct for pixels with logm < logm_max.

    Returns:
    - r_corr: corrected r band photometric data in nanomaggy.

    '''

    # data preparation.
    # g, r, i bands.
    # input.
    g_fits_path = f'{galaxy_name}/MGE/{galaxy_name}_frame-g.fits'
    r_fits_path = f'{galaxy_name}/MGE/{galaxy_name}_frame-r.fits'
    i_fits_path = f'{galaxy_name}/MGE/{galaxy_name}_frame-i.fits'

    # cut output.
    g_output_path = f'{galaxy_name}/MGE/{galaxy_name}_cut_g_dust_corr.fits'
    r_output_path = f'{galaxy_name}/MGE/{galaxy_name}_cut_r_dust_corr.fits'
    i_output_path = f'{galaxy_name}/MGE/{galaxy_name}_cut_i_dust_corr.fits'

    # mask map from sextractor.
    # be used to isolate the target galaxy.
    # this map is based on the rotated r-band data.
    seg = fits.open(f'{galaxy_name}/MGE/segmentation.fits')[0].data.astype('int')

    # after using the image_cutout function, the galaxy would be rotated such that the major axis would be in the horizontal level.
    # g band.
    g_data = image_cutout(fits_path = g_fits_path, ra = ra, dec = dec, scale = scale,
                          cut_size = cut_size, output_path = g_output_path,
                          vmin = 0, vmax = 1, rotation = True,
                          align_major = True, kin_pa = kin_pa, counterclock = counterclock)

    # r band.
    r_data = image_cutout(fits_path = r_fits_path, ra = ra, dec = dec, scale = scale,
                          cut_size = cut_size, output_path = r_output_path,
                          vmin = 0, vmax = 1, rotation = True,
                          align_major = True, kin_pa = kin_pa, counterclock = counterclock)
    # i band.
    i_data = image_cutout(fits_path = i_fits_path, ra = ra, dec = dec, scale = scale,
                          cut_size = cut_size, output_path = i_output_path,
                          vmin = 0, vmax = 1, rotation = True,
                          align_major = True, kin_pa = kin_pa, counterclock = counterclock)

    # debug.
    if not (seg.shape == g_data.shape == r_data.shape == i_data.shape):
        raise ValueError('data shape does not match!')

    # the galaxy_mask is used to isolate the target galaxy.
    galaxy_mask = (seg == target_label)

    # only consider those pixels with a flux value > e.g., 1% of the peak flux value.
    g_peak = np.nanmax(g_data[galaxy_mask])
    r_peak = np.nanmax(r_data[galaxy_mask])
    i_peak = np.nanmax(i_data[galaxy_mask])

    flux_good = (
            (g_data > flux_frac * g_peak) &
            (r_data > flux_frac * r_peak) &
            (i_data > flux_frac * i_peak)
    )

    # good pixel mask.
    good_pixel = flux_good & galaxy_mask

    # whether or not to do the convolution before constructing the g-i plot.
    # convolution ensures both bands show dust the same way.
    if convolve:
        # resolution matching.
        # convolve i data to match the resolution of g data.
        if psf_g <= psf_i:
            raise ValueError('the convolution in the code assumes psf_g > psf_i!')

        # sigma in pxiel.
        sigma_g = psf_g / scale / (2 * np.sqrt(2 * np.log(2)))
        sigma_i = psf_i / scale / (2 * np.sqrt(2 * np.log(2)))
        sigma_diff = np.sqrt(sigma_g ** 2 - sigma_i ** 2)

        i_data = gaussian_filter(i_data, sigma_diff)

    # convert from nMgy to Pogson magnitude.
    g_mag = nMgy_to_mag(g_data)
    i_mag = nMgy_to_mag(i_data)

    # g-i color image.
    g_i = np.full_like(r_data, np.nan, dtype = float)
    g_i[good_pixel] = g_mag[good_pixel] - i_mag[good_pixel]

    # visualization.
    plt.figure()
    plt.imshow(g_i, origin = 'lower',  cmap = 'RdYlBu_r')
    plt.colorbar()
    plt.title('g-i image')
    plt.show()

    # pixels on the same isophote (same elliptical distance m) are compared.
    # elliptical radius.
    yy, xx = np.indices(r_data.shape)
    dx = xx - xc
    dy = yy - yc

    # semi-major radius m.
    m = np.sqrt(dx ** 2 + (dy / q) ** 2)
    log_m = np.log10(np.maximum(m, 1e-3))

    # final mask used for fitting.
    mask = np.isfinite(g_i) & np.isfinite(log_m) & good_pixel

    # prepare arrays for fitting.
    x = log_m[mask]
    y = g_i[mask]

    # robust linear fit (iterative sigma-clip).
    # create a boolean array the same shape as x, filled with True.
    # this mask selects which data points are currently used in the fit.
    # starting with all True means 'use all points at first'.
    # later fits use a smaller set after bad points are removed.
    fit_mask = np.ones_like(x, dtype = bool)
    a = b = 0
    for _ in range(max_iter):
        # fit_mask.sum() counts how many points are left to fit.
        if fit_mask.sum() < 10:
            break

        a_new, b_new = np.polyfit(x[fit_mask], y[fit_mask], 1)
        resid = y - (a_new * x + b_new)
        sigma = np.nanstd(resid[fit_mask])

        if sigma == 0:
            a, b = a_new, b_new
            break

        keep = (np.abs(resid) <= sigma_clip * sigma)

        if np.array_equal(keep, fit_mask):
            a, b = a_new, b_new
            break

        fit_mask = keep
        a, b = a_new, b_new

    # best fitting line.
    xr = np.linspace(np.nanmin(x), np.nanmax(x), 1000)
    yr = a * xr + b

    # visualization.
    plt.figure(figsize = (10, 9))
    ax = plt.gca()
    ax.scatter(x, y, s = 18, marker = 'D', facecolors = 'none', edgecolors = 'k', linewidths = 0.6, alpha = 0.9)
    ax.plot(xr, yr, color = 'tab:red', lw = 2)
    # threshold line.
    ax.plot(xr, yr + E_thresh, color ='blue', lw = 2, linestyle = '--')
    if logm_max is not None:
        plt.axvline(logm_max, color='grey', linestyle='--', linewidth=2)

    # labels & ticks.
    ax.set_xlabel('log semi-major axis distance', fontsize = 10)
    ax.set_ylabel('g - i', fontsize = 10)
    ax.set_xlim(np.min(xr), np.max(xr))
    ax.set_ylim(np.nanmin(y), np.nanmax(y))

    plt.tight_layout()
    plt.show()

    # predicted intrinsic color.
    color_pred = a * log_m + b
    # color excess.
    E_gi = np.full_like(g_i, np.nan)
    E_gi[mask] = g_i[mask] - color_pred[mask]

    # visualization.
    plt.figure()
    plt.imshow(E_gi, origin = 'lower', cmap = 'RdYlBu_r')
    plt.colorbar()
    plt.title('color excess image')
    plt.show()

    # dusty pixels.
    dust_mask = np.zeros_like(g_i, dtype = bool)
    dust_mask[mask] = E_gi[mask] > float(E_thresh)

    # only spaxels with logm < logm_max will be corrected for dust.
    if logm_max is not None:
        dust_mask &= (log_m <= float(logm_max))

    # r-band extinction.
    Ar = np.full_like(g_i, np.nan)
    Ar[dust_mask] = 1.15 * E_gi[dust_mask]

    # correct r flux in the original unit (nanomaggy).
    r_corr = r_data.copy()
    r_mask = (r_data > 0) & np.isfinite(r_data) & dust_mask & np.isfinite(E_gi) & np.isfinite(Ar)
    r_corr[r_mask] = r_data[r_mask] * (10 ** (0.4 * Ar[r_mask]))

    return r_corr
#-----------------------------------------------------------------------------------






