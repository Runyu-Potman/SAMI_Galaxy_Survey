from astropy.io import fits
from SAMI_MGE import image_cutout
import numpy as np
import matplotlib.pyplot as plt
#----------------------------------------------------------------
# some constants.
ra = 212.85930
dec = 1.28652
scale = 0.396
cut_size = 200
# some constants from find galaxy function based on the original image.
# galaxy center.
xc = 356.81
yc = 356.79
# ellipticity.
q = 0.95
# after shifting the north to the up direction, add another pa value such that the
# major axis would be in the horizontal level.
kin_pa = 44.9
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
                           scale = 0.396, cut_size = 200, counterclock = True,
                           flux_frac = 0.01, max_iter = 10, sigma_clip = 3):
    # data preparation.
    # g, r, i bands.
    # input.
    g_fits_path = f'{galaxy_name}/MGE/{galaxy_name}_frame-g.fits'
    r_fits_path = f'{galaxy_name}/MGE/{galaxy_name}_frame-r.fits'
    i_fits_path = f'{galaxy_name}/MGE/{galaxy_name}_frame-i.fits'

    # cut output.
    g_output_path = f'{galaxy_name}/MGE/{galaxy_name}_cut_g.fits'
    r_output_path = f'{galaxy_name}/MGE/{galaxy_name}_cut_r.fits'
    i_output_path = f'{galaxy_name}/MGE/{galaxy_name}_cut_i.fits'

    # mask map from sextractor.
    # be used to isolate the target galaxy.
    # this map is based on the rotated r-band image.
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

    if not (seg.shape == g_data.shape == r_data.shape == i_data.shape):
        raise ValueError('data shape does not match!')

    # the galaxy_mask is used to isolate the target galaxy.
    galaxy_mask = (seg == target_label)

    # only consider those pixels with a flux value > e.g., 1% of the peak flux value.
    g_peak = np.nanmax(g_data[galaxy_mask])
    r_peak = np.nanmax(r_data[galaxy_mask])
    i_peak = np.nanmax(i_data[galaxy_mask])

    flux_good = (g_data > flux_frac * g_peak) & (r_data > flux_frac * r_peak) & (i_data > flux_frac * i_peak)

    # good pixel mask.
    good_pixel = flux_good & galaxy_mask

    # convert from nMgy to Pogson magnitude.
    g_mag = nMgy_to_mag(g_data)
    i_mag = nMgy_to_mag(i_data)

    # g-i color image.
    g_i = np.full_like(r_data, np.nan, dtype = float)
    g_i[good_pixel] = g_mag[good_pixel] - i_mag[good_pixel]

    # visulization.
    plt.figure()
    plt.imshow(g_i, origin = 'lower')
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

    # final mask.
    mask = np.isfinite(g_i) & np.isfinite(log_m) & good_pixel

    # prepare arrays for fit.
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
        p = np.polyfit(x[fit_mask], y[fit_mask], 1)
        a_new, b_new = p[0], p[1]
        resid = y - (a_new * x + b_new)
        sigma = np.nanstd(resid[fit_mask])
        if sigma == 0:
            a, b = a_new, b_new
            break
        keep = np.abs(resid) <= sigma_clip * sigma
        if np.array_equal(keep, fit_mask):
            a, b = a_new, b_new
            break
        fit_mask = keep
        a, b = a_new, b_new

    # best fitting line.
    xr = np.linspace(np.nanmin(x), np.nanmax(x), 400)
    yr = a * xr + b

    # visulization.
    plt.figure(figsize=(10, 9))
    ax = plt.gca()
    ax.scatter(x, y, s = 18, marker = 'D', facecolors = 'none', edgecolors = 'k', linewidths = 0.6, alpha = 0.9)
    ax.plot(xr, yr, color = 'tab:red', lw = 2)

    # labels & ticks like the example
    ax.set_xlabel('log semi-major axis distance', fontsize = 10)
    ax.set_ylabel('g - i', fontsize = 10)
    ax.set_xlim(np.min(xr), np.max(xr))
    ax.set_ylim(np.nanmin(y) - 0.1, np.nanmax(y) + 0.1)

    plt.tight_layout()
    plt.show()

r_band_dust_correction(galaxy_name = 227266, target_label = 30, ra = ra, dec = dec, xc = xc, yc = yc, q = q, kin_pa = kin_pa)








