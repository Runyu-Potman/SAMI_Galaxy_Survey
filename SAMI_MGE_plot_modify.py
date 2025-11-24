"""
Copyright (C) 1999-2025, Michele Cappellari

E-mail: michele.cappellari_at_physics.ox.ac.uk

Updated versions of the software are available from my web page
https://purl.org/cappellari/software

If you have found this software useful for your research,
I would appreciate an acknowledgement to the use of
"the MGE fitting method and software by Cappellari (2002)".

https://ui.adsabs.harvard.edu/abs/2002MNRAS.333..400C

This software is provided as is without any warranty whatsoever.
Permission to use, for non-commercial purposes is granted.
Permission to modify for personal or internal use is granted,
provided this copyright and disclaimer are included unchanged
at the beginning of the file. All other rights are reserved.
In particular, redistribution of the code is not allowed.

##############################################################################

Changelog
---------
- V1.0.0 : First implementation, Padova, February 1999, Michele Cappellari.
- V2.0.0 : Major revisions, Leiden, January 2000, MC.
- V2.1.0 : Updated documentation, Leiden, 8 October 2001, MC.
- V2.2.0 : Implemented MGE PSF, Leiden, 29 October 2001, MC.
- V2.3.0 : Added MODEL keyword, Leiden, 30 October 2001, MC.
- V2.3.1 : Added compilation options, MC, Leiden, 20 May 2002.
- V2.3.2 : Used N_ELEMENTS instead of KEYWORD_SET to test
           non-logical keywords, Leiden, 9 May 2003, MC.
- V2.4.0 : Convolved image with a Gaussian kernel instead of using
           the SMOOTH function before binning. Always showed contours
           in steps of 0.5 mag/arcsec². Replaced LOGRANGE and NLEVELS
           keywords with MAGRANGE, Leiden, 30 April 2005, MC.
- V2.4.1 : Added /CONVOL keyword, MC, Oxford, 23 September 2008.
- V2.4.2 : Used Coyote Library to select red contour levels for MGE model,
           MC, Oxford, 8 August 2011.
- V3.0.0 : Translated from IDL into Python, MC, Aspen Airport, 8 February 2014.
- V3.0.1 : Used input scale to label axes if given, avoided use of log,
           used data rather than model as reference for contour levels,
           allowed scalar ``sigmaPSF``, MC, Oxford, 18 September 2014.
- V3.0.2 : Fixed extent in contour plot, MC, Oxford, 18 June 2015.
- V3.0.3 : Fixed bug introduced by contour change in Matplotlib 1.5,
           MC, Oxford, 19 January 2016.
- V3.0.4 : Removed warning about non-integer indices in Numpy 1.11,
           MC, Oxford, 20 January 2017.
- V3.0.5 : Included ``mask`` keyword to identify bad pixels on the contour.
           Updated documentation, MC, Oxford, 20 March 2017.
- V3.0.6 : Fixed MatplotlibDeprecationWarning in Matplotlib V2.2,
- Vx.x.xx: Additional changes are now documented in the global CHANGELOG.rst
           file of the MgeFit package

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, special


# ----------------------------------------------------------------------------

def _gauss2d_mge(n, xc, yc, sx, sy, pos_ang):
    """
    Returns a 2D Gaussian image with size N[0]xN[1], center (XC,YC),
    sigma (SX,SY) along the principal axes and position angle POS_ANG, measured
    from the positive Y axis to the Gaussian major axis (positive counter-clockwise).

    """
    ang = np.radians(pos_ang - 90.)
    x, y = np.ogrid[-xc:n[0] - xc, -yc:n[1] - yc]

    xcosang = np.cos(ang) / (np.sqrt(2.) * sx) * x
    ysinang = np.sin(ang) / (np.sqrt(2.) * sx) * y
    xsinang = np.sin(ang) / (np.sqrt(2.) * sy) * x
    ycosang = np.cos(ang) / (np.sqrt(2.) * sy) * y

    im = (xcosang + ysinang) ** 2 + (ycosang - xsinang) ** 2

    return np.exp(-im)


# ----------------------------------------------------------------------------

def _multi_gauss(pars, img, sigmaPSF, normPSF, xpeak, ypeak, theta):
    lum, sigma, q = pars

    # Analytic convolution with an MGE circular Gaussian
    # Eq.(4,5) in Cappellari (2002)
    #
    u = 0.
    for lumj, sigj, qj in zip(lum, sigma, q):
        for sigP, normP in zip(sigmaPSF, normPSF):
            sx = np.sqrt(sigj ** 2 + sigP ** 2)
            sy = np.sqrt((sigj * qj) ** 2 + sigP ** 2)
            g = _gauss2d_mge(img.shape, xpeak, ypeak, sx, sy, theta)
            u += lumj * normP / (2. * np.pi * sx * sy) * g

    # Analytic integral of the MGE on the central pixel (with dx=1) ignoring rotation
    sx = np.sqrt(sigma ** 2 + sigmaPSF[:, None] ** 2)
    sy = np.sqrt((sigma * q) ** 2 + sigmaPSF[:, None] ** 2)
    u[round(xpeak), round(ypeak)] = (
                lum * normPSF[:, None] * special.erf(2 ** -1.5 / sx) * special.erf(2 ** -1.5 / sy)).sum()

    return u


# ----------------------------------------------------------------------------

def mge_print_contours(img, ang, xc, yc, sol, binning=1, magrange=10, magstep=0.5,
                       mask=None, minlevel=None, normpsf=1, scale=None, sigmapsf=0):
    """
    mge.fit_sectors
    ===============

    Purpose
    -------
    Produces a contour plot comparing a convolved MGE model to the original
    fitted image.

    Calling Sequence
    ----------------
    .. code-block:: python

        import mgefit as mge

        mge.print_contours(img, ang, xc, yc, sol, binning=None, magrange=10,
                           mask=None, minlevel=None, normpsf=1, scale=None,
                           sigmapsf=0)

    Parameters
    ----------
    img : array-like
        The image array fitted by ``mge_fit_sectors``.
    ang : float
        The common position angle of the Gaussians (in degrees), measured
        counterclockwise from the image Y-axis to the Gaussians' major axis,
        as determined by ``find_galaxy``.
    xc : float
        The X coordinate of the Gaussian center (in pixels).
    yc : float
        The Y coordinate of the Gaussian center (in pixels).
    sol : array-like, shape (3, Ngauss)
        The best-fitting MGE solution produced by ``mge_fit_sectors``:
            - ``sol[0]``: TotalCounts of each Gaussian component.
            - ``sol[1]``: Sigma, the Gaussian dispersion (in pixels).
            - ``sol[2]``: qObs, the observed axial ratio of the Gaussian components.

    binning : int, optional
        Number of pixels to bin together before plotting. Reduces file size
        (default: no binning).
    magrange : float, optional
        The range in magnitudes for equally spaced contours, in steps of
        0.5 mag/arcsec², starting from the model's maximum value
        (default: 10 magnitudes below the model's maximum).
    mask : array-like of bool, optional
        A Boolean array matching the size of ``img``, with ``False`` indicating
        excluded pixels (shown in golden color).
    minlevel : float, optional
        The minimum contour level to display, in the same units as the image.
        When provided, contours begin at this specified level (default: None).
    normpsf : scalar or array-like, optional
        PSF normalization values. If provided as a vector, it must match
        the size of ``sigmapsf`` and satisfy ``sum(normpsf) = 1``
        (default: 1).
    scale : float, optional
        Pixel scale in arcsec/pixel for the plot axes (default: 1).
    sigmapsf : scalar or array-like, optional
        Sigma of the PSF, in pixels, or an array representing the MGE model
        for the circular PSF (default: no convolution).

    Returns
    -------
    ``QuadContourSet`` Creates a contour plot.

    Notes
    -----
    This function is designed to compare the convolved MGE model against the
    original image fitted by ``mge_fit_sectors``. Contours are plotted in
    steps of 0.5 mag/arcsec².

    Examples
    --------
    See ``mge_fit_example.py`` for usage examples.

    """

    sigmapsf = np.atleast_1d(sigmapsf)
    normpsf = np.atleast_1d(normpsf)

    assert normpsf.size == sigmapsf.size, "sigmaPSF and normPSF must have the same length"
    assert round(np.sum(normpsf), 2) == 1, "PSF not normalized to normpsf.sum() = 1"

    s = img.shape
    if mask is not None:
        assert mask.dtype == bool, "MASK must be a boolean array"
        assert mask.shape == s, "MASK and IMG must have the same shape"

    model = _multi_gauss(sol, img, sigmapsf, normpsf, xc, yc, ang)
    peak = img[int(round(xc)), int(round(yc))]
    if minlevel is None:  # contours start from the peak and decrease
        levels = 0.9 * peak * 10 ** (-0.4 * np.arange(0, magrange, magstep)[::-1])  # magstep mag/arcsec^2 steps
    else:  # contours start from minlevel and increase
        magrange = 2.5 * np.log10(peak / minlevel)  # margange is ignored
        levels = minlevel * 10 ** (0.4 * np.arange(0, magrange, magstep))  # magstep mag/arcsec^2 steps

    if binning != 1:
        model = ndimage.filters.gaussian_filter(model, binning / 2.355)
        model = ndimage.zoom(model, 1. / binning, order=1)
        img = ndimage.filters.gaussian_filter(img, binning / 2.355)
        img = ndimage.zoom(img, 1. / binning, order=1)

    ax = plt.gca()
    ax.axis('equal')
    ax.set_adjustable('box')
    extent = np.array([-yc, s[1] - 1 - yc, -xc, s[0] - 1 - xc])

    if scale is None:
        plt.xlabel("pixels")
        plt.ylabel("pixels")
    else:
        extent = extent * scale
        plt.xlabel("arcsec")
        plt.ylabel("arcsec")

    cnt = ax.contour(img, levels, colors='dimgray', linestyles='solid', extent=extent, linewidths = 1)
    ax.contour(model, levels, colors='r', linestyles='solid', extent=extent, linewidths=1.5)
    if mask is not None:
        a = np.ma.masked_array(mask, mask)
        ax.imshow(a, cmap='autumn_r', interpolation='nearest', origin='lower',
                  extent=extent, zorder=2, alpha=0.7)

    return cnt

# ----------------------------------------------------------------------------
