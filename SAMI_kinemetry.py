from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kinemetry import kinemetry

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
from astropy.io import ascii
#from cap_plot_velfield import plot_velfield
from plotbin.plot_velfield import plot_velfield
#from plotbin.display_pixels import display_pixels
#from plotbin.display_bins_generators import display_bins_generators

#from cap_display_pixels import display_pixels
#from cap_display_bins_generators import display_bins_generators
from matplotlib.patches import Ellipse

import matplotlib.ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import gridspec
import matplotlib as mpl
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'

from kinemetry import kinemetry

import kinemetry as kin

import time
from os import path
#import inspect

#------------------------------------------------------------
def plot_kinemetry_profiles_velocity(k, fitcentre=False, name=None):
    """
    Based on the kinemetry results (passed in k), this routine plots radial
    profiles of the position angle (PA), flattening (Q), k1 and k5 terms.
    Last two plots are for X0,Y0 and systemic velocity

    """

    k0 = k.cf[:, 0]
    k1 = np.sqrt(k.cf[:, 1] ** 2 + k.cf[:, 2] ** 2)
    k5 = np.sqrt(k.cf[:, 5] ** 2 + k.cf[:, 6] ** 2)
    k51 = k5 / k1
    erk1 = (np.sqrt((k.cf[:, 1] * k.er_cf[:, 1]) ** 2 + (k.cf[:, 2] * k.er_cf[:, 2]) ** 2)) / k1
    erk5 = (np.sqrt((k.cf[:, 5] * k.er_cf[:, 5]) ** 2 + (k.cf[:, 6] * k.er_cf[:, 6]) ** 2)) / k5
    erk51 = (np.sqrt(((k5 / k1) * erk1) ** 2 + erk5 ** 2)) / k1

    fig, ax = plt.subplots(figsize=(7, 8))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])

    ax1 = plt.subplot(gs[0])
    ax1.errorbar(k.rad, k.pa, yerr=[k.er_pa, k.er_pa], fmt='--o', mec='k', mew=1.2, color='skyblue', mfc='skyblue',
                 capsize=3)
    ax1.set_ylabel('PA [deg]', fontweight='bold')
    if name:
        ax1.set_title(name, fontweight='bold')

    ax1.tick_params(axis='both', which='both', top=True, right=True)
    ax1.tick_params(axis='both', which='major', labelsize=10)
    ax1.yaxis.set_tick_params(length=6)
    ax1.xaxis.set_tick_params(width=2)
    ax1.yaxis.set_tick_params(width=2)
    ax1.xaxis.set_tick_params(length=6)
    ax1.tick_params(which='minor', length=3)
    ax1.tick_params(which='minor', width=1)
    ax1.tick_params(axis='both', which='both', top=True, right=True)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for tick in ax1.yaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(2)
    ax1.get_xaxis().set_ticklabels([])

    ax2 = plt.subplot(gs[1])
    ax2.errorbar(k.rad, k.q, yerr=[k.er_q, k.er_q], fmt='--o', mec='k', mew=1.2, color='skyblue', mfc='skyblue',
                 capsize=3)
    ax2.set_ylabel('Q ', fontweight='bold')
    # ax2.set_xlabel('R [arsces]')
    ax2.set_ylim(0, 1)
    if fitcentre:
        ax2.set_title('Velocity, fit centre', fontweight='bold')
    else:
        ax2.set_title('Velocity, fixed centre', fontweight='bold')

    ax2.tick_params(axis='both', which='both', top=True, right=True)
    ax2.tick_params(axis='both', which='major', labelsize=10)
    ax2.yaxis.set_tick_params(length=6)
    ax2.xaxis.set_tick_params(width=2)
    ax2.yaxis.set_tick_params(width=2)
    ax2.xaxis.set_tick_params(length=6)
    ax2.tick_params(which='minor', length=3)
    ax2.tick_params(which='minor', width=1)
    ax2.tick_params(axis='both', which='both', top=True, right=True)
    for tick in ax2.xaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for tick in ax2.yaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for axis in ['top', 'bottom', 'left', 'right']:
        ax2.spines[axis].set_linewidth(2)
    ax2.get_xaxis().set_ticklabels([])

    ax3 = plt.subplot(gs[2])
    ax3.errorbar(k.rad, k1, yerr=[erk1, erk1], fmt='--o', mec='k', mew=1.2, color='skyblue', mfc='skyblue', capsize=3)
    ax3.set_ylabel('$k_1$ [km/s]', fontweight='bold')

    ax3.tick_params(axis='both', which='both', top=True, right=True)
    ax3.tick_params(axis='both', which='major', labelsize=10)
    ax3.yaxis.set_tick_params(length=6)
    ax3.xaxis.set_tick_params(width=2)
    ax3.yaxis.set_tick_params(width=2)
    ax3.xaxis.set_tick_params(length=6)
    ax3.tick_params(which='minor', length=3)
    ax3.tick_params(which='minor', width=1)
    ax3.tick_params(axis='both', which='both', top=True, right=True)
    for tick in ax3.xaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for tick in ax3.yaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for axis in ['top', 'bottom', 'left', 'right']:
        ax3.spines[axis].set_linewidth(2)
    ax3.get_xaxis().set_ticklabels([])

    ax4 = plt.subplot(gs[3])
    ax4.errorbar(k.rad, k51, yerr=[erk51, erk51], fmt='--o', mec='k', mew=1.2, color='skyblue', mfc='skyblue',
                 capsize=3)
    ax4.set_ylabel('$k_{51}$', fontweight='bold')
    ax4.set_xlabel('R [arsces]', fontweight='bold')

    ax4.tick_params(axis='both', which='both', top=True, right=True)
    ax4.tick_params(axis='both', which='major', labelsize=10)
    ax4.yaxis.set_tick_params(length=6)
    ax4.xaxis.set_tick_params(width=2)
    ax4.yaxis.set_tick_params(width=2)
    ax4.xaxis.set_tick_params(length=6)
    ax4.tick_params(which='minor', length=3)
    ax4.tick_params(which='minor', width=1)
    ax4.tick_params(axis='both', which='both', top=True, right=True)
    for tick in ax4.xaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for tick in ax4.yaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for axis in ['top', 'bottom', 'left', 'right']:
        ax4.spines[axis].set_linewidth(2)
    ax4.get_xaxis().set_ticklabels([])

    ax5 = plt.subplot(gs[4])
    ax5.errorbar(k.rad, k.xc, yerr=k.er_xc, fmt='--o', mec='k', mew=1.2, color='skyblue', mfc='skyblue', capsize=3,
                 label='Xc')
    ax5.errorbar(k.rad, k.yc, yerr=k.er_yc, fmt='--o', mec='k', mew=1.2, color='salmon', mfc='salmon', capsize=3,
                 label='Yc')
    ax5.set_ylabel('$X_c, Y_c$ [arsces]', fontweight='bold')
    ax5.set_xlabel('R [arsces]', fontweight='bold')
    ax5.legend()

    ax5.tick_params(axis='both', which='both', top=True, right=True)
    ax5.tick_params(axis='both', which='major', labelsize=10)
    ax5.yaxis.set_tick_params(length=6)
    ax5.xaxis.set_tick_params(width=2)
    ax5.yaxis.set_tick_params(width=2)
    ax5.xaxis.set_tick_params(length=6)
    ax5.tick_params(which='minor', length=3)
    ax5.tick_params(which='minor', width=1)
    ax5.tick_params(axis='both', which='both', top=True, right=True)
    for tick in ax5.xaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for tick in ax5.yaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for axis in ['top', 'bottom', 'left', 'right']:
        ax5.spines[axis].set_linewidth(2)

    ax6 = plt.subplot(gs[5])
    ax6.errorbar(k.rad, k0, yerr=k.er_cf[:, 0], fmt='--o', mec='k', mew=1.2, color='skyblue', mfc='skyblue', capsize=3)
    ax6.hlines(np.median(k0), 5, 20, linestyles='dashed', colors='skyblue', label='median $V_{sys}$')
    ax6.set_ylabel('V$_{sys}$ [km/s]', fontweight='bold')
    ax6.set_xlabel('R [arsces]', fontweight='bold')
    ax6.legend()

    ax6.tick_params(axis='both', which='both', top=True, right=True)
    ax6.tick_params(axis='both', which='major', labelsize=10)
    ax6.yaxis.set_tick_params(length=6)
    ax6.xaxis.set_tick_params(width=2)
    ax6.yaxis.set_tick_params(width=2)
    ax6.xaxis.set_tick_params(length=6)
    ax6.tick_params(which='minor', length=3)
    ax6.tick_params(which='minor', width=1)
    ax6.tick_params(axis='both', which='both', top=True, right=True)
    for tick in ax6.xaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for tick in ax6.yaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for axis in ['top', 'bottom', 'left', 'right']:
        ax6.spines[axis].set_linewidth(2)

    fig.tight_layout()




#--------------------------------------------------------------
def plot_kinemetry_maps(xbin, ybin, velbin, k, sigma=False):
    """
    Based on the kinemetry results (k) and original coordinates (xbin,ybin) and
    the analysed moment (i.e. velocity), this routine plots the original moment
    (i.e. velocity) map with overplotted best fitted ellispes, reconstructed
    (rotation) map and a map based on the full Fourier analysis.

    """

    k0 = k.cf[:, 0]
    k1 = np.sqrt(k.cf[:, 1] ** 2 + k.cf[:, 2] ** 2)

    vsys = np.median(k0)
    if sigma:
        mx = np.max(k0)
        mn = np.min(k0)
        vsys = 0
    else:
        mx = np.max(k1)
        mn = -mx
        vsys = np.median(k0)

    tmp = np.where(k.velcirc < 123456789)

    fig, ax = plt.subplots(figsize=(12, 4))

    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.06])

    ax1 = plt.subplot(gs[0])
    im1 = plot_velfield(xbin, ybin, velbin - vsys, colorbar=False, label='km/s', nodots=True, vmin=mn, vmax=mx)
    ax1.plot(k.Xellip, k.Yellip, ',', label='ellipse locations')
    ax1.set_ylabel('arcsec', fontweight='bold')
    ax1.set_xlabel('arcsec', fontweight='bold')
    if sigma:
        ax1.set_title('$\sigma$', fontweight='bold')
    else:
        ax1.set_title('V', fontweight='bold')
    ax1.legend()

    ax1.tick_params(axis='both', which='both', top=True, right=True)
    ax1.tick_params(axis='both', which='major', labelsize=10)
    ax1.yaxis.set_tick_params(length=6)
    ax1.xaxis.set_tick_params(width=2)
    ax1.yaxis.set_tick_params(width=2)
    ax1.xaxis.set_tick_params(length=6)
    ax1.tick_params(which='minor', length=3)
    ax1.tick_params(which='minor', width=1)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for tick in ax1.yaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(2)

    ax1 = plt.subplot(gs[1])
    im1 = plot_velfield(xbin[tmp], ybin[tmp], k.velcirc[tmp] - vsys, colorbar=False, label='km/s', nodots=True, vmin=mn,
                        vmax=mx)
    ax1.set_xlabel('arcsec', fontweight='bold')
    if sigma:
        ax1.set_title('$\sigma_0$', fontweight='bold')
    else:
        ax1.set_title('V$_{disk}$', fontweight='bold')
    ax1.plot(k.xc, k.yc, '+', label='(Xc,Yc)')
    ax1.legend()

    ax1.tick_params(axis='both', which='both', top=True, right=True)
    ax1.tick_params(axis='both', which='major', labelsize=10)
    ax1.yaxis.set_tick_params(length=6)
    ax1.xaxis.set_tick_params(width=2)
    ax1.yaxis.set_tick_params(width=2)
    ax1.xaxis.set_tick_params(length=6)
    ax1.tick_params(which='minor', length=3)
    ax1.tick_params(which='minor', width=1)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for tick in ax1.yaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(2)

    ax1 = plt.subplot(gs[2])
    im1 = plot_velfield(xbin[tmp], ybin[tmp], k.velkin[tmp] - vsys, colorbar=False, label='km/s', nodots=True, vmin=mn,
                        vmax=mx)
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = fig.colorbar(im1, cax=cax)
    cb.ax.tick_params(labelsize=12, width=2)
    cb.set_ticks([mn, 0, mx], update_ticks=True)

    for axis in ['top', 'bottom', 'left', 'right']:
        cb.ax.spines[axis].set_linewidth(5)
    ax1.set_xlabel('arcsec', fontweight='bold')
    if sigma:
        ax1.set_title('$\sigma_{kin}$', fontweight='bold')
        cb.set_label('$\sigma$ [km/s]', fontweight='bold')
    else:
        ax1.set_title('V$_{kin}$', fontweight='bold')
        cb.set_label('V [km/s]', fontweight='bold')

    ax1.tick_params(axis='both', which='both', top=True, right=True)
    ax1.tick_params(axis='both', which='major', labelsize=10)
    ax1.yaxis.set_tick_params(length=6)
    ax1.xaxis.set_tick_params(width=2)
    ax1.yaxis.set_tick_params(width=2)
    ax1.xaxis.set_tick_params(length=6)
    ax1.tick_params(which='minor', length=3)
    ax1.tick_params(which='minor', width=1)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for tick in ax1.yaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(2)

    fig.tight_layout()


#---------------------------------------------------------------
def pa_and_k1_plot(k, y_lim, y_range):

    k1 = np.sqrt(k.cf[:, 1] ** 2 + k.cf[:, 2] ** 2)

    erk1 = (np.sqrt((k.cf[:, 1] * k.er_cf[:, 1]) ** 2 + (k.cf[:, 2] * k.er_cf[:, 2]) ** 2)) / k1

    fig, axs = plt.subplots(2, 1, figsize = (10/3, 6), sharex = True, gridspec_kw = {'hspace': 0})

    axs[0].errorbar(k.rad * 0.5, k.pa, yerr = k.er_pa, fmt = 'o', color = 'black', ecolor = 'black', capsize = 2.5, markersize = 2.5)
    axs[0].plot(k.rad * 0.5, k.pa, color = 'grey', linewidth = 1)
    axs[0].set_ylabel(r'PA$_{kin}$ (degrees)')
    axs[0].set_ylim([0, 185])
    axs[0].set_yticks(np.arange(0, 186, 20))
    axs[0].tick_params(labelbottom = False)

    # make these (major) ticks longer.
    axs[0].tick_params(axis = 'both', which = 'major', length = 4, width = 1, direction = 'in')
    # add minor ticks (shorter, no labels).
    axs[0].xaxis.set_minor_locator(AutoMinorLocator())
    axs[0].yaxis.set_minor_locator(AutoMinorLocator())
    axs[0].tick_params(axis = 'both', which = 'minor', length = 2, width = 1, direction = 'in')

csv_file = pd.read_csv('CATID/CATID_quality_cut_stellar_velocity_map.csv')


xbin = csv_file['x_arcsec'].values
ybin = csv_file['y_arcsec'].values
velbin = csv_file['vel'].values
er_velbin = csv_file['vel_err'].values

k = kinemetry(xbin = xbin, ybin = ybin, moment = velbin, error = er_velbin, x0 = 0, y0 = 0, scale = 0.5, rangeQ= [0.3, 0.8], npa = 41, nq = 41, plot = True)

plot_kinemetry_profiles_velocity(k)
plot_kinemetry_maps(xbin, ybin, velbin, k)

plt.show()


