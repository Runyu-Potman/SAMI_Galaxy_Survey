'''
This script can be used to do the quality cut process for stellar velocity maps of galaxies in SAMI galaxy survey. Based on the
article: 'The SAMI Galaxy survey: the third and final data release' (https://doi.org/10.1093/mnras/stab229, Scott M. Croom et al.),
for SAMI DR3 data, the following quality criteria to the stellar kinematic maps should be applied:
two moments: 1. S/N > 3, sig > 35 km/s, vel_err < 30 km/s, sig_err < sig * 0.1 +25 km/s.
four moments:  S/N > 20, sig > 70 km/s.
'''

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

# open the stellar velocity fits file: velocity (PRIMARY), velocity error (VEL_ERR), S/N (SNR).
vel_map = fits.open('7969_A_stellar-velocity_default_two-moment.fits')
vel_data = vel_map[0].data # [0]: PRIMARY.
vel_err_data = vel_map[1].data # [1]: VEL_ERR.
vel_SNR_data = vel_map[4].data # [4]: SNR.

# open the stellar velocity dispersion fits file: dispersion (PRIMARY), dispersion error (SIG_ERR), S/N (SNR).
sig_map = fits.open('7969_A_stellar-velocity-dispersion_default_two-moment.fits')
sig_data = sig_map[0].data # [0]: PRIMARY.
sig_err_data = sig_map[1].data # [1]: SIG_ERR.
sig_SNR_data = sig_map[4].data # [4]: SNR.

# mask NaN values in the initial velocity, velocity error, velocity SNR, dispersion, dispersion error, dispersion SNR maps.
# if any of the six maps have a NaN value at a specific spaxel, this spaxel should be masked (excluded).
vel_data = np.ma.masked_invalid(vel_data)
vel_err_data = np.ma.masked_invalid(vel_err_data)
vel_SNR_data = np.ma.masked_invalid(vel_SNR_data)

sig_data = np.ma.masked_invalid(sig_data)
sig_err_data = np.ma.masked_invalid(sig_err_data)
sig_SNR_data = np.ma.masked_invalid(sig_SNR_data)

# apply the two moments quality cut criteria.
vel_data = np.ma.masked_where(vel_SNR_data <= 3, vel_data) # S/N > 3.
vel_data = np.ma.masked_where(sig_SNR_data <= 3, vel_data) # S/N > 3.
vel_data = np.ma.masked_where(sig_data <= 35, vel_data) # sig > 35 km/s.
vel_data = np.ma.masked_where(vel_err_data >= 30, vel_data) # vel_err < 30 km/s.
vel_data = np.ma.masked_where(sig_err_data >= (sig_data * 0.1 + 25), vel_data) # sig_err < sig * 0.1 + 25 km/s.

# combined_mask: all the above mask.
combined_mask = np.ma.getmask(vel_data)
combined_mask = np.ma.mask_or(combined_mask, np.ma.getmask(vel_err_data))
combined_mask = np.ma.mask_or(combined_mask, np.ma.getmask(vel_SNR_data))

combined_mask = np.ma.mask_or(combined_mask, np.ma.getmask(sig_data))
combined_mask = np.ma.mask_or(combined_mask, np.ma.getmask(sig_err_data))
combined_mask = np.ma.mask_or(combined_mask, np.ma.getmask(sig_SNR_data))

# apply the combined_mask to the stellar velocity map.
vel_data = np.ma.masked_array(vel_data, mask = combined_mask)

# plot the quality cut stellar velocity map.
plt.figure(figsize=(10, 8))

# change the maximum and minimum value for the color bar.
vmin = -75
vmax = 75

# interpolation = 'nearest' copies the nearest value to a pixel, results in a pixelated appearance.
plt.imshow(vel_data, origin = 'lower', aspect = 'auto', cmap = 'jet', interpolation = 'nearest', vmin = vmin, vmax = vmax)

# add a color bar.
plt.colorbar(label = 'km/s')

# add labels and title.
plt.title('Quality Cut Stellar Kinematic Map for Galaxy 7969')
plt.xlabel('SPAXEL')
plt.ylabel('SPAXEL')

# show the plot.
plt.show()

# close the fits files after use.
vel_map.close()
sig_map.close()