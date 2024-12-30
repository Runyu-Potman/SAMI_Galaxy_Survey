'''
This script can be used to do the quality cut process for stellar velocity maps of galaxies in SAMI galaxy survey. Based on the
article: 'The SAMI Galaxy survey: the third and final data release' (https://doi.org/10.1093/mnras/stab229, Scott M. Croom et al.),
for SAMI DR3 data, the following quality cut criteria to the stellar kinematic maps should be applied:
two moments: 1. S/N > 3, sig > 35 km/s, vel_err < 30 km/s, sig_err < sig * 0.1 +25 km/s.
four moments:  S/N > 20, sig > 70 km/s.

In the meantime, this script can be used to export CSV file in preparation for position angle calculation based on the
fit_kinematic_pa code.
'''

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

# prepare the csv data for the position angle calculation.
nx, ny = vel_data.shape

x = np.arange(nx)
y = np.arange(ny)

data_to_save=[]

'''
In order to use the fit_kinematic_pa code, the coordinate (0, 0) should be an estimate of the centre of rotation. 
For SAMI, the dimension for the spaxels is 50 * 50, and the center of the galaxy is approximately located at (25, 25) 
or (26, 26) (check the flux map to see which spaxel has a higher value). So the center should be shifted such that 
(0, 0) being the center of rotation.

When checking the flux values in fits files, be careful about the indexing definition in fits file and in Python. 
Fits files use 1-based indexing, meaning the first pixel is indexed as (1, 1); Python use 0-based indexing, meaning 
the first pixel is indexed as (0, 0).

Another thing to mention is that in Python (and in Fits), (row, column) -> (y, x).
'''

for i in range(ny):
    for j in range(nx):
        if (not vel_data.mask[i, j] and not vel_err_data.mask[i, j] and not vel_SNR_data.mask[i, j]
                and not sig_data.mask[i, j] and not sig_err_data.mask[i, j] and not sig_SNR_data.mask[i, j]):
             print(f'{x[j] - 24}, {y[i] - 24}, {vel_data[i, j]}, {vel_err_data[i, j]}')
             data_to_save.append((x[j] - 24, y[i] - 24, vel_data[i, j], vel_err_data[i, j]))

output_file = 'stellar_velocity_quality_cut_7969.csv'
with open(output_file, 'w') as f:
    f.write('x, y, vel, vel_err\n')
    for entry in data_to_save:
        f.write(f'{entry[0]}, {entry[1]}, {entry[2]}, {entry[3]}\n')

# read the csv file.
quality_cut_map = pd.read_csv('stellar_velocity_quality_cut_7969.csv')

# strip any leading whitespace from column names.
quality_cut_map.columns = quality_cut_map.columns.str.strip()

# extract unique x and y values
x_values = np.unique(quality_cut_map['x'])
y_values = np.unique(quality_cut_map['y'])

velocity_grid = np.full((len(y_values), len(x_values)), np.nan)

for index, row in quality_cut_map.iterrows():
    x_grid = np.where(x_values == row['x'])[0][0]
    y_grid = np.where(y_values == row['y'])[0][0]
    velocity_grid[y_grid, x_grid] = row['vel']

# plot the stellar velocity map based on csv file.
plt.figure(figsize = (10, 8))

# change the maximum and minimum value for the color bar.
vmin = -75
vmax = 75

# interpolation = 'nearest' copies the nearest value to a pixel, results in a pixelated appearance.
plt.imshow(velocity_grid, origin = 'lower', aspect = 'auto', cmap = 'jet', interpolation = 'nearest', vmin = vmin, vmax = vmax)

# add a color bar.
plt.colorbar(label = 'km/s')

# add labels, title and ticks.
plt.title('Quality Cut Stellar Kinematic Map for Galaxy 7969')
plt.xlabel('SPAXEL')
plt.ylabel('SPAXEL')

plt.xticks(np.arange(len(x_values)), x_values)
plt.yticks(np.arange(len(y_values)), y_values)

# show the plot.
plt.show()

# close the fits files after use.
vel_map.close()
sig_map.close()