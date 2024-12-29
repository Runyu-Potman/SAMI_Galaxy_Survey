'''
This script can be used to do the quality cut process for stellar velocity maps of all the galaxy samples (3068 in total for DR3)
in SAMI galaxy survey, the same quality cut criteria is applied as in stellar_velocity_quality_cut_script.py. The ifs_velocity directory
contains 3068 galaxy directories (stellar velocity map), the ifs_velocity_dispersion directory contains 3068 galaxy directories
(corresponding velocity dispersion map). For example, for galaxy with CATID 6821, the directory structure is: (ifs_velocity) -> (6821) ->
(6821_A_stellar-velocity_default_two-moment.fits).
'''

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# define the base directories for the stellar velocity and velocity dispersion maps.
# get the current directory where the script is running.
cwd = os.getcwd()
# adjusted to the velocity base directory which contains 3068 galaxy directories.
vel_base_dir = os.path.join(cwd, 'ifs_velocity')
# adjusted to the dispersion base directory which contains 3068 galaxy directories.
sig_base_dir = os.path.join(cwd, 'ifs_velocity_dispersion')

# debug: list all galaxy directories (e.g., '6821', '6837', etc.) in the velocity base directory.
gal_dirs = [d for d in os.listdir(vel_base_dir) if os.path.isdir(os.path.join(vel_base_dir, d))]
print(f'Found galaxy directories: {gal_dirs}')

# loop over each galaxy directory.
for gal_dir in gal_dirs:
    # construct the full paths for the velocity and velocity dispersion directories.
    vel_dir_full = os.path.join(vel_base_dir, gal_dir)
    sig_dir_full = os.path.join(sig_base_dir, gal_dir)

    # get the list of stellar velocity fits files for the current galaxy.
    # update the pattern to account for the 'A' in the filename (e.g., '6821_A'), change to _B/_C/_D/_E in the case where one galaxy has multiple fits files.
    vel_fits = glob.glob(os.path.join(vel_dir_full, f'{gal_dir}_A_stellar-velocity_default_two-moment.fits'))

    # if no stellar velocity files are found, skip this galaxy.
    if not vel_fits:
        print(f'No stellar velocity fits file found for galaxy {gal_dir}. Skipping.')
        continue

    # loop through each stellar velocity fits file.
    for vel_file in vel_fits:
        # extract the base name (e.g., '6821') from the velocity file name.
        base_name = os.path.basename(vel_file).split('_')[0]

        # correct the filename pattern to match the dispersion file name.
        # explicitly append '_A' to the base name when searching for the dispersion file, change to _B/_C/_D/_E in the case where one galaxy has multiple fits files.
        sig_file = os.path.join(sig_dir_full, f'{base_name}_A_stellar-velocity-dispersion_default_two-moment.fits')

        #check if the corresponding dispersion file exists
        if not os.path.exists(sig_file):
            print(f'Warning: Corresponding dispersion fits file for {vel_file} not found. Skipping.')
            continue

        # open the stellar velocity fits file: velocity (PRIMARY), velocity error (VEL_ERR), S/N (SNR).
        vel_map = fits.open(vel_file)
        vel_data = vel_map[0].data  # [0]: PRIMARY.
        vel_err_data = vel_map[1].data  # [1]: VEL_ERR.
        vel_SNR_data = vel_map[4].data  # [4]: SNR.

        # open the stellar velocity dispersion fits file: dispersion (PRIMARY), dispersion error (SIG_ERR), S/N (SNR).
        sig_map = fits.open(sig_file)
        sig_data = sig_map[0].data  # [0]: PRIMARY.
        sig_err_data = sig_map[1].data  # [1]: SIG_ERR.
        sig_SNR_data = sig_map[4].data  # [4]: SNR.

        # mask NaN values in the initial velocity, velocity error, velocity SNR, dispersion, dispersion error, dispersion SNR maps.
        # if any of the six maps have a NaN value at a specific spaxel, this spaxel should be masked (excluded).
        vel_data = np.ma.masked_invalid(vel_data)
        vel_err_data = np.ma.masked_invalid(vel_err_data)
        vel_SNR_data = np.ma.masked_invalid(vel_SNR_data)

        sig_data = np.ma.masked_invalid(sig_data)
        sig_err_data = np.ma.masked_invalid(sig_err_data)
        sig_SNR_data = np.ma.masked_invalid(sig_SNR_data)

        # apply the two moments quality cut criteria.
        vel_data = np.ma.masked_where(vel_SNR_data <= 3, vel_data)  # S/N > 3.
        vel_data = np.ma.masked_where(sig_SNR_data <= 3, vel_data)  # S/N > 3.
        vel_data = np.ma.masked_where(sig_data <= 35, vel_data)  # sig > 35 km/s.
        vel_data = np.ma.masked_where(vel_err_data >= 30, vel_data)  # vel_err < 30 km/s.
        vel_data = np.ma.masked_where(sig_err_data >= (sig_data * 0.1 + 25), vel_data)  # sig_err < sig * 0.1 + 25 km/s.

        # combined_mask: all the above mask.
        combined_mask = np.ma.getmask(vel_data)
        combined_mask = np.ma.mask_or(combined_mask, np.ma.getmask(vel_err_data))
        combined_mask = np.ma.mask_or(combined_mask, np.ma.getmask(vel_SNR_data))

        combined_mask = np.ma.mask_or(combined_mask, np.ma.getmask(sig_data))
        combined_mask = np.ma.mask_or(combined_mask, np.ma.getmask(sig_err_data))
        combined_mask = np.ma.mask_or(combined_mask, np.ma.getmask(sig_SNR_data))

        # apply the combined_mask to the stellar velocity map.
        vel_data = np.ma.masked_array(vel_data, mask=combined_mask)

        # plot the quality cut stellar velocity map.
        plt.figure(figsize = (10, 8))

        # interpolation = 'nearest' copies the nearest value to a pixel, results in a pixelated appearance.
        plt.imshow(vel_data, origin = 'lower', aspect = 'auto', cmap = 'jet', interpolation = 'nearest')

        # add a color bar.
        plt.colorbar(label = 'km/s')

        # add labels and title.
        plt.title(f'Quality Cut Stellar Kinematic Map for Galaxy {base_name}')
        plt.xlabel('SPAXEL')
        plt.ylabel('SPAXEL')

        # save the plot as a PNG file in the current directory, change to _B/_C/_D/_E in the case where one galaxy has multiple fits files.
        plot_filename = os.path.join(cwd, f"quality_cut_stellar_velocity_map_{base_name}_A.png")
        plt.savefig(plot_filename)

        # close the plot to release memory.
        plt.close()

        # close the fits files after use.
        vel_map.close()
        sig_map.close()




