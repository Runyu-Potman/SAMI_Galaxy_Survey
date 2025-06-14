'''
Three slightly different functions to apply the quality cut criteria on the SAMI stellar velocity maps.
(1). The first function: apply the quality cut criteria and make the plot.
(2). The second function: apply the quality cut criteria, make the plot and produce a CSV file (x, y, vel, vel_err).
(3). The third function: global command to deal with lots of maps together.

Based on the article: 'The SAMI Galaxy survey: the third and final data release' (https://doi.org/10.1093/mnras/stab229),
for SAMI DR3 data, the following quality criteria should be applied to the stellar kinematic maps:
two moments (Q1 and Q2): S/N > 3 /Å, sig > 35 km/s, vel_err < 30 km/s, sig_err < sig * 0.1 +25 km/s.
four moments (Q3):  S/N > 20 /Å, sig > 70 km/s.

Note that all functions here focus on two moments situation.
'''

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

#-------------------------------------------------------------------------------------------------
def quality_cut_stellar_velocity_map(vel_fits_path, sig_fits_path, vmin = None, vmax = None):
    '''
    Apply the quality cut criteria and make plot with x_axis and y_axis in pixel unit,
    note that the center of the galaxy is located at around (25, 25).

    Parameters:
    - vel_fits_path: str, path to the stellar velocity fits file.
    - sig_fits_path: str, path to the stellar velocity dispersion fits file.
    - vmin: int, minimum value for the color bar.
    - vmax: int, maximum value for the color bar.

    Returns:
    - None
    '''

    # read the stellar velocity fits file: velocity (PRIMARY), velocity error (VEL_ERR), S/N (SNR).
    vel_map = fits.open(vel_fits_path)
    vel_data = vel_map[0].data # [0]: PRIMARY.
    vel_err_data = vel_map[1].data # [1]: VEL_ERR.
    vel_SNR_data = vel_map[4].data # [4]: SNR.

    # read the stellar velocity dispersion fits file: dispersion (PRIMARY), dispersion error (SIG_ERR), S/N (SNR).
    sig_map = fits.open(sig_fits_path)
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
    vel_data = np.ma.masked_where(vel_SNR_data <= 5, vel_data) # S/N > 5.
    vel_data = np.ma.masked_where(sig_SNR_data <= 5, vel_data) # S/N > 5.
    vel_data = np.ma.masked_where(sig_data <= 35, vel_data) # sig > 35 km/s.
    vel_data = np.ma.masked_where(vel_err_data >= 30, vel_data) # vel_err < 30 km/s.
    vel_data = np.ma.masked_where(sig_err_data >= (sig_data * 0.1 + 25), vel_data) # sig_err < sig * 0.1 + 25 km/s.

    sig_data = np.ma.masked_where(sig_SNR_data <= 5, sig_data)
    sig_data = np.ma.masked_where(sig_data <= 35, sig_data)
    sig_data = np.ma.masked_where(sig_err_data >= (sig_data * 0.1 + 25), sig_data)
    
    # get the mask from other maps.
    combined_mask = np.ma.getmask(vel_err_data)
    combined_mask = np.ma.mask_or(combined_mask, np.ma.getmask(vel_SNR_data))

    combined_mask = np.ma.mask_or(combined_mask, np.ma.getmask(sig_data))
    combined_mask = np.ma.mask_or(combined_mask, np.ma.getmask(sig_err_data))
    combined_mask = np.ma.mask_or(combined_mask, np.ma.getmask(sig_SNR_data))

    # apply the combined_mask to the stellar velocity map.
    cleaned_vel_data = np.ma.masked_array(vel_data, mask = combined_mask)

    # plot the quality cut stellar velocity map.
    plt.figure(figsize=(10, 8))

    plt.imshow(
        cleaned_vel_data, origin = 'lower', aspect = 'auto',
        cmap = 'jet', vmin = vmin, vmax = vmax
    )

    plt.colorbar(label = 'km/s')

    plt.title('Quality Cut Stellar Kinematic Map')
    plt.xlabel('SPAXEL')
    plt.ylabel('SPAXEL')

    plt.show()

    # close the fits files after use.
    vel_map.close()
    sig_map.close()

    return combined_mask, cleaned_vel_data

#----------------------------------------------------------------------------------------------------
def quality_cut_stellar_velocity_map_csv(vel_fits_path, sig_fits_path, output_file, pixel_to_arc = True):
    '''
    Apply the quality cut criteria and make plot with x_axis and y_axis in arcsec unit, note that the center of
    the galaxy in this function is now shifted to (0, 0), in preparation for e.g., position angle calculation.

    In order to use the fit_kinematic_pa code, the coordinate (0, 0) should be an estimate of the centre of rotation.
    For SAMI, the dimension for the maps is 50 * 50, and the center of the galaxy is approximately located at (25, 25).
    So the center should be shifted such that (0, 0) being the center of rotation.

    Be careful about the indexing definition in fits file and in Python. Fits files use 1-based indexing, meaning
    the first pixel is indexed as (1, 1); Python use 0-based indexing, meaning the first pixel is indexed as (0, 0).

    Another thing to mention is that in Python (and in Fits), (row, column) -> (y, x).

    Parameters:
    - vel_fits_path: str, path to the stellar velocity fits file.
    - sig_fits_path: str, path to the stellar velocity dispersion fits file.
    - output_file: str, path to the output file.
    - pixel_to_arc: bool, whether to transfer from pixel scale into arc scale.

    Returns:
    - None.
    '''

    # read the stellar velocity fits file: velocity (PRIMARY), velocity error (VEL_ERR), S/N (SNR).
    vel_map = fits.open(vel_fits_path)
    vel_data = vel_map[0].data # [0]: PRIMARY.
    vel_err_data = vel_map[1].data # [1]: VEL_ERR.
    vel_SNR_data = vel_map[4].data # [4]: SNR.

    # read the stellar velocity dispersion fits file: dispersion (PRIMARY), dispersion error (SIG_ERR), S/N (SNR).
    sig_map = fits.open(sig_fits_path)
    sig_data = sig_map[0].data # [0]: PRIMARY.
    sig_err_data = sig_map[1].data # [1]: SIG_ERR.
    sig_SNR_data = sig_map[4].data # [4]: SNR.

    # mask NaN values in the initial velocity, velocity error, velocity SNR, dispersion, dispersion error, dispersion SNR maps.
    # if any of the six maps have a NaN value at a specific spaxel, this spaxel should be excluded.
    vel_data = np.ma.masked_invalid(vel_data)
    vel_err_data = np.ma.masked_invalid(vel_err_data)
    vel_SNR_data = np.ma.masked_invalid(vel_SNR_data)

    sig_data = np.ma.masked_invalid(sig_data)
    sig_err_data = np.ma.masked_invalid(sig_err_data)
    sig_SNR_data = np.ma.masked_invalid(sig_SNR_data)

    # apply the two moments quality cut criteria.
    vel_data = np.ma.masked_where(vel_SNR_data <= 5, vel_data) # S/N > 5.
    vel_data = np.ma.masked_where(sig_SNR_data <= 5, vel_data) # S/N > 5.
    vel_data = np.ma.masked_where(sig_data <= 35, vel_data) # sig > 35 km/s.
    vel_data = np.ma.masked_where(vel_err_data >= 30, vel_data) # vel_err < 30 km/s.
    vel_data = np.ma.masked_where(sig_err_data >= (sig_data * 0.1 + 25), vel_data) # sig_err < sig * 0.1 + 25 km/s.

    sig_data = np.ma.masked_where(sig_SNR_data <= 5, sig_data)
    sig_data = np.ma.masked_where(sig_data <= 35, sig_data)
    sig_data = np.ma.masked_where(sig_err_data >= (sig_data * 0.1 + 25), sig_data)

    # prepare the csv data for the position angle calculation.
    ny, nx = vel_data.shape

    data_to_save = []

    for i in range(ny):
        for j in range(nx):
            if (not vel_data.mask[i, j] and not vel_err_data.mask[i, j] and not vel_SNR_data.mask[i, j]
                    and not sig_data.mask[i, j] and not sig_err_data.mask[i, j] and not sig_SNR_data.mask[i, j]):

                if pixel_to_arc:
                    x_arcsec = (j - 25) * 0.5
                    y_arcsec = (i - 25) * 0.5
                else:
                    x_arcsec = j - 25
                    y_arcsec = i - 25

                print(f'{x_arcsec}, {y_arcsec}, {vel_data[i, j]}, {vel_err_data[i, j]}, {sig_data[i, j]}')
                data_to_save.append((x_arcsec, y_arcsec, vel_data[i, j], vel_err_data[i, j], sig_data[i, j]))

    with open(output_file, 'w') as f:
        f.write('x_arcsec,y_arcsec,vel,vel_err,sig\n')
        for entry in data_to_save:
            f.write(f'{entry[0]}, {entry[1]}, {entry[2]}, {entry[3]}, {entry[4]}\n')

    # close the fits files after use.
    vel_map.close()
    sig_map.close()

#--------------------------------------------------------------------------------------------------
def quality_cut_stellar_velocity_map_global(vel_base_dir, sig_base_dir, output_dir):
    '''
    This script can be used to do the quality cut process for stellar velocity maps of all the galaxy samples
    in SAMI galaxy survey (3068 in total for DR3), the same quality cut criteria is applied (Q1 + Q2).

    The ifs_velocity directory contains 3068 galaxy directories (stellar velocity maps), the ifs_velocity_dispersion
    directory contains 3068 galaxy directories (corresponding velocity dispersion maps).

    For example, for galaxy with CATID 6821, the directory structure is:
    (ifs_velocity) -> (6821) -> (6821_A_stellar-velocity_default_two-moment.fits).
    (ifs_velocity_dispersion) -> (6821) -> (6821_A_stellar-velocity-dispersion_default_two-moment.fits).

    Parameters:
    - vel_base_dir: str, path to the velocity base directory.
    - sig_base_dir: str, path to the velocity dispersion base directory.
    - output_dir: str, path to the directory where the output PNG files will be saved.

    Returns:
    - None
    '''

    # debug: list all galaxy directories (e.g., '6821', '6837', etc.) in the velocity base directory.
    gal_dirs = [d for d in os.listdir(vel_base_dir) if os.path.isdir(os.path.join(vel_base_dir, d))]
    print(f'Found galaxy directories: {gal_dirs}')

    # loop over each galaxy directory.
    for gal_dir in gal_dirs:
        # construct the full paths for the velocity and velocity dispersion directories.
        vel_dir_full = os.path.join(vel_base_dir, gal_dir)
        sig_dir_full = os.path.join(sig_base_dir, gal_dir)

        # get the list of stellar velocity fits files for the current galaxy.
        # update the pattern to account for the 'A' in the filename, change to _B/_C/_D/_E in the case where one galaxy has multiple fits files.
        vel_fits = glob.glob(os.path.join(vel_dir_full, f'{gal_dir}_A_stellar-velocity_default_two-moment.fits'))

        # if no stellar velocity files are found, skip this galaxy.
        if not vel_fits:
            print(f'No stellar velocity fits file found for galaxy {gal_dir}. Skipping.')
            continue

        # loop through each stellar velocity fits file.
        for vel_file in vel_fits:
            # extract the base name (e.g., 6821) from the velocity filename.
            base_name = os.path.basename(vel_file).split('_')[0]

            # correct the filename pattern to match the dispersion filename.
            # explicitly append '_A' to the base name when searching for the dispersion file, change to _B/_C/_D/_E in the case where one galaxy has multiple fits files.
            sig_file = os.path.join(sig_dir_full, f'{base_name}_A_stellar-velocity-dispersion_default_two-moment.fits')

            if not os.path.exists(sig_file):
                print(f'Warning: Corresponding dispersion fits file for {vel_file} not found. Skipping.')
                continue

            # read the stellar velocity fits file: velocity (PRIMARY), velocity error (VEL_ERR), S/N (SNR).
            vel_map = fits.open(vel_file)
            vel_data = vel_map[0].data  # [0]: PRIMARY.
            vel_err_data = vel_map[1].data  # [1]: VEL_ERR.
            vel_SNR_data = vel_map[4].data  # [4]: SNR.

            # read the stellar velocity dispersion fits file: dispersion (PRIMARY), dispersion error (SIG_ERR), S/N (SNR).
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
            vel_data = np.ma.masked_where(vel_SNR_data <= 5, vel_data)  # S/N > 5.
            vel_data = np.ma.masked_where(sig_SNR_data <= 5, vel_data)  # S/N > 5.
            vel_data = np.ma.masked_where(sig_data <= 35, vel_data)  # sig > 35 km/s.
            vel_data = np.ma.masked_where(vel_err_data >= 30, vel_data)  # vel_err < 30 km/s.
            vel_data = np.ma.masked_where(sig_err_data >= (sig_data * 0.1 + 25),
                                          vel_data)  # sig_err < sig * 0.1 + 25 km/s.

            # get the mask from other maps.
            combined_mask = np.ma.getmask(vel_err_data)
            combined_mask = np.ma.mask_or(combined_mask, np.ma.getmask(vel_SNR_data))

            combined_mask = np.ma.mask_or(combined_mask, np.ma.getmask(sig_data))
            combined_mask = np.ma.mask_or(combined_mask, np.ma.getmask(sig_err_data))
            combined_mask = np.ma.mask_or(combined_mask, np.ma.getmask(sig_SNR_data))

            # apply the combined_mask to the stellar velocity map.
            vel_data = np.ma.masked_array(vel_data, mask = combined_mask)

            # plot the quality cut stellar velocity map.
            plt.figure(figsize = (10, 8))

            plt.imshow(
                vel_data, origin = 'lower', aspect = 'auto',
                cmap = 'jet'
            )

            plt.colorbar(label = 'km/s')

            plt.title(f'Quality Cut Stellar Kinematic Map for Galaxy {base_name}')
            plt.xlabel('SPAXEL')
            plt.ylabel('SPAXEL')

            plot_filename = os.path.join(output_dir, f'{base_name}_A_quality_cut_stellar_velocity_map.png')

            plt.savefig(plot_filename)
            plt.close()

            # close the fits files after use.
            vel_map.close()
            sig_map.close()

# example usage:
#cwd = os.getcwd()
# ifs_velocity contains 3068 galaxy directories.
#vel_base_dir = os.path.join(cwd, 'ifs_velocity')
# ifs_velocity_dispersion contains 3068 galaxy directories.
#sig_base_dir = os.path.join(cwd, 'ifs_velocity_dispersion')
#output_dir = cwd

#quality_cut_stellar_velocity_map_global(vel_base_dir, sig_base_dir, output_dir)
#---------------------------------------------------------------------------------------------------
def quality_cut_gaseous_velocity_map_csv(vel_fits_path, sig_fits_path, Halpha_fits_path, output_file, pixel_to_arc = True):
    '''
    Apply the quality cut criteria and prepare csv file with x_axis and y_axis in arcsec unit, note that the center
    of the galaxy in this function is now shifted to (0, 0), in preparation for e.g., position angle calculation.

    In order to use the fit_kinematic_pa code, the coordinate (0, 0) should be an estimate of the centre of rotation.
    For SAMI, the dimension for the spaxels is 50 * 50, and the center of the galaxy is approximately located at (25, 25).
    So the center should be shifted such that (0, 0) being the center of rotation.

    Be careful about the indexing definition in fits file and in Python. Fits files use 1-based indexing, meaning
    the first pixel is indexed as (1, 1); Python use 0-based indexing, meaning the first pixel is indexed as (0, 0).

    Another thing to mention is that in Python (and in Fits), (row, column) -> (y, x).

    Parameters:
    - vel_fits_path: str, path to the gaseous velocity fits file.
    - sig_fits_path: str, path to the gaseous velocity dispersion fits file.
    - Halpha_fits_path: str, path to the Halpha flux fits file.
    - output_file: str, path to the output csv file.

    Returns:
    - None
    '''

    # read the gaseous velocity fits file: velocity (PRIMARY), velocity error (VEL_ERR).
    vel_map = fits.open(vel_fits_path)
    vel_data = np.squeeze(vel_map[0].data) # [0]: PRIMARY.
    vel_err_data = np.squeeze(vel_map[1].data) # [1]: VEL_ERR.

    # read the gaseous velocity dispersion fits file: dispersion (PRIMARY), dispersion error (SIG_ERR).
    sig_map = fits.open(sig_fits_path)
    sig_data = np.squeeze(sig_map[0].data) # [0]: PRIMARY.
    sig_err_data = np.squeeze(sig_map[1].data) # [1]: SIG_ERR.

    # read the Halpha flux fits file.
    Halpha_map = fits.open(Halpha_fits_path)
    Halpha_data = Halpha_map[0].data
    Halpha_err_data = Halpha_map[1].data

    # here we extract the total component.
    # for the 1-component situation, [0, :, :] is the same as [1, :, :].
    Halpha_data = Halpha_data[0, :, :]
    Halpha_err_data = Halpha_err_data[0, :, :]

    # mask NaN values in the initial velocity, velocity error, dispersion, dispersion error, Halpha, Halpha error.
    # if any of the six maps have a NaN value at a specific spaxel, this spaxel should be excluded.
    vel_data = np.ma.masked_invalid(vel_data)
    vel_err_data = np.ma.masked_invalid(vel_err_data)

    sig_data = np.ma.masked_invalid(sig_data)
    sig_err_data = np.ma.masked_invalid(sig_err_data)

    Halpha_data = np.ma.masked_invalid(Halpha_data)
    Halpha_err_data = np.ma.masked_invalid(Halpha_err_data)

    # calculate S/N.
    SNR_data = Halpha_data / Halpha_err_data

    # apply the quality cut criteria.
    vel_data = np.ma.masked_where(SNR_data <= 5, vel_data) # S/N > 5.
    vel_data = np.ma.masked_where(vel_err_data >= 30, vel_data) # vel_err < 30 km/s.

    sig_data = np.ma.masked_where(SNR_data <= 5, sig_data)

    # prepare the csv data for the position angle calculation.
    ny, nx = vel_data.shape

    data_to_save = []

    for i in range(ny):
        for j in range(nx):
            if (not vel_data.mask[i, j] and not vel_err_data.mask[i, j]
                    and not sig_data.mask[i, j] and not sig_err_data.mask[i, j]
                    and not Halpha_data.mask[i, j] and not Halpha_err_data.mask[i, j]):

                if pixel_to_arc:
                    x_arcsec = (j - 25) * 0.5
                    y_arcsec = (i - 25) * 0.5
                else:
                    x_arcsec = j - 25
                    y_arcsec = i - 25

                print(f'{x_arcsec}, {y_arcsec}, {vel_data[i, j]}, {vel_err_data[i, j]}, {sig_data[i, j]}')
                data_to_save.append((x_arcsec, y_arcsec, vel_data[i, j], vel_err_data[i, j], sig_data[i, j]))

    with open(output_file, 'w') as f:
        f.write('x_arcsec,y_arcsec,vel,vel_err,sig\n')
        for entry in data_to_save:
            f.write(f'{entry[0]}, {entry[1]}, {entry[2]}, {entry[3]}, {entry[4]}\n')

    # close the fits files after use.
    vel_map.close()
    sig_map.close()
    Halpha_map.close()