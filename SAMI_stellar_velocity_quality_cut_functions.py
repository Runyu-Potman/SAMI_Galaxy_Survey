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

version_02: 02/07/2025
we include the four-moment situation and make preparations for DYNAMITE, however, based on the paper of Santucci et al.,
they do not apply the Q3 but increase the error for those measurements which do not meet Q3.

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
def quality_cut_stellar_velocity_map_csv(vel_fits_path, sig_fits_path, output_file, pixel_to_arc = True,
                                         center_x = 25, center_y = 25):
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
    - center_x: float, center x coordinate of the galaxy.
    - center_y: float, center y coordinate of the galaxy.

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
                    x_arcsec = (j - center_x) * 0.5
                    y_arcsec = (i - center_y) * 0.5
                else:
                    x_arcsec = j - center_x
                    y_arcsec = i - center_y

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

#-------------------------------------------------------------------------------------------------------------------
def quality_cut_stellar_velocity_map_four_moment(
        vel_fits_path, sig_fits_path, h3_fits_path, h4_fits_path, output_filename,
        Q3 = False, plot = False, dynamite = False, center_x = 25, center_y = 25,
        vmin=None, vmax=None):
    '''
    Apply the quality cut criteria and make plot with x_axis and y_axis in pixel unit,
    note that the center of the galaxy is located at around (25, 25).If dynamite = True,
    then a fits file will be prepared for dynamite input, the center will be shifted to (0, 0).

    Parameters:
    - vel_fits_path: str, path to the stellar velocity fits file.
    - sig_fits_path: str, path to the stellar velocity dispersion fits file.
    - h3_fits_path: str, path to the skewness fits file.
    - h4_fits_path: str, path to the kurtosis fits file.
    - output_filename: str, name and path of the output fits file (if dynamite = True).
    - Q3: the Q3 quality cut criteria when considering four moment, default = False.
    - plot: plot the vel, sig, h3 and h4.
    - dynamite: whether to prepare for dynamite input or not.
    - center_x: the center of the galaxy in pixel unit, default = 25.
    - center_y: the center of the galaxy in pixel unit, default = 25.
    - vmin: int, minimum value for the color bar.
    - vmax: int, maximum value for the color bar.

    Returns:
    - None
    '''

    # read the stellar velocity fits file: velocity (PRIMARY), velocity error (VEL_ERR), S/N (SNR).
    with fits.open(vel_fits_path) as vel_map:
        vel_data = vel_map[0].data  # [0]: PRIMARY.
        vel_err_data = vel_map[1].data  # [1]: VEL_ERR.
        vel_SNR_data = vel_map[4].data  # [4]: SNR.

        # flux for dynamite.
        vel_flux = vel_map[2].data # [2]: FLUX.

    # read the stellar velocity dispersion fits file: dispersion (PRIMARY), dispersion error (SIG_ERR), S/N (SNR).
    with fits.open(sig_fits_path) as sig_map:
        sig_data = sig_map[0].data  # [0]: PRIMARY.
        sig_err_data = sig_map[1].data  # [1]: SIG_ERR.
        sig_SNR_data = sig_map[4].data  # [4]: SNR.

        # flux for dynamite.
        sig_flux = sig_map[2].data # [2]: FLUX.

    # read the skewness fits file: h3 (PRIMARY), h3 error (H3_ERR), S/N  (SNR).
    with fits.open(h3_fits_path) as h3_map:
        h3_data = h3_map[0].data # [0]: PRIMARY.
        h3_err_data = h3_map[1].data # [1]: H3_ERR.
        h3_SNR_data = h3_map[4].data # [4]: SNR.

        # flux for dynamite.
        h3_flux = h3_map[2].data # [2]: FLUX.

    # read the kurtosis fits file: h4 (PRIMARY), h4 error (H4_ERR), S/N  (SNR).
    with fits.open(h4_fits_path) as h4_map:
        h4_data = h4_map[0].data # [0]: PRIMARY.
        h4_err_data = h4_map[1].data # [1]: H4_ERR.
        h4_SNR_data = h4_map[4].data # [4]: SNR.

        # flux for dynamite.
        h4_flux = h4_map[2].data # [2]: FLUX.

    # mask NaN values in the initial velocity, velocity error, velocity SNR, dispersion, dispersion error, dispersion SNR,
    # h3, h3 error, h3 SNR, h4, h4 err, h4 SNR maps.
    # if any of the twelve maps have a NaN value at a specific spaxel, this spaxel should be masked (excluded).
    vel_data = np.ma.masked_invalid(vel_data)
    vel_err_data = np.ma.masked_invalid(vel_err_data)
    vel_SNR_data = np.ma.masked_invalid(vel_SNR_data)

    sig_data = np.ma.masked_invalid(sig_data)
    sig_err_data = np.ma.masked_invalid(sig_err_data)
    sig_SNR_data = np.ma.masked_invalid(sig_SNR_data)

    h3_data = np.ma.masked_invalid(h3_data)
    h3_err_data = np.ma.masked_invalid(h3_err_data)
    h3_SNR_data = np.ma.masked_invalid(h3_SNR_data)

    h4_data = np.ma.masked_invalid(h4_data)
    h4_err_data = np.ma.masked_invalid(h4_err_data)
    h4_SNR_data = np.ma.masked_invalid(h4_SNR_data)

    # in principle, the SNR maps across all kinematic fits files should be the same for SAMI.
    SNR_comparison_all = (np.all(vel_SNR_data == sig_SNR_data)
                          and np.all(sig_SNR_data == h3_SNR_data)
                          and np.all(h3_SNR_data == h4_SNR_data))

    if SNR_comparison_all:
        print("SNR across all input fits files are the same, continue...")
        SNR_data = vel_SNR_data.copy()
    else:
        raise ValueError("SNR are not the same across all input fits files, check the input data!")

    if Q3 == True:
        # apply Q1, Q2 and Q3.
        vel_data = np.ma.masked_where(SNR_data <= 20.5, vel_data)  # S/N > 20.5.
        vel_data = np.ma.masked_where(sig_data <= 70, vel_data)  # sig > 70 km/s.
        vel_data = np.ma.masked_where(vel_err_data >= 30, vel_data)  # vel_err < 30 km/s.
        vel_data = np.ma.masked_where(sig_err_data >= (sig_data * 0.1 + 25), vel_data)  # sig_err < sig * 0.1 + 25 km/s.

        sig_data = np.ma.masked_where(SNR_data <= 20.5, sig_data)
        sig_data = np.ma.masked_where(sig_data <= 70, sig_data)
        sig_data = np.ma.masked_where(sig_err_data >= (sig_data * 0.1 + 25), sig_data)

        h3_data = np.ma.masked_where(SNR_data <= 20.5, h3_data)
        h4_data = np.ma.masked_where(SNR_data <= 20.5, h4_data)

    else:
        # apply the two moments quality cut criteria.
        vel_data = np.ma.masked_where(SNR_data <= 5, vel_data)  # S/N > 5.
        vel_data = np.ma.masked_where(sig_data <= 35, vel_data)  # sig > 35 km/s.
        vel_data = np.ma.masked_where(vel_err_data >= 30, vel_data)  # vel_err < 30 km/s.
        vel_data = np.ma.masked_where(sig_err_data >= (sig_data * 0.1 + 25), vel_data)  # sig_err < sig * 0.1 + 25 km/s.

        sig_data = np.ma.masked_where(SNR_data <= 5, sig_data)
        sig_data = np.ma.masked_where(sig_data <= 35, sig_data)
        sig_data = np.ma.masked_where(sig_err_data >= (sig_data * 0.1 + 25), sig_data)

        h3_data = np.ma.masked_where(SNR_data <= 5, h3_data)
        h4_data = np.ma.masked_where(SNR_data <= 5, h4_data)

    # get the mask from all kinematic maps.
    combined_mask = np.ma.getmask(vel_data)
    combined_mask = np.ma.mask_or(combined_mask, np.ma.getmask(vel_err_data))
    combined_mask = np.ma.mask_or(combined_mask, np.ma.getmask(SNR_data))

    combined_mask = np.ma.mask_or(combined_mask, np.ma.getmask(sig_data))
    combined_mask = np.ma.mask_or(combined_mask, np.ma.getmask(sig_err_data))

    combined_mask = np.ma.mask_or(combined_mask, np.ma.getmask(h3_data))
    combined_mask = np.ma.mask_or(combined_mask, np.ma.getmask(h3_err_data))

    combined_mask = np.ma.mask_or(combined_mask, np.ma.getmask(h4_data))
    combined_mask = np.ma.mask_or(combined_mask, np.ma.getmask(h4_err_data))

    # apply the combined_mask to all kinematic maps.
    cleaned_vel_data = np.ma.masked_array(vel_data, mask = combined_mask)
    cleaned_vel_err_data = np.ma.masked_array(vel_err_data, mask = combined_mask)

    cleaned_sig_data = np.ma.masked_array(sig_data, mask = combined_mask)
    cleaned_sig_err_data = np.ma.masked_array(sig_err_data, mask = combined_mask)

    cleaned_h3_data = np.ma.masked_array(h3_data, mask = combined_mask)
    cleaned_h3_err_data = np.ma.masked_array(h3_err_data, mask = combined_mask)

    cleaned_h4_data = np.ma.masked_array(h4_data, mask = combined_mask)
    cleaned_h4_err_data = np.ma.masked_array(h4_err_data, mask = combined_mask)

    if plot:
        def plot(data, title, label = None):
            plt.figure(figsize = (10, 8))

            plt.imshow(data, origin = 'lower', aspect = 'auto',
                       cmap = 'jet', vmin = vmin, vmax = vmax)

            plt.colorbar(label = label)
            plt.title(title)
            plt.xlabel('SPAXEL')
            plt.ylabel('SPAXEL')

            plt.show()

        # velocity map.
        plot(cleaned_vel_data, title = 'cleaned stellar velocity map', label = 'km/s')

        # velocity dispersion map.
        plot(cleaned_sig_data, title = 'cleaned stellar velocity dispersion map', label = 'km/s')

        # h3 map.
        plot(cleaned_h3_data, title = 'cleaned h3 map')

        # h4 map.
        plot(cleaned_h4_data, title = 'cleaned h4 map')

    if dynamite:
        # prepare the flux.
        vel_flux = np.ma.masked_invalid(vel_flux)
        sig_flux = np.ma.masked_invalid(sig_flux)
        h3_flux = np.ma.masked_invalid(h3_flux)
        h4_flux = np.ma.masked_invalid(h4_flux)

        # in principle, the flux maps should be the same across all kinematic maps.
        flux_comparison_all = (np.all(vel_flux == sig_flux)
                               and np.all(sig_flux == h3_flux)
                               and np.all(h3_flux == h4_flux))

        if flux_comparison_all:
            print("flux maps across all input fits files are the same, continue...")
            flux = vel_flux.copy()

            flux = np.ma.masked_where(flux < 0, flux)
            flux = np.ma.masked_array(flux, mask = combined_mask)

        else:
            raise ValueError("flux maps are not the same across all input fits files, check the input data!")

        # get dimensions.
        ny, nx = flux.shape

        # generate x, y coordinates with center at (0, 0).
        x_coords = (np.arange(nx) - center_x) * 0.5
        y_coords = (np.arange(ny) - center_y) * 0.5
        # the order is correct here, do not use y, x = np.meshgrid(y_coords, x_coords)
        x, y = np.meshgrid(x_coords, y_coords)

        # flatten all data and apply mask.
        # the mask.
        flat_mask = ~flux.mask.flatten()

        # each valid spaxel gets a unique bin ID.
        n_valid = flat_mask.sum()
        BIN_ID = np.arange(1, n_valid + 1).astype(int)

        x_flat = x.flatten()[flat_mask]
        y_flat = y.flatten()[flat_mask]

        flux_flat = flux.flatten()[flat_mask]

        v_flat = cleaned_vel_data.flatten()[flat_mask]
        dv_flat = cleaned_vel_err_data.flatten()[flat_mask]

        sig_flat = cleaned_sig_data.flatten()[flat_mask]
        dsig_flat = cleaned_sig_err_data.flatten()[flat_mask]

        h3_flat = cleaned_h3_data.flatten()[flat_mask]
        dh3_flat = cleaned_h3_err_data.flatten()[flat_mask]

        h4_flat = cleaned_h4_data.flatten()[flat_mask]
        dh4_flat = cleaned_h4_err_data.flatten()[flat_mask]

        if BIN_ID.size == x_flat.size == y_flat.size == flux_flat.size \
            == v_flat.size == dv_flat.size == sig_flat.size == dsig_flat.size \
            == h3_flat.size == dh3_flat.size == h4_flat.size == dh4_flat.size:

            total_index = BIN_ID.size
            print('total valid index number:', total_index)

        else:
            raise ValueError('the size of the flatten valid data do no match!')

        # create binary table for extension [1].
        col1 = fits.Column(name = 'BIN_ID', array = BIN_ID, format = 'J')
        col2 = fits.Column(name = 'X', array = x_flat, format = 'E')
        col3 = fits.Column(name = 'Y', array = y_flat, format = 'E')
        col4 = fits.Column(name = 'FLUX', array = flux_flat, format = 'E')
        col5 = fits.Column(name = 'V', array = v_flat, format = 'E')
        col6 = fits.Column(name = 'DV', array = dv_flat, format = 'E')
        col7 = fits.Column(name = 'SIG', array = sig_flat, format = 'E')
        col8 = fits.Column(name = 'DSIG', array = dsig_flat, format = 'E')
        col9 = fits.Column(name = 'H3', array = h3_flat, format = 'E')
        col10 = fits.Column(name = 'DH3', array = dh3_flat, format = 'E')
        col11 = fits.Column(name = 'H4', array = h4_flat, format = 'E')
        col12 = fits.Column(name = 'DH4', array = dh4_flat, format = 'E')

        cols = fits.ColDefs([col1, col2, col3,
                             col4, col5, col6,
                             col7, col8, col9,
                             col10, col11, col12])

        hdul = fits.BinTableHDU.from_columns(cols)
        hdul.name = 'STEKIN_UNBIN'

        hdu0 = fits.PrimaryHDU()

        hdul = fits.HDUList([hdu0, hdul])
        hdul.writeto(output_filename, overwrite = True)

    return combined_mask, cleaned_vel_data

#---------------------------------------------------------------------------------










