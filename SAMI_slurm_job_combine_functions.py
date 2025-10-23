import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from matplotlib.ticker import AutoMinorLocator

def slurm_job_combine(base_dir, center_x = 25, center_y = 25):
    '''

    Parameters:
    - base_dir:
    - center_x:
    - center_y:

    Returns:
    - None
    '''

    # initialize the full maps for mean and standard deviation.
    age_map = np.full((50, 50), np.nan)
    metal_map = np.full_like(age_map, np.nan)

    age_std_map = np.full_like(age_map, np.nan)
    metal_std_map = np.full_like(age_map, np.nan)

    # loop over the tasks (job_id: 0 - 24).
    for job_id in range(25):
        # load the partial maps (mean and standard deviation).
        age_part = np.load(f'{base_dir}/output/age_map_part_{job_id}.npy')
        metal_part = np.load(f'{base_dir}/output/metal_map_part_{job_id}.npy')

        age_std_part = np.load(f'{base_dir}/output/age_std_part_{job_id}.npy')
        metal_std_part = np.load(f'{base_dir}/output/metal_std_part_{job_id}.npy')

        # calculate the slice indices for inserting the data.
        start_y = job_id * 2
        end_y = start_y + age_part.shape[0]

        # insert the partial maps into full maps.
        age_map[start_y:end_y, :] = age_part
        metal_map[start_y:end_y, :] = metal_part

        age_std_map[start_y:end_y, :] = age_std_part
        metal_std_map[start_y:end_y, :] = metal_std_part

    age_full = age_map.copy()
    metal_full = metal_map.copy()

    np.save(f'{base_dir}/age_map_full.npy', age_map)
    np.save(f'{base_dir}/metal_map_full.npy', metal_map)
    np.save(f'{base_dir}/age_std_map_full.npy', age_std_map)
    np.save(f'{base_dir}/metal_std_map_full.npy', metal_std_map)

    # begin making gradient plots.
    x_bar = np.load(f'{base_dir}/x_bar.npy')
    y_bar = np.load(f'{base_dir}/y_bar.npy')

    # shift the center back.
    x_bar = x_bar + center_x
    y_bar = y_bar + center_y

    # find the nearest pixel.
    x_bar = np.rint(x_bar).astype(int)
    y_bar = np.rint(y_bar).astype(int)

    r_all = np.sqrt((x_bar - center_x)** 2 + (y_bar - center_y) ** 2) * 0.5

    age_map = age_map[y_bar, x_bar]
    metal_map = metal_map[y_bar, x_bar]
    age_std_map = age_std_map[y_bar, x_bar]
    metal_std_map = metal_std_map[y_bar, x_bar]

    # mask invalid data.
    valid = ~np.isnan(age_map) & ~np.isnan(metal_map) & ~np.isnan(age_std_map) & ~np.isnan(metal_std_map)
    r_all = r_all[valid]
    age_map = age_map[valid]
    metal_map = metal_map[valid]

    age_array = age_map.copy()
    metal_array = metal_map.copy()

    return age_full, metal_full, age_array, metal_array, r_all
#-----------------------------------------------------------------------------------
base_dir = 'CATID_adap_binned_age_Z_v02'
slurm_job_combine(base_dir)