import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def slurm_job_combine(base_dir, center_x = 24, center_y = 24):
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
        start_x = job_id * 2
        end_x = start_x + age_part.shape[0]

        # insert the partial maps into full maps.
        age_map[start_x:end_x, :] = age_part
        metal_map[start_x:end_x, :] = metal_part

        age_std_map[start_x:end_x, :] = age_std_part
        metal_std_map[start_x:end_x, :] = metal_std_part

    np.save(f'{base_dir}/age_map_full.npy', age_map)
    np.save(f'{base_dir}/metal_map_full.npy', metal_map)
    np.save(f'{base_dir}/age_std_map_full.npy', age_std_map)
    np.save(f'{base_dir}/metal_std_map_full.npy', metal_std_map)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(age_map - 9, origin = 'lower', cmap = 'jet')
    plt.colorbar(label = 'log(Age(Gyr))')
    plt.title('spatially resolved log(Age(Gyr)) map')

    plt.subplot(1, 2, 2)
    plt.imshow(metal_map, origin  = 'lower', cmap = 'jet')
    plt.colorbar(label = '[M/H]')
    plt.title('spatially resolved metallicity map')

    plt.tight_layout()
    plt.savefig(f'{base_dir}/spatial_maps.png', dpi = 300)
    plt.show()

    plt.figure(figsize = (12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(age_std_map, origin = 'lower', cmap = 'jet')
    plt.colorbar(label = 'log(Age(Gyr)) std')
    plt.title('spatially resolved log(Age(Gyr)) standard deviation map')

    plt.subplot(1, 2, 2)
    plt.imshow(metal_std_map, origin = 'lower', cmap = 'jet')
    plt.colorbar(label = '[M/H] std')
    plt.title('spatially resolved metallicity standard deviation map')

    plt.tight_layout()
    plt.show()

    # begin making gradient plots.
    # create spatial grid and compute radius from center.
    x, y = np.meshgrid(np.arange(50), np.arange(50), indexing = 'ij')
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)

    # flatten all arrays.
    r = r.flatten()
    age_map = age_map.flatten()
    metal_map = metal_map.flatten()
    age_std_map = age_std_map.flatten()
    metal_std_map = metal_std_map.flatten()

    # mask invalid data.
    valid = ~np.isnan(age_map) & ~np.isnan(metal_map) & ~np.isnan(age_std_map) & ~np.isnan(metal_std_map)
    r = r[valid]
    age_map = age_map[valid]
    metal_map = metal_map[valid]
    age_std_map = age_std_map[valid]
    metal_std_map = metal_std_map[valid]

    # fit linear regression: age vs radius.
    slope_age, intercept_age, r_value_age, _, _ = linregress(r, age_map)
    fit_age = slope_age * r + intercept_age

    # fit linear regression: metal vs radius.
    slope_metal, intercept_metal, r_value_metal, _, _ = linregress(r, metal_map)
    fit_metal = slope_metal * r + intercept_metal

    # sort values for plotting the regression line smoothly.
    sort_idx = np.argsort(r)
    r_sorted = r[sort_idx]
    fit_age_sorted = fit_age[sort_idx]
    fit_metal_sorted = fit_metal[sort_idx]

    # Plotting
    plt.figure(figsize = (12, 5))

    # Age Gradient
    plt.subplot(1, 2, 1)
    sc1 = plt.scatter(r, age_map - 9, c = r, cmap = 'viridis', s = 10, alpha = 0.7, label = 'Pixels')
    plt.plot(r_sorted, fit_age_sorted - 9, 'r-', linewidth = 2, label = f'Fit: slope = {slope_age:.3f}')
    plt.xlabel('radius (pixels)')
    plt.ylabel('log(Age(Gyr))')
    plt.title('age gradient')
    plt.colorbar(sc1, ax = plt.gca(), label = 'Radius')
    plt.legend()

    # Metallicity Gradient
    plt.subplot(1, 2, 2)
    sc2 = plt.scatter(r, metal_map, c = r, cmap = 'viridis', s = 10, alpha = 0.7, label = 'Pixels')
    plt.plot(r_sorted, fit_metal_sorted, 'r-', linewidth = 2, label = f'Fit: slope = {slope_metal:.3f}')
    plt.xlabel('radius (pixels)')
    plt.ylabel('[M/H]')
    plt.title('metallicity gradient')
    plt.colorbar(sc2, ax = plt.gca() ,label = 'Radius')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{base_dir}/gradients.png', dpi = 300)
    plt.show()

    # plots showing the error bar.
    # Plotting
    plt.figure(figsize = (12, 5))

    # Age Gradient
    plt.subplot(1, 2, 1)
    plt.errorbar(r, age_map - 9, yerr = age_std_map, fmt = 'o', markersize = 3, alpha = 0.3, label= 'Pixels',
                 color = 'blue')
    plt.plot(r_sorted, fit_age_sorted - 9, 'r-', linewidth = 2, label = f'Fit: slope = {slope_age:.3f}')
    plt.xlabel('radius (pixels)')
    plt.ylabel('log(Age(Gyr))')
    plt.title('age gradient')
    plt.legend()

    # Metallicity Gradient
    plt.subplot(1, 2, 2)
    plt.errorbar(r, metal_map, yerr = metal_std_map, fmt = 'o', markersize = 5, alpha = 0.2, label = 'Pixels',
                 color = 'green')
    plt.plot(r_sorted, fit_metal_sorted, 'r-', linewidth = 2, label = f'Fit: slope = {slope_metal:.3f}')
    plt.xlabel('radius (pixels)')
    plt.ylabel('[M/H]')
    plt.title('metallicity gradient')
    plt.legend()

    plt.tight_layout()
    plt.show()
#-----------------------------------------------------------------------------------
base_dir = 'CATID_adap_binned_age_Z_v02'
slurm_job_combine(base_dir)