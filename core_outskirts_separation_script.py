import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# separate core and outskirts regions.
vel_map = pd.read_csv('stellar_velocity_quality_cut_CATID.csv')

# strip any leading whitespace from column names.
vel_map.columns = vel_map.columns.str.strip()

x = vel_map['x'].values
y = vel_map['y'].values
vel = vel_map['vel'].values
dvel = vel_map['vel_err'].values

# define core radius (this could be adjusted if needed, the value 8 is determined by visual inspection).
core_radius = 8

# distance of each spaxel from center (0, 0).
distance_to_center = np.sqrt(x**2 + y**2)

# extract core and outskirts data.
core_data = distance_to_center <= core_radius
x_core = x[core_data]
y_core = y[core_data]
vel_core = vel[core_data]
dvel_core = dvel[core_data]

outskirts_data = distance_to_center > core_radius
x_outskirts = x[outskirts_data]
y_outskirts = y[outskirts_data]
vel_outskirts = vel[outskirts_data]
dvel_outskirts = dvel[outskirts_data]

# create dataframes for core and outskirts data.
core_dataframe = pd.DataFrame({
    'x': x_core,
    'y': y_core,
    'vel': vel_core,
    'vel_err': dvel_core
})

outskirts_dataframe = pd.DataFrame({
    'x': x_outskirts,
    'y': y_outskirts,
    'vel': vel_outskirts,
    'vel_err': dvel_outskirts
})

# save the data to CSV files in preparation for position angle calculation.
core_dataframe.to_csv('stellar_velocity_quality_cut_core_region_CATID.csv', index = False)
outskirts_dataframe.to_csv('stellar_velocity_quality_cut_outskirts_region_CATID.csv', index = False)

# read the csv file of core region.
vel_map_core = pd.read_csv('stellar_velocity_quality_cut_core_region_CATID.csv')

# strip any leading whitespace from column names.
vel_map_core.columns = vel_map_core.columns.str.strip()

# extract unique x and y values.
x_core = np.unique(vel_map_core['x'])
y_core = np.unique(vel_map_core['y'])

vel_core_grid = np.full((len(y_core), len(x_core)), np.nan)

for index, row in vel_map_core.iterrows():
    x_core_grid = np.where(x_core == row['x'])[0][0]
    y_core_grid = np.where(y_core == row['y'])[0][0]
    vel_core_grid[y_core_grid, x_core_grid] = row['vel']

# read the csv file of outskirts region.
vel_map_outskirts = pd.read_csv('stellar_velocity_quality_cut_outskirts_region_CATID.csv')

# strip any leading whitespace from column names.
vel_map_outskirts.columns = vel_map_outskirts.columns.str.strip()

# extract unique x and y values.
x_outskirts = np.unique(vel_map_outskirts['x'])
y_outskirts = np.unique(vel_map_outskirts['y'])

vel_outskirts_grid = np.full((len(y_outskirts), len(x_outskirts)), np.nan)

for index, row in vel_map_outskirts.iterrows():
    x_outskirts_grid = np.where(x_outskirts == row['x'])[0][0]
    y_outskirts_grid = np.where(y_outskirts == row['y'])[0][0]
    vel_outskirts_grid[y_outskirts_grid, x_outskirts_grid] = row['vel']

# create a 1x2 plot for core and outskirts regions.
fig, axs = plt.subplots(1, 2, figsize = (15, 6))

# plot for core region.
axs[0].imshow(vel_core_grid, origin = 'lower', aspect = 'auto', cmap = 'jet', interpolation = 'nearest', vmin = -20, vmax = 55)
axs[0].set_title('Core Region')
axs[0].set_xlabel('SPAXEL')
axs[0].set_ylabel('SPAXEL')
axs[0].set_xticks(np.arange(len(x_core)))
axs[0].set_xticklabels(x_core)
axs[0].set_yticks(np.arange(len(y_core)))
axs[0].set_yticklabels(y_core)
plt.colorbar(axs[0].imshow(vel_core_grid, origin = 'lower', aspect = 'auto', cmap = 'jet', interpolation = 'nearest'), ax = axs[0], label = 'km/s')

# plot for outskirts region.
axs[1].imshow(vel_outskirts_grid, origin = 'lower', aspect = 'auto', cmap = 'jet', interpolation = 'nearest', vmin = -20, vmax = 55)
axs[1].set_title('Outskirts Region')
axs[1].set_xlabel('SPAXEL')
axs[1].set_ylabel('SPAXEL')
axs[1].set_xticks(np.arange(len(x_outskirts)))
axs[1].set_xticklabels(x_outskirts)
axs[1].set_yticks(np.arange(len(y_outskirts)))
axs[1].set_yticklabels(y_outskirts)
plt.colorbar(axs[1].imshow(vel_outskirts_grid, origin = 'lower', aspect = 'auto', cmap = 'jet', interpolation = 'nearest'), ax = axs[1], label = 'km/s')

# show the plot.
plt.tight_layout()
plt.show()
