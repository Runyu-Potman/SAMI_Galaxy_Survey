from poster import *

######################################################################################
# 227266 gas kinematic output.
gas_output_file_227266_kinematics = '227266/kinematic/227266_quality_cut_gas_velocity_map.csv'

# 227266 stellar kinematic output.
star_output_file_227266_kinematics = '227266/kinematic/227266_quality_cut_stellar_velocity_map.csv'

# --------------------------------------------------------------------------
fig, axs = plt.subplots(2, 3, figsize=(10, 6))
fig.suptitle('Galaxy 227266 Which Hosts a Kinematically Distinct Core', fontsize = 15)

# colormap consistent with DYNAMITE.
vel_cmap = cmr.get_sub_cmap('twilight_shifted', 0.05, 0.95)
sig_cmap = cmr.get_sub_cmap('twilight_shifted', 0.05, 0.6)

# 227266 plotting.
# we should mask the particular spaxel!

plot_vel_or_sig(csv_path=gas_output_file_227266_kinematics, cmap=vel_cmap, cbar_label=r'$V_\mathrm{gas}$ (km/s)',
                value_type='vel', ax=axs[0, 1], PAs=[17], line_length=9, plot_psf=True, psffwhm=2.108)

# directly copied from stellar kinematics.
plot_vel_or_sig(csv_path=star_output_file_227266_kinematics, value_type='vel', ax=axs[0, 0], cmap=vel_cmap,
                cbar_label=r'$V_{\bigstar}$ (km/s)',
                plot_psf=True, fontsize=12, psffwhm=2.108, vmin=-55, vmax=55, PAs=[171.75, 34.53], line_length=[3, 10],
                pa_center_x=0.75, pa_center_y=0.75)

plot_vel_or_sig(csv_path = star_output_file_227266_kinematics, value_type = 'sig', ax = axs[1, 0], cmap = sig_cmap,
                cbar_label = r'$\sigma_{\bigstar}$ (km/s)', plot_psf = True, fontsize = 12, psffwhm = 2.108, vmin = 110, vmax = 230)

axs[0, 0].set_title('Stellar Velocity', fontsize=12)
axs[1, 0].set_title('Stellar Velocity Dispersion', fontsize=12)
axs[0, 1].set_title('Ionized Gas Velocity', fontsize=12)

axs[0, 1].text(-4, 9.4, r'$\mathrm{PA}_\mathrm{\,gas}$=$-163^\circ$', color='black', fontsize=10, ha='left')
axs[0, 0].text(-11, -4.4, r'$\mathrm{PA}_\bigstar$=$171.75^\circ$', color='black', fontsize=10, ha='left')
axs[0, 0].text(-6.4, 9.8, r'$\mathrm{PA}_\bigstar$=$34.53^\circ$', color='black', fontsize=10, ha='left')

##########################################################################################
Ha_fits_path = '227266/emission_line/227266_A_Halpha_default_recom-comp.fits'
Hb_fits_path = '227266/emission_line/227266_A_Hbeta_default_recom-comp.fits'
OIII_fits_path = '227266/emission_line/227266_A_OIII5007_default_recom-comp.fits'
NII_fits_path = '227266/emission_line/227266_A_NII6583_default_recom-comp.fits'
classification, x_coords, y_coords = plot_nii_spatial(
    ax = axs[1, 1],
    Ha_fits_path=Ha_fits_path,
    Hb_fits_path=Hb_fits_path,
    OIII_fits_path=OIII_fits_path,
    NII_fits_path=NII_fits_path,
    threshold=5.0,
    scale=0.5, bpt_AGN = True
)

#########################################################
base_dir = '227266/age_z'
age_227266, metal_227266, age_227266_array, metal_227266_array, age_227266_std, metal_227266_std, r_all_227266 = slurm_job_combine(
    base_dir)

plot_age_and_Z(axs, axs_x = 0, age_full=age_227266, metal_full=metal_227266, r_all=r_all_227266,
               age_array=age_227266_array,
               metal_array=metal_227266_array, age_std_array=age_227266_std, metal_std_array=metal_227266_std,
               r_dash=3.0, title = True)

############################################################
fits_filename_227266 = '227266/dynamite/dynamite_fits/enclosed_mass_profiles.fits'
reproduce_mass_plot(fits_filename_227266, ax=axs[1, 2], name='227266', r_kdc=3.0, extrap_start=7.5, label = True)
axs[1, 2].set_title('Mass Profile', fontsize=10)

fits_orbit_227266 = '227266/dynamite/dynamite_fits/orbit_density.fits'
reproduce_orbit_plot(fits_orbit_227266, ax = axs[1, 3], name = '227266', r_kdc = 3.0, text = True)
axs[1, 3].set_title('Orbit Distribution', fontsize=10)

#############################################################
add_psf(ax = axs[0, 3], psffwhm = 2.108)
add_psf(ax = axs[1, 0], psffwhm = 2.108)
add_psf(ax = axs[1, 1], psffwhm = 2.108)
# ---------------------------------------------------------------------------
# same axis ratio.
axs[0, 0].set_box_aspect(1)
axs[0, 1].set_box_aspect(1)
axs[0, 2].set_box_aspect(1)
axs[0, 3].set_box_aspect(1)

axs[1, 0].set_box_aspect(1)
axs[1, 1].set_box_aspect(1)
axs[1, 2].set_box_aspect(1)
axs[1, 3].set_box_aspect(1)
#-------------------------------------------------------------------------
plt.tight_layout(h_pad=0, w_pad=0)
plt.savefig('final/poster_227266.png', dpi=300, bbox_inches='tight')
plt.show()