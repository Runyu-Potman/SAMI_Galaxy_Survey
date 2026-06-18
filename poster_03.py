

#------------------------------------------------------------------------
fits_paths = [
        '7969/dynamite/dynamite_fits/kinematics.fits', '143287/dynamite/dynamite_fits/kinematics.fits',
        '227266/dynamite/dynamite_fits/kinematics.fits', '230776/dynamite/dynamite_fits/kinematics.fits',
        '300787/dynamite/dynamite_fits/kinematics.fits', '9239900248/dynamite/dynamite_fits/kinematics.fits'
    ]

labels = ["Galaxy 7969", "Galaxy 143287", "Galaxy 227266", "Galaxy 230776", "Galaxy 300787", "Galaxy 9239900248"]

fig = plot_6x2_velocity_grid(fits_paths, labels=labels)
plt.show()











