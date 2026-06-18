
            vmax_i = np.nanmax(np.abs(data_vel))
            if not np.isfinite(vmax_i) or vmax_i == 0:
                vmax_i = 1.0
            vlims.append((-vmax_i, vmax_i))

    fig, axs = plt.subplots(2, n_gal, figsize=figsize, constrained_layout=True)

    if n_gal == 1:
        axs = np.array(axs).reshape(2, 1)

    def style_map_axis(ax):
        ax.set_xlim(-9, 9)
        ax.set_ylim(-9, 9)
        ax.set_xticks([-5, 0, 5])
        ax.set_yticks([-5, 0, 5])
        ax.tick_params(direction='in', length=5, labelbottom=False, labelleft=False)

    def style_colorbar(cbar, vmax_val):
        cbar.set_ticks([])
        cbar.ax.tick_params(length=0, labelleft=False, labelright=False)
        cbar.outline.set_linewidth(1.0)

        vmax_int = int(np.rint(vmax_val))
        vmin_int = -vmax_int

        cbar.ax.text(
            0.5, 0.97, f'{vmax_int:d}',
            transform=cbar.ax.transAxes,
            ha='center', va='top',
            fontsize=10
        )
        cbar.ax.text(
            0.5, 0.03, f'{vmin_int:d}',
            transform=cbar.ax.transAxes,
            ha='center', va='bottom',
            fontsize=10
        )

    for i in range(n_gal):
        extent = extents[i]
        vmin, vmax = vlims[i]

        im_data = axs[0, i].imshow(
            data_maps[i],
            origin='lower',
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect='equal',
            extent=extent
        )
        axs[0, i].set_title(labels[i], fontsize=12)
        style_map_axis(axs[0, i])

        im_model = axs[1, i].imshow(
            model_maps[i],
            origin='lower',
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect='equal',
            extent=extent
        )
        style_map_axis(axs[1, i])

        if i == 0:
            axs[0, i].set_ylabel('data vel', fontsize=12)
            axs[1, i].set_ylabel('model vel', fontsize=12)

        cbar1 = fig.colorbar(im_data, ax=axs[0, i], fraction=0.046, pad=0.01)
        style_colorbar(cbar1, vmax)

        cbar2 = fig.colorbar(im_model, ax=axs[1, i], fraction=0.046, pad=0.01)
        style_colorbar(cbar2, vmax)

    return fig

#------------------------------------------------------------------------
fits_paths = [
        '7969/dynamite/dynamite_fits/kinematics.fits', '143287/dynamite/dynamite_fits/kinematics.fits',
        '227266/dynamite/dynamite_fits/kinematics.fits', '230776/dynamite/dynamite_fits/kinematics.fits',
        '300787/dynamite/dynamite_fits/kinematics.fits', '9239900248/dynamite/dynamite_fits/kinematics.fits'
    ]

labels = ["Galaxy 7969", "Galaxy 143287", "Galaxy 227266", "Galaxy 230776", "Galaxy 300787", "Galaxy 9239900248"]

fig = plot_6x2_velocity_grid(fits_paths, labels=labels)
plt.show()











