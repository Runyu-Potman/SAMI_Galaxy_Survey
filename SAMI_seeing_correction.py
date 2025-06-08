import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pyx.style import linestyle, linewidth

mpl.rcParams['text.usetex'] = True

def correct_lambda_R(lambda_R_obs, Re_arcsec, ellipticity, sersic_n, fwhm=2.1):
    # Step 1: Convert FWHM to sigma_PSF
    sigma_psf = fwhm / 2.355

    # transform circularized effective radius to effective radius.
    Re_arcsec = Re_arcsec / np.sqrt(1 - ellipticity)

    # Step 2: Compute x
    x = sigma_psf / Re_arcsec

    # Step 3: Compute f(x)
    fx = 7.44 / (1 + np.exp(4.87 * x ** 1.68 + 3.03)) - 0.34

    # Step 4: Compute f(ellip, n)
    f_en = 0.011 * np.log10(ellipticity) - 0.278 * np.log10(sersic_n) + 0.098

    # Step 5: Total correction
    delta_log_lambda = fx + x * f_en

    # Step 6: Corrected lambda_R
    log_lambda_corr = np.log10(lambda_R_obs) - delta_log_lambda
    lambda_corr = 10 ** log_lambda_corr

    return lambda_corr, ellipticity

lambda_7969, ellipticity_7969 = correct_lambda_R(0.045, 2.859, 0.098, 5.073, fwhm = 1.561)

lambda_227266, ellipticity_227266 = correct_lambda_R(0.052, 7.136, 0.113, 4.095, fwhm = 2.108)

lambda_230776, ellipticity_230776 = correct_lambda_R(0.104, 23.81, 0.342, 3.661, fwhm = 2.144)

# make the lambda_re verse ellipticity plot.
# Axes limits
x_lim = (0.0, 0.5)
y_lim = (0.0, 0.5)
x_range = np.linspace(x_lim[0], x_lim[1], 1000)

# different difinitions.
emsellem_2007 = np.full_like(x_range, 0.1)

emsellem_2011 = 0.31 * np.sqrt(x_range)

# Cappellari (2016).
x_cap_2016 = x_range[x_range < 0.4]
cap_2016 = 0.08 + x_cap_2016 / 4
cap_2016_limit = 0.08 + 0.4 / 4  # λ value at limit

# Sande (2021).
x_sande = x_range[x_range < 0.428]
sande = 0.12 + x_sande / 4
sande_limit = 0.12 + 0.428 / 4  # λ value at limit

# Plotting
plt.figure(figsize=(20/3, 6))
plt.plot(x_range, emsellem_2007, color = 'grey', linestyle = '--', linewidth = 2, label = 'Emsellem et al. (2007)')
plt.plot(x_range, emsellem_2011, color = 'blue', linestyle = ':', linewidth = 2, label = 'Emsellem et al. (2011)')
plt.plot(x_cap_2016, cap_2016, color = 'green', linestyle = '-.', linewidth = 2, label = 'Cappellari et al. (2016)')
plt.plot(x_sande, sande, color = 'red', linestyle = '-', linewidth = 2, label = 'Sande et al. (2021)')

# Vertical lines restricted to curve range
plt.vlines(0.4, ymin = 0, ymax = cap_2016_limit, colors = 'green', linestyles = '-.', linewidth = 2)
plt.vlines(0.428, ymin = 0, ymax = sande_limit, colors = 'red', linestyles = '-', linewidth = 2)

# Minor ticks and ticks direction
plt.minorticks_on()
plt.tick_params(axis = 'both', which = 'major', length = 4, width = 1, direction = 'in')
plt.tick_params(axis = 'both', which = 'minor', length = 2, width = 1, direction = 'in')

# three galaxies.
# Define galaxies with (ε_e, λ_Re, color, marker)
galaxies = {
    'Galaxy 7969':     (ellipticity_7969, lambda_7969, 'yellow', 'o'),
    'Galaxy 227266':   (ellipticity_227266, lambda_227266, 'salmon', 'o'),
    'Galaxy 230776':   (ellipticity_230776, lambda_230776, 'cyan', 'o'),
}
# Add galaxy points
for name, (eps, lam, color, marker) in galaxies.items():
    plt.scatter(eps, lam, color = color, marker = marker, s = 100,
                edgecolors = 'k', label = name, zorder = 5)











# Formatting
plt.xlabel(r'$\epsilon_{\mathrm{e}}$', fontsize = 15)
plt.ylabel(r'$\lambda_{R_{\mathrm{e}}}$', fontsize = 15)
plt.xlim(x_lim)
plt.ylim(y_lim)
plt.legend(fontsize = 10)
plt.tight_layout()
plt.show()

