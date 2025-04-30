import os
import numpy as np
from astropy.io import fits
import re
import matplotlib.pyplot as plt

def extract_miles_ssp_no_alpha(fits_dir, fwhm_value, output_filename, quiet = False):
    '''
    note that fake mass is used. also baseFe

    Paremeters:
    - fits_dir:
    - fwhm_value:
    - output_filename:
    - quiet:

    Returns:

    '''

    # list to store the extracted data.
    ages = set()
    metals = set()
    spectra_data = []

    # variable to store the reference wavelength array.
    reference_wavelength = None

    # loop over all fits files in the directory.
    for fits_file in os.listdir(fits_dir):
        if fits_file.endswith('.fits'):

            if not quiet:
                print(f"Processing file: {fits_file}")

            # Parse filename to extract age (Gyr) and metallicity ([M/H]).
            match_age = re.search(r'T(\d+\.\d+)', fits_file)
            match_metal = re.search(r'Z([mp])(\d+\.\d+)', fits_file)

            if not quiet:
                if match_age:
                    print(f"Found age: {match_age.group(1)}")
                else:
                    raise ValueError(f"Age not found in {fits_file}.")

                if match_metal:
                    print(f"Found metallicity: {match_metal.group(0)}")
                else:
                    raise ValueError(f"Metallicity not found in {fits_file}.")

            if match_age and match_metal:
                age = float(match_age.group(1))

                # fix for metallicity: convert 'p' to '+' and 'm' to '-'.
                sign = '-' if match_metal.group(1) == 'm' else '+'
                metallicity = float(sign + match_metal.group(2))

                # add to corresponding sets.
                ages.add(age)
                metals.add(metallicity)

                # open the fits file and extract the spectrum and header.
                with fits.open(os.path.join(fits_dir, fits_file)) as hdul:
                    spectrum_data = hdul[0].data
                    header = hdul[0].header

                    crval1 = header['CRVAL1']
                    cdelt1 = header['CDELT1']
                    crpix1 = header['CRPIX1']
                    naxis1 = header['NAXIS1']

                    wavelength = crval1 + (np.arange(naxis1) + 1 - crpix1) * cdelt1

                    # check wavelength consistency.
                    if reference_wavelength is None:
                        reference_wavelength = wavelength
                    else:
                        if not np.allclose(reference_wavelength, wavelength, rtol = 1e-8, atol = 1e-10):
                            raise ValueError(f"Wavelength grid mismatch in file: {fits_file}")

                    # store the spectrum data for later reshaping.
                    spectra_data.append((spectrum_data, age, metallicity))

    # covert the sets to sorted lists.
    ages = sorted(ages)
    metals = sorted(metals)

    if not quiet:
        print(f"Unique ages: {ages}")
        print(f"Unique metallicities: {metals}")

    # number of unique values for each parameter.
    n_ages = len(ages)
    n_metals = len(metals)

    if not quiet:
        print('Number of ages:', n_ages)
        print('Number of metallicities:', n_metals)
        print(
            'Wavelength range: {:.1f} - {:.1f} Å ({} pixels)'.format(reference_wavelength[0], reference_wavelength[-1],
                                                                     len(reference_wavelength)))

    # create an empty array for templates.
    templates = np.zeros((len(reference_wavelength), n_ages, n_metals))

    # fill the templates array with data from the spectra_data list.
    for spectrum_data, age, metallicity in spectra_data:
        age_idx = ages.index(age)
        metal_idx = metals.index(metallicity)

        # find the corresponding position in the templates array and insert the spectrum.
        templates[:, age_idx, metal_idx] = spectrum_data

    # create an array for fwhm (constant across all pixels and all templates).
    fwhm = np.full(len(reference_wavelength), fwhm_value)

    # fake mass.
    #################################################################
    # important !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # we use fake mass because we only want to estimate light-weighted age and [M/Z] with pPXF, no M/L.
    masses = np.full((n_ages, n_metals), 100.0)
    ################################################################

    # save everything into an .npz file in preparation for pPXF.
    np.savez_compressed(output_filename, templates=templates, lam=reference_wavelength, fwhm=fwhm, ages=ages,
                        metals=metals, masses=masses)

    if not quiet:
        print('Beginning checking the final saved .npz file...')

        data = np.load(output_filename)
        print('Arrays in the file:', data.files)

        # access specific arrays by their names.
        templates = data['templates']
        masses = data['masses']
        lam = data['lam']
        ages = data['ages']
        metals = data['metals']
        fwhm = data['fwhm']

        print('Age list:', ages)
        print('Metallicity list:', metals)
        print("Templates shape:", templates.shape)
        print("Ages shape:", ages.shape)
        print("Metals shape:", metals.shape)
        print("masses shape:", masses.shape)
        print("Wavelengths shape:", lam.shape)
        print("FWHM shape:", fwhm.shape)
        print('start wavelength:', lam[0])
        print('end wavelength:', lam[-1])

        # Select the first template (index 0) for a given metallicity (e.g., metallicity index 0)
        template_flux = templates[:, 0, 0]  # 1st age, 1st metallicity

        # Plot this template
        plt.figure(figsize=(10, 6))
        plt.plot(lam, template_flux, label="Template 0 (Age = {}, Metallicity = {})".format(ages[0], metals[0]))
        plt.xlabel("Wavelength (Å)")
        plt.ylabel("Flux")
        plt.title("First Template Spectrum")
        plt.legend()
        plt.show()

        # Close the file
        data.close()


fits_dir = 'miles_ssp_basti_ch_baseFe'
fwhm_value = 2.51
output_filename = 'miles_ssp_model_ch_basti.npz'
extract_miles_ssp_no_alpha(fits_dir, fwhm_value, output_filename, quiet = False)






















