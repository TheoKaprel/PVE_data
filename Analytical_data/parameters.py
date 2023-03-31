import numpy as np




def get_psf_params(machine):
    assert (machine in ['ge-discovery', 'siemens-intevo'])

    # Units in mm

    if machine=="ge-discovery":
        # holes diameter
        d = 1.5
        # holes length
        l = 35
    elif machine=="siemens-intevo":
        d = 1.11
        l = 24.05


    # mass attenuation coefficient at 150 keV in Pb
    mu = 1.91 * 11.35 * 10

    # Effective length
    leff = l - 2 / mu

    sigma0_psf = d / (2*np.sqrt(2*np.log(2)))
    alpha_psf= d / (2*np.sqrt(2*np.log(2))) / leff

    print(f'FWHM(d)={d} + {d/leff} d')
    print(f'FWHM(10cm) = {d*(1 + 100 / leff)}')

    print(sigma0_psf)
    print(alpha_psf)
    return sigma0_psf, alpha_psf




def get_FWHM_b(machine, b):
    assert (machine in ['ge-discovery', 'siemens-intevo'])

    # Units in mm

    if machine=="ge-discovery":
        # holes diameter
        d = 1.5
        # holes length
        l = 35
    elif machine=="siemens-intevo":
        d = 1.11
        l = 24.05

    # mass attenuation coefficient at 150 keV in Pb
    mu = 1.91 * 11.35 * 10

    # Effective length
    leff = l - 2 / mu

    FWHM_b = d + b * d / leff
    return FWHM_b