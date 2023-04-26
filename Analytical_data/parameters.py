import numpy as np
import argparse

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

    print(f'sigma0 = {sigma0_psf}')
    print(f'alpha = {alpha_psf}')
    return sigma0_psf, alpha_psf


def get_detector_params(machine):
    assert (machine in ['ge-discovery', 'siemens-intevo'])

    if machine=='ge-discovery':
        size = 128
        spacing = 4.41806
    elif machine=='siemens-intevo':
        size = 256
        spacing = 2.3976
    return size,spacing

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




if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--spect_system', default="ge-discovery", choices=['ge-discovery', 'siemens-intevo'],
                        help='SPECT system simulated for PVE projections')

    args = parser.parse_args()
    _,__ = get_psf_params(machine = args.spect_system)
