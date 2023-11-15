import numpy as np
import argparse

def get_psf_params(machine, verbose = True):
    assert (machine in ['ge-discovery', 'siemens-intevo'])

    # Units in mm

    if machine=="ge-discovery":
        # holes diameter
        d = 1.5
        # holes length
        l = 35
        # septal thickness
        t = 0.2

    elif machine=="siemens-intevo":
        d1 = 1.11 # diameters across the flats
        d2 = 2/np.sqrt(3) * d1  # long diameters
        d_mean = d1 * 3 *np.log(3)/np.pi # mean diameter

        d = d1

        l = 24.05
        t = 0.16


    # mass attenuation coefficient at 150 keV in Pb
    mu = 1.91 * 11.35 / 10 # (mm)^-1

    # Effective length
    leff = l - 2 / mu

    sigma0_psf = d / (2*np.sqrt(2*np.log(2)))
    alpha_psf= d / (2*np.sqrt(2*np.log(2))) / leff

    w = t * l / (2 * d + t)
    septal_penetration = np.exp(-w * mu)

    K = 0.26  # hexagonal holes
    efficiency = (K * (d / leff) * d / (d + t)) ** 2
    sensitivity = efficiency * 60 * 3.7 * 10 ** 4

    if verbose:
        print(f'FWHM(d)={d} + {d/leff} d')
        print(f'FWHM(10cm) = {d*((leff + 100) / leff)}')
        print(f'sigma0 = {sigma0_psf}')
        print(f'alpha = {alpha_psf}')
        print(f'septal penetration : {round(septal_penetration*100,3)} %')
        print(f'efficiency : {round(efficiency*100, 8)} %  ({round(sensitivity, 1)} cpm/microCi)')

    return sigma0_psf, alpha_psf,efficiency


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
    sigma0_psf, alpha_psf,_ = get_psf_params(machine=machine, verbose=False)
    FWHM_b = (2*np.sqrt(2*np.log(2))) * (sigma0_psf + b * alpha_psf)
    return FWHM_b





if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--spect_system', default="ge-discovery", choices=['ge-discovery', 'siemens-intevo'],
                        help='SPECT system simulated for PVE projections')

    args = parser.parse_args()
    _,__,___ = get_psf_params(machine = args.spect_system)
