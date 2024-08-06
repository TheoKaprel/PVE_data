import numpy as np
import argparse

spect_systems_list = ['ge-discovery', 'siemens-intevo-lehr',  'siemens-intevo-megp',
                        'siemens-intevo-lehr-analytic','siemens-intevo-megp-analytic']

def get_psf_params(machine, verbose = True):
    assert (machine in spect_systems_list)

    # Units in mm

    c = (2 * np.sqrt(2 * np.log(2)))
    if machine=="ge-discovery":
        # holes diameter
        d = 1.5
        # holes length
        l = 35
        # septal thickness
        t = 0.2

        # mass attenuation coefficient at 150 keV in Pb
        mu = 1.91 * 11.35 / 10  # (mm)^-1

        # Effective length
        leff = l - 2 / mu

        sigma0_psf = d / (2 * np.sqrt(2 * np.log(2)))
        alpha_psf = d / (2 * np.sqrt(2 * np.log(2))) / leff

        w = t * l / (2 * d + t)
        septal_penetration = np.exp(-w * mu)

        K = 0.26  # hexagonal holes
        efficiency = (K * (d / leff) * d / (d + t)) ** 2
        sensitivity = efficiency * 60 * 3.7 * 10 ** 4

    elif machine=="siemens-intevo-lehr-analytic":
        d1 = 1.11 # diameters across the flats
        # d2 = 2/np.sqrt(3) * d1  # long diameters
        # d_mean = d1 * 3 *np.log(3)/np.pi # mean diameter
        d = d1
        l = 24.05
        t = 0.16

        # mass attenuation coefficient at 150 keV in Pb
        mu = 1.91 * 11.35 / 10 # (mm)^-1

        # Effective length
        leff = l - 2 / mu

        sigma0_psf = d / (2*np.sqrt(2*np.log(2)))
        alpha_psf= d / (2*np.sqrt(2*np.log(2))) / leff
        alpha_fwhm, sigma_fwhm = c * alpha_psf, c * sigma0_psf
        w = t * l / (2 * d + t)
        septal_penetration = np.exp(-w * mu)

        K = 0.26  # hexagonal holes
        efficiency = (K * (d / leff) * d / (d + t)) ** 2
        sensitivity = efficiency * 60 * 3.7 * 10 ** 4
    elif machine=="siemens-intevo-megp-analytic":
        d1 = 2.94 # diameters across the flats
        d = d1
        l = 40.64
        t = 1.14

        # linear attenuation coefficient at 200 keV in Pb (Lead)
        mu = 0.936 * 11.35 / 10 # (mm)^-1

        # Effective length
        leff = l - 2 / mu

        sigma_fwhm = d
        alpha_fwhm = d/leff

        sigma0_psf = d / (2*np.sqrt(2*np.log(2)))
        alpha_psf= d / (2*np.sqrt(2*np.log(2))) / leff

        w = t * l / (2 * d + t)
        septal_penetration = np.exp(-w * mu)

        K = 0.26  # hexagonal holes
        efficiency = (K * (d / leff) * d / (d + t)) ** 2
        sensitivity = efficiency * 60 * 3.7 * 10 ** 4


    elif machine=="siemens-intevo-lehr":
        # experimental fit

        # cf calc_exp_psf / rtk_psf
        sigma0_psf = 1.9111
        alpha_psf = 0.01767
        # cf abstract HOA MIC
        efficiency = 0.0096/100

        alpha_fwhm, sigma_fwhm = c * alpha_psf, c * sigma0_psf
    elif machine=="siemens-intevo-megp":
        alpha_psf = 0.03235363042582603
        sigma0_psf= 1.1684338873367237
        efficiency = 0.00012387387387387 # cf intevo siemens doc
        alpha_fwhm, sigma_fwhm = c * alpha_psf, c * sigma0_psf


    if verbose:
        print(f'FWHM(d)={sigma_fwhm} + {alpha_fwhm} d')
        print(f'FWHM(10cm) = {sigma_fwhm + 100* alpha_fwhm}')
        print(f'sigma0 = {sigma0_psf}')
        print(f'alpha = {alpha_psf}')
        print(f'efficiency : {round(efficiency*100, 8)} %')
        print(f"--sigmazero {sigma0_psf} --alphapsf {alpha_psf}")

    return sigma0_psf, alpha_psf, efficiency


def get_detector_params(machine):
    assert (machine in spect_systems_list)

    if machine=='ge-discovery':
        size = 128
        spacing = 4.41806
    elif machine=='siemens-intevo-lehr':
        size = 256
        spacing = 2.3976
    elif machine == 'siemens-intevo-megp':
        size = 128
        spacing = 4.7952

    return size,spacing

def get_FWHM_b(machine, b):
    sigma0_psf, alpha_psf,_ = get_psf_params(machine=machine, verbose=False)
    FWHM_b = (2*np.sqrt(2*np.log(2))) * (sigma0_psf + b * alpha_psf)
    return FWHM_b





if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--spect_system', default="ge-discovery", choices=spect_systems_list,
                        help='SPECT system simulated for PVE projections')

    args = parser.parse_args()
    _,__,___ = get_psf_params(machine = args.spect_system)
