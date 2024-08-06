#!/usr/bin/env python3

import argparse

import matplotlib.pyplot as plt
import numpy as np


def main():
    print(args)

    dict = {"LEHR": {'d': 1.11, 'l': 24.05, 'mu': 1.91,'t':0.16, 'c': 'blue', 'rint': 4}, # 150keV
            "MEGP": {'d': 2.94, 'l': 40.64, 'mu': 0.936,'t': 1.14, 'c': 'orangered', 'rint': 3.4}, # 200keV
            # "hehs": {'d': 4, 'l': 59.7, 'mu': 0.150} # 500keV
            }

    b = np.linspace(0,20,100)
    K = 0.26
    fig,ax = plt.subplots()
    for key,params in dict.items():
        l_eff = params['l'] - 2/(params['mu']*11.35/10)
        fwhm_b = (params['d']*(l_eff + b*10)/l_eff)
        eff = (K * (params['d'] / l_eff) * params['d'] / (params['d'] + params['t'])) ** 2
        print('-----')
        print(key)
        print(f'Efficiency: {eff*100}')
        print(f"FWHM(10cm): {params['d']*(l_eff + 100)/l_eff}")
        print('-----')

        ax.plot(b, fwhm_b, label=key, color=params['c'], linewidth=2)
    ax.set_xlabel('Source to collimator distance (cm)', fontsize=20)
    ax.set_ylabel('FWHM (mm)', fontsize=20)
    ax.set_ylim([0, 20])
    plt.title('Collimator Resolution', fontsize=20)
    plt.legend(fontsize=20)
    params_plt = {'mathtext.default': 'regular'}
    plt.rcParams.update(params_plt)
    fig,ax = plt.subplots()
    for key,params in dict.items():
        l_eff = params['l'] - 2/(params['mu']*11.35/10)
        fwhm_b = (params['d']*(l_eff + b*10)/l_eff)
        r_sys = np.sqrt(params['rint']**2 + fwhm_b**2)
        ax.plot(b, fwhm_b, label=key+" ($FWHM_{int}$=0)", color=params['c'], linewidth=2, linestyle="dashed")
        ax.plot(b, r_sys, label=key+" ($FWHM_{int}$="+f"{params['rint']}mm)", color=params['c'], linewidth=2)
    ax.set_xlabel('Source to collimator distance (cm)', fontsize=20)
    ax.set_ylabel('FWHM (mm)', fontsize=20)
    ax.set_ylim([0,20])
    plt.title('System Resolution', fontsize=20)
    plt.legend(fontsize=20)
    plt.rcParams["savefig.directory"] = "/export/home/tkaprelian/Desktop/MANUSCRIPT/CHAP3_PVE_SOTA_PVC"

    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    main()
