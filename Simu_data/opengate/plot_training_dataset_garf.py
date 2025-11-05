#!/usr/bin/env python3

import argparse
import uproot
import matplotlib.pyplot as plt
import numpy as np


def main():
    print(args)
    ax = None
    fig_angles, ax_angles = plt.subplots(1,len(args.rootfile))
    for _,rf in enumerate(args.rootfile):
        print(f"---- {rf} ------")
        h = uproot.open(rf)
        a = h['ARF (training);1']
        dataset_array = a.arrays(library='numpy')
        print(f"{dataset_array['E'].shape=}")
        E = dataset_array['E']
        theta = dataset_array['Theta']
        phi = dataset_array['Phi']
        true_window = dataset_array["window"]
        windows = np.unique(true_window)
        print(f"{windows=}")
        nw = len(windows)
        if ax is None:
            fig,ax =plt.subplots(1,nw)

        ax_angles[_].hist2d(theta[true_window==1], phi[true_window==1],bins=100)
        ax_angles[_].set_title(rf)

        for k in range(nw):
            ax[k].hist(E[true_window==k]*1000,bins=100, label="rf", alpha = 0.7, density= True)
            ax[k].set_xlabel("Energy (keV)")
            ax[k].set_title(f"EW {k}")

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--rootfile", nargs="+")
    parser.add_argument("--rr")
    args = parser.parse_args()

    main()
