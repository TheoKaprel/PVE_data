#!/usr/bin/env python3

import argparse
import opengate as gate
from gaga_phsp.spect_intevo_helpers import *
from gaga_phsp.gaga_helpers_tests import get_tests_folder
import opengate.tests.utility as utility
from pathlib import Path

def main():
    print(args)
    # units
    mm = gate.g4_units.mm
    Bq = gate.g4_units.Bq
    sec = gate.g4_units.second
    deg = gate.g4_units.deg


    # spect options
    simu = SpectIntevoSimulator('standalone_torch', "test004_main5_standalone_torch")
    simu.output_folder = Path(args.output_folder)
    simu.ct_image = args.ct  # (needed to position the source)
    simu.activity_image = args.source
    simu.radionuclide = args.radionuclide
    simu.gantry_angles = [0 * deg, 100 * deg, 230 * deg]

    simu.duration = 30 * sec
    simu.number_of_threads = 1
    simu.total_activity = args.activity * Bq
    # simu.visu = True

    simu.image_size = [96, 96]
    simu.image_spacing = [4.7951998710632 * mm , 4.7951998710632 * mm]

    simu.gaga_source.pth_filename = args.gan_pth
    simu.garf_detector.pth_filename = args.garf_pth
    simu.garf_detector.hit_slice_flag = False
    simu.gaga_source.batch_size = int(args.batchsize)  # 5e5 best on nvidia linux
    simu.gaga_source.backward_distance = 30 * mm # ????
    simu.gaga_source.energy_threshold_MeV = 0.15

    # run the simulation
    simu.generate_projections()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-a","--activity", type = float, default = 2e7)
    parser.add_argument("-s", "--source", type=str)
    parser.add_argument("--ct", type=str)
    parser.add_argument("--radionuclide", type=str, choices=['Tc99m', 'Lu177'])
    parser.add_argument("--bs", type=float)
    parser.add_argument("--gan_pth", type=str)
    parser.add_argument("--garf_pth", type=str)
    parser.add_argument("--output_folder", type=str)
    args = parser.parse_args()

    main()
