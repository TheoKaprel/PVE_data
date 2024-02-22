#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from test203_helpers import *
from path import Path
from box import Box
import sys
import time
import argparse

def main():
    print(args)


    mm = gate.g4_units.mm
    Bq = gate.g4_units.Bq
    sec = gate.g4_units.second

    # parameters
    p = Box()
    p.argv = sys.argv
    p.simu_mode = ""  # (visu) | speed | run

    p.radius = args.sid * mm
    p.detector_offset = 0 * mm
    p.colli_type = "lehr"
    p.radionuclide = "Tc99m"

    # p.data_folder = Path("training_data")
    p.ct_image = args.ct
    p.activity_image = args.source
    p.garf_pth_filename = args.pthgarf
    p.gaga_pth_filename = args.pthgaga


    p.size = [256, 256]
    p.spacing = [4.7951998710632 * mm / 2, 4.7951998710632 * mm / 2]

    p.number_of_threads = 8
    p.duration = 1 * sec
    p.activity = args.activity * Bq
    p.backward_distance = 530 * mm

    p.nprojs = args.nprojs

    p.batchsize = args.batchsize

    # fake init for output_folder
    init_test_simu(p, __file__, args.folder)

    # go
    run_standalone(p)

    # print
    print(f"Time = {p.duration:0.1f} seconds ")
    print(f"PPS = {p.pps} ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a","--activity", type = float)
    parser.add_argument("--ct", type = str)
    parser.add_argument("-s", "--source", type=str)
    parser.add_argument("--pthgaga", type=str)
    parser.add_argument("--pthgarf", type=str)
    parser.add_argument("-b","--batchsize", type = float, default = 100000)
    parser.add_argument("-f", "--folder", type = str)
    # parser.add_argument("-o", "--output", type = str)
    parser.add_argument("-n","--nprojs", type = int)
    parser.add_argument("--sid", type=float)
    args = parser.parse_args()

    main()
