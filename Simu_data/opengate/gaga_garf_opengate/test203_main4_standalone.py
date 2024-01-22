#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from test203_helpers import *
from path import Path
from box import Box
import sys

if __name__ == "__main__":
    mm = gate.g4_units.mm
    Bq = gate.g4_units.Bq
    sec = gate.g4_units.second

    # parameters
    p = Box()
    p.argv = sys.argv
    p.simu_mode = ""  # (visu) | speed | run

    p.radius = 380 * mm
    p.detector_offset = 0 * mm
    p.colli_type = "lehr"
    p.radionuclide = "Tc99m"

    p.data_folder = Path("training_data")
    p.ct_image = "ct_4mm.mhd"
    p.activity_image = "source_2spheres.mhd"
    p.garf_pth_filename = "pth/garf_tc99m_lehr_rr50.pth"
    p.gaga_pth_filename = "pth/gaga.pth"
    
    p.size = [256, 256]
    p.spacing = [4.7951998710632 * mm / 2, 4.7951998710632 * mm / 2]

    p.number_of_threads = 1
    p.duration = 20 * sec
    p.activity = 1e6 * Bq
    p.backward_distance = 280 * mm

    # fake init for output_folder
    init_test_simu(p, __file__, "outputs/david")

    # go
    run_standalone(p)

    # print
    print(f"Time = {p.duration:0.1f} seconds ")
    print(f"PPS = {p.pps} ")
