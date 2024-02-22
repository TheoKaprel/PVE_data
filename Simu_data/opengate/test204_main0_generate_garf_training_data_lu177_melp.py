#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import opengate as gate
from opengate.actors.digitizers import energy_windows_peak_scatter

import sys
sys.path.append('/export/home/tkaprelian/Desktop/PVE/PVE_data/Simu_data/spect_siemens_intevo/')
import spect_siemens_intevo as gate_intevo
import os
import argparse

def main():
    # create the simulation
    sim = gate.Simulation()
    simu_name = "garf_training_dataset_lu177_melp_rr50"
    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)

    # units
    mm = gate.g4_units.mm
    keV = gate.g4_units.keV
    Bq = gate.g4_units.Bq

    # options
    ui = sim.user_info
    # ui.visu = True
    ui.number_of_threads = 8
    if ui.visu:
        ui.number_of_threads = 1
    ui.visu_type = "vrml"

    # spect info
    colli_type = "melp"
    activity = 5e9 * Bq
    radius = 500 * mm

    # channels
    p1 = 112.9498 * keV
    p2 = 208.3662 * keV
    channels = [
        *energy_windows_peak_scatter("peak113", "scatter1", "scatter2", p1, 0.2, 0.1),
        *energy_windows_peak_scatter("peak208", "scatter3", "scatter4", p2, 0.2, 0.1),
    ]

    # spect
    arf, ew = gate_intevo.create_simu_for_arf_training_dataset(
        sim, colli_type, 300 * keV, activity, rr=50, channels=channels, radius=radius
    )
    # output
    arf.output = f"{output_folder}/{simu_name}.root"

    print(sim.user_info)
    # run
    # output = sim.run()
    sim.run()
    output = sim.output

    # print results at the end
    stats = output.get_actor("stats")
    stats.write(f"{output_folder}/{simu_name}_stats.txt")
    print(stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a","--activity", type = float, default = 5e9)
    # parser.add_argument("--data", type=str)
    parser.add_argument("--output_folder", type=str)
    args = parser.parse_args()
    main()