#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import opengate as gate
import sys
sys.path.append('/export/home/tkaprelian/Desktop/PVE/PVE_data/Simu_data/spect_siemens_intevo/')
# print(sys.path)
import spect_siemens_intevo as gate_intevo
import os

if __name__ == "__main__":
    # create the simulation
    sim = gate.Simulation()
    simu_name = "garf_training_dataset_tc99m_lehr_rr50"
    output_folder = "test203"
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
    colli_type = "lehr"
    activity = 5e9 * Bq
    radius = 500 * mm

    # channels
    channels = [
        {"name": f"scatter", "min": 108.5 * keV, "max": 129.5 * keV},
        {"name": f"peak140", "min": 129.5 * keV, "max": 150.5 * keV},
    ]
    # spect
    arf, ew = gate_intevo.create_simu_for_arf_training_dataset(
        sim, colli_type, 160 * keV, activity, rr=50, channels=channels, radius=radius
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