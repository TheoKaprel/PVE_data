#!/usr/bin/env python3

import argparse
from test203_helpers import *
import opengate
from opengate.sources.generic import set_source_rad_energy_spectrum
from opengate.actors.digitizers import energy_windows_peak_scatter

import os
import sys
sys.path.append('/export/home/tkaprelian/Desktop/PVE/PVE_data/Simu_data/spect_siemens_intevo/')
import spect_siemens_intevo as gate_intevo


def main():
    # create the simulation
    sim = gate.Simulation()
    simu_name = "analog_simu_pt_src"
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

    # source/ct parameters
    source = sim.add_source('GenericSource', 'pt_src')
    source.particle = "gamma"
    set_source_rad_energy_spectrum(source = source, rad = args.radionuclide)
    source.activity = args.activity * Bq / ui.number_of_threads
    source.position.type = 'box'
    source.position.dimension = [2 * mm, 2 * mm, 2 * mm]
    source.direction.type = 'iso'

    # spect info
    if (args.radionuclide).lower()=="tc99m":
        colli_type = "lehr"
        channels = [
            {"name": f"peak140", "min": 129.5 * keV, "max": 150.5 * keV},
            {"name": f"scatter", "min": 108.5 * keV, "max": 129.5 * keV},
        ]
    elif (args.radionuclide).lower()=="lu177":
        colli_type = "melp"
        p1 = 112.9498 * keV
        p2 = 208.3662 * keV
        channels = [
            *energy_windows_peak_scatter("peak113", "scatter1", "scatter2", p1, 0.2, 0.1),
            *energy_windows_peak_scatter("peak208", "scatter3", "scatter4", p2, 0.2, 0.1),
        ]

    radius = args.sid * mm

    # channels
    head,colli,crystal = gate_intevo.add_intevo_spect_head(sim, "spect", colli_type, debug=ui.visu)
    digit = gate_intevo.add_digitizer_v2(sim, crystal.name, "digit")
    gate_intevo.set_head_orientation(head, colli_type, radius)


    # output projection (not needed)
    ew = sim.get_actor_user_info("digit_energy_window")
    ew.channels = channels
    proj = digit.find_first_module("projection")
    channel_names = [c["name"] for c in ew.channels]
    proj.input_digi_collections = channel_names
    proj.output = os.path.join(output_folder, args.output_projs)

    sim.add_actor("SimulationStatisticsActor", "stats")

    print(sim.user_info)
    # run
    sim.run()
    output = sim.output

    # print results at the end
    stats = output.get_actor("stats")
    stats.write(f"{output_folder}/{simu_name}_stats.txt")
    print(stats)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--activity", type=float)
    parser.add_argument("--sid", type=float)
    parser.add_argument("--radionuclide", type=str)
    parser.add_argument("--output_folder")
    parser.add_argument("--output_projs")
    args = parser.parse_args()

    main()
