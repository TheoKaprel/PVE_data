#!/usr/bin/env python3
import argparse
from test203_helpers import *
from box import Box
import sys
import os
sys.path.append(os.path.join(os.getcwd(),"../spect_siemens_intevo"))
import spect_siemens_intevo as gate_intevo
from opengate.actors.digitizers import energy_windows_peak_scatter


def main():
    # create the simulation
    sim = gate.Simulation()
    simu_name = "analog_simu_lu177_megp"
    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)

    # units
    mm = gate.g4_units.mm
    keV = gate.g4_units.keV
    Bq = gate.g4_units.Bq
    sec = gate.g4_units.second

    # options
    ui = sim.user_info
    # ui.visu = True
    ui.number_of_threads = 8
    if ui.visu:
        ui.number_of_threads = 1
    ui.visu_type = "vrml"

    # source/ct parameters
    p = Box()
    p.data_folder = args.data
    p.ct_image = args.ct
    p.activity_image = args.source
    p.radionuclide = "Lu177"
    p.activity = args.activity * Bq
    p.duration = 1 * sec
    patient = add_ct_image(sim, p)
    source = add_vox_source(sim, p, patient)

    # spect info
    colli_type = "melp"
    radius = args.sid * mm

    # channels
    head,colli,crystal = gate_intevo.add_intevo_spect_head(sim, "spect", colli_type, debug=ui.visu)
    digit = gate_intevo.add_digitizer_v2(sim, crystal.name, "digit")
    gate_intevo.set_head_orientation(head, colli_type, radius, gantry_angle = args.angle)

    # channels
    p1 = 112.9498 * keV
    p2 = 208.3662 * keV
    channels = [
        *energy_windows_peak_scatter("peak113", "scatter1", "scatter2", p1, 0.2, 0.1),
        *energy_windows_peak_scatter("peak208", "scatter3", "scatter4", p2, 0.2, 0.1),
    ]
    # output projection (not needed)
    ew = sim.get_actor_user_info("digit_energy_window")
    ew.channels = channels
    proj = digit.find_first_module("projection")
    channel_names = [c["name"] for c in ew.channels]
    proj.input_digi_collections = channel_names

    if args.output_projs is None:
        output_projs_fn = "projs.mhd"
    else:
        output_projs_fn = args.output_projs

    proj.output = os.path.join(output_folder, output_projs_fn)

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
    parser.add_argument("--angle", type=float)
    parser.add_argument("--source")
    parser.add_argument("--ct")
    parser.add_argument("--data")
    parser.add_argument("--output_folder")
    parser.add_argument("--output_projs")
    args = parser.parse_args()


    main()
