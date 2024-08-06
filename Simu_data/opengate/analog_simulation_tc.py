#!/usr/bin/env python3

import argparse
import opengate as gate
from test203_helpers import *
from box import Box
import sys
sys.path.append('/export/home/tkaprelian/Desktop/PVE/PVE_data/Simu_data/spect_siemens_intevo_loc/')
import spect_siemens_intevo as gate_intevo
import os
from opengate.sources.generic import get_rad_gamma_energy_spectrum


def main():
    # create the simulation
    sim = gate.Simulation()
    simu_name = "garf_training_dataset_tc99m_lehr_rr50"
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
    # p = Box()
    # p.data_folder = args.data
    # p.ct_image = args.ct
    # p.activity_image = args.source
    # p.radionuclide = "Tc99m"
    # p.activity = args.activity * Bq
    # p.duration = 1 * sec
    # patient = add_ct_image(sim, p)
    # source = add_vox_source(sim, p, patient)

    world = sim.world
    m = gate.g4_units.m
    world.size = [2 * m, 2 * m, 3 * m]
    world.material = "G4_AIR"
    mm = gate.g4_units.mm
    sim.physics_manager.set_production_cut("world", "all", 1 * mm)
    source = sim.add_source("GenericSource", "pointsource")
    w, e = get_rad_gamma_energy_spectrum("Tc99m")
    source.mother = "world"
    source.particle = "gamma"
    source.energy.type = "spectrum_lines"
    source.energy.spectrum_weight = w
    source.energy.spectrum_energy = e
    source.direction.type = "iso"
    source.position.type = "sphere"
    Bq = gate.g4_units.Bq
    ui = sim.user_info
    source.activity = args.activity * Bq / ui.number_of_threads


    # spect info
    colli_type = "lehr"
    radius = args.sid * mm

    # channels


    head,colli,crystal = gate_intevo.add_intevo_spect_head(sim, "spect", colli_type, debug=ui.visu)
    digit = gate_intevo.add_digitizer_v2(sim, crystal.name, "digit")
    # digit = gate_intevo.add_digitizer_validated_Tc99m(sim, head, crystal,args.output_folder)
    gate_intevo.set_head_orientation(head, colli_type, radius)

    channels = [
        {"name": f"peak140", "min": 129.5 * keV, "max": 150.5 * keV},
        {"name": f"scatter", "min": 108.5 * keV, "max": 129.5 * keV},
    ]
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
    parser.add_argument("--source")
    parser.add_argument("--ct")
    parser.add_argument("--data")
    parser.add_argument("--output_folder")
    parser.add_argument("--output_projs")
    args = parser.parse_args()


    main()
