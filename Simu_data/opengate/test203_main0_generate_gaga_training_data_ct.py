#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from test203_helpers import *
import opengate as gate

import argparse

def main():
    # create the simulation
    sim = gate.Simulation()
    simu_name = "gaga_training_dataset_ct_large"
    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)

    # units
    mm = gate.g4_units.mm
    Bq = gate.g4_units.Bq
    sec = gate.g4_units.second

    # options
    ui = sim.user_info
    ui.number_of_threads = 8
    # ui.visu = True
    ui.visu_type = "vrml"

    # parameters
    p = Box()
    p.data_folder = args.data
    p.ct_image = args.ct
    p.activity_image = args.source
    p.radionuclide = args.radionuclide
    p.activity = args.activity * Bq
    p.duration = 1 * sec

    # add CT phantom
    patient = add_ct_image(sim, p)

    # cylinder for phsp
    sim.add_parallel_world("sphere_world")
    sph_surface = sim.add_volume("Sphere", "phase_space_sphere")
    sph_surface.rmin = 610 * mm
    sph_surface.rmax = 611 * mm
    sph_surface.color = [0, 1, 0, 1]
    sph_surface.material = "G4_AIR"
    sph_surface.mother = "sphere_world"

    # source uniform (limited FOV)
    source = add_vox_source(sim, p, patient)

    # stats
    stats = sim.add_actor("SimulationStatisticsActor", "stats")
    stats.output = f"{output_folder}/{simu_name}_stats.txt"

    # phsp
    phsp = sim.add_actor("PhaseSpaceActor", "phase_space")
    phsp.mother = "phase_space_sphere"
    phsp.attributes = [
        "KineticEnergy",
        "PrePosition",
        "PreDirection",
        "TimeFromBeginOfEvent",
        "EventID",
        "EventKineticEnergy",
        "EventPosition",
        "EventDirection",
    ]
    phsp.output = f"{output_folder}/{simu_name}.root"
    # this option allow to store all events even if absorbed
    phsp.store_absorbed_event = True
    f = sim.add_filter("ParticleFilter", "f")
    f.particle = "gamma"
    phsp.filters.append(f)
    print(phsp)
    print(phsp.output)

    # physic list
    sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option3"
    sim.physics_manager.set_production_cut("world", "all", 1 * mm)

    # run
    sim.run()

    # print results at the end
    stats = sim.output.get_actor("stats")
    print(stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a","--activity", type = float, default = 2e7)
    parser.add_argument("-s", "--source", type=str)
    parser.add_argument("--ct", type=str)
    parser.add_argument("--radionuclide", type=str, choices=['Tc99m', 'Lu177'])
    parser.add_argument("--data", type=str)
    parser.add_argument("--output_folder", type=str)
    args = parser.parse_args()
    main()