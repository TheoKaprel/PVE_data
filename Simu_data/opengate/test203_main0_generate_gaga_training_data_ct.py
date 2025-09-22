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
    keV = gate.g4_units.keV
    MeV = gate.g4_units.MeV

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
    sph_surface.rmin = 350 * mm
    sph_surface.rmax = 351 * mm
    sph_surface.color = [0, 1, 0, 1]
    sph_surface.material = "G4_AIR"
    sph_surface.mother = "sphere_world"

    # source uniform (limited FOV)
    # source = add_vox_source(sim, p, patient)
    source = sim.add_source("VoxelSource", "vox_source")
    source.attached_to = patient.name
    source.particle = "gamma"
    source.energy.type = "mono"
    source.energy.mono = 0.208366 * MeV
    source.image = p.activity_image
    source.direction.type = "iso"


    Bq = gate.g4_units.Bq
    ui = sim.user_info
    sec = gate.g4_units.second
    source.activity = p.activity / ui.number_of_threads

    ne = int((p.activity / Bq)) * p.duration / sec
    print(f"Vox source translation: {source.position.translation}")
    print(f"Vox source total activity: {p.activity/Bq} Bq")
    print(f"Expected events: {ne}")

    # stats
    stats = sim.add_actor("SimulationStatisticsActor", "stats")
    stats.output_filename = f"{output_folder}/{simu_name}_stats.txt"

    # phsp
    phsp = sim.add_actor("PhaseSpaceActor", "phase_space")
    phsp.attached_to = "phase_space_sphere"
    phsp.attributes = [
        "KineticEnergy",
        "PrePosition",
        "PreDirection",
        # "TimeFromBeginOfEvent",
        # "EventID",
        # "EventKineticEnergy",
        "EventPosition",
        "EventDirection",
    ]
    phsp.output_filename = f"{output_folder}/{simu_name}.root"
    # this option allow to store all events even if absorbed
    phsp.store_absorbed_event = False
    f = sim.add_filter("ParticleFilter", "f")
    f.particle = "gamma"
    phsp.filters.append(f)
    fk = sim.add_filter("KineticEnergyFilter", "fk")
    fk.energy_min = 150 * keV
    phsp.filters.append(fk)
    print(phsp)
    print(phsp.output_filename)

    # physic list
    sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option3"
    sim.physics_manager.set_production_cut("world", "all", 1 * gate.g4_units.km)

    # sim.physics_manager.energy_range_min = 150 * gate.g4_units.keV
    # sim.physics_manager.energy_range_max = 210 * gate.g4_units.keV

    # run
    sim.run()

    stats = sim.get_actor("stats")
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