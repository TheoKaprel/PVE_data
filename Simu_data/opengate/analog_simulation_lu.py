#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import opengate as gate
from opengate.actors.digitizers import energy_windows_peak_scatter
from opengate.contrib.spect import siemens_intevo as gate_spect
from scipy.spatial.transform import Rotation  # used to describe a rotation matrix

import os
import argparse
import numpy as np

def main():
    # create the simulation
    sim = gate.Simulation()
    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)

    # units
    cm = gate.g4_units.cm
    m = gate.g4_units.m
    km = gate.g4_units.km
    Bq = gate.g4_units.Bq
    keV = gate.g4_units.keV
    sec = gate.g4_units.second

    # main options
    sim.g4_verbose = False
    sim.g4_verbose_level = 1
    sim.number_of_threads = 4
    sim.visu = args.visu
    sim.random_seed = int(np.random.rand()*100000)
    sim.output_dir = output_folder
    sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option4"
    sim.physics_manager.global_production_cuts.all = 1 * km


    world = sim.world
    world.size = [3 * m, 3 * m, 3 * m]
    world.material = "G4_AIR"

    # spect head

    angle_step = 3
    n_angles = args.n
    sim.run_timing_intervals = [[k * sec, (k + 1) * sec] for k in range(n_angles)]

    spect, colli, crystal = gate_spect.add_spect_head(
        sim, f"spect", collimator_type="melp", debug=sim.visu
    )
    crystal_name = f"{spect.name}_crystal"

    # channels = [{"name": f"peak208_{spect.name}", "min": 192.4 * keV, "max": 223.6 * keV}]
    channels = [{"name": f"peak208_{spect.name}", "min": 187.56 * keV, "max": 229.24 * keV}]
    digitizer = gate_spect.add_digitizer(sim, spect, crystal,channels)
    digitizer[f'Projection_{crystal_name}'].output_filename = args.output

    gate_spect.rotate_gantry(head=spect,
                             radius=30*cm,
                             start_angle_deg=args.start*angle_step,
                             step_angle_deg=angle_step,
                             nb_angle=n_angles,
                             initial_rotation=None)

    source = sim.add_source("VoxelSource", "vox_source")
    spectrum = gate.sources.utility.get_spectrum("Lu177", "gamma", database="icrp107")
    print(f"{spectrum=}")
    source.particle = "gamma"
    source.energy.type = "spectrum_discrete"
    source.energy.spectrum_energies = spectrum.energies
    source.energy.spectrum_weights = spectrum.weights
    source.image = args.source
    source.direction.type = "iso"
    source.activity = args.activity * Bq / sim.number_of_threads


    # add stat actor
    stats = sim.add_actor("SimulationStatisticsActor", "stats")
    stats.track_types_flag = True
    stats.output_filename = "test043_arf_training_dataset_stats.txt"
    stats.stats.write_to_disk = True

    # start simulation
    sim.run(start_new_process=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a","--activity", type = float, default = 5e9)
    parser.add_argument("--source", type = str)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--visu",action="store_true")
    parser.add_argument("--start",type=int)
    parser.add_argument("-n",type=int)
    parser.add_argument("--output",type=str)
    args = parser.parse_args()
    main()