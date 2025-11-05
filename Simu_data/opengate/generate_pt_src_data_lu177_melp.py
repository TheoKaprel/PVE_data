#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import opengate as gate
from opengate.actors.digitizers import energy_windows_peak_scatter
from opengate.contrib.spect import siemens_intevo as gate_spect
from scipy.spatial.transform import Rotation  # used to describe a rotation matrix

import os
import argparse

def main():
    # create the simulation
    sim = gate.Simulation()
    simu_name = "garf_training_dataset_lu177_melp_rr50"
    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)

    # main options
    sim.g4_verbose = False
    sim.g4_verbose_level = 1
    sim.number_of_threads = 1
    sim.visu = args.visu
    sim.random_seed = 12365
    sim.output_dir = output_folder

    # units
    nm = gate.g4_units.nm
    mm = gate.g4_units.mm
    cm = gate.g4_units.cm
    m = gate.g4_units.m
    km = gate.g4_units.km
    Bq = gate.g4_units.Bq
    keV = gate.g4_units.keV
    MeV = gate.g4_units.MeV

    # activity
    activity = int(args.activity) * Bq / sim.number_of_threads

    world = sim.world
    world.size = [3 * m, 3 * m, 3 * m]
    world.material = "G4_AIR"

    # spect head
    spect, colli, crystal = gate_spect.add_spect_head(
        sim, "spect", collimator_type="melp", debug=sim.visu
    )
    crystal_name = f"{spect.name}_crystal"


    # detector input plane
    pos, crystal_dist, psd = gate_spect.compute_plane_position_and_distance_to_crystal("melp")
    pos -= 1 * nm  # to avoid overlap
    print(f"plane position     {pos / mm} mm")
    print(f"crystal distance   {crystal_dist / mm} mm")

    # detector input plane
    detector_plane = sim.add_volume("Box", "detPlane")
    detector_plane.mother = spect.name
    detector_plane.size = [1 * nm, 44.6 * cm,57.6 * cm ]
    detector_plane.translation = [pos,0,0]
    detector_plane.material = "G4_Galactic"
    detector_plane.color = [1, 0, 0, 1]

    sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option4"
    sim.physics_manager.global_production_cuts.all = 1 * km

    # source
    s1 = sim.add_source("GenericSource", "s1")
    s1.particle = "gamma"
    s1.activity = activity
    s1.position.type = "sphere"
    s1.position.radius = 2 * cm
    s1.position.translation = [-28 * cm,0,0]
    s1.position.rotation = Rotation.from_euler('y', 90, degrees=True).as_matrix()
    s1.direction.type = "iso"
    s1.energy.type = "mono"
    s1.energy.mono = 208.5 * keV
    s1.direction.acceptance_angle.volumes = [detector_plane.name]
    s1.direction.acceptance_angle.intersection_flag = True


    digitizzzer = gate_spect.add_digitizer_lu177(sim, crystal_name,spect.name)

    arf = sim.add_actor("ARFActor", "arf")
    arf.attached_to = detector_plane.name
    arf.output_filename = "projection_arf2.mha"
    arf.image_size = [128, 128]
    arf.image_spacing = [4.7952 * mm, 4.7952 * mm]
    arf.verbose_batch = True
    arf.distance_to_crystal = crystal_dist
    arf.pth_filename = args.arf
    arf.batch_size = 2e5
    arf.gpu_mode = "auto"
    arf.plane_axis = [1, 2, 0]

    proj = sim.add_actor("DigitizerProjectionActor", "Projection")
    proj.attached_to = crystal  # Attach to crystal volume
    proj.input_digi_collections = ["scatter1","peak113","scatter2", "scatter3","peak208","scatter4"]  # Use multiple energy channels
    # proj.input_digi_collections = ["peak208"]  # Use multiple energy channels
    proj.spacing = [4.7952 * mm, 4.7952 * mm]  # Set pixel spacing in mm
    proj.size = [128, 128]  # Image size in pixels (128x128)
    proj.origin_as_image_center = True  # Origin is not at image center
    proj.output_filename = 'projection_analog2.mha'
    proj.detector_orientation_matrix = Rotation.from_euler('y', 90, degrees=True).as_matrix()

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
    parser.add_argument("--arf", type=str)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--visu",action="store_true")
    args = parser.parse_args()
    main()