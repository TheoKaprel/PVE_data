#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from test203_helpers import *
import opengate as gate

if __name__ == "__main__":
    # create the simulation
    sim = gate.Simulation()
    simu_name = "gaga_training_dataset_ct_large"
    output_folder = "training_data"
    os.makedirs(output_folder, exist_ok=True)

    # units
    cm = gate.g4_units.cm
    cm3 = gate.g4_units.cm3
    mm = gate.g4_units.mm
    nm = gate.g4_units.nm
    keV = gate.g4_units.keV
    Bq = gate.g4_units.Bq
    kBq = 1000 * Bq
    MBq = 1000 * kBq
    BqmL = Bq / cm3
    sec = gate.g4_units.second

    # options
    ui = sim.user_info
    ui.number_of_threads = 8
    # ui.visu = True
    ui.visu_type = "vrml"

    # parameters
    p = Box()
    p.data_folder = Path("training_data")
    # p.ct_image = "53_CT_bg_crop_4mm_vcrop.mhd"
    # p.activity_image = "uniform_4mm_vcrop.mhd"
    p.ct_image = "AA_attmap_rec_fp.mhd"
    p.activity_image = "src_uniform_AA_4mm.mhd"
    p.radionuclide = "Tc99m"
    # p.activity = 4e6 * Bq
    p.activity = 2e7 * Bq
    p.duration = 1 * sec
    if ui.visu:
        p.activity = 100 * Bq

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
