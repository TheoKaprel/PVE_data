#!/usr/bin/env python3
import argparse
from test203_helpers import *
from box import Box
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),"../spect_siemens_intevo_loc"))
import spect_siemens_intevo as gate_intevo
from opengate.actors.digitizers import energy_windows_peak_scatter


def main():
    # units
    m = gate.g4_units.m
    mm = gate.g4_units.mm
    Bq = gate.g4_units.Bq
    sec = gate.g4_units.second

    # create the simulation
    sim = gate.Simulation()
    simu_name = f"analog_simu_lu177_megp_{args.activity}_{args.sid}mm"
    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)

    # world size
    world = sim.world
    world.size = [2 * m, 2 * m, 3 * m]
    world.material = "G4_AIR"

    # basic physics
    sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option3"
    sim.physics_manager.set_production_cut("world", "all", 1e3 * m)

    # options
    ui = sim.user_info
    ui.number_of_threads = 8

    print(f"Simu          : {simu_name}")
    print(f"Output folder : {output_folder}")
    print(f"Nb threads    : {ui.number_of_threads}")

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

    # parameters
    p_garf = Box()
    p_garf.size = [128,128]
    p_garf.spacing = [4.7951998710632 * mm, 4.7951998710632 * mm]
    p_garf.plane_size = [p_garf.size[0] * p_garf.spacing[0], p_garf.size[1] * p_garf.spacing[1]]
    p_garf.radius = args.sid * mm
    p_garf.detector_offset = 0 * mm
    p_garf.colli_type = "melp"
    p_garf.radionuclide = "Lu177"
    p_garf.garf_pth_filename = args.garf
    p_garf.simu_name = simu_name
    p_garf.output_folder = output_folder

    # SPECT
    add_intevo_head_arf(sim, p_garf, "arf", 0, angle=args.angle)
    # add_intevo_head_arf(sim, p_garf, "arf2", 1, angle=180+args.angle)


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
    parser.add_argument("--garf")
    parser.add_argument("--data")
    parser.add_argument("--output_folder")
    parser.add_argument("--output_projs")
    args = parser.parse_args()


    main()
