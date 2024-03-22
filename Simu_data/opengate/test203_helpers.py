#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import opengate as gate
import opengate.exception
import sys
sys.path.append(os.path.join(os.getcwd(),"../spect_siemens_intevo"))
import spect_siemens_intevo as gate_intevo
from opengate.sources.generic import get_rad_gamma_energy_spectrum
from gaga_phsp import gaga_helpers_gate

import os
from path import Path
import inspect
import numpy as np
from scipy.spatial.transform import Rotation
from box import Box
import itk
import time


def get_calling_module_file():
    # Get the calling frame
    frame = inspect.currentframe().f_back

    # Get the globals dictionary of the calling module
    calling_globals = frame.f_globals

    # Retrieve the __file__ attribute from the globals dictionary
    calling_module_file = calling_globals.get("__file__")

    if calling_module_file is not None:
        # Get the absolute path of the calling module's file
        calling_module_file = os.path.abspath(calling_module_file)

    return calling_module_file


def get_simu_name(f):
    script_path = os.path.abspath(f)
    simu_name = os.path.splitext(os.path.basename(script_path))[0]
    return simu_name


def init_test_simu(p, f, out_folder=None):
    p.simu_name = get_simu_name(f)
    p.output_folder = p.simu_name.split("_")[0]

    # arg?
    if len(p.argv) > 1:
        p.simu_mode = p.argv[1]

    # output ?
    if out_folder is not None:
        p.output_folder = out_folder

    # create the simulation
    sim = gate.Simulation()

    # options
    ui = sim.user_info
    ui.check_overlap = False
    ui.visu_type = "vrml"
    ui.number_of_threads = p.number_of_threads

    # units
    m = gate.g4_units.m
    sec = gate.g4_units.second
    Bq = gate.g4_units.Bq

    if p.simu_mode == "visu":
        p.output_folder = f"{p.output_folder}_visu"
        ui.visu = True
        ui.number_of_threads = 1
        p.activity = 3 * Bq
        if len(p.argv) > 2:
            if p.argv[2] == "noct":
                p.simu_mode = "noct"

    if p.simu_mode == "noct":
        p.output_folder = f"{p.output_folder}_noct"

    if p.simu_mode == "speed":
        p.output_folder = f"{p.output_folder}_speed"
        ui.number_of_threads = 1
        p.activity = 1e6 * Bq
        p.duration = 1 * sec

    # size
    p.plane_size = [p.size[0] * p.spacing[0], p.size[1] * p.spacing[1]]

    # output folder
    p.output_folder = Path(p.output_folder)
    os.makedirs(p.output_folder, exist_ok=True)

    # world size
    world = sim.world
    world.size = [2 * m, 2 * m, 3 * m]
    world.material = "G4_AIR"

    # colli ?
    p.colli_type = None
    if p.radionuclide == "Tc99m":
        p.colli_type = "lehr"
    if p.radionuclide == "Lu177":
        p.colli_type = "melp"
    if p.colli_type is None:
        opengate.exception.fatal(f"Unknown radionuclide {p.radionuclide}")

    # stats
    stats = sim.add_actor("SimulationStatisticsActor", "stats")
    stats.output = p.output_folder / f"{p.simu_name}_stats.txt"

    # basic physics
    sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option3"
    sim.physics_manager.set_production_cut("world", "all", 1e3 * m)

    print(f"Simu          : {p.simu_name}")
    print(f"Output folder : {p.output_folder}")
    print(f"Run mode      : {p.simu_mode}")
    print(f"Nb threads    : {ui.number_of_threads}")
    return sim


def add_intevo_head(sim, p, name, n, angle=0):
    ui = sim.user_info
    # create head
    head, colli, crystal = gate_intevo.add_intevo_spect_head(
        sim, name, p.colli_type, debug=ui.visu
    )
    # initial default orientation
    rot = gate_intevo.set_head_orientation(head, p.colli_type, p.radius, angle)

    # offset
    head.translation[2] = p.detector_offset

    # adapt digitizer to radionuclide (default is Lu177)
    keV = gate.g4_units.keV
    digit = gate_intevo.add_digitizer_v2(sim, crystal.name, f"{name}_digit")
    if p.radionuclide == "Tc99m":
        channels = [
            {"name": f"scatter", "min": 114 * keV, "max": 126 * keV},
            {"name": f"peak140", "min": 126 * keV, "max": 154 * keV},
        ]
        channel_names = [c["name"] for c in channels]
        cc = digit.find_first_module("energy_window")
        cc.channels = channels
        proj = digit.find_first_module("projection")
        proj.input_digi_collections = channel_names
    else:
        proj = digit.get_last_module()
    proj.output = f"{p.output_folder}/{p.simu_name}_{n}.mhd"

    # size (FIXME why p is a boxlist ?)
    proj.size = [p.size[0], p.size[1]]
    proj.spacing = [p.spacing[0], p.spacing[1]]

    # associated physics
    mm = gate.g4_units.mm
    sim.physics_manager.set_production_cut(head.name, "all", 1 * mm)


def add_intevo_head_arf(sim, p, name, n, angle):
    arf = add_garf_detector(
        sim,
        name,
        p.size,
        p.spacing,
        p.colli_type,
        p.radius,
        angle,
        p.detector_offset,
        p.garf_pth_filename,
    )
    arf.output = f"{p.output_folder}/{p.simu_name}_{n}.mhd"

    # print
    pos, crystal_dist, psd = gate_intevo.compute_plane_position_and_distance_to_crystal(
        p.colli_type
    )
    print(f"ARF colli            : {p.colli_type}")
    print(f"ARF crystal distance : {crystal_dist}")
    print(f"ARF detector size    : {p.plane_size}")
    print(f"ARF detector offset  : {p.detector_offset}")


def add_ct_image(sim, p):
    patient = sim.add_volume("Image", "patient")
    patient.image = p.ct_image
    patient.material = "G4_AIR"
    f1 = os.path.join(p.data_folder,"Schneider2000MaterialsTable.txt")
    f2 = os.path.join(p.data_folder,"Schneider2000DensitiesTable.txt")
    gcm3 = gate.g4_units.g_cm3
    tol = 0.05 * gcm3
    vm, materials = gate.geometry.materials.HounsfieldUnit_to_material(sim, tol, f1, f2)
    patient.voxel_materials = vm
    print(f"CT image: {patient.image}")
    print(f"CT image translation: {patient.translation}")
    print(f"CT image tol: {tol/gcm3} g/cm3")
    print(f"CT image mat: {len(patient.voxel_materials)} materials")

    # associated physics
    mm = gate.g4_units.mm
    sim.physics_manager.set_production_cut("patient", "all", 1 * mm)

    return patient


def add_waterbox(sim, p):
    mm = gate.g4_units.mm
    wb = sim.add_volume("Box", "waterbox")
    wb.size = [200 * mm, 200 * mm, 2000 * mm]
    wb.material = "G4_WATER"
    return wb


def run_and_write(sim, p):
    sim.run_timing_intervals = [[0, p.duration]]

    # output stats file
    so = sim.get_actor_user_info("stats").output

    # run
    sim.run()

    # print results at the end
    stats = sim.output.get_actor("stats")
    print(stats)
    print(so)


def add_arf_actor(sim, detector_plane, size, spacing, crystal_dist, name, pth_filename):
    # arf actor
    arf = sim.add_actor("ARFActor", f"arf_{name}")
    arf.mother = detector_plane.name
    arf.batch_size = 1e5
    arf.image_size = size
    arf.image_spacing = spacing
    arf.verbose_batch = True
    arf.distance_to_crystal = crystal_dist
    arf.gpu_mode = "auto"
    arf.enable_hit_slice = False
    arf.pth_filename = pth_filename
    return arf


def add_garf_detector(
    sim,
    name,
    size,
    spacing,
    colli_type,
    radius,
    gantry_angle,
    detector_offset,
    pth_filename,
):
    # garf plane
    plane_size = [size[0] * spacing[0], size[1] * spacing[1]]
    arf_plane = gate_intevo.add_detection_plane_for_arf(
        sim, plane_size, colli_type, radius, gantry_angle, f"{name}_garf_plane"
    )
    # table offset
    arf_plane.translation[2] = detector_offset
    # get crystal dist
    pos, crystal_dist, psd = gate_intevo.compute_plane_position_and_distance_to_crystal(
        colli_type
    )
    # garf actor
    arf = add_arf_actor(
        sim, arf_plane, size, spacing, crystal_dist, f"{name}_garf", pth_filename
    )
    return arf


def add_debug_phsp_plane(sim, p):
    sim.add_parallel_world("phsp_world")
    # plane_size = [p.size[0] * p.spacing[0], p.size[1] * p.spacing[1]]
    mm = gate.g4_units.mm
    plane_size = [533 * mm, 387 * mm]
    phsp_plane = gate_intevo.add_detection_plane_for_arf(
        sim, plane_size, p.colli_type, p.radius, 0, f"{p.simu_name}_garf_plane_phsp"
    )
    phsp_plane.mother = "phsp_world"
    phsp = sim.add_actor("PhaseSpaceActor", "phsp")
    phsp.mother = phsp_plane.name
    phsp.attributes = [
        "KineticEnergy",
        "PrePosition",
        "PreDirection",
    ]
    phsp.output = f"{p.output_folder}/{p.simu_name}_phsp.root"


def add_vox_source(sim, p, patient):
    source = sim.add_source("VoxelsSource", "vox_source")
    w, e = get_rad_gamma_energy_spectrum(p.radionuclide)
    source.mother = patient.name
    source.particle = "gamma"
    source.energy.type = "spectrum_lines"
    source.energy.spectrum_weight = w
    source.energy.spectrum_energy = e
    source.image = p.activity_image
    source.direction.type = "iso"
    if patient.name != "world" and patient.name != "waterbox":
        source.position.translation = gate.image.get_translation_between_images_center(
            patient.image, source.image
        )

    Bq = gate.g4_units.Bq
    ui = sim.user_info
    sec = gate.g4_units.second
    source.activity = p.activity / ui.number_of_threads

    ne = int((p.activity / Bq) * np.sum(w) * p.duration / sec)
    print(f"Vox source translation: {source.position.translation}")
    print(f"Vox source total activity: {source.activity/Bq} Bq")
    print(f"{p.radionuclide} yield: {np.sum(w)}")
    print(f"Expected events: {ne}")
    return source


def add_gaga_source(sim, p):
    keV = gate.g4_units.keV
    Bq = gate.g4_units.Bq
    sec = gate.g4_units.second
    ui = sim.user_info

    w, e = get_rad_gamma_energy_spectrum(p.radionuclide)
    ryield = np.sum(w)

    # source
    source = sim.add_source("GANSource", "gaga")
    source.particle = "gamma"
    source.pth_filename = p.gaga_pth_filename
    source.position_keys = ["PrePosition_X", "PrePosition_Y", "PrePosition_Z"]
    if p.ideal_pos_flag:
        source.position_keys = [
            "IdealPosition_X",
            "IdealPosition_Y",
            "IdealPosition_Z",
        ]
    source.backward_distance = p.backward_distance
    source.direction_keys = ["PreDirection_X", "PreDirection_Y", "PreDirection_Z"]
    source.energy_key = "KineticEnergy"
    source.energy_min_threshold = 10 * keV
    source.skip_policy = "ZeroEnergy"
    source.weight_key = None
    source.time_key = None
    source.backward_force = True  # because we don't care about timing
    source.batch_size = p.batch_size
    source.verbose_generator = False
    source.gpu_mode = "auto"
    source.activity = p.activity / ui.number_of_threads

    # condition
    cond_gen = gate.sources.gansources.VoxelizedSourceConditionGenerator(
        p.data_folder / p.activity_image
    )
    cond_gen.compute_directions = True
    gen = gate.sources.gansources.GANSourceConditionalGenerator(
        source, cond_gen.generate_condition
    )
    source.generator = gen

    # The (source position) conditions are in Geant4 (world) coordinate system
    # (because it is build from phsp Event Position)
    # so the vox sampling should transform the source image points in G4 world
    # This is done by translating from centers of CT and source images
    tr = gate.image.get_translation_between_images_center(
        "data/53_CT_bg_crop_4mm_vcrop.mhd", p.data_folder / p.activity_image
    )
    print(f"Translation from source coord system to G4 world {tr=} (rotation not done)")
    cond_gen.translation = tr

    # print
    ne = int(source.activity / Bq * ui.number_of_threads * p.duration / sec)
    print(f"Vox source total activity: {source.activity/Bq} Bq")
    print(f"{p.radionuclide} yield: {ryield}")
    print(f"Expected events: {ne}")
    return source, cond_gen


def run_standalone(p):
    # crystal dist
    _, p.crystal_dist, _ = gate_intevo.compute_plane_position_and_distance_to_crystal(
        p.colli_type
    )
    print(f"crystal dist: {p.crystal_dist} mm")

    # translation
    tr = gate.image.get_translation_between_images_center(
        p.ct_image, p.activity_image
    )
    print(f"Translation from source coord system to G4 world: {tr}")

    # create param
    gaga_user_info = Box(
        {
            "pth_filename": p.gaga_pth_filename,
            "activity_source": p.activity_image,
            "batch_size": p.batchsize,
            "gpu_mode": "gpu",
            "backward_distance": p.backward_distance,
            "verbose": 1,
            "cond_translation": tr,
        }
    )
    garf_user_info = Box(
        {
            "pth_filename": p.garf_pth_filename,
            "image_size": p.size,
            "image_spacing": p.spacing,
            "plane_distance": p.radius,
            "detector_offset": [0, 0],  # FIXME
            "distance_to_crystal": p.crystal_dist,
            "batch_size": p.batchsize,
            "gpu_mode": "gpu",
            "verbose": 0,
            "hit_slice": False,
        }
    )

    # initialize gaga and garf (read the NN)
    gaga_helpers_gate.gaga_garf_generate_spect_initialize(gaga_user_info, garf_user_info)

    # rotation wrt intevo
    # r1 = rotation like the detector (needed), see in digitizer ProjectionActor
    r1 = Rotation.from_euler("yx", (90, 90), degrees=True)
    # r2 = rotation X180 is to set the detector head-foot, rotation Z90 is the gantry angle
    r2 = Rotation.from_euler("xz", (180, 90), degrees=True)
    r = r2 * r1
    garf_user_info.plane_rotation = r

    # angles
    angles=[]
    nprojs=p.nprojs
    l_angles = np.linspace(0, 360, nprojs+1)[:-1]
    for angle in l_angles:
        angles.append(Rotation.from_euler("z", angle, degrees=True))

    # nb
    Bq = gate.g4_units.Bq
    sec = gate.g4_units.second
    n = (p.activity / Bq) * (p.duration / sec)
    print(f"Total number of particles: {n}")

    # Go
    p.start_time = time.time()
    images = gaga_helpers_gate.gaga_garf_generate_spect(gaga_user_info, garf_user_info, n, angles)
    p.stop_time = time.time()
    p.duration = p.stop_time - p.start_time
    p.pps = int(n / p.duration)

    # save image
    i = 0
    for image in images:
        output = f"{p.output_folder}/{p.simu_name}_{i}.mhd"
        print(f"Done, saving output in {output}")
        itk.imwrite(image, output)
        i += 1
