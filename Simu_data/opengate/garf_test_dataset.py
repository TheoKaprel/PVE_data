import os
import click
from pathlib import Path
from box import Box
import opengate as gate
import opengate.contrib.spect_ge_nm670 as gate_spect
from garf_training_dataset import *
import torch
import itk

import sys
sys.path.append("/export/home/tkaprelian/Desktop/External_repositories/listMode_SPECT")
from scatter_correction_dew import scatter_correction_dew

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('-n','number_of_particles', help = 'number of particles')
@click.option('--src_ref')
@click.option('--pth')
@click.option('--visu', is_flag = True, default = False)
def generate_garf_test_dataset(number_of_particles,src_ref, pth, visu):
    paths = Box()
    paths.pwd = Path(os.getcwd())
    paths.data = paths.pwd / 'data'

    sim = gate.Simulation()
    ui = sim.user_info
    ui.g4_verbose = False
    ui.visu = visu
    ui.visu_versbose = True
    ui.number_of_threads = 1
    ui.check_volumes_overlap = False
    sim.add_material_database(paths.data / "GateMaterials.db")

    # world size
    world = sim.world
    world.size = [2 * m, 2 * m, 2 * m]
    world.material = 'G4_AIR'

    # fake spect head
    spect_length = 19 * cm
    spect_translation = 38 * cm
    SPECThead = sim.add_volume("Box", "SPECThead")
    SPECThead.size = [57.6 * cm, 44.6 * cm, spect_length]
    SPECThead.translation = [0, 0, -spect_translation]
    SPECThead.material = "G4_AIR"
    SPECThead.color = [1, 0, 1, 1]

    detPlane = sim_set_detector_plane(sim, SPECThead.name)


    p = sim.get_physics_user_info()
    p.physics_list_name = "G4EmStandardPhysics_option4"
    sim.set_cut("world", "all", 1 * km)

    activity = int(float(number_of_particles)) * Bq
    source = sim.add_source('Voxels', 'source')
    source.mother = world.name
    source.image = Path(f'{src_ref}.mhd')
    source.particle = 'gamma'
    source.activity = activity / ui.number_of_threads
    source.direction.type = 'iso'
    source.energy.mono = 140.5 * keV


    #arf actor
    arf = sim.add_actor("ARFActor", "arf")
    arf.mother = detPlane.name
    arf.output = Path(f'{src_ref}_garf_eww.mhd')
    arf.batch_size = 2e5
    arf.image_size = [128, 128]
    arf.image_spacing = [4.41806 * mm, 4.41806 * mm]
    arf.verbose_batch = True
    arf.distance_to_crystal = 74.625 * mm
    arf.pth_filename = pth

    # add stat actor
    stats = sim.add_actor('SimulationStatisticsActor', 'Stats')
    stats.track_types_flag=True

    # create G4 objects
    sim.initialize()

    # start simulation
    sim.start()

    gate.delete_run_manager_if_needed(sim)


    image = itk.imread(f'{src_ref}_garf_ew.mhd')
    res = scatter_correction_dew(input_image=image,head=1,energy_window=3,primary=2,scatter=1,factor=1.1)
    itk.imwrite(res,f'{src_ref}_garf.mhd')






def make_physics_list(sim):
    # physic list
    p = sim.get_physics_user_info()
    p.physics_list_name = 'G4EmStandardPhysics_option4'
    p.enable_decay = False
    cuts = p.production_cuts
    cuts.world.gamma = 10 * mm
    cuts.world.electron = 10 * mm
    cuts.world.positron = 10 * mm
    cuts.world.proton = 10 * mm
    cuts.spect.gamma = 0.1 * mm
    cuts.spect.electron = 0.01 * mm
    cuts.spect.positron = 0.1 * mm

# units
m = gate.g4_units('m')
cm = gate.g4_units('cm')
km = gate.g4_units('km')
keV = gate.g4_units('keV')
MeV = gate.g4_units('MeV')
mm = gate.g4_units('mm')
nm = gate.g4_units('nm')
Bq = gate.g4_units('Bq')
sec = gate.g4_units('second')
gcm3 = gate.g4_units("g/cm3")


if __name__=='__main__':
    generate_garf_test_dataset()