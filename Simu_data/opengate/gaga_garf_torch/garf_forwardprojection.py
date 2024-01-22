import os
import click
from pathlib import Path
from box import Box
import itk
import opengate as gate
from garf_helpers import *

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('-n','number_of_particles', help = 'number of PRIMARY particles. Ex: 4e7', required = True)
@click.option('-s','--source', 'source_image', required = True)
@click.option('--pth', help='Path to the nn-ARF .pth file', required = True)
@click.option('--batchsize', default = 2e5)
@click.option('-o', '--output', 'output_filename', help='Output projection filename', required = True)
@click.option('--visu', is_flag = True, default = False)
def generate_garf_projection(number_of_particles,source_image, pth,batchsize, output_filename, visu):


    paths = Box()
    # paths.pwd = Path(os.getcwd())
    paths.pwd = Path(os.path.dirname(__file__))
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


    nb_primaries=int(float(number_of_particles))
    activity = nb_primaries * Bq
    source = sim.add_source('Voxels', 'source')
    source.mother = world.name
    source.image = Path(source_image)
    source.particle = 'gamma'
    source.activity = activity / ui.number_of_threads
    source.direction.type = 'iso'
    source.energy.mono = 140.5 * keV


    #arf actor
    arf = sim.add_actor("ARFActor", "arf")
    arf.mother = detPlane.name

    arf.output = Path(output_filename)
    arf.batch_size = int(float(batchsize))
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

    arf = sim.get_actor('arf')
    nb_detected = arf.detected_particles
    print(f'Total Number of particles that have reached to Detector Plane : {nb_detected}')
    print(f'Ratio detected/primaries: {nb_detected / nb_primaries}')

    output_img = arf.output_image
    output_img_arr = itk.array_view_from_image(output_img)
    output_img_arr = output_img_arr / nb_primaries
    output_img_scaled = itk.image_from_array(output_img_arr)
    output_img_scaled.CopyInformation(output_img)
    if '.mha' in output_filename:
        output_scaled_filename = output_filename.replace('.mha', '_scaled.mha')
    elif '.mhd' in output_filename:
        output_scaled_filename = output_filename.replace('.mhd', '_scaled.mhd')
    itk.imwrite(output_img_scaled, output_scaled_filename)



    stats = sim.get_actor('Stats')
    print(stats)
    # stat_filename = output_filename.replace('.mha', '_stats.txt')
    # stats.write(Path(stat_filename))

    gate.delete_run_manager_if_needed(sim)







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
    generate_garf_projection()