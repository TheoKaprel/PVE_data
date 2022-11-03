import os
import click
from pathlib import Path
from box import Box
import opengate as gate
import opengate.contrib.spect_ge_nm670 as gate_spect


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--ct', help = "path to the ct")
@click.option('--actmap', help = "path to the activity map")
@click.option('--activity', '-a', 'upactivity')
@click.option('--rot', is_flag = True, default = False)
@click.option('--nproj', '-n', type = int, default = 60)
@click.option('--mt', type = int, default = 1, help = 'Multi Thread')
@click.option('--visu', is_flag = True, default = False)
@click.option('--debug', 'updebug', is_flag = True, default = False)
@click.option('--output_folder', '-o')
def sim_full_spect(ct, actmap, upactivity,rot, nproj,mt, visu, updebug, output_folder):
    paths = Box()
    paths.pwd = Path(os.getcwd())
    paths.data = paths.pwd / 'data'

    if output_folder:
        paths.output = paths.pwd / output_folder
    else:
        paths.output = paths.pwd / 'ouputs'

    sim = gate.Simulation()

    # main options
    ui = sim.user_info
    ui.g4_verbose = False
    ui.visu = visu
    ui.visu_versbose = True
    ui.number_of_threads = mt
    ui.check_volumes_overlap = False

    sim.add_material_database(paths.data / "GateMaterials.db")


    # world size
    world = sim.world
    world.size = [2 * m, 2 * m, 2 * m]
    world.material = 'G4_AIR'

    spect = gate_spect.add_ge_nm67_spect_head(sim, name='spect', collimator_type="lehr", debug=updebug)
    sdd = 38 * cm
    psd = 6.11 * cm
    pos = [0, 0, -(sdd + psd)]
    spect.translation, spect.rotation = gate.get_transform_orbiting(pos, 'y', 0)

    make_physics_list(sim)


    # patient_ct = sim.add_volume("Image", "patient_ct")
    # patient_ct.image = ct
    # patient_ct.mother = world.name
    # f1 = str(paths.data / "Schneider2000MaterialsTable.txt")
    # f2 = str(paths.data / "Schneider2000DensitiesTable.txt")
    # tol = 0.05 * gcm3
    # patient_ct.voxel_materials, materials = gate.HounsfieldUnit_to_material(tol, f1, f2)
    # print(f"tol = {tol} g/cm3")
    # print(f"mat : {len(patient_ct.voxel_materials)} materials")

    activity = int(float(upactivity)) * Bq
    source = sim.add_source('Voxels', 'source')
    source.mother = world.name
    source.image = actmap
    source.particle = 'gamma'
    source.activity = activity / ui.number_of_threads
    source.direction.type = 'iso'
    source.energy.mono = 140.5 * keV


    # add stat actor
    stats = sim.add_actor('SimulationStatisticsActor', 'Stats')
    stats.track_types_flag=True

    """ 
        The order of the actors is important !
        1. Hits
        2. Singles
        3. EnergyWindows
    """
    # hits collection
    hc = sim.add_actor('HitsCollectionActor', 'Hits')
    l = sim.get_all_volumes_user_info()
    crystal = l[[k for k in l if 'crystal' in k][0]]
    hc.mother = crystal.name
    hc.output = paths.output / 'full_spect_hc.root'
    hc.attributes = ['PostPosition', 'TotalEnergyDeposit', 'TrackVolumeCopyNo','PreStepUniqueVolumeID', 'PostStepUniqueVolumeID','GlobalTime', 'KineticEnergy', 'ProcessDefinedStep']
    # singles collection
    sc = sim.add_actor('HitsAdderActor', 'Singles')
    sc.mother = crystal.name
    sc.input_hits_collection = 'Hits'
    sc.policy = 'EnergyWinnerPosition'
    sc.skip_attributes = ['ProcessDefinedStep', 'KineticEnergy']
    sc.output = hc.output
    # EnergyWindows
    cc = sim.add_actor('HitsEnergyWindowsActor', 'EnergyWindows')
    cc.mother = crystal.name
    cc.input_hits_collection = 'Singles'
    cc.channels = [{'name': 'scatter', 'min': 114 * keV, 'max': 126 * keV},
                   {'name': 'peak140', 'min': 126 * keV, 'max': 154.55 * keV},
                   {'name': 'spectrum', 'min': 0 * keV, 'max': 5000 * keV}  # should be strictly equal to 'Singles'
                   ]
    cc.output = hc.output
    # Projection
    proj = sim.add_actor('HitsProjectionActor', 'Projection')
    proj.mother = crystal.name
    proj.input_hits_collections = ['spectrum', 'scatter', 'peak140']
    proj.spacing = [4.41806 * mm, 4.41806 * mm]
    proj.size = [128, 128]
    # proj.plane = 'XY' # not implemented yet
    proj.output = paths.output / f'proj_{upactivity}.mhd'

    if rot:
        # motion of the spect, create also the run time interval
        motion = sim.add_actor('MotionVolumeActor', 'Move')
        motion.mother = spect.name
        sim.run_timing_intervals = []
        arc = 360
        motion.translations, motion.rotations = gate.volume_orbiting_transform('y', 0, arc, nproj, spect.translation, spect.rotation)
        motion.priority = 5

        sim.run_timing_intervals = gate.range_timing(0, 1 * sec, nproj)
        print(f'Run intervals: {sim.run_timing_intervals}')

    sim.initialize()
    sim.start()

    stats = sim.get_actor('Stats')
    print(stats)
    stats.write(paths.output / 'stats.txt')


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
keV = gate.g4_units('keV')
mm = gate.g4_units('mm')
Bq = gate.g4_units('Bq')
sec = gate.g4_units('second')
gcm3 = gate.g4_units("g/cm3")

if __name__=='__main__':
    sim_full_spect()



