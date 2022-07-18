import os
import click
from pathlib import Path
from box import Box
import gam_gate as gam
import contrib.spect_ge_nm670 as gam_spect




CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--activity', '-a', 'upactivity', type = int, default = 1)
@click.option('--visu', is_flag = True, default = False)
@click.option('--debug', 'updebug', is_flag = True, default = False)
def sim_full_spect(upactivity, visu, updebug):
    paths = Box()
    paths.pwd = Path(os.getcwd())
    paths.data = paths.pwd / 'data'
    paths.output = paths.pwd / 'outputs'

    sim = gam.Simulation()

    # main options
    ui = sim.user_info
    ui.g4_verbose = False
    ui.visu = visu
    ui.visu_versbose = True
    ui.number_of_threads = 1
    ui.check_volumes_overlap = False

    # units
    m = gam.g4_units('m')
    cm = gam.g4_units('cm')
    keV = gam.g4_units('keV')
    mm = gam.g4_units('mm')
    Bq = gam.g4_units('Bq')

    # world size
    world = sim.world
    world.size = [2 * m, 2 * m, 2 * m]
    world.material = 'G4_AIR'

    spect = gam_spect.add_ge_nm67_spect_head(sim, 'spect', collimator=True, debug=updebug)
    sdd = 38 * cm
    psd = 6.11 * cm
    pos = [0, 0, -(sdd + psd)]
    spect.translation, spect.rotation = gam.get_transform_orbiting(pos, 'y', 0)


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

    # default source for tests
    # activity = upactivity * Bq
    # beam1 = sim.add_source('Generic', 'beam1')
    # beam1.mother = world.name
    # beam1.particle = 'gamma'
    # beam1.energy.mono = 140.5 * keV
    # beam1.position.type = 'sphere'
    # beam1.position.radius = 3 * cm
    # beam1.position.translation = [0, 0, 0 * cm]
    # beam1.direction.type = 'momentum'
    # beam1.direction.type = 'iso'
    # beam1.activity = activity / ui.number_of_threads

    activity = upactivity * Bq
    source = sim.add_source('Voxels', 'source')
    source.mother = world.name
    source.particle = 'gamma'
    source.activity = activity / ui.number_of_threads
    source.image = paths.data / 'my_iec_test.mhd'
    # source.direction.type = 'iso'
    source.direction.type = 'momentum'
    source.direction.momentum = [0,0,-1]
    source.energy.mono = 140.5 * keV



    # add stat actor
    stats = sim.add_actor('SimulationStatisticsActor', 'Stats')
    stats.track_types_flag=True


    # hits collection
    hc = sim.add_actor('HitsCollectionActor', 'Hits')
    # get crystal volume by looking for the word crystal in the name
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
    """ 
        The order of the actors is important !
        1. Hits
        2. Singles
        3. EnergyWindows
    """

    proj = sim.add_actor('HitsProjectionActor', 'Projection')
    proj.mother = crystal.name
    proj.input_hits_collections = ['spectrum', 'scatter', 'peak140']
    proj.spacing = [4.41806 * mm, 4.41806 * mm]
    proj.size = [128, 128]
    # proj.plane = 'XY' # not implemented yet
    proj.output = paths.output / 'proj.mhd'


    sim.initialize()
    sim.start()

    stats = sim.get_actor('Stats')
    print(stats)
    stats.write(paths.output / 'stats.txt')




if __name__=='__main__':
    sim_full_spect()
