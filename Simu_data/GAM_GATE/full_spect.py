import os
import gam_gate as gam
import contrib.spect_ge_nm670 as gam_spect


def sim_full_spect():
    pwd = os.getcwd()

    sim = gam.Simulation()


    # main options
    ui = sim.user_info
    ui.g4_verbose = False
    ui.visu = True
    ui.visu_versbose = True
    ui.number_of_threads = 1
    ui.check_volumes_overlap = False

    # units
    m = gam.g4_units('m')
    cm = gam.g4_units('cm')
    keV = gam.g4_units('keV')
    mm = gam.g4_units('mm')
    Bq = gam.g4_units('Bq')
    kBq = 1000 * Bq

    # world size
    world = sim.world
    world.size = [1 * m, 1 * m, 1 * m]
    world.material = 'G4_AIR'

    spect = gam_spect.add_ge_nm67_spect_head(sim, 'spect', collimator=True, debug=False)

    # waterbox
    waterbox = sim.add_volume('Box', 'waterbox')
    waterbox.size = [15 * cm, 15 * cm, 15 * cm]
    waterbox.material = 'G4_WATER'
    blue = [0, 1, 1, 1]
    waterbox.color = blue

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
    activity = 1 * Bq
    beam1 = sim.add_source('Generic', 'beam1')
    beam1.mother = waterbox.name
    beam1.particle = 'gamma'
    beam1.energy.mono = 140.5 * keV
    beam1.position.type = 'sphere'
    beam1.position.radius = 3 * cm
    beam1.position.translation = [0, 0, 0 * cm]
    beam1.direction.type = 'momentum'
    beam1.direction.type = 'iso'
    beam1.activity = activity / ui.number_of_threads

    # add stat actor
    sim.add_actor('SimulationStatisticsActor', 'Stats')

    # hits collection
    hc = sim.add_actor('HitsCollectionActor', 'Hits')
    # get crystal volume by looking for the word crystal in the name
    l = sim.get_all_volumes_user_info()
    crystal = l[[k for k in l if 'crystal' in k][0]]
    hc.mother = crystal.name
    print('Crystal : ', crystal.name)
    hc.output = os.path.join(pwd, 'full_spect_hc.root')
    hc.attributes = ['PostPosition', 'TotalEnergyDeposit', 'TrackVolumeCopyNo',
                     'PreStepUniqueVolumeID', 'PostStepUniqueVolumeID',
                     'GlobalTime', 'KineticEnergy', 'ProcessDefinedStep']

    # singles collection
    sc = sim.add_actor('HitsAdderActor', 'Singles')
    sc.mother = crystal.name
    sc.input_hits_collection = 'Hits'
    sc.policy = 'EnergyWinnerPosition'
    # sc.policy = 'EnergyWeightedCentroidPosition'
    sc.skip_attributes = ['KineticEnergy', 'ProcessDefinedStep', 'KineticEnergy']
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

    sim.initialize()
    sim.start()




if __name__=='__main__':
    # print(os.getcwd())
    sim_full_spect()
