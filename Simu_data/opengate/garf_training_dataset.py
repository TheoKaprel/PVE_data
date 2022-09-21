import os
import click
from pathlib import Path
from box import Box
import opengate as gate
import opengate.contrib.spect_ge_nm670 as gate_spect


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('-n','number_of_particles', help = 'number of particles')
@click.option('--visu', is_flag = True, default = False)
@click.option('--mt',type=int, default = 1)
@click.option('--output', '-o')
def generate_garf_training_dataset(number_of_particles, visu,mt, output):
    paths = Box()
    paths.pwd = Path(os.getcwd())
    paths.data = paths.pwd / 'data'


    sim = gate.Simulation()
    ui = sim.user_info
    ui.g4_verbose = False
    ui.visu = visu
    ui.visu_versbose = True
    ui.number_of_threads = mt
    ui.check_volumes_overlap = False
    # sim.add_material_database(paths.data / "GateMaterials.db")

    # world size
    world = sim.world
    world.size = [1 * m, 1 * m, 0.7 * m]
    world.material = 'G4_AIR'

    spect = gate_spect.add_ge_nm67_spect_head(sim, name='spect', collimator_type="lehr", debug=visu)
    sdd = 10 * cm
    psd = 6.11 * cm
    pos = [0, 0, -(sdd + psd)]
    spect.translation, spect.rotation = gate.get_transform_orbiting(pos, 'y', 0)
    crystal_name = f"{spect.name}_crystal"

    detPlane = sim_set_detector_plane(sim, spect_name=spect.name)

    make_physics_list(sim)

    # faire le plan source ici
    source_plane = sim.add_volume('Box', 'source_plane')
    source_plane.size = [800 * mm, 800 * mm, 1 * mm]
    source_plane.translation,source_plane.rotation = gate.get_transform_orbiting([0, 0, - 1 * cm], 'y', 0)
    source_plane.material = 'G4_AIR'
    source_plane.color = [1, 0.7, 0.7, 0.8]

    activity = int(number_of_particles) * Bq
    source = sim.add_source('Generic', 'source')
    source.mother = 'source_plane'
    source.position.type = 'disc'
    source.position.radius = 300 * mm
    source.particle = 'gamma'
    source.activity = activity / ui.number_of_threads
    source.direction.type = 'iso'
    source.energy.type = "range"
    source.energy.min_energy = 10 * keV
    source.energy.max_energy = 154 * keV
    source.direction.acceptance_angle.volumes = [detPlane.name]
    source.direction.acceptance_angle.intersection_flag = True

    # digitizer
    channels = [
        {"name": f"scatter2_{spect.name}", "min": 114 * keV, "max": 126 * keV},
        {"name": f"peak140_{spect.name}", "min": 126 * keV, "max": 154 * keV},
    ]
    cc = gate_spect.add_digitizer_energy_windows(sim, crystal_name, channels)


    # arf actor for building the training dataset
    if visu==False:
        arf = sim.add_actor("ARFTrainingDatasetActor", "ARF_training")
        arf.mother = detPlane.name

        if output:
            arf.output = Path(output)
        else:
            arf.output = paths.pwd / 'outputs' / "arf_training_dataset.root"
        arf.energy_windows_actor = cc.name
        arf.russian_roulette = 100

    # add stat actor
    stats = sim.add_actor('SimulationStatisticsActor', 'Stats')
    stats.track_types_flag=True

    # create G4 objects
    sim.initialize()

    # start simulation
    sim.start()





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
nm = gate.g4_units('nm')
Bq = gate.g4_units('Bq')
sec = gate.g4_units('second')
gcm3 = gate.g4_units("g/cm3")


def sim_set_detector_plane(sim, spect_name):
    # detector input plane
    detector_plane = sim.add_volume("Box", "detPlane")
    detector_plane.mother = spect_name
    detector_plane.size = [57.6 * cm, 44.6 * cm, 1 * nm]
    """
    the detector is 'colli_trd' located in head, size and translation depends on the collimator type
    - lehr = 4.02 + 4.18 /2 = 6.13 + tiny shift (cm)
    - megp = 5.17 + 6.48 / 2.0 = 8.41 + tiny shift (cm)
    """
    # detector_plane.translation = [0, 0, 8.42 * cm]
    detector_plane.translation = [0, 0, 6.14 * cm]
    detector_plane.material = "G4_Galactic"
    detector_plane.color = [1, 0, 0, 1]

    return detector_plane


if __name__=='__main__':
    generate_garf_training_dataset()