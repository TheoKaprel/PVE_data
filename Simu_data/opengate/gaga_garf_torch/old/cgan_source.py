#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import opengate as gate
import opengate.contrib.spect_ge_nm670 as gate_spect
import opengate.contrib.phantom_nema_iec_body as gate_iec
import opengate.contrib.spect_ideal_timed_reconstruction as spect_recon
import numpy as np
import os
from box import Box
import pathlib

def create_simulation(sim, paths, colli="lehr"):
    # units
    m = gate.g4_units("m")
    cm = gate.g4_units("cm")
    cm3 = gate.g4_units("cm3")
    keV = gate.g4_units("keV")
    mm = gate.g4_units("mm")
    Bq = gate.g4_units("Bq")
    BqmL = Bq / cm3

    # main parameters
    ui = sim.user_info
    ui.check_volumes_overlap = True
    ui.random_seed = 4123456
    ac = 1e5 * BqmL
    # ac = 3e3 * BqmL / ui.number_of_threads
    ui.visu = False
    # ui.running_verbose_level = gate.EVENT
    # ui.g4_verbose = True

    # world size
    world = sim.world
    world.size = [1.5 * m, 1.5 * m, 1.5 * m]
    world.material = "G4_AIR"

    # test phase space to check with reference
    phase_space_sphere = sim.add_volume("Sphere", "phase_space_sphere")
    phase_space_sphere.rmin = 212 * mm
    phase_space_sphere.rmax = 213 * mm
    phase_space_sphere.color = [1, 1, 1, 1]
    phase_space_sphere.material = "G4_AIR"

    # spect head
    distance = 30 * cm
    psd = 6.11 * cm
    p = [0, 0, -(distance + psd)]
    spect1, crystal = gate_spect.add_ge_nm67_spect_head(
        sim, "spect1", collimator_type=colli, debug=ui.visu
    )
    spect1.translation, spect1.rotation = gate.get_transform_orbiting(p, "x", 180)

    # physic list
    # sim.set_production_cut("world", "all", 1 * mm)

    # activity parameters
    spheres_diam = [10, 13, 17, 22, 28, 37]
    spheres_activity_concentration = [ac * 6, ac * 5, ac * 4, ac * 3, ac * 2, ac]

    # initialisation for conditional
    spheres_radius = [x / 2.0 for x in spheres_diam]
    (
        spheres_centers,
        spheres_volumes,
    ) = gate_iec.get_default_sphere_centers_and_volumes()
    spheres_activity_ratio = []
    spheres_activity = []
    for diam, ac, volume, center in zip(
        spheres_diam, spheres_activity_concentration, spheres_volumes, spheres_centers
    ):
        activity = ac * volume
        print(
            f"Sphere {diam}: {str(center):<30} {volume / cm3:7.3f} cm3 "
            f"{activity / Bq:7.0f} Bq  {ac / BqmL:7.1f} BqmL"
        )
        spheres_activity.append(activity)

    total_activity = sum(spheres_activity)
    print(f"Total activity {total_activity / Bq:.0f} Bq")
    for activity in spheres_activity:
        spheres_activity_ratio.append(activity / total_activity)
    print("Activity ratio ", spheres_activity_ratio, sum(spheres_activity_ratio))

    # unique (reproducible) random generator
    rs = gate.get_rnd_seed(123456)

    class GANTest:
        def __init__(self):
            # will store all conditional info (position, direction)
            # (not needed, only for test)
            self.all_cond = None

        def __getstate__(self):
            print("getstate GANTest")
            for v in self.__dict__:
                print("state", v)
            self.all_cond = None
            return {}  # self.__dict__

        def generate_condition(self, n):
            n_samples = gate_iec.get_n_samples_from_ratio(n, spheres_activity_ratio)
            cond = gate_iec.generate_pos_dir_spheres(
                spheres_centers, spheres_radius, n_samples, shuffle=True, rs=rs
            )

            if self.all_cond is None:
                self.all_cond = cond
            else:
                self.all_cond = np.column_stack((self.all_cond, cond))

            return cond

    # GAN source
    gsource = sim.add_source("GANSource", "gaga")
    gsource.particle = "gamma"
    # no phantom, we consider attached to the world at origin
    # gsource.mother = f'{iec_phantom.name}_interior'
    gsource.activity = total_activity
    gsource.pth_filename = paths.gate / "pth2" / "test001_GP_0GP_10_50000.pth"
    gsource.position_keys = ["PrePosition_X", "PrePosition_Y", "PrePosition_Z"]
    gsource.backward_distance = 5 * cm
    gsource.direction_keys = ["PreDirection_X", "PreDirection_Y", "PreDirection_Z"]
    gsource.energy_key = "KineticEnergy"
    # gsource.energy_threshold = 0.001 * keV
    gsource.energy_min_threshold = 10 * keV
    # gsource.skip_policy = "SkipEvents"
    # SkipEvents is a bit faster than Energy zero,
    # but it changes the nb of events,so force ZeroEnergy
    gsource.skip_policy = "ZeroEnergy"
    gsource.weight_key = None
    gsource.time_key = "TimeFromBeginOfEvent"
    gsource.relative_timing = True
    gsource.batch_size = 1e5
    gsource.verbose_generator = True

    # GANSourceConditionalGenerator manages the conditional GAN
    # GANTest manages the generation of the conditions, we use a class here to store the total
    # list of conditions (only needed for the test)
    condition_generator = GANTest()
    gen = gate.GANSourceConditionalGenerator(
        gsource, condition_generator.generate_condition
    )
    gsource.generator = gen

    # it is possible to use acceptance angle. Not done here to check exiting phsp
    # gsource.direction.acceptance_angle.volumes = [spect1.name]
    # gsource.direction.acceptance_angle.intersection_flag = True

    # add stat actor
    stat = sim.add_actor("SimulationStatisticsActor", "Stats")
    stat.output = paths.output / "test038_gan_stats.txt"

    # add default digitizer (it is easy to change parameters if needed)
    gate_spect.add_simplified_digitizer_Tc99m(
        sim, "spect1_crystal", paths.output / "test038_gan_proj.mhd"
    )
    # gate_spect.add_ge_nm670_spect_simplified_digitizer(sim, 'spect2_crystal', paths.output / 'test033_proj_2.mhd')
    singles_actor = sim.get_actor_user_info(f"Singles_spect1_crystal")
    singles_actor.output = paths.output / "test038_gan_singles.root"

    phsp_actor = sim.add_actor("PhaseSpaceActor", "phsp")
    phsp_actor.mother = phase_space_sphere.name
    phsp_actor.attributes = [
        "KineticEnergy",
        "PrePosition",
        "PostPosition",
        "PreDirection",
        "PostDirection",
        "GlobalTime",
        "EventPosition",
        "EventDirection",
        "EventKineticEnergy",
        "TimeFromBeginOfEvent"
    ]
    phsp_actor.output = paths.output / "test038_gan_phsp.root"

    return condition_generator


if __name__=='__main__':
    paths = Box()
    paths.current = pathlib.Path(__file__).parent.resolve()

    paths.gate = pathlib.Path("/export/home/tkaprelian/opengate_venv/lib/python3.9/site-packages/opengate/tests/data/gate/gate_test038_gan_phsp_spect")
    paths.output = paths.current / "output"

    # create the simulation
    sim = gate.Simulation()
    condition_generator = create_simulation(sim, paths, None)

    # change output names
    stat = sim.get_actor_user_info("Stats")
    stat.output = paths.output / "test038_gan_aa_stats.txt"
    proj = sim.get_actor_user_info("Projection_spect1_crystal")
    proj.output = paths.output / "test038_gan_aa_proj.mhd"
    singles = sim.get_actor_user_info("Singles_spect1_crystal")
    singles.output = paths.output / "test038_gan_aa_singles.root"

    # go (cannot be spawn in another process)
    output = sim.start(False)



    print()
    stats = output.get_actor("Stats")
    print(stats)



    # Recons
    spect_recon.go_one(file_input=os.path.join(paths.output , "test038_gan_phsp.root"),
                       n = - 1,
                       output = os.path.join(paths.output , "recons.mhd"),
                       output_folder=None,
                       shuffle=False)

