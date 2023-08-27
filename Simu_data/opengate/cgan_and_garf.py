#!/usr/bin/env python3

import argparse
import opengate.contrib.phantom_nema_iec_body as gate_iec
import numpy as np
import pathlib
import os
from pathlib import Path
from box import Box
import itk
from garf_helpers import *

def main():
    print(args)
    paths = Box()
    paths.current = pathlib.Path(__file__).parent.resolve()

    paths.gate = pathlib.Path("/export/home/tkaprelian/opengate_venv/lib/python3.9/site-packages/opengate/tests/data/gate/gate_test038_gan_phsp_spect")
    paths.output = paths.current / "output"

    # create the simulation
    sim = gate.Simulation()

    # main parameters
    ui = sim.user_info
    ui.check_volumes_overlap = True
    ui.random_seed = 4123456
    ac = 1e6 * BqmL
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

    p = sim.get_physics_user_info()
    p.physics_list_name = "G4EmStandardPhysics_option4"
    sim.set_cut("world", "all", 1 * km)


    # fake spect head
    spect_length = 19 * cm
    spect_translation = 38 * cm
    SPECThead = sim.add_volume("Box", "SPECThead")
    SPECThead.size = [57.6 * cm, 44.6 * cm, spect_length]
    SPECThead.translation = [0, 0, -spect_translation]
    SPECThead.material = "G4_AIR"
    SPECThead.color = [1, 0, 1, 1]

    detPlane = sim_set_detector_plane(sim, SPECThead.name)

    arf = sim.add_actor("ARFActor", "arf")
    arf.mother = detPlane.name

    arf.output = os.path.join(args.output, "proj_cgan_garf.mhd")
    arf.batch_size = int(float(args.batchsize))
    arf.image_size = [128, 128]
    arf.image_spacing = [4.41806 * mm, 4.41806 * mm]
    arf.verbose_batch = True
    arf.distance_to_crystal = 74.625 * mm
    arf.pth_filename = os.path.join(paths.current , "outputs/arf/arf_5x10_9.pth")


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

    # output_img = arf.output_image
    # output_img_fn = os.path.join(args.output, "cgan_garf_proj.mhd")
    # itk.imwrite(output_img, output_img_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--activity", type=float)
    parser.add_argument("-o", "--output", type=str)
    parser.add_argument("-b", "--batchsize", type=int, default= 100000)
    args = parser.parse_args()

    m = gate.g4_units("m")
    cm = gate.g4_units("cm")
    km = gate.g4_units("km")
    cm3 = gate.g4_units("cm3")
    keV = gate.g4_units("keV")
    mm = gate.g4_units("mm")
    Bq = gate.g4_units("Bq")
    BqmL = Bq / cm3


    main()
