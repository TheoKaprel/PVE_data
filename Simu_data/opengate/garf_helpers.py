import opengate as gate

nm = gate.g4_units('nm')
mm = gate.g4_units('mm')
cm = gate.g4_units('cm')



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