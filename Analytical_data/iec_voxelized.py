#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import opengate as gate
import opengate.contrib.phantom_nema_iec_body as gate_iec
import opengate_core as g4
import itk
import json
import click
from pathlib import Path


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--spacing', required = True)
@click.option('-o', '--output', required = True)
def create_iec(spacing, output):
    output = Path(output)

    # create the simulation
    sim = gate.Simulation()

    ui = sim.user_info
    ui.g4_verbose = 1

    # shhhht !
    gate.log.setLevel(gate.NONE)

    # world
    m = gate.g4_units("m")
    sim.world.size = [1 * m, 1 * m, 1 * m]

    # add a iec phantom
    iec = gate_iec.add_phantom(sim)



    # initialize only (no source but no start).
    # initialization is needed because it builds the hierarchy of G4 volumes
    # that are needed by the "voxelize" function
    sim.initialize()

    # create an empty image with the size (extent) of the volume
    # add one pixel margin
    image = gate.create_image_with_volume_extent(sim, iec.name, spacing=[float(spacing), float(spacing), float(spacing)], margin=1)
    info = gate.get_info_from_image(image)
    print(f"Image : {info.size} {info.spacing} {info.origin}")

    # voxelized a volume
    print("Starting voxelization ...")
    labels, image = gate.voxelize_volume(sim, iec.name, image)
    print(f"Output labels: {labels}")

    # materials = ["G4_WATER", "G4_LUNG_ICRP", "G4_AIR", "IEC_PLASTIC", "G4_LEAD_OXIDE"]
    #
    # for mat in materials:
    #     print(gate.dump_material_like_Gate(mat))
    #


    n = g4.G4NistManager.Instance()
    water = n.FindMaterial("G4_WATER")
    print(water)



    # n.PrintG4Material("G4_WATER")

    # write labels
    lf = str(output).replace('.mhd', '.json')
    outfile = open(lf, "w")
    json.dump(labels, outfile, indent=4)

    # write image
    print(f"Write image {output}")
    itk.imwrite(image, str(output))

if __name__ == "__main__":
    create_iec()