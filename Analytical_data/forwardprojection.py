

"""
Forward projection of an input 3D acitvity map


Can be used either as a function callable from command line or as a module


"""

import click
import os
import itk
import numpy as np
import matplotlib.pyplot as plt
import subprocess

sigma0pve_default = 0.9008418065898374
alphapve_default = 0.025745123547513887

def get_filename(file_path):
    return file_path[file_path.rfind('/')+1:][:-4]

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings = CONTEXT_SETTINGS)
@click.option('--inputsrc', '-i', help = 'path to the input 3D image to forward project')
@click.option('--output_folder', '-o', help = 'output folder. The output files will be ${inputsrc}_PVE.mhd and ${inputsrc}_PVfree.mhd')
@click.option('--geom', '-g', default = None, help = 'If the geometry file you want to use is already created, precise here the path to the xml file')
@click.option('--nproj',type=int, default = None, help = 'Precise the number of projections needed')
@click.option('--pve',is_flag = True, default = False, help = 'To project the input source without partial volume effect')
@click.option('--pvfree', is_flag = True, default = False, help = 'To project the input source without partial volume effect')
@click.option('--sigma0pve', default = sigma0pve_default,type = float, help = 'sigma at distance 0 of the detector', show_default=True)
@click.option('--alphapve', default = alphapve_default, type = float, help = 'Slope of the PSF against the detector distance', show_default=True)
@click.option('--output_ref', default = None, type = str, help = 'ref to append to output_filename')
@click.option('--output_name', type=str, help='if no ref, name of the output')
def forwardproject_click(inputsrc, output_folder,geom, nproj,pve, pvfree, sigma0pve, alphapve, output_ref,output_name):
    forwardproject(inputsrc, output_folder,geom, nproj,pve, pvfree, sigma0pve, alphapve, output_ref,output_name)



def forwardproject(inputsrc, output_folder,geom, nproj,pve, pvfree, sigma0pve=sigma0pve_default, alphapve=alphapve_default, output_ref=None,output_name=None):
    # projection parameters
    if nproj and not(geom):
        geom_fn = f'./data/geom_{nproj}.xml'

        create_proj = subprocess.run(['rtksimulatedgeometry', "-o", geom_fn, "-f", "0", "-n", f'{nproj}' , "-a", "360", "--sdd", "0", "--sid", "380", "--proj_iso_x", "-280.54680999999999", "--proj_iso_y", "-280.54680999999999"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if (create_proj.returncode != 0):
            print(f'Error in the creation of the geometry...!')
    elif nproj and geom:
        geom_fn = geom
    elif geom and not(nproj):
        geom_fn = geom
        nproj=1
    else:
        geom_fn = "./data/geom_1.xml"
        nproj = 1


    projector = "Zeng"
    attenuationmap = "./data/acf_ct_air.mhd"

    if output_name:
        filename = output_name
    else:
        filename = get_filename(inputsrc) # get input filename (without path and extension)

    if output_ref==None:
        output_ref=''


    if pvfree:
        sigma0PVfree = 0
        alphapsfPVfree = 0
        outputPVfree = f'{output_folder}/{filename}{output_ref}_PVfree.mhd'

        # RTK computing of the PVfree forwardprojection
        resPVfree = subprocess.run(['rtkforwardprojections', "-g", geom_fn, "-i", inputsrc, "-o", outputPVfree , "-f", projector, "--attenuationmap", attenuationmap,"--sigmazero", f'{sigma0PVfree}', "--alphapsf", f'{alphapsfPVfree}', "--dimension", f"128,128,{nproj}", "--spacing", "4.41806,4.41806,1"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if (resPVfree.returncode != 0):
            print(f'Error in the PVfree projection for input : {inputsrc}')
            print(resPVfree.stdout)
            print(resPVfree.stderr)
            exit()
    if  pve:

        outputPVE = f'{output_folder}/{filename}{output_ref}_PVE.mhd'

        # RTK computing of the PVE forwardprojection
        resPVE = subprocess.run(['rtkforwardprojections', "-g", geom_fn, "-i", inputsrc, "-o", outputPVE , "-f", projector, "--attenuationmap", attenuationmap,"--sigmazero", f'{sigma0pve}', "--alphapsf", f'{alphapve}', "--dimension",  f"128,128,{nproj}", "--spacing", "4.41806,4.41806,1"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if (resPVE.returncode!=0):
            print(f'Error in the PVE projection for input : {inputsrc}')
            print(resPVE.stdout)
            print(resPVE.stderr)
            exit()



if __name__ == '__main__':
    forwardproject_click()


