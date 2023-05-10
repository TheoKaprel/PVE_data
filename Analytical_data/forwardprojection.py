#!/usr/bin/env python3


"""
Forward projection of an input 3D acitvity map


Can be used either as a function callable from command line or as a module


"""

import click
import os
import itk
import numpy as np
from itk import RTK as rtk

try:
    from .parameters import get_psf_params
except:
    from parameters import get_psf_params



CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings = CONTEXT_SETTINGS)
@click.option('--inputsrc', '-i', help = 'path to the input 3D image to forward project')
@click.option('--output_folder', '-o', help = 'output folder.')
@click.option('--geom', '-g', 'geometry_filename', default = None, help = 'If the geometry file you want to use is already created, precise here the path to the xml file')
@click.option('--nproj', type=int, default = None, help = 'if not geom: Precise the number of projections needed (ex 120)')
@click.option('--sid', type=float, default = None, help = 'if not geom: Precise the detector-to-isocenter distance (mm)')
@click.option('--attmap', '-a', help = 'Path to the attenuation map if the default is not ok)')
@click.option('--pve',is_flag = True, default = False, help = 'To project the input source without partial volume effect')
@click.option('--pvfree', is_flag = True, default = False, help = 'To project the input source without partial volume effect')
@click.option('--spect_system', default = "ge-discovery",type = str, help = 'spect system for psf', show_default=True)
@click.option('--size', default = 128, show_default = True)
@click.option('--spacing', default = 4.41806, show_default = True)
@click.option('--type', default = 'mhd', show_default = True)
@click.option('--noise', is_flag = True, default = False, help= 'Apply poisson noise to the projection')
@click.option('--save_src', is_flag = True, default = False, help= 'to save the scaled source (the one that is actually projected)')
@click.option('--total_count', default = 5e5, show_default = True)
@click.option('--output_ref', default = None, type = str, help = 'ref to append to output_filename')
def forwardproject_click(inputsrc, output_folder,geometry_filename,attmap, nproj, sid,pve, pvfree, spect_system, size,spacing, type,noise, save_src,total_count, output_ref):

    forwardprojectRTK(inputsrc=inputsrc, output_folder=output_folder,geometry_filename=geometry_filename,attmap=attmap,
                      nproj=nproj,sid = sid, pve=pve, pvfree=pvfree,size=size,spacing=spacing, type = type,save_src=save_src,
                      spect_system=spect_system, noise=noise,total_count=total_count, output_ref=output_ref)



def forwardprojectRTK(inputsrc, output_folder,geometry_filename,attmap, nproj, sid,pve, pvfree, size,spacing,type,total_count, spect_system,save_src=False, noise=False, output_ref=None):
    # projection parameters
    offset = (-spacing*size + spacing)/2

    if ((geometry_filename is not None) and (nproj is None) and (sid is None)):
        xmlReader = rtk.ThreeDCircularProjectionGeometryXMLFileReader.New()
        xmlReader.SetFilename(geometry_filename)
        xmlReader.GenerateOutputInformation()
        geometry = xmlReader.GetOutputObject()
        nproj = len(geometry.GetGantryAngles())
    elif ((geometry_filename is None) and (nproj is not None) and (sid is not None)):
        list_angles = np.linspace(0,360,nproj+1)
        geometry = rtk.ThreeDCircularProjectionGeometry.New()
        for i in range(nproj):
            geometry.AddProjection(sid, 0, list_angles[i], offset, offset)
    else:
        print('ERROR: give me geom xor (nproj and sid)')
        exit(0)

    pixelType = itk.F
    imageType = itk.Image[pixelType, 3]

    source_image= itk.imread(inputsrc, itk.F)
    source_array = itk.array_from_image(source_image)
    source_array_act = source_array / np.sum(source_array) * float(total_count) * spacing ** 2 / (source_image.GetSpacing()[0] ** 3)
    source_image_act = itk.image_from_array(source_array_act).astype(itk.F)
    source_image_act.SetOrigin(source_image.GetOrigin())
    source_image_act.SetSpacing(source_image.GetSpacing())

    if save_src:
        itk.imwrite(source_image_act, inputsrc.replace(".mhd", "_scaled.mhd"))


    # readerType = itk.ImageFileReader[imageType]
    # source_image_reader = readerType.New()
    # source_image_reader.SetFileName(inputsrc)
    # source_image_reader.Update()







    output_spacing = [spacing,spacing, 1]
    offset = (-spacing * size + spacing) / 2
    output_offset = [offset, offset, (-nproj+1)/2]

    output_image = rtk.ConstantImageSource[imageType].New()
    output_image.SetSpacing(output_spacing)
    output_image.SetOrigin(output_offset)
    output_image.SetSize([size, size, nproj])
    output_image.SetConstant(0.)

    forward_projector = rtk.ZengForwardProjectionImageFilter.New()
    forward_projector.SetInput(0, output_image.GetOutput())
    forward_projector.SetInput(1, source_image_act)
    if attmap:
        attenuation_image = itk.imread(attmap, itk.F)
        forward_projector.SetInput(2, attenuation_image)

    forward_projector.SetGeometry(geometry)

    if output_ref is None:
        _,output_ref = os.path.split(inputsrc)
        output_ref = output_ref.replace(".mhd", "").replace(".mha", "")


    if pvfree:
        forward_projector.SetSigmaZero(0)
        forward_projector.SetAlpha(0)
        forward_projector.Update()

        output_forward_PVfree = forward_projector.GetOutput()

        output_filename_PVfree = os.path.join(output_folder,f'{output_ref}_PVfree.{type}')
        itk.imwrite(output_forward_PVfree,output_filename_PVfree)

    if pve:
        sigma0_psf, alpha_psf = get_psf_params(machine=spect_system)
        forward_projector.SetSigmaZero(sigma0_psf)
        forward_projector.SetAlpha(alpha_psf)
        forward_projector.Update()

        output_forward_PVE = forward_projector.GetOutput()
        output_filename_PVE = os.path.join(output_folder,f'{output_ref}_PVE.{type}')
        itk.imwrite(output_forward_PVE,output_filename_PVE)

        if noise:
            output_forward_PVE_array = itk.array_from_image(output_forward_PVE)
            noisy_projection_array = np.random.poisson(lam=output_forward_PVE_array, size=output_forward_PVE_array.shape).astype(float)
            output_forward_PVE_noisy = itk.image_from_array(noisy_projection_array)
            output_forward_PVE_noisy.SetSpacing(output_forward_PVE.GetSpacing())
            output_forward_PVE_noisy.SetOrigin(output_forward_PVE.GetOrigin())
            output_filename_PVE_noisy = os.path.join(output_folder, f'{output_ref}_PVE_noisy.{type}')
            itk.imwrite(output_forward_PVE_noisy, output_filename_PVE_noisy)




if __name__ == '__main__':
    forwardproject_click()


