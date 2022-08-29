

"""
Forward projection of an input 3D acitvity map


Can be used either as a function callable from command line or as a module


"""

import click
import os
import itk
import numpy as np
from itk import RTK as rtk

from parameters import sigma0pve_default, alphapve_default


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings = CONTEXT_SETTINGS)
@click.option('--inputsrc', '-i', help = 'path to the input 3D image to forward project')
@click.option('--output_folder', '-o', help = 'output folder.')
@click.option('--geom', '-g', 'geometry_filename', default = None, help = 'If the geometry file you want to use is already created, precise here the path to the xml file')
@click.option('--attmap', '-a', default = "./data/acf_ct_air.mhd",help = 'Path to the attenuation map if the default is not ok)')
@click.option('--nproj',type=int, default = None, help = 'Precise the number of projections needed')
@click.option('--pve',is_flag = True, default = False, help = 'To project the input source without partial volume effect')
@click.option('--pvfree', is_flag = True, default = False, help = 'To project the input source without partial volume effect')
@click.option('--sigma0pve', default = sigma0pve_default,type = float, help = 'sigma at distance 0 of the detector', show_default=True)
@click.option('--alphapve', default = alphapve_default, type = float, help = 'Slope of the PSF against the detector distance', show_default=True)
@click.option('--size', default = 128, show_default = True)
@click.option('--spacing', default = 4.41806, show_default = True)
@click.option('--noise', is_flag = True, default = False, help= 'Apply poisson noise to the projection')
@click.option('--output_ref', default = None, type = str, help = 'ref to append to output_filename')
def forwardproject_click(inputsrc, output_folder,geometry_filename,attmap, nproj,pve, pvfree, sigma0pve, alphapve, size,spacing,noise, output_ref):

    forwardprojectRTK(inputsrc=inputsrc, output_folder=output_folder,geometry_filename=geometry_filename,attmap=attmap,
                      nproj=nproj,pve=pve, pvfree=pvfree,size=size,spacing=spacing,
                      sigma0pve=sigma0pve, alphapve=alphapve, noise=noise, output_ref=output_ref)



def forwardprojectRTK(inputsrc, output_folder,geometry_filename,attmap, nproj,pve, pvfree, size,spacing, sigma0pve=sigma0pve_default, alphapve=alphapve_default, noise=False, output_ref=None):
    # projection parameters
    offset = (-spacing*size + spacing)/2

    if (geometry_filename and not nproj):
        xmlReader = rtk.ThreeDCircularProjectionGeometryXMLFileReader.New()
        xmlReader.SetFilename(geometry_filename)
        xmlReader.GenerateOutputInformation()
        geometry = xmlReader.GetOutputObject()
        nproj = len(geometry.GetGantryAngles())
    elif (nproj and not geometry_filename):

        list_angles = np.linspace(0,360,nproj+1)
        geometry = rtk.ThreeDCircularProjectionGeometry.New()
        for i in range(nproj):
            geometry.AddProjection(380, 0, list_angles[i], offset, offset)
        # geom_fn = f'./data/geom_{nproj}.xml'
        # xmlWriter = rtk.ThreeDCircularProjectionGeometryXMLFileWriter.New()
        # xmlWriter.SetFilename(geom_fn)
        # xmlWriter.SetObject(geometry)
        # xmlWriter.WriteFile()
    else:
        print('ERROR: give me geom xor nproj')
        exit(0)


    # source_image= itk.imread(inputsrc, itk.F)
    pixelType = itk.F
    imageType = itk.Image[pixelType, 3]
    readerType = itk.ImageFileReader[imageType]
    source_image_reader = readerType.New()
    source_image_reader.SetFileName(inputsrc)
    source_image_reader.Update()

    attenuation_image = itk.imread(attmap, itk.F)



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
    forward_projector.SetInput(1, source_image_reader.GetOutput())
    forward_projector.SetInput(2, attenuation_image)

    forward_projector.SetGeometry(geometry)

    if pvfree:
        forward_projector.SetSigmaZero(0)
        forward_projector.SetAlpha(0)
        forward_projector.Update()

        output_forward_PVfree = forward_projector.GetOutput()

        output_filename_PVfree = os.path.join(output_folder,f'{output_ref}_PVfree.mha')
        itk.imwrite(output_forward_PVfree,output_filename_PVfree)

    if pve:
        forward_projector.SetSigmaZero(sigma0pve)
        forward_projector.SetAlpha(alphapve)
        forward_projector.Update()

        output_forward_PVE = forward_projector.GetOutput()

        if noise:
            output_forward_PVE_array = itk.array_from_image(output_forward_PVE)
            noisy_projection_array = np.random.poisson(lam=output_forward_PVE_array, size=output_forward_PVE_array.shape).astype(float)
            output_forward_PVE_noisy = itk.image_from_array(noisy_projection_array)
            output_forward_PVE_noisy.SetSpacing(output_forward_PVE.GetSpacing())
            output_forward_PVE_noisy.SetOrigin(output_forward_PVE.GetOrigin())
            output_forward_PVE = output_forward_PVE_noisy

        output_filename_PVE = os.path.join(output_folder,f'{output_ref}_PVE.mha')
        itk.imwrite(output_forward_PVE,output_filename_PVE)





if __name__ == '__main__':
    forwardproject_click()


