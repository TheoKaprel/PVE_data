import click
import itk
from itk import RTK as rtk
import numpy as np
import os

from forwardprojection import alphapve_default,sigma0pve_default


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--input', '-i', help = 'input projections')
@click.option('--outputfilename', '-o')
@click.option('--like')
@click.option('--data_folder', help = 'Location of the folder containing : geom_120.xml and acf_ct_air.mhd')
@click.option('--geom', '-g')
@click.option('--attenuationmap', '-a')
@click.option('--beta', type = float, default = 0, show_default = True)
@click.option('--pvc', is_flag = True, default = False, help = 'if --pvc, resolution correction')
@click.option('--nprojpersubset', type = int, default = 10, show_default = True)
@click.option('--niterations', type = int, default = 5, show_default = True)
@click.option('--FB', 'projector_type', default = "Zeng")
def osem_reconstruction_click(input, outputfilename,like, data_folder, geom,attenuationmap,beta, pvc, nprojpersubset, niterations, projector_type):
    osem_reconstruction(input=input, outputfilename=outputfilename,like=like, data_folder=data_folder, geom=geom,attenuationmap=attenuationmap,
                        beta= beta, pvc=pvc, nprojpersubset=nprojpersubset, niterations=niterations, projector_type=projector_type)

def osem_reconstruction(input, outputfilename,like, data_folder, geom,attenuationmap,beta, pvc, nprojpersubset, niterations, projector_type):
    print('Begining of reconstruction ...')

    Dimension = 3
    pixelType = itk.F
    imageType = itk.Image[pixelType, Dimension]


    print('Creating output image...')
    like_image = itk.imread(like, pixelType)
    output_image = rtk.ConstantImageSource[imageType].New()
    output_image.SetSpacing(like_image.GetSpacing())
    output_image.SetOrigin(like_image.GetOrigin())
    output_image.SetSize(itk.size(like_image))
    output_image.SetConstant(1)

    print('Reading input projections...')
    projections = itk.imread(input, pixelType)
    nproj = itk.size(projections)[2]
    print(f'{nproj} projections detected')

    print('Reading geometry file ...')
    if (data_folder or geom):
        if (data_folder and not(geom)):
            geom_filename = os.path.join(data_folder, f'geom_{nproj}.xml')
        elif (geom and not (data_folder)):
            geom_filename = geom
        else:
            print('Error in geometry arguments')
            exit(0)
        xmlReader = rtk.ThreeDCircularProjectionGeometryXMLFileReader.New()
        xmlReader.SetFilename(geom_filename)
        xmlReader.GenerateOutputInformation()
        geometry = xmlReader.GetOutputObject()
        print(geom_filename + ' is opened!')
    else:
        if projector_type=="Zeng":
            Offset = projections.GetOrigin()
        else:
            Offset = [0,0]

        list_angles = np.linspace(0,360,nproj+1)
        geometry = rtk.ThreeDCircularProjectionGeometry.New()
        for i in range(nproj):
            geometry.AddProjection(380, 0, list_angles[i], Offset[0], Offset[1])
        print(f'Created geom file with {nproj} angles and Offset = {Offset[0]},{Offset[1]}')





    print('Reading attenuation map ...')
    if (data_folder and not(attenuationmap)):
        attmap_filename = os.path.join(data_folder, f'acf_ct_air.mhd')
    elif (attenuationmap and not (data_folder)):
        attmap_filename = attenuationmap
    else:
        print('Error in attenuationmap arguments')
        exit(0)

    attenuation_map = itk.imread(attmap_filename, pixelType)

    print('Set OSEM parameters ...')
    OSEMType = rtk.OSEMConeBeamReconstructionFilter[imageType, imageType]
    osem = OSEMType.New()
    osem.SetInput(0, output_image.GetOutput())
    osem.SetInput(1, projections)

    osem.SetGeometry(geometry)

    osem.SetNumberOfIterations(niterations)
    osem.SetNumberOfProjectionsPerSubset(nprojpersubset)

    osem.SetBetaRegularization(beta)

    if (projector_type=='Zeng'):
        osem.SetInput(2, attenuation_map)

        FP = osem.ForwardProjectionType_FP_ZENG
        BP = osem.BackProjectionType_BP_ZENG

        if pvc:
            osem.SetSigmaZero(sigma0pve_default)
            osem.SetAlpha(alphapve_default)
        else:
            osem.SetSigmaZero(0)
            osem.SetAlpha(0)
    elif projector_type=='Joseph':
        FP = osem.ForwardProjectionType_FP_JOSEPH
        BP = osem.BackProjectionType_BP_JOSEPH


    osem.SetForwardProjectionFilter(FP)
    osem.SetBackProjectionFilter(BP)



    print('Reconstruction ...')
    osem.Update()

    # Writer
    print("Writing output image...")
    itk.imwrite(osem.GetOutput(), outputfilename)

    print('Done!')





if __name__ =='__main__':

    osem_reconstruction_click()
