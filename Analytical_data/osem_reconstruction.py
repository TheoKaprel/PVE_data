import click
import itk
from itk import RTK as rtk
import os

from forwardprojection import alphapve_default,sigma0pve_default


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--input', '-i', help = 'input projections')
@click.option('--outputfilename', '-o')
@click.option('--like')
@click.option('--data_folder', help = 'Location of the folder containing : geom_60.xml and acf_ct_air.mhd')
@click.option('--pvc', is_flag = True, default = False, help = 'if --pvc, resolution correction')
@click.option('--nprojpersubset', type = int, default = 10)
@click.option('--niterations', type = int, default = 5)
def osem_reconstruction_click(input, outputfilename,like, data_folder,pvc, nprojpersubset, niterations):
    osem_reconstruction(input, outputfilename,like, data_folder,pvc, nprojpersubset, niterations)

def osem_reconstruction(input, outputfilename,like, data_folder,pvc, nprojpersubset, niterations):
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
    geom_filename = os.path.join(data_folder, f'geom_{nproj}.xml')
    xmlReader = rtk.ThreeDCircularProjectionGeometryXMLFileReader.New()
    xmlReader.SetFilename(geom_filename)
    xmlReader.GenerateOutputInformation()
    geometry = xmlReader.GetOutputObject()
    print(geom_filename+ ' is opened!')


    print('Reading attenuation map ...')
    attmap_filename = os.path.join(data_folder, f'acf_ct_air.mhd')
    attenuation_map = itk.imread(attmap_filename, pixelType)

    print('Set OSEM parameters ...')
    OSEMType = rtk.OSEMConeBeamReconstructionFilter[imageType, imageType]
    osem = OSEMType.New()
    osem.SetInput(0, output_image.GetOutput())
    osem.SetInput(1, projections)
    osem.SetInput(2, attenuation_map)
    osem.SetGeometry(geometry)
    if pvc:
        osem.SetSigmaZero(sigma0pve_default)
        osem.SetAlpha(alphapve_default)
    else:
        osem.SetSigmaZero(0)
        osem.SetAlpha(0)
    osem.SetNumberOfIterations(niterations)
    osem.SetNumberOfProjectionsPerSubset(nprojpersubset)
    osem.SetForwardProjectionFilter(osem.ForwardProjectionType_FP_ZENG)
    osem.SetBackProjectionFilter(osem.BackProjectionType_BP_ZENG)

    print('Reconstruction ...')
    osem.Update()

    # Writer
    print("Writing output image...")
    itk.imwrite(osem.GetOutput(), outputfilename)

    print('Done!')





if __name__ =='__main__':

    osem_reconstruction_click()
