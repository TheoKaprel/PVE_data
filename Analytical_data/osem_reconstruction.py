import click
import itk
from itk import RTK as rtk
import numpy as np
import os

from forwardprojection import alphapve_default,sigma0pve_default

print(alphapve_default)
print(sigma0pve_default)

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--input', '-i', help = 'input projections')
@click.option('--outputfilename', '-o', help = 'Output filename of desired type (mhd/mha)')
@click.option('--start')
@click.option('--like')
@click.option('--size', type = int)
@click.option('--spacing', type = float)
@click.option('--data_folder', help = 'Location of the folder containing : geom_120.xml and acf_ct_air.mhd')
@click.option('--geom', '-g')
@click.option('--attenuationmap', '-a')
@click.option('--beta', type = float, default = 0, show_default = True)
@click.option('--pvc', is_flag = True, default = False, help = 'if --pvc, resolution correction')
@click.option('--nprojpersubset', type = int, default = 10, show_default = True)
@click.option('-n','--niterations', type = int, default = 5, show_default = True)
@click.option('--FB', 'projector_type', default = "Zeng", show_default = True)
@click.option('--output-every', type = int)
@click.option('--iteration-filename', help = 'If output-every is not null, iteration-filename to output intermediate iterations with %d as a placeholder for iteration number')
def osem_reconstruction_click(input,start, outputfilename,like,size,spacing, data_folder, geom,attenuationmap,beta, pvc, nprojpersubset, niterations, projector_type, output_every, iteration_filename):
    osem_reconstruction(input=input,start=start, outputfilename=outputfilename,like=like,size=size,spacing=spacing, data_folder=data_folder, geom=geom,attenuationmap=attenuationmap,
                        beta= beta, pvc=pvc, nprojpersubset=nprojpersubset, niterations=niterations, projector_type=projector_type, output_every=output_every, iteration_filename=iteration_filename)

def osem_reconstruction(input,start, outputfilename,like,size,spacing, data_folder, geom,attenuationmap,beta, pvc, nprojpersubset, niterations, projector_type, output_every, iteration_filename):
    print('Begining of reconstruction ...')

    Dimension = 3
    pixelType = itk.F
    imageType = itk.Image[pixelType, Dimension]

    print('Creating the first output image...')
    if start:
        output_image = itk.imread(start)
    elif (size and spacing):
        output_array = np.ones((size,size,size))
        output_image = itk.image_from_array(output_array)
        output_image.SetSpacing([spacing, spacing, spacing])
        offset = (-size * spacing + spacing) / 2
        output_image.SetOrigin([offset, offset, offset])
        output_image = output_image.astype(pixelType)

    else:
        like_image = itk.imread(like, pixelType)
        constant_image = rtk.ConstantImageSource[imageType].New()
        constant_image.SetSpacing(like_image.GetSpacing())
        constant_image.SetOrigin(like_image.GetOrigin())
        constant_image.SetSize(itk.size(like_image))
        constant_image.SetConstant(1)
        output_image = constant_image.GetOutput()

    print('Reading input projections...')
    projections = itk.imread(input, pixelType)
    nproj = itk.size(projections)[2]
    print(f'{nproj} projections')

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
        att_corr = True
    elif (attenuationmap and not (data_folder)):
        attmap_filename = attenuationmap
        att_corr = True
    else:
        att_corr = False
        print('no att map but ok')

    if att_corr:
        attenuation_map = itk.imread(attmap_filename, pixelType)


    print('Set OSEM parameters ...')
    OSEMType = rtk.OSEMConeBeamReconstructionFilter[imageType, imageType]
    osem = OSEMType.New()
    osem.SetInput(0, output_image)
    osem.SetInput(1, projections)

    osem.SetGeometry(geometry)

    osem.SetNumberOfIterations(niterations)
    osem.SetNumberOfProjectionsPerSubset(nprojpersubset)

    osem.SetBetaRegularization(beta)

    if att_corr:
        osem.SetInput(2, attenuation_map)

    if (projector_type=='Zeng'):
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



    global iter
    iter = 0
    def callback():
        global iter
        iter+=1
        if iter%output_every==0:
            output_iter = osem.GetOutput()
            itk.imwrite(output_iter, iteration_filename.replace('%d', str(iter)))
            print(f'end of iteration {iter}')

    if output_every:
        if iteration_filename:
            try:
                assert ('%d' in iteration_filename)
            except:
                print(f'Error in iteration filename {iteration_filename}. Should contain a %d to be replaced by the iteration number')
                exit(0)
        else:
            iteration_filename = outputfilename.replace('.mh', '_%d.mh')
        osem.AddObserver(itk.IterationEvent(), callback)


    print('Reconstruction ...')
    osem.Update()

    # Writer
    print("Writing output image...")
    itk.imwrite(osem.GetOutput(), outputfilename)

    print('Done!')





if __name__ =='__main__':

    osem_reconstruction_click()
