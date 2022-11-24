import itk
import numpy as np
import click
import os
import random
import string
from itk import RTK as rtk
import time
import json

from parameters import sigma0pve_default, alphapve_default


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.command(context_settings = CONTEXT_SETTINGS)
@click.option('--nb_data','-n', type = int, required = True, help = 'number of desired data = (src,projPVE,projPVfree)')
@click.option('--size_volume', type = int, default = 128, help = 'Size of the desired image i.e. number of voxels per dim', show_default=True)
@click.option('--spacing_volume', type = float, default = 4, help = 'Spacing of the desired image i.e phyisical length of a voxels (mm)', show_default=True)
@click.option('--size_proj', type = int, default = 128, help = 'Size of the desired projections', show_default=True)
@click.option('--spacing_proj', type = float, default = 4.41806, help = 'Spacing of the desired projection', show_default=True)
@click.option('--type', default = 'mha', show_default = True, help = "Create mha or mhd image")
@click.option('--like', default = None, help = "Instead of specifying spacing/size, you can specify an image as a metadata model", show_default=True)
@click.option('--min_radius', default = 4, help = 'minimum radius of the random spheres', show_default = True)
@click.option('--max_radius', default = 32, help = 'max radius of the random spheres', show_default = True)
@click.option('--max_activity', default = 1, help = 'max activity in spheres', show_default = True)
@click.option('--nspheres', default = 1, help = 'max number of spheres to generate on each source', show_default= True)
@click.option('--background', default = None, help = 'If you want background activity specify the maximal activity:background ratio. For example --background 10 for a maximum 1/10 background activity.')
@click.option('--ellipse', is_flag = True, default= False, help = "if --ellipse, activity spheres are in fact ellipses")
@click.option('--geom', '-g', default = None, help = 'geometry file to forward project. Default is the proj on one detector')
@click.option('--attenuationmap', '-a',default = None, help = 'path to the attenuation map file')
@click.option('--output_folder','-o', default = './dataset', help = " Absolute or relative path to the output folder", show_default=True)
@click.option('--sigma0pve', default = sigma0pve_default,type = float, help = 'sigma at distance 0 of the detector', show_default=True)
@click.option('--alphapve', default = alphapve_default, type = float, help = 'Slope of the PSF against the detector distance', show_default=True)
@click.option('--save_src', is_flag = True, default = False, help = "if you want to also save the source that will be forward projected")
@click.option('--noise', is_flag = True, default = False, help = "Add Poisson noise ONLY to ProjPVE")
def generate(nb_data, output_folder,size_volume, spacing_volume,size_proj,spacing_proj, type,  like,min_radius, max_radius,max_activity, nspheres,background,ellipse, geom,attenuationmap, sigma0pve, alphapve, save_src, noise):
    dataset_infos = {}
    current_date = time.strftime("%d_%m_%Y_%Hh_%Mm_%Ss", time.localtime())
    dataset_infos['date'] = current_date
    dataset_infos['nb_data']= nb_data
    dataset_infos['output_folder']=output_folder
    dataset_infos['size_volume']=size_volume
    dataset_infos['spacing_volume']=spacing_volume
    dataset_infos['size_proj']=size_proj
    dataset_infos['spacing_proj']=spacing_proj
    dataset_infos['like']=like
    dataset_infos['type']=type
    dataset_infos['min_radius'] = min_radius
    dataset_infos['max_radius'] = max_radius
    dataset_infos['max_activity'] = max_activity
    dataset_infos['nspheres'] = nspheres
    dataset_infos['background'] = background
    dataset_infos['ellipse'] = ellipse
    dataset_infos['geom'] = geom
    dataset_infos['attenuationmap'] = attenuationmap
    dataset_infos['sigma0pve'] = sigma0pve
    dataset_infos['alphapve'] = alphapve
    dataset_infos['save_src'] = save_src
    dataset_infos['noise'] = noise
    print(json.dumps(dataset_infos, indent = 3))

    t0 = time.time()

    # get output image parameters
    if like is not None:
        im_like = itk.imread(like)
        vSpacing = np.array(im_like.GetSpacing())
        vSize = np.array(itk.size(im_like))
        vOffset = np.array(im_like.GetOrigin())
    else:
        vSize = np.array([size_volume,size_volume,size_volume])
        vSpacing = np.array([spacing_volume,spacing_volume,spacing_volume])
        offset = (-spacing_volume*size_volume + spacing_volume)/2
        vOffset = np.array([offset,offset,offset])

    # matrix settings
    lengths = vSize*vSpacing
    lspaceX = np.linspace(-vSize[0] * vSpacing[0] / 2, vSize[0] * vSpacing[0] / 2, vSize[0])+vSpacing[0] / 2
    lspaceY = np.linspace(-vSize[1] * vSpacing[1] / 2, vSize[1] * vSpacing[1] / 2, vSize[1])+vSpacing[1] / 2
    lspaceZ = np.linspace(-vSize[2] * vSpacing[2] / 2, vSize[2] * vSpacing[2] / 2, vSize[2])+vSpacing[2] / 2
    X, Y, Z = np.meshgrid(lspaceX,lspaceY,lspaceZ)


    # Prepare Forward Projection
    xmlReader = rtk.ThreeDCircularProjectionGeometryXMLFileReader.New()
    xmlReader.SetFilename(geom)
    xmlReader.GenerateOutputInformation()
    geometry = xmlReader.GetOutputObject()
    nproj = len(geometry.GetGantryAngles())


    pixelType = itk.F
    imageType = itk.Image[pixelType, 3]
    output_spacing = [spacing_proj,spacing_proj, 1]
    offset = (-spacing_proj * size_proj + spacing_proj) / 2
    output_offset = [offset, offset, (-nproj+1)/2]
    output_image = rtk.ConstantImageSource[imageType].New()
    output_image.SetSpacing(output_spacing)
    output_image.SetOrigin(output_offset)
    output_image.SetSize([size_proj, size_proj, nproj])
    output_image.SetConstant(0.)

    forward_projector_PVfree = rtk.ZengForwardProjectionImageFilter.New()
    forward_projector_PVfree.SetInput(0, output_image.GetOutput())

    forward_projector_PVfree.SetGeometry(geometry)
    forward_projector_PVfree.SetSigmaZero(0)
    forward_projector_PVfree.SetAlpha(0)

    forward_projector_PVE = rtk.ZengForwardProjectionImageFilter.New()
    forward_projector_PVE.SetInput(0, output_image.GetOutput())

    forward_projector_PVE.SetGeometry(geometry)
    forward_projector_PVE.SetSigmaZero(sigma0pve)
    forward_projector_PVE.SetAlpha(alphapve)

    if attenuationmap is not None:
        attenuation_image = itk.imread(attenuationmap, itk.F)
        forward_projector_PVfree.SetInput(2, attenuation_image)
        forward_projector_PVE.SetInput(2, attenuation_image)

    if background is not None:
        background_radius_min,background_radius_max = 160,240
        print(f'Background radius between {background_radius_min} and {background_radius_max} mm')

    total_counts_in_proj_min,total_counts_in_proj_max = 2e4,1e5
    print(f'Total counts in projections between {total_counts_in_proj_min} and {total_counts_in_proj_max}')

    for n in range(nb_data):
        src_array = np.zeros_like(X)

        if background:
            bg_center = np.random.randint(-50,50,3)
            bg_radius = np.random.randint(background_radius_min, background_radius_max, 3)
            bg_level = round(np.random.rand(),3)*1/float(background)

            src_array += (bg_level) * ((((X - bg_center[0]) / bg_radius[0]) ** 2 + ((Y - bg_center[1]) / bg_radius[1]) ** 2 + (
                        (Z - bg_center[2]) / bg_radius[2]) ** 2) < 1).astype(float)


        random_nb_of_sphers = np.random.randint(1,nspheres+1)

        for s  in range(random_nb_of_sphers):
            random_activity = round(np.random.rand(),3)*(max_activity-1)+1

            # random radius and center
            if ellipse:
                radius = np.random.rand(3)*(max_radius-min_radius) + min_radius
            else:
                radius = np.random.rand()*(max_radius-min_radius) + min_radius
                radius = [radius, radius, radius]

            center = (2*np.random.rand(3)-1)*(lengths/2-np.max(radius)) # the sphere borders remain inside the phantom
            src_array += random_activity  * ( ( ((X-center[0]) / radius[0]) ** 2 + ((Y-center[1]) / radius[1]) ** 2 + ((Z-center[2])/ radius[2]) ** 2  ) < 1).astype(float)

        total_counts_in_proj = np.random.randint(total_counts_in_proj_min,total_counts_in_proj_max)
        src_array_normedToTotalCounts = src_array / np.sum(src_array) * total_counts_in_proj * spacing_proj**2 / spacing_volume**3

        src_img_normedToTotalCounts = itk.image_from_array(src_array_normedToTotalCounts.astype(np.float32))
        src_img_normedToTotalCounts.SetSpacing(vSpacing)
        src_img_normedToTotalCounts.SetOrigin(vOffset)

        # Random output filename
        letters = string.ascii_uppercase
        filenamelength = 5
        source_ref = ''.join(random.choice(letters) for i in range(filenamelength))


        # saving of source 3D image
        if save_src:
            src_img = itk.image_from_array(src_array.astype(np.float32))
            src_img.SetSpacing(vSpacing)
            src_img.SetOrigin(vOffset)
            source_path = os.path.join(output_folder,f'{source_ref}.{type}')
            itk.imwrite(src_img,source_path)


        #compute fowardprojections :
        print(source_ref)

        # proj PVfree
        forward_projector_PVfree.SetInput(1, src_img_normedToTotalCounts)
        forward_projector_PVfree.Update()
        output_forward_PVfree = forward_projector_PVfree.GetOutput()
        output_filename_PVfree = os.path.join(output_folder,f'{source_ref}_PVfree.{type}')
        itk.imwrite(output_forward_PVfree,output_filename_PVfree)

        # proj PVE
        forward_projector_PVE.SetInput(1, src_img_normedToTotalCounts)
        forward_projector_PVE.Update()
        output_forward_PVE = forward_projector_PVE.GetOutput()
        output_filename_PVE = os.path.join(output_folder,f'{source_ref}_PVE.{type}')
        itk.imwrite(output_forward_PVE,output_filename_PVE)


        if noise:
            output_forward_PVE_array = itk.array_from_image(output_forward_PVE)
            noisy_projection_array = np.random.poisson(lam=output_forward_PVE_array, size=output_forward_PVE_array.shape).astype(float)
            output_forward_PVE_noisy = itk.image_from_array(noisy_projection_array)
            output_forward_PVE_noisy.SetSpacing(output_forward_PVE.GetSpacing())
            output_forward_PVE_noisy.SetOrigin(output_forward_PVE.GetOrigin())
            output_filename_PVE_noisy = os.path.join(output_folder, f'{source_ref}_PVE_noisy.{type}')
            itk.imwrite(output_forward_PVE_noisy, output_filename_PVE_noisy)


    tf = time.time()
    elapsed_time = round(tf - t0)
    elapsed_time_min = round(elapsed_time/60)
    dataset_infos['elapsed_time_s'] = elapsed_time
    print(f'Total time elapsed for data generation : {elapsed_time_min} min    (i.e. {elapsed_time} s)')

    formatted_dataset_infos = json.dumps(dataset_infos, indent=4)
    output_info_json = os.path.join(output_folder, 'dataset_infos.json')
    jsonfile = open(output_info_json, "w")
    jsonfile.write(formatted_dataset_infos)
    jsonfile.close()


if __name__ == '__main__':
    generate()
