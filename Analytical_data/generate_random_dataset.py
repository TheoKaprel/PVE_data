import itk
import numpy as np
import click
import os
import random
import string
from itk import RTK as rtk
import time
import json
from scipy.spatial.transform import Rotation as R


from parameters import sigma0pve_default, alphapve_default


def strParamToArray(str_param):
    array_param = np.array(str_param.split(','))
    array_param = array_param.astype(np.float)
    if len(array_param) == 1:
        array_param = np.array([array_param[0].astype(np.float)] * 3)
    return array_param[::-1]


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.command(context_settings = CONTEXT_SETTINGS)
@click.option('--nb_data','-n', type = int, required = True, help = 'number of desired data = (src,projPVE,projPVfree)')
@click.option('--size_volume', type = str, default = "150", help = 'Size of the desired image i.e. number of voxels per dim', show_default=True)
@click.option('--spacing_volume', type = str, default = "4", help = 'Spacing of the desired image i.e phyisical length of a voxels (mm)', show_default=True)
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
        vSize = strParamToArray(size_volume).astype(int)
        vSpacing = strParamToArray(spacing_volume)
        vOffset = [(-sp*size + sp)/2 for (sp,size) in zip(vSpacing,vSize)]


    # matrix settings
    lengths = vSize*vSpacing
    lspaceX = np.linspace(-vSize[0] * vSpacing[0] / 2, vSize[0] * vSpacing[0] / 2, vSize[0])
    lspaceY = np.linspace(-vSize[1] * vSpacing[1] / 2, vSize[1] * vSpacing[1] / 2, vSize[1])
    lspaceZ = np.linspace(-vSize[2] * vSpacing[2] / 2, vSize[2] * vSpacing[2] / 2, vSize[2])

    X,Y,Z = np.meshgrid(lspaceX,lspaceY,lspaceZ, indexing='ij')

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
        background_radius_x_mean = 200
        background_radius_x_std = 20
        background_radius_y_mean = 120
        background_radius_y_std = 10

    total_counts_in_proj_min,total_counts_in_proj_max = 2e4,1e5
    print(f'Total counts in projections between {total_counts_in_proj_min} and {total_counts_in_proj_max}')

    for n in range(nb_data):
        src_array = np.zeros_like(X)

        if background is not None:
            # backgroun = cylinder with revolution axis = Y
            bg_center = np.random.randint(-50,50,2)
            bg_radius_x =  background_radius_x_std*np.random.randn() + background_radius_x_mean
            bg_radius_y = background_radius_y_std*np.random.randn() + background_radius_y_mean
            rotation = np.random.rand()*2*np.pi
            bg_level = round(np.random.rand(),3)*1/float(background)

            background_array = (bg_level) * (((((X - bg_center[0])*np.cos(rotation) - (Z - bg_center[1])*np.sin(rotation) )/ bg_radius_x) ** 2 +
                                              (((X - bg_center[0])*np.sin(rotation) + (Z - bg_center[1])*np.cos(rotation) ) / bg_radius_y) ** 2) < 1).astype(float)
            src_array += background_array


        random_nb_of_sphers = np.random.randint(1,nspheres+1)

        for s  in range(random_nb_of_sphers):
            random_activity = round(np.random.rand(),3)*(max_activity-1)+1

            if background is None:
                center = (2 * np.random.rand(3) - 1) * (lengths / 2)
            else:
                # center of the sphere inside the background
                center_index = np.random.randint(X.shape)
                while (background_array[center_index[0], center_index[1], center_index[2]]==0):
                    center_index = np.random.randint(X.shape)
                center = [lspaceX[center_index[0]], lspaceY[center_index[1]], lspaceZ[center_index[2]]]


            if ellipse:
                radius = np.random.rand(3)*(max_radius-min_radius) + min_radius
                rotation_angles = np.random.rand(3)*2*np.pi
                rot = R.from_rotvec([[rotation_angles[0], 0, 0], [0, rotation_angles[1], 0], [0, 0, rotation_angles[2]]])
                rotation_matrices = rot.as_matrix()
                rot_matrice = rotation_matrices[0].dot((rotation_matrices[1].dot(rotation_matrices[2])))
                lesion = random_activity * ((  (    (  (X-center[0])*rot_matrice[0,0] + (Y-center[1])*rot_matrice[0,1] + (Z - center[2])*rot_matrice[0,2])**2/(radius[0]**2) +
                                                    ( (X-center[0])*rot_matrice[1,0] + (Y-center[1])*rot_matrice[1,1] + (Z - center[2])*rot_matrice[1,2])**2/(radius[1]**2) +
                                                    ( (X-center[0])*rot_matrice[2,0] + (Y-center[1])*rot_matrice[2,1] + (Z - center[2])*rot_matrice[2,2])**2/(radius[2]**2) < 1)
                                            ).astype(float))
            else:
                radius = np.random.rand()*(max_radius-min_radius) + min_radius
                radius = [radius, radius, radius]
                lesion = random_activity * ((((X - center[0]) / radius[0]) ** 2 + ((Y - center[1]) / radius[1]) ** 2 + ((Z - center[2]) / radius[2]) ** 2) < 1).astype(float)

            src_array += lesion

        total_counts_in_proj = np.random.randint(total_counts_in_proj_min,total_counts_in_proj_max)
        src_array_normedToTotalCounts = src_array / np.sum(src_array) * total_counts_in_proj * spacing_proj**2 / (vSpacing[0]*vSpacing[1]*vSpacing[2])

        src_img_normedToTotalCounts = itk.image_from_array(src_array_normedToTotalCounts.astype(np.float32))
        src_img_normedToTotalCounts.SetSpacing(vSpacing[::-1])
        src_img_normedToTotalCounts.SetOrigin(vOffset)

        # Random output filename
        letters = string.ascii_uppercase
        filenamelength = 5
        source_ref = ''.join(random.choice(letters) for _ in range(filenamelength))


        # saving of source 3D image
        if save_src:
            src_img = itk.image_from_array(src_array.astype(np.float32))
            src_img.SetSpacing(vSpacing[::-1])
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
