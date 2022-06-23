import itk
import numpy as np
import click
import os
import sys
sys.path.append("/export/home/tkaprelian/Desktop/External_repositories/syd_algo")
from faf_ACF_image import faf_ACF_image


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.command(context_settings = CONTEXT_SETTINGS)
@click.option('--size', type = int, default = 128, help = 'Size of the desired image i.e. number of voxels per dim', show_default=True)
@click.option('--spacing', type = float, default = 4, help = 'Spacing of the desired image i.e phyisical length of a voxels (mm)', show_default=True)
@click.option('--like', default = None, help = "Instead of specifying spacing/size, you can specify a .mhd image as a metadata model")
@click.option('--value', type = float, default = -1024, help = 'Default value to assign pixels for the homogeneous phantom', show_default=True)
@click.option('--output_folder','--of', required = True, help = " Absolute or relative path to the output folder")
@click.option('--output_name','-o', required = True, help = "Name of the ct output will be : ct_{output_name}.mhd/raw")
@click.option('--attmap/--no-attmap', is_flag=True, show_default=True,default = False, help = "If True, compute the attenuation correction factor map. Name of the attenuation map will be : acf_{output_name}.mhd/raw")
def create_homogene_phantom_click(size, spacing, value, like,output_folder, output_name, attmap):
    create_homogene_phantom(size, spacing, value, like,output_folder, output_name,attmap)


def create_homogene_phantom(size, spacing, value, like,output_folder, output_name, attmap):
    if like:
        im_like = itk.imread(like)
        vSpacing = np.array(im_like.GetSpacing())
        vSize = np.array(itk.size(im_like))
        vOffset = np.array(im_like.GetOrigin())
    else:
        vSize = np.array([size,size,size])
        vSpacing = np.array([spacing,spacing,spacing])
        offset = (-spacing*size + spacing)/2
        vOffset = np.array([offset,offset,offset])

    ct_array = value*np.ones(shape=vSize)
    ct_im = itk.image_from_array(ct_array)
    ct_im.SetSpacing(vSpacing)
    ct_im.SetOrigin(vOffset)
    output_path = os.path.join(output_folder, f'ct_{output_name}.mhd')
    itk.imwrite(ct_im,output_path)
    print(f'{output_path} ok')

    if attmap:

        ctCoeff = [0.2068007,0.57384408]
        spectCoeff = [0.0001668,0.15459051,0.28497715]
        acf_img = faf_ACF_image(ct_im, ctCoeff, spectCoeff, weight=None, proj=None)
        output_path_acf = os.path.join(output_folder, f'acf_{output_name}.mhd')
        itk.imwrite(acf_img, output_path_acf)
        print(f'{output_path_acf} ok')


if __name__ == '__main__':
    create_homogene_phantom_click()