import itk
import numpy as np
import click
import os

""""
Creation of either 
- a hamogeneous ct with a userdefined value in HouseField Units
- an activity map containing a single sphere in the center of the phantom

"""



CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.command(context_settings = CONTEXT_SETTINGS)
@click.option('--type','-t', required = True, type = click.Choice(['ct','source']))
@click.option('--size', type = int, default = 128, help = 'Size of the desired image i.e. number of voxels per dim', show_default=True)
@click.option('--spacing', type = float, default = 4, help = 'Spacing of the desired image i.e phyisical length of a voxels (mm)', show_default=True)
@click.option('--like', default = None, help = "Instead of specifying spacing/size, you can specify a .mhd image as a metadata model")
@click.option('--output_folder', required = True, help = " Absolute or relative path to the output folder")
@click.option('--output_name', required = True, help = "Name of the desired output_name.mhd/raw file")
@click.option('--value', type = float, default = -1024, help = '[IF ct] Default value to assign pixels for the homogeneous phantom', show_default=True)
@click.option('--radius', type = float, default = 64, help = '[IF source] Radius of activity source (mm)', show_default=True)
@click.option('--center',nargs = 3, default = (0,0,0), type = (int,int,int), help = '[IF source] Center of the point source (Ox,Oy,Oz) (mm)')
def create(type,output_folder, output_name,size,spacing, value, like,center, radius):
    if type=='ct':
        create_homogene_phantom(output_ct_folder = output_folder, output_ct_name = output_name,size=size,spacing=spacing,value=value)
    elif type=='source':
        create_point_source(output_src_folder = output_folder, output_src_name = output_name,size=size, spacing=spacing, like=like,center = center, radius= radius)

def create_homogene_phantom(output_ct_folder, output_ct_name,size, spacing, value):
    ct_array = value*np.ones((size,size,size))
    ct_im = itk.image_from_array(ct_array)
    offset = (-spacing*size + spacing)/2
    ct_im.SetSpacing((spacing,spacing,spacing))
    ct_im.SetOrigin((offset,offset,offset))
    output_path = os.path.join(output_ct_folder, f'{output_ct_name}.mhd')
    itk.imwrite(ct_im,output_path)


def create_point_source(output_src_folder, output_src_name,size, spacing, like, center, radius):
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

    center = [center[1], center[2], center[0]]


    lspaceX = np.linspace(-vSize[0] * vSpacing[0] / 2, vSize[0] * vSpacing[0] / 2, vSize[0])
    lspaceY = np.linspace(-vSize[1] * vSpacing[1] / 2, vSize[1] * vSpacing[1] / 2, vSize[1])
    lspaceZ = np.linspace(-vSize[2] * vSpacing[2] / 2, vSize[2] * vSpacing[2] / 2, vSize[2])
    X, Y, Z = np.meshgrid(lspaceX,lspaceY,lspaceZ)
    src_array = ( ( ((X - center[0]) / radius) ** 2 + ((Y - center[1]) / radius) ** 2 + ((Z - center[2])/ radius) ** 2  ) < 1).astype(float)

    src_img = itk.image_from_array(src_array)
    src_img.SetSpacing(vSpacing)
    src_img.SetOrigin(vOffset)

    output_path = os.path.join(output_src_folder,f'{output_src_name}.mhd')
    itk.imwrite(src_img,output_path)



if __name__ == '__main__':
    create()