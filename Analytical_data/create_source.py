import itk
import numpy as np
import click
import os



CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.command(context_settings = CONTEXT_SETTINGS)
@click.option('--size', type = int, default = 128, help = 'Size of the desired image i.e. number of voxels per dim', show_default=True)
@click.option('--spacing', type = float, default = 4, help = 'Spacing of the desired image i.e phyisical length of a voxels (mm)', show_default=True)
@click.option('--like', default = None, help = "Instead of specifying spacing/size, you can specify a .mhd image as a metadata model")
@click.option('--n_source', default = 1, type = int, help = "Number of spherical sources")
@click.option('--value', type = float,multiple = True, default = [1], help = 'Activity concentration to assign to the source', show_default=True)
@click.option('--radius', type = float,multiple = True, default = [64], help = 'Radius of activity source (mm)', show_default=True)
@click.option('--center', type = (int,int,int), multiple = True, help = 'Center of the point source (Ox,Oy,Oz) (mm)')
@click.option('--output_folder', required = True, help = " Absolute or relative path to the output folder")
@click.option('--output_name', required = True, help = "Name of the desired output_name.mhd/raw file")
def create_source_click(size,spacing, value, like,n_source, center, radius, output_folder, output_name):
    """
    Creates source
    Usage :
    for One spherical source :
    python create_source.py --size 128 --spacing 4 --n_source 1 --value 1 --radius 4 --center 0 0 0  --output_folder folder/ --output_name name

    for multiple sources :
    python create_source.py --size 128 --spacing 4 --n_source 2 --value 1 --radius 4 --center 0 0 0 --value 1 --radius 8 --center 128 0 0  --output_folder folder/ --output_name name

    """
    create_source(size=size, spacing=spacing, like=like,n_source = n_source, value=value, center=center, radius=radius, output_folder=output_folder, output_name=output_name)


def create_source(size, spacing, like,n_source, value, center, radius, output_folder, output_name):
    if (n_source!= len(value) or n_source!=len(center) or n_source!=len(radius)):
        print('ERROR : problem in the number of source/parameters per sources...')
        exit(0)

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





    lspaceX = np.linspace(-vSize[0] * vSpacing[0] / 2, vSize[0] * vSpacing[0] / 2, vSize[0])
    lspaceY = np.linspace(-vSize[1] * vSpacing[1] / 2, vSize[1] * vSpacing[1] / 2, vSize[1])
    lspaceZ = np.linspace(-vSize[2] * vSpacing[2] / 2, vSize[2] * vSpacing[2] / 2, vSize[2])
    X, Y, Z = np.meshgrid(lspaceX,lspaceY,lspaceZ)
    src_array = np.zeros_like(X)

    for s in range(n_source):
        center_s = [center[s][1], center[s][2], center[s][0]]
        radius_s = radius[s]
        src_array += value[s]*(( ( ((X - center_s[0]) / radius_s) ** 2 + ((Y - center_s[1]) / radius_s) ** 2 + ((Z - center_s[2])/ radius_s) ** 2  ) < 1).astype(float))



    src_img = itk.image_from_array(src_array)
    src_img.SetSpacing(vSpacing)
    src_img.SetOrigin(vOffset)

    output_path = os.path.join(output_folder,f'{output_name}.mhd')
    itk.imwrite(src_img,output_path)
    print(f'{output_path} ok!')



if __name__ == '__main__':
    create_source_click()