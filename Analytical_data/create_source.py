import itk
import numpy as np
import click


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.command(context_settings = CONTEXT_SETTINGS)
@click.option('--size', type = int, default = 128, help = 'Size of the desired image i.e. number of voxels per dim', show_default=True)
@click.option('--spacing', type = float, default = 4, help = 'Spacing of the desired image i.e phyisical length of a voxels (mm)', show_default=True)
@click.option('--like', default = None, help = "Instead of specifying spacing/size, you can specify an image as a metadata model")
@click.option('--n_source', default = 1, type = int, help = "Number of spherical sources")
@click.option('--value', type = float,multiple = True, default = [1], help = 'Activity concentration to assign to the source', show_default=True)
@click.option('--type', type = str, default = "sphere", help = 'sphere or square')
@click.option('--radius', type = float,multiple = True, default = [64], help = 'Radius of activity source (mm)', show_default=True)
@click.option('--center', type = (int,int,int), multiple = True, help = 'Center of the point source (Ox,Oy,Oz) (mm)')
@click.option('--output','-o', 'output_filename', required = True, help = "Output filename (should be .mha or .mhd)")
def create_source_click(size,spacing, value,type, like,n_source, center, radius, output_filename):
    """
    Creates source
    Usage :
    for One spherical source :
    python create_source.py --size 128 --spacing 4 --n_source 1 --value 1 --radius 4 --center 0 0 0  --output ./path/to/source.mhd

    for multiple sources :
    python create_source.py --size 128 --spacing 4 --n_source 2 --value 1 --radius 4 --center 0 0 0 --value 1 --radius 8 --center 128 0 0 --output ./path/to/source.mhd

    """
    create_source(size=size, spacing=spacing, like=like,n_source = n_source, value=value, type = type,
                  center=center, radius=radius, output_filename=output_filename)


def create_source(size, spacing, like,n_source, value,type, center, radius, output_filename):
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
        if type == 'sphere':
            src_array += value[s]*(( ( ((X - center_s[0]) / radius_s) ** 2 + ((Y - center_s[1]) / radius_s) ** 2 + ((Z - center_s[2])/ radius_s) ** 2  ) < 1).astype(float))
        elif type=='square':
            src_array += value[s]*(((np.abs(X-center_s[0])<radius_s)*(np.abs(Y-center_s[1])<radius_s)*(np.abs(Z-center_s[2])<radius_s)).astype(float))



    src_img = itk.image_from_array(src_array)
    src_img.SetSpacing(vSpacing)
    src_img.SetOrigin(vOffset)


    itk.imwrite(src_img,output_filename)
    print(f'{output_filename} ok!')



if __name__ == '__main__':
    create_source_click()