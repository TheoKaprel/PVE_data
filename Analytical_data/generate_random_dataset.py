import itk
import numpy as np
import click
import os
import random
import string
import forwardprojection


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.command(context_settings = CONTEXT_SETTINGS)
@click.option('--nb_data','-n', type = int, required = True, help = 'number of desired data = (src,projPVE,projPVfree)')
@click.option('--size', type = int, default = 128, help = 'Size of the desired image i.e. number of voxels per dim', show_default=True)
@click.option('--spacing', type = float, default = 4, help = 'Spacing of the desired image i.e phyisical length of a voxels (mm)', show_default=True)
@click.option('--like', default = None, help = "Instead of specifying spacing/size, you can specify a .mhd image as a metadata model", show_default=True)
@click.option('--min_radius', default = 4, help = 'minimum radius of the random spheres', show_default = True)
@click.option('--max_radius', default = 32, help = 'max radius of the random spheres', show_default = True)
@click.option('--max_activity', default = 1, help = 'max activity in spheres', show_default = True)
@click.option('--nspheres', default = 1, help = 'max number of spheres to generate on each source', show_default= True)
@click.option('--background', default = None, help = 'If you want background activity specify the activity:background ratio. For example --background 10 for 1/10 background activity.')
@click.option('--ellipse', is_flag = True, default= False, help = "if --ellipse, activity spheres are in fact ellipses")
@click.option('--geom', '-g', default = None, help = 'geometry file to forward project. Default is the proj on one detector')
@click.option('--output_folder','-o', default = './dataset', help = " Absolute or relative path to the output folder", show_default=True)
@click.option('--sigma0pve', default = forwardprojection.sigma0pve_default,type = float, help = 'sigma at distance 0 of the detector', show_default=True)
@click.option('--alphapve', default = forwardprojection.alphapve_default, type = float, help = 'Slope of the PSF against the detector distance', show_default=True)
def generate(nb_data, output_folder,size, spacing, like,min_radius, max_radius,max_activity, nspheres,background,ellipse, geom, sigma0pve, alphapve):
    # get output image parameters
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

    # matrix settings
    lengths = vSize*vSpacing
    lspaceX = np.linspace(-vSize[0] * vSpacing[0] / 2, vSize[0] * vSpacing[0] / 2, vSize[0])+vSpacing[0] / 2
    lspaceY = np.linspace(-vSize[1] * vSpacing[1] / 2, vSize[1] * vSpacing[1] / 2, vSize[1])+vSpacing[1] / 2
    lspaceZ = np.linspace(-vSize[2] * vSpacing[2] / 2, vSize[2] * vSpacing[2] / 2, vSize[2])+vSpacing[2] / 2
    X, Y, Z = np.meshgrid(lspaceX,lspaceY,lspaceZ)




    for n in range(nb_data):
        src_array = np.zeros_like(X)

        if background:
            bg_center = [0,0,0]
            bg_radius = np.random.randint(180, 217)
            src_array += (1/float(background)) * ((((X - bg_center[0]) / bg_radius) ** 2 + ((Y - bg_center[1]) / bg_radius) ** 2 + (
                        (Z - bg_center[2]) / bg_radius) ** 2) < 1).astype(float)


        random_nb_of_sphers = np.random.randint(1,nspheres+1)

        for s  in range(random_nb_of_sphers):
            random_activity = np.random.randint(1, max_activity+1)

            # random radius and center
            if ellipse:
                radius = np.random.rand(3)*(max_radius-min_radius) + min_radius
            else:
                radius = np.random.rand()*(max_radius-min_radius) + min_radius
                radius = [radius, radius, radius]

            center = (2*np.random.rand(3)-1)*(lengths/2-np.max(radius)) # the sphere borders remain inside the phantom
            src_array += random_activity  * ( ( ((X-center[0]) / radius[0]) ** 2 + ((Y-center[1]) / radius[1]) ** 2 + ((Z-center[2])/ radius[2]) ** 2  ) < 1).astype(float)


        src_img = itk.image_from_array(src_array)
        src_img.SetSpacing(vSpacing)
        src_img.SetOrigin(vOffset)

        # Random output filename
        letters = string.ascii_uppercase
        filenamelength = 5
        randomfn = ''.join(random.choice(letters) for i in range(filenamelength))

        # saving of source 3D image
        source_path = os.path.join(output_folder,f'{randomfn}.mhd')
        itk.imwrite(src_img,source_path)
        
        if geom == None:
            geom = './data/geom_1.xml'
        

        #compute the foward projection :
        print(source_path)
        forwardprojection.forwardproject(inputsrc=source_path, output_folder=output_folder,geom=geom, nproj=1,pve=True, pvfree=True, sigma0pve=sigma0pve, alphapve=alphapve)


if __name__ == '__main__':
    generate()
