import itk
import numpy as np
import matplotlib.pyplot as plt
import glob
import click

FWHM = 17.79000820647356

def get_list_of_img_size(path_src_images):
    id_d = path_src_images.find("%d")
    list_imgs = glob.glob(f'{path_src_images[:id_d]}*{path_src_images[id_d+2:]}')
    list_img_size = [(img,int(img[id_d:img.find(path_src_images[id_d+2:])])) for img in list_imgs]
    return sorted(list_img_size, key = lambda img_size: img_size[1])


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('-s', '--sources','sources_paths', help = "Define the sources filename containing their size with %d. Ex : -s source_%d.mha")
def illustration(sources_paths):
    list_img_size = get_list_of_img_size(path_src_images=sources_paths)

    list_size = []
    list_RC = []

    fig_img,ax_img = plt.subplots(3,5)

    for k,img_size in enumerate(list_img_size):
        src_img_fn = img_size[0]
        size = img_size[1]

        src_array_img = itk.array_from_image(itk.imread(src_img_fn))
        rec_img_fn = f'recons_{size}.mha'
        rec_array_img = itk.array_from_image(itk.imread(rec_img_fn))

        RC = (np.max(rec_array_img[src_array_img==np.max(src_array_img)])) / 1
        list_size.append(2*size/FWHM)
        list_RC.append(RC)

        ax_img[0,k].imshow(src_array_img[63,:,:], cmap = 'Greys', vmin = 0, vmax = 1)
        ax_img[1,k].imshow(rec_array_img[63,:,:], cmap = 'Greys', vmin = 0, vmax = 1)
        ax_img[2,k].plot(rec_array_img[63,63,:], color = 'black', linewidth = 1.5)

        ax_img[0, k].set_xlim([25, 100])
        ax_img[0, k].set_ylim([25, 100])
        ax_img[1, k].set_xlim([25, 100])
        ax_img[1, k].set_ylim([25, 100])
        ax_img[2,k].set_xlim([25,100])
        ax_img[2,k].set_ylim([0,1.1])

        ax_img[0,k].set_title(f'Diameter : {2*size} mm', fontsize=18)


    ax_img[0,0].set_ylabel('Source', fontsize=18)
    ax_img[1,0].set_ylabel('Slice of \n reconstructed images', fontsize=14)
    ax_img[2,0].set_ylabel('Profile of \n reconstructed images', fontsize=14)



    fig,ax = plt.subplots()
    ax.plot(list_size, list_RC, linewidth = 2, marker = 'o', color = 'black')
    ax.set_xlabel('Size / FWHM', fontsize=18)
    ax.set_ylabel('Recovery Coefficient', fontsize=18)
    plt.show()






if __name__=='__main__':
    illustration()