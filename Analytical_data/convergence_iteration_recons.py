import matplotlib.pyplot as plt
import numpy as np
import itk
import glob
import click

def get_list_of_iter_img(path_iterations):
    id_d = path_iterations.find("%d")
    list_imgs = glob.glob(f'{path_iterations[:id_d]}*{path_iterations[id_d+2:]}')
    list_iter = [int(img[id_d:img.find(path_iterations[id_d+2:])]) for img in list_imgs]
    return list_imgs,list_iter



CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('-s', '--source',)
@click.option('-p', 'path_iterations', help="path to reconstructed images with %d instead of iteration number. ex rec_img_120_%d.mhd")
def plot_conv_iter_recons(source, path_iterations):

    source_array = itk.array_from_image(itk.imread(source))

    list_img, list_iter = get_list_of_iter_img(path_iterations)
    list_err = []

    for img_file, iter in zip(list_img,list_iter):

        img_array = itk.array_from_image(itk.imread(img_file))
        list_err.append(np.mean((img_array-source_array)**2)/np.mean(img_array**2))

    fig,ax = plt.subplots()
    ax.scatter(list_iter, list_err, s=10, marker='o')
    plt.show()







if __name__=='__main__':
    plot_conv_iter_recons()