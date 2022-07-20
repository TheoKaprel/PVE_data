import itk
import numpy as np
import click
import os
import matplotlib.pyplot as plt

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--source', required = True)
@click.option('--folder')
@click.option('--input', '-i', required = True, multiple = True, help = 'Selected images will be input_%d.mhd where %d is the iteration')
@click.option('-n',type = int ,required = True, help = 'Max number of iteration')
def comp_rec_images(folder,source, input, n):
    source_array = itk.array_from_image(itk.imread(source))

    iteration_array = np.arange(1,n+1)
    RMSE_array = np.zeros((len(input),n))

    fig,ax = plt.subplots()

    for i in range(len(input)):
        one_input = input[i]
        for k in range(1,n+1):
            img_filename = os.path.join(folder, f'{one_input}_{k}.mhd')
            img_array = itk.array_from_image(itk.imread(img_filename))
            RMSE_array[i,k-1] = np.sqrt(np.mean( (img_array - source_array)**2 ))
        ax.plot(iteration_array,RMSE_array[i,:],'o-', label = one_input)
    plt.legend()
    plt.show()




if __name__=='__main__':
    comp_rec_images()