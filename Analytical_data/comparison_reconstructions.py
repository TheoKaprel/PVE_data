import itk
import numpy as np
import click
import os
import matplotlib.pyplot as plt

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--source', required = True)
@click.option('--folder', multiple = True)
@click.option('--input', '-i',multiple = True, required = True, help = 'Selected images will be input_%d.mha where %d is the iteration')
@click.option('-n',type = int ,multiple = True,required = True, help = 'Max number of iteration')
@click.option('--every', type= int, multiple = True)
@click.option('--add', type = (int,int),multiple = True, help = 'Add an iteration')
@click.option('-ns', type = int, multiple =True)
@click.option('--nproj', type = int, multiple = True)
def comp_rec_images(source,folder,input, n,every,s,nproj, add):
    assert(len(folder)==len(input))
    assert(len(folder)==len(n))
    assert(len(folder)==len(every))
    assert(len(folder)==len(s))
    assert(len(folder)==len(nproj))

    source_array = itk.array_from_image(itk.imread(source))
    norm = np.sqrt(np.sum(source_array**2))
    nfolder = len(folder)

    fig,ax = plt.subplots(2,1)


    for k in range(nfolder):
        folder_k = folder[k]
        input_k = input[k]
        n_k = n[k]
        every_k = every[k]
        s_k = s[k]
        nproj_k = nproj[k]

        iteration_array_k = np.arange(every_k, n_k + 1, every_k)

        for (id,new_iter) in add:
            if id==k:
                iteration_array_k = np.append(iteration_array_k,new_iter)

        RMSE_array_k = np.zeros(len(iteration_array_k))
        equivalent_iteration_array_k = np.zeros(len(iteration_array_k))

        for itit in range(len(iteration_array_k)):
            it = iteration_array_k[itit]
            equivalent_iteration_array_k[itit] = it*s_k
            img_filename = os.path.join(folder_k, f'{input_k}{it}.mha')
            img_array = itk.array_from_image(itk.imread(img_filename))
            RMSE_array_k[itit] = np.sqrt(np.sum( (img_array - source_array)**2 )) / norm
            ax[1].plot(img_array[63,63,:], label = it)
        ax[0].plot(equivalent_iteration_array_k,RMSE_array_k,'o-', label = f'{folder_k} | nproj = {nproj_k} | s = {s_k}')









    plt.legend()
    plt.show()




if __name__=='__main__':
    comp_rec_images()