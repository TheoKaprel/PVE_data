#!/usr/bin/env python3

import itk
import numpy as np
import click
import os
import matplotlib.pyplot as plt


def calc_norm(img,norm):
    if (norm==False or norm==None):
        return 1
    else:
        return img.max()

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--source', required = True)
@click.option('--images', '-i',multiple = True, required = True)
@click.option('--legend', '-l')
@click.option('-s','--slice', type = int)
@click.option('-p','--profile', type = int)
@click.option('--mse')
@click.option('--norm', is_flag = True, default = False)
def comp_rec_images(source,images,legend, slice, profile, mse, norm):
    if legend:
        assert(len(images) == len(legend))


    source_array = itk.array_from_image(itk.imread(source))

    fig_img,ax_img = plt.subplots(1,len(images)+1)



    norm_src = calc_norm(source_array, norm=norm)
    stack_slices = [source_array[slice,:,:] / norm_src]

    for img in (images):
        img_array = itk.array_from_image(itk.imread(img))
        norm_img = calc_norm(img_array, norm=norm)
        stack_slices.append(img_array[slice,:,:] / norm_img)


    vmin_ = min([np.min(sl) for sl in stack_slices])
    vmax_ = max([np.max(sl) for sl in stack_slices])


    imsh = ax_img[0].imshow(stack_slices[0], vmin = vmin_, vmax = vmax_)
    ax_img[0].set_title(source)
    for k in range(len(images)):
        imsh = ax_img[k+1].imshow(stack_slices[k+1], vmin = vmin_, vmax = vmax_)
        ax_img[k+1].set_title(images[k])

    fig_img.colorbar(imsh, ax=ax_img)

    plt.suptitle(f'Slice {slice}')

    if profile:
        fig_prof,ax_prof = plt.subplots()
        ax_prof.plot(stack_slices[0][profile, :], '-', label=source)
        for k in range(len(images)):
            ax_prof.plot(stack_slices[k+1][profile,:], '-',label = images[k])

        ax_prof.legend()


    plt.show()



if __name__=='__main__':
    comp_rec_images()