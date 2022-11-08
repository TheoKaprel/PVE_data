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
@click.option('--mse',  is_flag = True, default = False)
@click.option('--norm', is_flag = True, default = False)
def comp_rec_images(source,images,legend, slice, profile, mse, norm):
    if legend:
        assert(len(images) == len(legend))
        legends = list(legend)
    else:
        legends = list(images)


    source_array = itk.array_from_image(itk.imread(source))

    fig_img,ax_img = plt.subplots(1,len(images)+1)



    norm_src = calc_norm(source_array, norm=norm)
    stack_img = [source_array / norm_src]

    for img in (images):
        img_array = itk.array_from_image(itk.imread(img))
        norm_img = calc_norm(img_array, norm=norm)
        stack_img.append(img_array / norm_img)


    vmin_ = min([np.min(sl) for sl in stack_img])
    vmax_ = max([np.max(sl) for sl in stack_img])


    imsh = ax_img[0].imshow(stack_img[0][slice,:,:], vmin = vmin_, vmax = vmax_)
    ax_img[0].set_title(source)
    for k in range(len(images)):
        imsh = ax_img[k+1].imshow(stack_img[k+1][slice,:,:], vmin = vmin_, vmax = vmax_)
        ax_img[k+1].set_title(legends[k])

    fig_img.colorbar(imsh, ax=ax_img)

    plt.suptitle(f'Slice {slice}')

    if profile:
        fig_prof,ax_prof = plt.subplots()
        ax_prof.plot(stack_img[0][slice,profile, :], '-', label=source)
        for k in range(len(images)):
            ax_prof.plot(stack_img[k+1][slice,profile,:], '-',label = legends[k])

        ax_prof.legend()

    if mse:
        fig_mse,ax_mse = plt.subplots()
        lrmse = []
        for k in range(len(images)):
            rmse = np.sqrt(np.mean((stack_img[0] - stack_img[k+1])**2))
            lrmse.append(rmse)

        ax_mse.bar([k for k in range(len(images))],lrmse, tick_label = legends, color = 'black')
        ax_mse.set_ylabel('MSE', fontsize = 20)


    plt.show()



if __name__=='__main__':
    comp_rec_images()