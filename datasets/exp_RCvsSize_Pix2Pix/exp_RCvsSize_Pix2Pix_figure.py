import itk
import numpy as np
import matplotlib.pyplot as plt
import click
import glob

def get_list_of_img_size(path_src_images):
    id_d = path_src_images.find("%d")
    list_imgs = glob.glob(f'{path_src_images[:id_d]}*{path_src_images[id_d+2:]}')
    list_img_size = [(img,int(img[id_d:img.find(path_src_images[id_d+2:])])) for img in list_imgs]
    return sorted(list_img_size, key = lambda img_size: img_size[1])


FWHM = 17.79000820647356

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('-s', '--sources','sources_paths', help = "Define the sources filename containing their size with %d. Ex : -s source_%d.mha")
def res_exp(sources_paths):
    RC = 'mean'
    # RC = 'max'


    list_img_size = get_list_of_img_size(path_src_images=sources_paths)

    list_size = []
    list_RC_PVE_noPVC = []
    list_RC_PVE_PVC = []
    list_RC_noPVE_noPVC = []
    list_RC_PVE_DeepPVC = []


    for k,img_size in enumerate(list_img_size):
        src_img_fn = img_size[0]
        size = img_size[1]

        src_array_img = itk.array_from_image(itk.imread(src_img_fn))


        true_hot_act = np.max(src_array_img)

        if size!=12:
            list_size.append(2 * size / FWHM)
            rec_img_fn_PVE_noPVC = f'rec_source_{size}_PVE_noPVC.mha'
            rec_img_PVE_noPVC = itk.array_from_image(itk.imread(rec_img_fn_PVE_noPVC))
            if RC=='mean':
                RC_PVE_noPVC = (np.mean(rec_img_PVE_noPVC[src_array_img==true_hot_act])) / true_hot_act
            else:
                RC_PVE_noPVC = (np.max(rec_img_PVE_noPVC[src_array_img==true_hot_act])) / true_hot_act
            list_RC_PVE_noPVC.append(RC_PVE_noPVC)

            rec_img_fn_PVE_PVC = f'rec_source_{size}_PVE_PVC.mha'
            rec_img_PVE_PVC = itk.array_from_image(itk.imread(rec_img_fn_PVE_PVC))
            if RC=='mean':
                RC_PVE_PVC = (np.mean(rec_img_PVE_PVC[src_array_img==true_hot_act])) / true_hot_act
            else:
                RC_PVE_PVC = (np.max(rec_img_PVE_PVC[src_array_img==true_hot_act])) / true_hot_act
            list_RC_PVE_PVC.append(RC_PVE_PVC)

            rec_img_fn_noPVE_noPVC = f'rec_source_{size}_noPVE_noPVC.mha'
            rec_img_noPVE_noPVC = itk.array_from_image(itk.imread(rec_img_fn_noPVE_noPVC))

            if RC=='mean':
                RC_noPVE_noPVC = (np.mean(rec_img_noPVE_noPVC[src_array_img==true_hot_act])) / true_hot_act
            else:
                RC_noPVE_noPVC = (np.max(rec_img_noPVE_noPVC[src_array_img==true_hot_act])) / true_hot_act
            list_RC_noPVE_noPVC.append(RC_noPVE_noPVC)


        rec_img_fn_PVE_DeepPVC = f'rec_source_{size}_PVE_DeepPVC.mha'
        rec_img_PVE_DeepPVC = itk.array_from_image(itk.imread(rec_img_fn_PVE_DeepPVC))
        if RC=='mean':
            RC_PVE_DeepPVC = (np.mean(rec_img_PVE_DeepPVC[src_array_img==true_hot_act])) / true_hot_act
        else:
            RC_PVE_DeepPVC = (np.max(rec_img_PVE_DeepPVC[src_array_img==true_hot_act])) / true_hot_act
        list_RC_PVE_DeepPVC.append(RC_PVE_DeepPVC)


    fig,ax = plt.subplots()
    ax.plot(list_size, list_RC_PVE_noPVC, linewidth = 3, marker = 'o', color = 'red', label = 'PVE/noPVC')
    ax.plot(list_size, list_RC_PVE_PVC, linewidth = 3, marker = 'o', color = 'orange', label = 'PVE/PVC')
    ax.plot(list_size, list_RC_noPVE_noPVC, linewidth = 3, marker = 'o', color = 'green', label='noPVE/noPVC')
    ax.plot([2 * s / FWHM for s in [4,8,12,16,32,64]], list_RC_PVE_DeepPVC, linewidth = 3, marker = 'o', color = 'blue', label = 'PVE/DeepPVC')
    plt.legend()
    ax.set_xlabel('Size / FWHM', fontsize=18)
    ax.set_ylabel('Recovery Coefficient', fontsize=18)
    plt.show()







if __name__=='__main__':
    res_exp()