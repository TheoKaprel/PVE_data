import numpy as np
import itk
import json
import click



CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--lj','labels', help = 'Labels json')
@click.option('--li','voxels_labels', help = 'Labels image')
@click.option('--ct',help = 'Output ct filename')
@click.option('--source',help = 'Output source filename')
@click.option('--bg','background',type = float, help = 'Background ratio. if --bg 10, then the ratio background / source = 1/10')
@click.option('--iec-type', type = click.Choice(['air', 'water', 'iec'], case_sensitive=False), help = 'Attenuating material (iec = real, ie with plastic, ...)')
def labels_to_HU(labels, voxels_labels,ct,source, background, iec_type):

    if (voxels_labels==None) and (labels==None):
        print("ERROR: you must provide at least one of the label.json or label.mhd image")
        exit(0)
    else:
        if voxels_labels==None:
            voxels_labels = labels.replace('.json', '.mhd')
        if labels==None:
            labels = voxels_labels.replace('.mhd', '.json')


    dict_labels = json.loads(open(labels).read())

    img_labels = itk.imread(voxels_labels)

    if ct:
        make_ct_from_labels(dict_labels=dict_labels, img_labels=img_labels, output_hu=ct,iec_type=iec_type)
    if source:
        make_src_from_labels(dict_labels=dict_labels, img_labels=img_labels, output_src=source, background=background)



def make_ct_from_labels(dict_labels,img_labels, output_hu, iec_type):
    dict_mat_HU = {'G4_AIR': -1000,
                   'G4_WATER': 0,
                   'IEC_PLASTIC': 20,
                   'G4_LUNG_ICRP': -200}

    dict_labels_mat_hu ={}
    if iec_type=="air":
        for l in dict_labels:
            mat = 'G4_AIR'
            dict_labels_mat_hu[l] = [dict_labels[l], mat, dict_mat_HU[mat]]
    elif iec_type=="water":
        for l in dict_labels:
            if l == "world":
                mat = 'G4_AIR'
            else:
                mat = 'G4_WATER'
            dict_labels_mat_hu[l] = [dict_labels[l], mat, dict_mat_HU[mat]]
    elif iec_type=="iec":
        for l in dict_labels:
            mat = "IEC_PLASTIC"
            if l == "world":
                mat = 'G4_AIR'
            if 'sphere' in l:
                mat = 'G4_WATER'
            if 'capillary' in l:
                mat = 'G4_WATER'
            if 'shell' in l:
                mat = 'IEC_PLASTIC'
            if l == "iec_center_cylinder_hole":
                mat = 'G4_LUNG_ICRP'

            dict_labels_mat_hu[l] = [dict_labels[l], mat, dict_mat_HU[mat]]


    # write labels
    lf = str(output_hu).replace('.mhd', '.json')
    outfile = open(lf, "w")
    json.dump(dict_labels_mat_hu, outfile, indent=4)


    array_labels = itk.array_from_image(img_labels)
    array_HU = np.zeros(array_labels.shape)
    for l in dict_labels_mat_hu:
        label_info = dict_labels_mat_hu[l]
        array_HU[array_labels == label_info[0]] = label_info[2]

    img_HU = itk.image_from_array(array_HU)
    img_HU.CopyInformation(img_labels)
    itk.imwrite(img_HU, output_hu)
    print(f'ct saved in : {output_hu}')


def make_src_from_labels(dict_labels, img_labels, output_src, background):
    list_labels_with_src = ["iec_sphere_10mm",
                            "iec_sphere_13mm",
                            "iec_sphere_22mm",
                            "iec_sphere_17mm",
                            "iec_sphere_28mm",
                            "iec_sphere_37mm"]


    array_labels = itk.array_from_image(img_labels)
    array_src = np.zeros(array_labels.shape)

    # background activity
    array_src[array_labels >= 2] = 1 / (background)

    for l in dict_labels:
        # hot source activity
        if l in list_labels_with_src:
            array_src[array_labels == dict_labels[l]] = 1

    img_src = itk.image_from_array(array_src)
    img_src.CopyInformation(img_labels)
    itk.imwrite(img_src, output_src)
    print(f'src saved in : {output_src}')



if __name__=='__main__':
    labels_to_HU()