import os
import h5py
import glob

import numpy as np
import argparse
import itk


def convert():
    print(args)
    keys=args.keys.split(',')
    print(keys)
    ext_typ=args.type

    list_frstkey = glob.glob(os.path.join(args.folder, f"?????_{keys[0]}.{ext_typ}"))
    f = h5py.File(args.output, 'a')
    list_keys=list(f.keys())
    print(list_frstkey)
    for i,fn_src in enumerate(list_frstkey):
        ref = fn_src.split(f'_{keys[0]}.{ext_typ}')[0][-5:]
        # if ref not in f_keys:
        print(ref)

        save = True
        for key in keys:
            if not os.path.isfile(os.path.join(args.folder, f"{ref}_{key}.{ext_typ}")):
                print(f"{ref}_{key}.{ext_typ} not in {args.folder}")
                save=False

        if save==False:
            if ref in list_keys:
                del f[ref]
        else:
            if ref not in list_keys:
                grp = f.create_group(ref)
            else:
                grp = f[ref]

            for key in keys:
                save_key_in_grp(grp=grp,ref=ref,key=key,folder=args.folder,ext_typ=ext_typ)

    f.close()
    print('done!')


def save_key_in_grp(grp,ref,key,folder,ext_typ):
    array_fn = os.path.join(folder, f"{ref}_{key}.{ext_typ}")
    if ext_typ!="npy":
        array = itk.array_from_image(itk.imread(array_fn))
    else:
        array = np.load(array_fn)

    if key not in list(grp.keys()):
        dset_key = grp.create_dataset(key, array.shape, dtype='float16')
        dset_key[:, :, :] = array

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--folder')
    parser.add_argument('--keys', type=str)
    parser.add_argument('--type', type=str)
    parser.add_argument('--output')
    args = parser.parse_args()
    convert()