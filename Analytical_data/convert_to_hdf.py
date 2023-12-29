import os
import h5py
import glob

import numpy as np
import argparse


def convert():
    print(args)
    keys=args.keys.split(',')
    print(keys)

    list_frstkey = glob.glob(os.path.join(args.folder, f"?????_{keys[0]}.npy"))
    f = h5py.File(os.path.join(args.folder, args.output), 'a')
    list_keys=list(f.keys())
    print(list_frstkey)
    for i,fn_src in enumerate(list_frstkey):
        ref = fn_src.split(f'_{keys[0]}.npy')[0][-5:]
        # if ref not in f_keys:
        print(ref)

        save = True
        for key in keys:
            if not os.path.isfile(os.path.join(args.folder, f"{ref}_{key}.npy")):
                print(f"{ref}_{key}.npy not in {args.folder}")
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
                save_key_in_grp(grp=grp,ref=ref,key=key,folder=args.folder)

    print('done!')


def save_key_in_grp(grp,ref,key,folder):
    array_fn = os.path.join(folder, f"{ref}_{key}.npy")
    array = np.load(array_fn)
    dset_key = grp.create_dataset(key, array.shape, dtype='float16')
    dset_key[:, :, :] = array


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--folder')
    parser.add_argument('--keys', type=str)
    parser.add_argument('--output')
    args = parser.parse_args()
    convert()