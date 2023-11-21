import os
import h5py
import glob

import numpy as np
import argparse


def convert():
    list_src = glob.glob(os.path.join(args.folder, "?????_src.npy"))

    f = h5py.File(os.path.join(args.folder, args.output), 'w')
    f_keys = list(f.keys())

    for i,fn_src in enumerate(list_src):
        ref = fn_src.split('_src.npy')[0][-5:]
        if ref not in f_keys:
            print(ref)


            keys=['src', 'attmap', 'attmap_fp', 'PVE_att', 'PVE_att_noisy', 'PVfree_att', 'PVfree']

            save = True
            for key in keys:
                if not os.path.isfile(os.path.join(args.folder, f"{ref}_{key}.npy")):
                    print(f"{ref}_{key}.npy not in {args.folder}")
                    save=False
            if save==False:
                continue

            grp = f.create_group(ref)

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
    parser.add_argument('--output')
    args = parser.parse_args()
    convert()