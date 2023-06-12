import os
import h5py
import glob
import numpy as np
import argparse


def convert():
    list_noisy_PVE_PVfree = glob.glob(os.path.join(args.folder, "?????_noisy_PVE_PVfree.npy"))


    f = h5py.File(os.path.join(args.folfer, args.output), 'w')

    for i,fn_noisy_PVE_PVfree in enumerate(list_noisy_PVE_PVfree):
        print(fn_noisy_PVE_PVfree)
        fn_rec_fp = fn_noisy_PVE_PVfree.replace("_noisy_PVE_PVfree", "_rec_fp")

        array_noisy_PVE_PVfree = np.load(fn_noisy_PVE_PVfree)
        array_rec_fp = np.load(fn_rec_fp)

        grp = f.create_group(fn_noisy_PVE_PVfree.split("_noisy_PVE_PVfree.npy")[0][-5:])
        dset_PVE_noisy = grp.create_dataset("PVE_noisy",(120,256,256),dtype='float16')
        dset_PVE = grp.create_dataset("PVE",(120,256,256),dtype='float16')
        dset_PVfree = grp.create_dataset("PVfree",(120,256,256),dtype='float16')
        dset_rec_fp = grp.create_dataset("rec_fp",(120,256,256),dtype='float16')
        dset_PVE_noisy[:,:,:] = array_noisy_PVE_PVfree[:120,:,:]
        dset_PVE[:,:,:] = array_noisy_PVE_PVfree[120:240,:,:]
        dset_PVfree[:,:,:] = array_noisy_PVE_PVfree[240:,:,:]
        dset_rec_fp[:,:,:] = array_rec_fp

    print('done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--folder')
    # parser.add_argument('--opt1',action ="store_true", help = "One .h5 file per data, containing a (4,120,256,256) array")
    # parser.add_argument('--opt2',action ="store_true", help="One .h5 file per data, each one containing 4 datasets (PVE_noisy,PVE,PVfree,rec_fp)")
    # parser.add_argument('--opt3',action ="store_true", help = "One global .h5 file, divided in N groups, each one conaining 4 datasets (PVE_noisy,PVE,PVfree,rec_fp)")
    parser.add_argument('--output')
    args = parser.parse_args()
