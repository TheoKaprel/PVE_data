import itk
import os
import h5py
import glob
import argparse


def convert():
    list_noisy_PVE_PVfree = glob.glob(os.path.join(args.folder, "?????_noisy_PVE_PVfree.npy"))

    if args.opt3:
        f = h5py.File(os.path.join(args.folfer, "dataset.h5"), 'w')

    for i,fn_PVE in enumerate(list_noisy_PVE_PVfree):
        print(fn_PVE)
        fn_PVE_noisy = fn_PVE.replace("_PVE", "_PVE_noisy")
        fn_PVfree = fn_PVE.replace("_PVE", "_PVfree")
        fn_PVrec_fp = fn_PVE.replace("_PVE", "_rec_fp")
        fn_hdf= fn_PVE.replace('PVE.mhd', 'data.h5')

        array_PVE_noisy = itk.array_from_image(itk.imread(os.path.join(args.folder, fn_PVE_noisy)))
        array_PVE = itk.array_from_image(itk.imread(os.path.join(args.folder, fn_PVE)))
        array_PVfree = itk.array_from_image(itk.imread(os.path.join(args.folder, fn_PVfree)))
        array_rec_fp = itk.array_from_image(itk.imread(os.path.join(args.folder, fn_PVrec_fp)))

        # np.save(fn_npy,np.stack((array_PVE_noisy,array_PVE,array_PVfree,array_rec_fp)))

        f = h5py.File(os.path.join(args.folder, fn_hdf), 'w')
        # grp = f.create_group(fn_PVE.split("_PVE.mhd")[0][-5:])
        dset_PVE_noisy = f.create_dataset("PVE_noisy",(120,256,256),dtype='float16')
        dset_PVE = f.create_dataset("PVE",(120,256,256),dtype='float16')
        dset_PVfree = f.create_dataset("PVfree",(120,256,256),dtype='float16')
        dset_rec_fp = f.create_dataset("rec_fp",(120,256,256),dtype='float16')
        dset_PVE_noisy[:,:,:] = array_PVE_noisy
        dset_PVE[:,:,:] = array_PVE
        dset_PVfree[:,:,:] = array_PVfree
        dset_rec_fp[:,:,:] = array_rec_fp


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--folder')
    parser.add_argument('--inputtype')
    parser.add_argument('--opt1',action ="store_true", help = "One .h5 file per data, containing a (4,120,256,256) array")
    parser.add_argument('--opt2',action ="store_true", help="One .h5 file per data, each one containing 4 datasets (PVE_noisy,PVE,PVfree,rec_fp)")
    parser.add_argument('--opt3',action ="store_true", help = "One global .h5 file, divided in N groups, each one conaining 4 datasets (PVE_noisy,PVE,PVfree,rec_fp)")
    parser.add_argument('--output')
    args = parser.parse_args()
