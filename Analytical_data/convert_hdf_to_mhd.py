#!/usr/bin/env python3

import argparse
import h5py
import numpy as np
import itk
import os

def main():
    print(args)

    f = h5py.File(args.h5, 'r')
    keys = args.keys.split(',')
    dtype=np.float32
    spacing = np.array([float(s) for s in args.spacing.split(',')])


    for ref in list(f.keys()):
        data = f[ref]
        print(ref)
        for key in keys:
            array = np.array(data[key], dtype=dtype)
            img = itk.image_from_array(array)
            img_shape = np.array(img.shape)
            origin = [(-img_shape[k] * spacing[k] + spacing[k]) / 2 for k in range(3)]
            img.SetSpacing(spacing)
            img.SetOrigin(origin)
            itk.imwrite(img, os.path.join(args.output_folder, f"{ref}_{key}.mhd"))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5")
    parser.add_argument("--keys")
    parser.add_argument("--spacing", type = str, default = "4.7952,4.7952,4.7952")
    parser.add_argument("--output_folder")
    args = parser.parse_args()

    main()
