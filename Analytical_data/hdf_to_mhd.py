#!/usr/bin/env python3

import os
import h5py
import glob

import itk
import numpy as np
import argparse

def main():
    print(args)
    str_dtype = h5py.special_dtype(vlen=str)

    f = h5py.File(args.hdf, 'r')
    keys = list(f.keys())

    with h5py.File(args.hdf, 'r') as f:
        for key in keys:
            print(key)
            data = f[key]
            l_data = list(data.keys())
            print(l_data)

            for data_key in l_data:
                if data[data_key].dtype==str_dtype:
                    print(data[data_key][0])
                else:
                    array = np.array(data[data_key], dtype=np.float64)
                    itk_img = itk.image_from_array(array)
                    itk.imwrite(itk_img,os.path.join(args.output_folder, f'{key}_{data_key}.mhd'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf")
    parser.add_argument("--output_folder")
    args = parser.parse_args()

    main()
