#!/usr/bin/env python3

import argparse
import glob
import itk
import numpy as np
import gatetools
import os

def main():
    print(args)

    lsrc=glob.glob(os.path.join(args.folder, "?????_src.npy"))

    for src_fn in lsrc:
        rec_fn = src_fn.replace("_src.npy", "_rec.npy")
        if (os.path.isfile(src_fn) and os.path.isfile(rec_fn)):
            print(src_fn)

            src_array=np.load(src_fn)
            rec_array=np.load(rec_fn)
            src_img=itk.image_from_array(src_array)
            input_size = np.array(itk.size(src_img))
            input_spacing = [1,1,1]
            input_origin = [(-input_size[k] * input_spacing[k] + input_spacing[k]) / 2 for k in range(3)]

            src_img.SetSpacing(input_spacing)
            src_img.SetOrigin(input_origin)

            new_size=list(rec_array.shape)[::-1] # same size as rec (tant pis pour le spacing)

            src_img_reshaped = gatetools.applyTransformation(input=src_img, like=None, spacinglike=None, matrix=None, newsize=new_size,
                                                   neworigin=None, newspacing=None, newdirection=None, force_resample=True,
                                                   keep_original_canvas=None, adaptive=True, rotation=None,
                                                   rotation_center=None,
                                                   translation=None, pad=None, interpolation_mode="NN", bspline_order=2)

            src_array_reshaped = itk.array_from_image(src_img_reshaped)
            np.save(src_fn.replace("_src.npy", "_src_4mm.npy"), src_array_reshaped)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder")
    args = parser.parse_args()

    main()
