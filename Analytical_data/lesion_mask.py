#!/usr/bin/env python3

import argparse
import h5py
import numpy as np
from itk import RTK as rtk
import itk
import os
import time

def main():
    print(args)

    t0=time.time()

    h5file = h5py.File(args.hdf, 'r+')
    refs=list(h5file.keys())

    # Projections infos
    spacing_proj=2.3976
    nproj=120
    size_proj=256
    sid = 280

    # pixelType = itk.F
    pixelType = itk.ctype("float")
    imageType = itk.Image[pixelType, 3]
    output_spacing = [spacing_proj,spacing_proj, 1]
    offset = (-spacing_proj * size_proj + spacing_proj) / 2
    output_offset = [offset, offset, (-nproj+1)/2]
    output_proj = rtk.ConstantImageSource[imageType].New()
    output_proj.SetSpacing(output_spacing)
    output_proj.SetOrigin(output_offset)
    output_proj.SetSize([size_proj, size_proj, nproj])
    output_proj.SetConstant(0.)
    output_proj.Update()

    fov = np.array(("532,388").split(',')).astype(np.float64)
    fovmask_np = np.zeros((nproj,size_proj,size_proj),dtype=np.float32)
    pm = (fov/spacing_proj).astype(int)
    i1, i2=(size_proj - pm[0])//2 , (size_proj - pm[1])//2
    fovmask_np[:,i2:i2+pm[1],i1:i1+pm[0]]=1
    fovmask = itk.image_from_array(fovmask_np)
    fovmask.CopyInformation(output_proj.GetOutput())
    print(fovmask.GetSpacing())
    print(fovmask.GetOrigin())
    print(fovmask.GetLargestPossibleRegion())
    fov_maskmult=itk.MultiplyImageFilter[imageType,imageType,imageType].New()
    fov_maskmult.SetInput1(fovmask)

    list_angles = np.linspace(0, 360, nproj + 1)
    GeometryType = rtk.ThreeDCircularProjectionGeometry
    geometry = GeometryType.New()
    for i in range(nproj):
        if args.fp=="Zeng":
            geometry.AddProjection(sid, 0, list_angles[i], offset, offset)
        elif args.fp=="Joseph":
            geometry.AddProjection(sid, 0, list_angles[i],0,0)


    for ref in refs:
        grp=h5file[ref]
        src=np.array(grp['src'], dtype=np.float32)

        lesion_mask=(src>2).astype(np.float32)
        lesion_mask_img=itk.image_from_array(lesion_mask)
        spacing=[2.3976,2.3976,2.3976]
        size=list(lesion_mask_img.shape)
        origin= [-sp*(sz-1)/2 for (sp,sz) in zip(spacing,size)]
        lesion_mask_img.SetSpacing(spacing)
        lesion_mask_img.SetOrigin(origin[::-1])

        # forwardprojection of lesion mask
        if args.fp=="Zeng":
            forward_projector = rtk.ZengForwardProjectionImageFilter.New()
        elif args.fp=="Joseph":
            forward_projector = rtk.JosephForwardProjectionImageFilter[imageType,imageType].New()

        forward_projector.SetInput(0, output_proj.GetOutput())
        forward_projector.SetGeometry(geometry)
        if args.fp=="Zeng":
            forward_projector.SetSigmaZero(0)
            forward_projector.SetAlpha(0)

        forward_projector.SetInput(1, lesion_mask_img)
        forward_projector.Update()
        lesion_mask_fp = forward_projector.GetOutput()
        fov_maskmult.SetInput2(lesion_mask_fp)
        lesion_mask_fp = fov_maskmult.GetOutput()

        # gaussian filter
        gaussian_filter=itk.DiscreteGaussianImageFilter[imageType,imageType].New()
        gaussian_filter.SetInput(lesion_mask_fp)
        gaussian_filter.SetSigma(spacing_proj/2)
        gaussian_filter.Update()
        lesion_mask_fp = gaussian_filter.GetOutput()

        lesion_mask_fp_array=itk.array_from_image(lesion_mask_fp)
        lesion_mask_fp_array = (lesion_mask_fp_array>0).astype(np.float16)
        output_fn = os.path.join(args.outputfolder, f'{ref}_lesion_mask_fp.npy')
        np.save(output_fn, lesion_mask_fp_array)
        print(output_fn)

    print("Done!")
    print(f"Time Elapsed : {time.time() - t0} s")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf")
    parser.add_argument("--outputfolder")
    parser.add_argument("--fp", choices=['Zeng', 'Joseph'])
    args = parser.parse_args()

    main()
