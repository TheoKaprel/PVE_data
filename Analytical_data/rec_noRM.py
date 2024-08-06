#!/usr/bin/env python3

import argparse
import itk
from itk import RTK as rtk
import numpy as np
import glob
import os
import random

def main():
    print(args)

    Dimension = 3
    pixelType = itk.F
    imageType = itk.Image[pixelType, Dimension]

    spacing_projs = np.array([4.7952, 4.7952, 1])
    size_projs = np.array([112, 80, 120])
    origin_projs = np.array([(-sz * sp + sp) / 2 for (sz, sp) in zip(size_projs, spacing_projs)])
    print(origin_projs)

    spacing_volume = np.array([4.7952, 4.7952, 4.7952])


    # Geometry
    list_angles = np.linspace(0,360,args.nproj+1)
    geometry = rtk.ThreeDCircularProjectionGeometry.New()
    for i in range(args.nproj):
        geometry.AddProjection(args.sid, 0, list_angles[i], origin_projs[0], origin_projs[1])


    OSEMType = rtk.OSEMConeBeamReconstructionFilter[imageType, imageType]
    osem = OSEMType.New()
    osem.SetGeometry(geometry)

    osem.SetNumberOfIterations(args.niterations)
    osem.SetNumberOfProjectionsPerSubset(args.nprojpersubset)
    osem.SetBetaRegularization(0)
    FP = osem.ForwardProjectionType_FP_ZENG
    BP = osem.BackProjectionType_BP_ZENG
    osem.SetForwardProjectionFilter(FP)
    osem.SetBackProjectionFilter(BP)
    osem.SetSigmaZero(0)
    osem.SetAlphaPSF(0)

    base = args.baseref
    list_files = glob.glob(f'{args.folder}/?????_{base}.npy')
    output_ref = base+"_rec_noRM"
    list_ready_to_rec = [l for l in list_files if not os.path.exists(l.replace(base, output_ref))]
    list_to_rec_fp = random.sample(list_ready_to_rec, args.n)

    for proj_filename in list_to_rec_fp:
        while (os.path.exists(proj_filename.replace(base, output_ref))):
                proj_filename = random.choice(list_ready_to_rec)

        print(proj_filename)
        projections_np = np.load(proj_filename)
        projections = itk.image_from_array(projections_np.astype(np.float32))
        projections.SetSpacing(spacing_projs)
        projections.SetOrigin(origin_projs)

        attmap_fn = proj_filename.replace(base, "attmap_4mm")
        attmap_np = np.load(attmap_fn)
        origin_volume = np.array([(-sz*sp+sp)/2 for (sz,sp) in zip(attmap_np.shape[::-1], spacing_volume)])
        attmap = itk.image_from_array(attmap_np.astype(np.float32))
        attmap.SetSpacing(spacing_volume)
        attmap.SetOrigin(origin_volume)

        constant_image = rtk.ConstantImageSource[imageType].New()
        constant_image.SetSpacing(attmap.GetSpacing())
        constant_image.SetOrigin(attmap.GetOrigin())
        constant_image.SetSize(itk.size(attmap))
        constant_image.SetConstant(1)
        output_image = constant_image.GetOutput()


        osem.SetInput(0, output_image)
        osem.SetInput(1, projections)
        osem.SetInput(2, attmap)

        osem.Update()
        output_rec_filename = proj_filename.replace(base,output_ref)
        output_rec = osem.GetOutput()
        output_rec_np = itk.array_from_image(output_rec)
        np.save(output_rec_filename, output_rec_np)
        print(f"Saved rec: {output_rec_filename}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-n", type = int, default = -1, help = "number of data to rec&fp")
    parser.add_argument("--folder")
    parser.add_argument("--baseref")
    parser.add_argument("--nproj", type=int)
    parser.add_argument("--sid", type=float)
    parser.add_argument("--niterations",default = 1, type = int, help = "number of iterations")
    parser.add_argument("--nprojpersubset",default = 10, type = int, help = "number of projs per subset")
    args = parser.parse_args()

    main()
