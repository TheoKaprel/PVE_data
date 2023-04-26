#!/usr/bin/env python3

import argparse
import itk
from itk import RTK as rtk
import numpy as np
import glob

from parameters import get_psf_params,get_detector_params

def main():
    print(args)

    size_proj,spacing_proj = get_detector_params(machine=args.spect_system)

    if args.merged:
        list_files = glob.glob(f'{args.folder}/?????_noisy_PVE_PVfree.{args.filetype}')
    else:
        list_files = glob.glob(f'{args.folder}/?????_PVE_noisy.{args.filetype}')

    Dimension = 3
    pixelType = itk.F
    imageType = itk.Image[pixelType, Dimension]

    if args.like is not None:
        like_image = itk.imread(args.like, pixelType)
        constant_image = rtk.ConstantImageSource[imageType].New()
        constant_image.CopyInformation(like_image)
        constant_image.SetConstant(1)
        output_image = constant_image.GetOutput()
    elif (args.size and args.spacing):
        output_array = np.ones((args.size,args.size,args.size))
        output_image = itk.image_from_array(output_array)
        output_image.SetSpacing([args.spacing, args.spacing, args.spacing])
        offset = (-args.size * args.spacing + args.spacing) / 2
        output_image.SetOrigin([offset, offset, offset])
        output_image = output_image.astype(pixelType)

    xmlReader = rtk.ThreeDCircularProjectionGeometryXMLFileReader.New()
    xmlReader.SetFilename(args.geom)
    xmlReader.GenerateOutputInformation()
    geometry = xmlReader.GetOutputObject()
    nproj = len(geometry.GetGantryAngles())

    OSEMType = rtk.OSEMConeBeamReconstructionFilter[imageType, imageType]
    osem = OSEMType.New()
    osem.SetInput(0, output_image)


    osem.SetGeometry(geometry)

    osem.SetNumberOfIterations(args.niterations)
    osem.SetNumberOfProjectionsPerSubset(10)
    osem.SetBetaRegularization(0)
    FP = osem.ForwardProjectionType_FP_ZENG
    BP = osem.BackProjectionType_BP_ZENG
    osem.SetForwardProjectionFilter(FP)
    osem.SetBackProjectionFilter(BP)

    sigma0_psf, alpha_psf = get_psf_params(machine=args.spect_system)
    osem.SetSigmaZero(sigma0_psf)
    osem.SetAlpha(alpha_psf)

    forward_projector = rtk.ZengForwardProjectionImageFilter.New()
    output_proj_spacing = np.array([spacing_proj,spacing_proj, 1])
    proj_offset = (-spacing_proj * size_proj + spacing_proj) / 2
    output_proj_offset = [proj_offset, proj_offset, (-nproj+1)/2]
    output_proj_array = np.zeros((nproj,size_proj, size_proj), dtype=np.float32)
    output_proj = itk.image_from_array(output_proj_array)
    output_proj.SetSpacing(output_proj_spacing)
    output_proj.SetOrigin(output_proj_offset)
    forward_projector.SetInput(0, output_proj)
    forward_projector.SetGeometry(geometry)
    forward_projector.SetSigmaZero(0)
    forward_projector.SetAlpha(0)


    for proj_filename in list_files:
        print(proj_filename)

        if args.filetype=="npy":
            projections_np = np.load(proj_filename)
            if args.merged:
                projections = itk.image_from_array(projections_np[:nproj,:,:].astype(np.float32))
            else:
                projections = itk.image_from_array(projections_np.astype(np.float32))
            projections.CopyInformation(output_proj)
        else:
            projections = itk.imread(proj_filename, pixelType)
            if args.merged:
                projections_np = itk.array_from_image(projections)
                projections_ = itk.image_from_array(projections_np[:nproj,:,:])
                projections_.CopyInformation(projections)
                projections=projections_


        osem.SetInput(1, projections)
        osem.Update()

        # if args.merged:
        #     itk.imwrite(osem.GetOutput(), proj_filename.replace(f'_noisy_PVE_PVfree.{args.filetype}', f'_rec.{args.filetype}'))
        # else:
        #     itk.imwrite(osem.GetOutput(), proj_filename.replace(f'_PVE_noisy.{args.filetype}', f'_rec.{args.filetype}'))

        forward_projector.SetInput(1, osem.GetOutput())
        forward_projector.Update()

        if args.merged:
            output_proj_rec_fp_filename = proj_filename.replace(f'_noisy_PVE_PVfree.{args.filetype}', f'_rec_fp.{args.filetype}')
        else:
            output_proj_rec_fp_filename = proj_filename.replace(f'_PVE_noisy.{args.filetype}', f'_rec_fp.{args.filetype}')

        if args.filetype=='npy':
            output_proj_rec_fp = forward_projector.GetOutput()
            output_proj_rec_fp_np = itk.array_from_image(output_proj_rec_fp)
            np.save(output_proj_rec_fp_filename,output_proj_rec_fp_np)
        else:
            itk.imwrite(forward_projector.GetOutput(), output_proj_rec_fp_filename)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder")
    parser.add_argument("--geom")
    parser.add_argument('--spect_system', default="ge-discovery", choices=['ge-discovery', 'siemens-intevo'],help='SPECT system simulated for PVE projections')
    parser.add_argument('--like', type = str)
    parser.add_argument('--size', type = int)
    parser.add_argument('--spacing', type = float)
    parser.add_argument("-n", "--niterations",default = 1, type = int, help = "number of iterations")
    parser.add_argument("--filetype",default = "mhd", choices = ['mhd', 'mha', 'npy'])
    parser.add_argument("--merged",action ="store_true")
    args = parser.parse_args()

    main()
