import glob

import itk
import numpy as np
import argparse
import os
import random
from itk import RTK as rtk

from parameters import get_psf_params,get_detector_params
from generate_dataset_helpers import get_dtype,strParamToArray,chooseRandomRef,generate_convex,generate_cylinder,generate_sphere,generate_ellipse,generate_bg_cylinder,sample_activity,random_3d_function,save_me


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--input_folder", '-i')
parser.add_argument('--itype', default = 'mhd', help = "Create mha, mhd,npy image")
parser.add_argument('--otype', default = 'mha', help = "Create mha, mhd,npy image")
parser.add_argument('--dtype', default = 'float64', help = "if npy, image dtype")
parser.add_argument('--nproj',type = int, default = None, help = 'if no geom, precise nb of proj angles')
parser.add_argument('--sid',type = float, default = None, help = 'if no geom, precise detector-to-isocenter distance (mm)')
parser.add_argument('--fov', type=str,help="FOV (mm,mm) of the detector. Should be in the format --fov 532,388 ")
parser.add_argument('--output_folder','-o', default = './dataset', help = " Absolute or relative path to the output folder")
parser.add_argument('--spect_system', default = "ge-discovery", choices=['ge-discovery', 'siemens-intevo-lehr', "siemens-intevo-megp"], help = 'SPECT system simulated for PVE projections')
parser.add_argument("-v", "--verbose", action="store_true")
def generate(opt):
    print(opt)
    size_proj,spacing_proj = get_detector_params(machine=opt.spect_system)
    print(f'size / spacing derived from spect_system ({opt.spect_system}) : size={size_proj}    spacing={spacing_proj}')

    offset = (-spacing_proj * size_proj + spacing_proj) / 2 #proj offset

    # Geometry

    list_angles = np.linspace(0,360,opt.nproj+1)
    geometry = rtk.ThreeDCircularProjectionGeometry.New()
    for i in range(opt.nproj):
        geometry.AddProjection(opt.sid, 0, list_angles[i], offset, offset)
    nproj = opt.nproj

    # Projections infos
    dtype = get_dtype(opt.dtype)
    pixelType = itk.F
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

    if opt.fov is not None:
        fov_is_set = True
        fov = np.array((opt.fov).split(',')).astype(np.float64)
        fovmask_np = np.zeros((nproj,size_proj,size_proj),dtype=np.float32)
        pm = (fov/spacing_proj).astype(int)
        i1, i2=(size_proj - pm[0])//2 , (size_proj - pm[1])//2
        fovmask_np[:,i2:i2+pm[1],i1:i1+pm[0]]=1
        fovmask = itk.image_from_array(fovmask_np)

        fovmask.CopyInformation(output_proj.GetOutput())
        print(fovmask.GetSpacing())
        itk.imwrite(fovmask,os.path.join(opt.output_folder, 'fov.mhd'))

        fov_maskmult=itk.MultiplyImageFilter[imageType,imageType,imageType].New()
        fov_maskmult.SetInput1(fovmask)
    else:
        fov_is_set=False

    itype = opt.itype
    otype = opt.otype

    while (len(glob.glob(os.path.join(opt.output_folder, f'*_rec_fp.{otype}')))<len(glob.glob(os.path.join(opt.input_folder, f'*_rec.{itype}')))):

        rec =  random.choice(glob.glob(os.path.join(opt.input_folder, f'*_rec.{itype}')))
        ref = rec.split(f'_rec.{itype}')[0][-5:]
        while os.path.exists(os.path.join(opt.output_folder, f'{ref}_rec_fp.{otype}')):
            rec = random.choice(glob.glob(os.path.join(opt.input_folder, f'*_rec.{itype}')))
            ref = rec.split(f'_rec.{itype}')[0][-5:]

        print(ref)
        if itype in ['mhd', 'mha']:
            rec_img = itk.imread(rec)
        elif itype=="npy":
            rec_array = np.load(rec).astype(np.float32)
            rec_img = itk.image_from_array(rec_array)
            img_shape = np.array(rec_array.shape)
            spacing = [4.7952, 4.7952, 4.7952]
            origin = [(-img_shape[k] * spacing[k] + spacing[k]) / 2 for k in range(3)]
            rec_img.SetSpacing(spacing)
            rec_img.SetOrigin(origin)

        forward_projector_rec_fp = rtk.ZengForwardProjectionImageFilter.New()
        forward_projector_rec_fp.SetInput(0, output_proj.GetOutput())
        forward_projector_rec_fp.SetGeometry(geometry)
        forward_projector_rec_fp.SetSigmaZero(0)
        forward_projector_rec_fp.SetAlpha(0)

        # forward_projs
        forward_projector_rec_fp.SetInput(1, rec_img)
        forward_projector_rec_fp.Update()
        output_rec_fp = forward_projector_rec_fp.GetOutput()
        if fov_is_set:
            fov_maskmult.SetInput2(output_rec_fp)
            output_rec_fp = fov_maskmult.GetOutput()

        save_me(img=output_rec_fp, ftype=opt.otype, output_folder=opt.output_folder, src_ref=ref,
                ref="rec_fp", grp=None, dtype=dtype)



if __name__ == '__main__':
    opt = parser.parse_args()
    generate(opt=opt)
