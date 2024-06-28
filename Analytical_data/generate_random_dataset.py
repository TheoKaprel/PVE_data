import glob

import itk
import numpy as np
import argparse
import os
import random
from itk import RTK as rtk
import time
import json
import h5py
import gatetools

from parameters import get_psf_params,get_detector_params
from generate_dataset_helpers import get_dtype,strParamToArray,chooseRandomRef,generate_convex,generate_cylinder,generate_sphere,generate_ellipse,generate_bg_cylinder,sample_activity,random_3d_function,save_me


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--nb_data','-n', type = int, required = True, help = 'number of desired data = (src,projPVE,projPVfree)')
parser.add_argument('--size_volume', type = str, default = "150", help = 'Size of the desired image i.e. number of voxels per dim')
parser.add_argument('--spacing_volume', type = str, default = "4", help = 'Spacing of the desired image i.e phyisical length of a voxels (mm)')
parser.add_argument('--size_proj', type = int, default = None, help = 'Size of the desired projections')
parser.add_argument('--spacing_proj', type = float, default = None, help = 'Spacing of the desired projection. Ex intevo : 2.3976')
parser.add_argument('--type', default = 'mha', help = "Create mha, mhd,npy image")
parser.add_argument('--dtype', default = 'float64', help = "if npy, image dtype")
parser.add_argument('--like', default = None, help = "Instead of specifying spacing/size, you can specify an image as a metadata model")
parser.add_argument('--min_radius', default = 4,type = float, help = 'minimum radius of the random spheres')
parser.add_argument('--max_radius', default = 32,type = float, help = 'max radius of the random spheres')
parser.add_argument('--prop_radius', default = "uniform", choices=['uniform', 'squared_inv'], help = 'proportion of radius between min/max')
parser.add_argument('--min_ratio', default = 10,type = float, help = 'min bg:src ratio. If no background, it is the min activity')
parser.add_argument('--max_ratio', default = 20,type = float, help = 'max bg:src ratio. If no background, it is the max activity')
parser.add_argument('--min_activity', default = 10, type = float, help = "minimum activity in MBq. Then, N_min = A_min * 1e6 * 20s * efficiency ")
parser.add_argument('--max_activity', default = 100, type = float, help = "maximal activity in MBq. Then, N_min = A_max * 1e6 * 20s * efficiency")
parser.add_argument('--nspheres', default = 1,type = int, help = 'max number of spheres to generate on each source')
parser.add_argument('--background', action= 'store_true', help = 'If you want background add --background')
parser.add_argument('--sphere',type = float, default = 0, help = "if --sphere p, activity sources are spheres with proba p")
parser.add_argument('--ellipse',type = float, default = 0, help = "if --ellipse p, activity sources are ellipses with proba p")
parser.add_argument('--cylinder',type = float, default = 0, help = "if --cylinder p, activity sources are cylinders with proba p")
parser.add_argument('--convex',type = float, default = 0, help = "if --convex p, activity sources are convexs with proba p")
parser.add_argument('--grad_act',action ="store_true", help = "if --grad_act, hot sources are not homogeneous")
parser.add_argument('--geom', '-g', default = None, help = 'geometry file to forward project')
parser.add_argument('--nproj',type = int, default = None, help = 'if no geom, precise nb of proj angles')
parser.add_argument('--sid',type = float, default = None, help = 'if no geom, precise detector-to-isocenter distance (mm)')
parser.add_argument('--fov', type=str,help="FOV (mm,mm) of the detector. Should be in the format --fov 532,388 ")
parser.add_argument('--attenuationmapfolder',default = None, help = 'path to the attenuationmaps folder (random choice in the folder)')
parser.add_argument('--attmapaugmentation', action="store_true", help = "add this if data augmentation is needed for the attenuation map. Max rotation : (5,360,5) and max translation : (50,50,50) ")
parser.add_argument('--organlabels', type = str, help = "use --organlabels if you want to assign different activity ratios to main organs. For now: body, liver, kidneys, and bones.")
parser.add_argument('--organratios', type = str, help = "if --organlabels is specified, you have to also specify min/ma ratios for each organ in organlabels")
parser.add_argument('--output_folder','-o', default = './dataset', help = " Absolute or relative path to the output folder")
parser.add_argument('--spect_system', default = "ge-discovery", choices=['ge-discovery', 'siemens-intevo-lehr', "siemens-intevo-megp"], help = 'SPECT system simulated for PVE projections')
parser.add_argument('--save_src',action ="store_true", help = "if you want to also save the source that will be forward projected")
parser.add_argument('--lesion_mask',action ="store_true", help = "if you want to also save the source that will be forward projected")
parser.add_argument('--rec_fp',type = int, default = 0, help = "noisy projections are reconstructed with 1 osem-rm iter and forward-projected w/o rm to obtain ABCDE_rec_fp.mha")
parser.add_argument("-v", "--verbose", action="store_true")
def generate(opt):
    print(opt)
    current_date = time.strftime("%d_%m_%Y_%Hh_%Mm_%Ss", time.localtime())
    opt.date = current_date
    dataset_infos = vars(opt)

    t0 = time.time()

    sigma0_psf, alpha_psf,efficiency = get_psf_params(opt.spect_system)
    dataset_infos['sigma0_psf'] = sigma0_psf
    dataset_infos['alpha_psf'] = alpha_psf
    dataset_infos['efficiency'] = efficiency


    if (opt.spacing_proj is None) and (opt.size_proj is None) and (opt.spect_system is not None):
        size_proj,spacing_proj = get_detector_params(machine=opt.spect_system)
        print(f'size / spacing derived from spect_system ({opt.spect_system}) : size={size_proj}    spacing={spacing_proj}')
        dataset_infos['size_proj']=size_proj
        dataset_infos['spacing_proj']=spacing_proj
    else:
        size_proj,spacing_proj=opt.size_proj,opt.spacing_proj

    offset = (-spacing_proj * size_proj + spacing_proj) / 2 #proj offset

    # Geometry
    if ((opt.geom is not None) and (opt.nproj is None) and (opt.sid is None)):
        xmlReader = rtk.ThreeDCircularProjectionGeometryXMLFileReader.New()
        xmlReader.SetFilename(opt.geom)
        xmlReader.GenerateOutputInformation()
        geometry = xmlReader.GetOutputObject()
        nproj = len(geometry.GetGantryAngles())
        dataset_infos['nproj'] = nproj
        dataset_infos['sid'] = geometry.GetSourceToIsocenterDistances()[0]
    elif ((opt.geom is None) and (opt.nproj is not None) and (opt.sid is not None)):
        list_angles = np.linspace(0,360,opt.nproj+1)
        geometry = rtk.ThreeDCircularProjectionGeometry.New()
        for i in range(opt.nproj):
            geometry.AddProjection(opt.sid, 0, list_angles[i], offset, offset)
        nproj = opt.nproj
    else:
        print('ERROR: give me geom xor (nproj and sid)')
        exit(0)

    sid = geometry.GetSourceToIsocenterDistances()[0]

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

    min_ratio, Max_ratio = opt.min_ratio, opt.max_ratio
    if opt.background:
        background_radius_x_mean, background_radius_z_mean,background_radius_y_mean = 200, 120,300
        background_radius_x_std, background_radius_z_std,background_radius_y_std = 20, 10, 100
        dataset_infos['bg_shape_params'] = {'mean_xzy': f'({background_radius_x_mean},{background_radius_z_mean},{background_radius_y_mean})',
                                            'std_xzy': f'({background_radius_x_std},{background_radius_z_std},{background_radius_y_std})'}

    p_sphere,p_ellipse,p_cylinder,p_convex = opt.sphere,opt.ellipse,opt.cylinder,opt.convex
    assert(p_sphere+p_ellipse+p_cylinder+p_convex==1)

    time_per_proj = 20
    min_activity,max_activity = opt.min_activity, opt.max_activity
    min_count= int(min_activity * 1e6 * time_per_proj * efficiency)
    max_count= int(max_activity * 1e6 * time_per_proj * efficiency)

    print(f'Activity between {min_activity} MBq and {max_activity} MBq --> nb of counts between {min_count} and {max_count}')
    """
     Attention, my bad, c pas vrai que l activité de Lu177 ou Tc99m est de 20/100 MBq.
     Quand je dis ça je ne compte QUE les désintégrations qui produisent un gamma de 208 kev, c'est à dire 11% d'entre elles. 
     Donc 100MBq ici ça signifie 100 * 100/11 = 909 MBq de vrai Lu177
     """

    print(json.dumps(dataset_infos, indent = 3))

    dataset_infos['src_refs']=[]

    if opt.attenuationmapfolder is not None:
        attmap_refs_list = glob.glob(os.path.join(opt.attenuationmapfolder, '*_attmap_cropped_rot.mhd'))
        with_attmaps=True
        if opt.organlabels is not None:
            organ_labels = open(opt.organlabels).read()
            organ_labels = json.loads(organ_labels)

            organ_ratios = open(opt.organratios).read()
            organ_ratios = json.loads(organ_ratios)
            with_rois = True
        else:
            with_rois = False
    else:
        with_attmaps=False

    if (opt.type=='h5'):
        to_hdf=True
        hdf_file_filename = os.path.join(opt.output_folder, f'{os.path.basename(os.path.abspath(opt.output_folder))}.h5')
        print(hdf_file_filename)
        hdf_file = h5py.File(hdf_file_filename, 'a')
        str_dtype = h5py.special_dtype(vlen=str)
    else:
        to_hdf=False

    time_src=0

    for n in range(opt.nb_data):
        # Random output filename
        source_ref = chooseRandomRef(Nletters=5)

        if to_hdf:
            grp=hdf_file.create_group(source_ref)
        else:
            grp=None

        if with_attmaps:
            attmap_ref = random.choice(attmap_refs_list)
            if to_hdf:
                juste_ref = os.path.basename(os.path.abspath(attmap_ref))
                dset_attmap_ref = grp.create_dataset("attmap_ref",(len(juste_ref)+5), dtype=str_dtype)
                dset_attmap_ref[0] = juste_ref

        if opt.verbose:
            if with_attmaps:
                print(f'{attmap_ref} / {source_ref}')
            else:
                print(source_ref)

        forward_projector = rtk.ZengForwardProjectionImageFilter.New()
        forward_projector.SetInput(0, output_proj.GetOutput())
        forward_projector.SetGeometry(geometry)

        # get source image parameters
        if with_attmaps:
            attmap = itk.imread(attmap_ref,pixel_type=pixelType)
            if with_rois:
                labels_fn = attmap_ref.replace('_attmap_cropped_rot.mhd', '_rois_labels_cropped_rot.mhd')
                labels = itk.imread(labels_fn)


            if opt.attmapaugmentation:
                rot=np.random.rand(3)*[5,360,5]
                transl = np.random.rand(3)*100-50
                attmap = gatetools.applyTransformation(input=attmap, like=None, spacinglike=None, matrix=None, newsize=None,
                                              neworigin=None, newspacing=None, newdirection=None, force_resample=True,
                                              keep_original_canvas=None, adaptive=None, rotation=rot, rotation_center=None,
                                              translation=transl, pad=None, interpolation_mode=None, bspline_order=2)

                if with_rois:
                    labels = gatetools.applyTransformation(input=labels, like=None, spacinglike=None, matrix=None, newsize=None,
                                              neworigin=None, newspacing=None, newdirection=None, force_resample=True,
                                              keep_original_canvas=None, adaptive=None, rotation=rot, rotation_center=None,
                                              translation=transl, pad=None, interpolation_mode=None, bspline_order=2)

            if with_rois:
                labels_array = itk.array_from_image(labels)


            # save_me(img=attmap,ftype=opt.type,output_folder=opt.output_folder, src_ref=source_ref,
            #         ref="attmap", grp=grp,dtype=dtype)

            forward_projector_attmap = rtk.ZengForwardProjectionImageFilter.New()
            forward_projector_attmap.SetInput(0, output_proj.GetOutput())
            forward_projector_attmap.SetGeometry(geometry)
            forward_projector_attmap.SetInput(1, attmap)
            forward_projector_attmap.SetSigmaZero(0)
            forward_projector_attmap.SetAlpha(0)
            forward_projector_attmap.Update()
            attmap_fp = forward_projector_attmap.GetOutput()
            attmap_fp.DisconnectPipeline()
            if fov_is_set:
                fov_maskmult.SetInput2(attmap_fp)
                attmap_fp = fov_maskmult.GetOutput()

            save_me(img=attmap_fp,ftype=opt.type,output_folder=opt.output_folder, src_ref=source_ref,
                    ref="attmap_fp", grp=grp,dtype=dtype)

            attmap_np = itk.array_from_image(attmap)
            vSpacing = np.array(attmap.GetSpacing())[::-1]
            vSize = np.array(attmap_np.shape)
            vOffset = np.array(attmap.GetOrigin())[::-1]

            forward_projector_with_att = rtk.ZengForwardProjectionImageFilter.New()
            forward_projector_with_att.SetInput(0, output_proj.GetOutput())
            forward_projector_with_att.SetGeometry(geometry)

            forward_projector_with_att.SetInput(2, attmap)
        elif opt.like is not None:
            im_like = itk.imread(opt.like)
            vSpacing = np.array(im_like.GetSpacing())[::-1]
            vSize = np.array(itk.size(im_like))[::-1]
            vOffset = np.array(im_like.GetOrigin())[::-1]
        else:
            vSize = strParamToArray(opt.size_volume).astype(int)
            vSpacing = strParamToArray(opt.spacing_volume)
            vOffset = [(-sp * size + sp) / 2 for (sp, size) in zip(vSpacing, vSize)]

        time_src_0=time.time()
        # matrix settings
        lengths = vSize * vSpacing
        lspaceX = np.linspace(-vSize[0] * vSpacing[0] / 2, vSize[0] * vSpacing[0] / 2, vSize[0])
        lspaceY = np.linspace(-vSize[1] * vSpacing[1] / 2, vSize[1] * vSpacing[1] / 2, vSize[1])
        lspaceZ = np.linspace(-vSize[2] * vSpacing[2] / 2, vSize[2] * vSpacing[2] / 2, vSize[2])

        X, Y, Z = np.meshgrid(lspaceX, lspaceY, lspaceZ, indexing='ij')

        src_array = np.zeros_like(X)

        if opt.background:
            # background = cylinder with revolution axis = Y
            background_array = np.zeros_like(X)

            if ((with_attmaps) and (with_rois)):
                background_array[labels_array>0]=1
            elif ((with_attmaps) and (attmap_np.max()>0)):
                background_array[attmap_np>0.01]=1
            else:
                while (background_array.max()==0): # to avoid empty background
                    bg_center = np.random.randint(-50,50,3)
                    bg_radius_xzy = (background_radius_x_std, background_radius_z_std, background_radius_y_std) * np.random.randn(3) + (background_radius_x_mean, background_radius_z_mean, background_radius_y_mean)
                    bg_level = 1
                    background_array = generate_bg_cylinder(X,Y,Z,activity=bg_level,center=bg_center,radius_xzy=bg_radius_xzy)


            if opt.grad_act:
                M = 5
                background_array = background_array * random_3d_function(a0 = 10,xx = X,yy=Y,zz=Z,M=M)/10
                background_array[background_array<0] = 0

            src_array += background_array


        lesion_array=np.zeros_like(X)
        random_nb_of_sphers = np.random.randint(1,opt.nspheres)
        if opt.grad_act:
            M = 8
            rndm_grad_act = random_3d_function(a0=10, xx=X, yy=Y, zz=Z, M=M)/10

            # rndm_grad_act scaled between 0.5 and 1.5.
            rndm_grad_act_0_1 =  (rndm_grad_act - rndm_grad_act.min()) / (rndm_grad_act.max() - rndm_grad_act.min())
            min_scale, max_scale = 0.5, 1.5
            rndm_grad_act_scaled = rndm_grad_act_0_1 * (max_scale - min_scale) + min_scale

        if opt.organlabels is not None:
            for organ in organ_labels.keys():
                if ((organ!="body") and (np.random.rand()>2/3)): # choose each organ (except background body) with proba 1/3
                    if (labels_array==int(organ_labels[organ])).any(): # if the organ is present in the attmap, then...

                        min_ratio_rois = organ_ratios[organ][0]
                        max_ratio_rois = organ_ratios[organ][1]

                        organ_rndm_activity = np.random.rand() * (max_ratio_rois - min_ratio_rois) + min_ratio_rois

                        if opt.grad_act:
                            mean = np.mean(rndm_grad_act_scaled[labels_array==int(organ_labels[organ])])
                            rndm_grad_act_scaled_scaled = rndm_grad_act_scaled / mean
                            organ_act = (labels_array==int(organ_labels[organ])) * (rndm_grad_act_scaled_scaled * organ_rndm_activity)
                        else:
                            organ_act = (labels_array == int(organ_labels[organ])) * (organ_rndm_activity)

                        src_array+=organ_act

        for s in range(random_nb_of_sphers):
            # random_activity = sample_activity(min_r=min_ratio,max_r=Max_ratio,lbda=lbda,with_bg=opt.background)
            random_activity = np.random.rand()*(Max_ratio-min_ratio)+min_ratio

            if opt.background is None:
                center = (2 * np.random.rand(3) - 1) * (lengths / 2)
            else:
                # center of the sphere inside the background
                center_index = np.random.randint(X.shape)
                while (background_array[center_index[0], center_index[1], center_index[2]]==0):
                    center_index = np.random.randint(X.shape)
                center = [lspaceX[center_index[0]], lspaceY[center_index[1]], lspaceZ[center_index[2]]]


            rdn_shape = random.random()
            if rdn_shape<p_sphere: # sphere
                lesion = generate_sphere(center=center,X=X,Y=Y,Z=Z,min_radius=opt.min_radius, max_radius = opt.max_radius, prop_radius = opt.prop_radius)
            elif rdn_shape<p_sphere+p_ellipse: # ellipse
                lesion = generate_ellipse(center=center,X=X,Y=Y,Z=Z,min_radius=opt.min_radius, max_radius = opt.max_radius, prop_radius = opt.prop_radius)
            elif rdn_shape<p_sphere+p_ellipse+p_cylinder: # cylinder
                lesion = generate_cylinder(X=X, Y=Y, Z=Z, center=center, min_radius=opt.min_radius,max_radius=opt.max_radius, prop_radius=opt.prop_radius)
            else: # convex shape
                lesion = generate_convex(X=X,Y=Y,Z=Z,center=center,min_radius=opt.min_radius, max_radius = opt.max_radius, prop_radius = opt.prop_radius)


            if opt.grad_act:
                rndm_grad_act_scaled_scaled = rndm_grad_act_scaled / np.mean(rndm_grad_act_scaled[lesion>0])
                lesion = lesion * (rndm_grad_act_scaled_scaled*random_activity)
            else:
                lesion = random_activity * lesion

            lesion_array += lesion

        src_array[lesion_array > 0] = lesion_array[lesion_array > 0]

        time_src+=(time.time() - time_src_0)
        if opt.verbose:
            print('fp...')

        total_counts_per_proj = round(np.random.rand() * (max_count - min_count) + min_count)
        print(f"{total_counts_per_proj} ({total_counts_per_proj/(1e6 * time_per_proj * efficiency)} MBq)")
        src_array_normedToTotalCounts = src_array / np.sum(src_array) * total_counts_per_proj * spacing_proj**2 / (vSpacing[0]*vSpacing[1]*vSpacing[2])

        src_img_normedToTotalCounts = itk.image_from_array(src_array_normedToTotalCounts.astype(np.float32))
        src_img_normedToTotalCounts.SetSpacing(vSpacing[::-1])
        src_img_normedToTotalCounts.SetOrigin(vOffset[::-1])

        if opt.rec_fp>0:
            if with_attmaps:
                attmap_rec_fp = itk.imread(attmap_ref.replace('.mhd', '_4mm.mhd'), pixel_type=pixelType)
                if opt.attmapaugmentation:
                    # apply the same transformation that to attmap
                    attmap_rec_fp = gatetools.applyTransformation(input=attmap_rec_fp, like=None, spacinglike=None, matrix=None, newsize=None,
                                                  neworigin=None, newspacing=None, newdirection=None, force_resample=True,
                                                  keep_original_canvas=None, adaptive=None, rotation=rot, rotation_center=None,
                                                  translation=transl, pad=None, interpolation_mode=None, bspline_order=2)
                save_me(img=attmap_rec_fp, ftype=opt.type, output_folder=opt.output_folder, src_ref=source_ref,
                        ref="attmap_4mm", grp=grp, dtype=dtype)

                attmap_rec_fp_np = itk.array_from_image(attmap_rec_fp)
                vSpacing_recpfp = np.array(attmap_rec_fp.GetSpacing())
                vSize_recfp = np.array(attmap_rec_fp_np.shape)[::-1]
                vOffset_recfp = np.array(attmap_rec_fp.GetOrigin())
            else:
                vSpacing_recpfp = np.array([4.7952, 4.7952, 4.7952])
                vSize_recfp = np.array([128, 128, 128])
                vOffset_recfp = [(-sp * size + sp) / 2 for (sp, size) in zip(vSpacing_recpfp, vSize_recfp)]


        # saving of source 3D image
        if opt.save_src:
            save_me(img=src_img_normedToTotalCounts, ftype=opt.type, output_folder=opt.output_folder, src_ref=source_ref,
                    ref="src", grp=grp, dtype=dtype)

            if opt.rec_fp>0:
                src_img_normedToTotalCounts_4mm = gatetools.applyTransformation(input=src_img_normedToTotalCounts, like=attmap_rec_fp, spacinglike=None, matrix=None, newsize=None,
                                                       neworigin=None, newspacing=None, newdirection=None,
                                                       force_resample=True,
                                                       keep_original_canvas=None, adaptive=None, rotation=None,
                                                       rotation_center=None,
                                                       translation=None, pad=None, interpolation_mode="NN",
                                                       bspline_order=2)

                save_me(img=src_img_normedToTotalCounts_4mm, ftype=opt.type, output_folder=opt.output_folder, src_ref=source_ref,
                        ref="src_4mm", grp=grp, dtype=dtype)


        if opt.lesion_mask:
            lesion_mask = (lesion_array > 0).astype(np.float32)
            lesion_mask_img = itk.image_from_array(lesion_mask)
            lesion_mask_img.CopyInformation(src_img_normedToTotalCounts)


            lesion_mask_4mm = gatetools.applyTransformation(input=lesion_mask_img,
                                                                            like=attmap_rec_fp, spacinglike=None,
                                                                            matrix=None, newsize=None,
                                                                            neworigin=None, newspacing=None,
                                                                            newdirection=None,
                                                                            force_resample=True,
                                                                            keep_original_canvas=None, adaptive=None,
                                                                            rotation=None,
                                                                            rotation_center=None,
                                                                            translation=None, pad=None,
                                                                            interpolation_mode="NN",
                                                                            bspline_order=2)
            save_me(img=lesion_mask_4mm,ftype=opt.type, output_folder = opt.output_folder, src_ref = source_ref,
                    ref = "lesion_mask_4mm", grp = grp, dtype = np.uint16)

            forward_projector.SetInput(1, lesion_mask_img)

            forward_projector.SetSigmaZero(sigma0_psf)
            forward_projector.SetAlpha(alpha_psf)
            forward_projector.Update()
            output_forward_lesion_mask = forward_projector.GetOutput()
            output_forward_lesion_mask.DisconnectPipeline()
            if fov_is_set:
                fov_maskmult.SetInput2(output_forward_lesion_mask)
                output_forward_lesion_mask = fov_maskmult.GetOutput()

            lesion_mask_fp_array = itk.array_from_image(output_forward_lesion_mask)
            lesion_mask_fp_array = (lesion_mask_fp_array > 0.05*lesion_mask_fp_array.max()).astype(np.float32)

            save_me(array=lesion_mask_fp_array,ftype=opt.type, output_folder = opt.output_folder, src_ref = source_ref,
                    ref = "lesion_mask_fp", grp = grp, dtype = np.uint16, img_like = output_forward_lesion_mask)


        # fowardprojections :
        forward_projector.SetInput(1, src_img_normedToTotalCounts)

        #proj PVfree
        forward_projector.SetSigmaZero(0)
        forward_projector.SetAlpha(0)
        forward_projector.Update()
        output_forward_PVfree = forward_projector.GetOutput()
        output_forward_PVfree.DisconnectPipeline()
        if fov_is_set:
            fov_maskmult.SetInput2(output_forward_PVfree)
            output_forward_PVfree = fov_maskmult.GetOutput()

        save_me(img=output_forward_PVfree, ftype=opt.type, output_folder=opt.output_folder, src_ref=source_ref,
                ref="PVfree", grp=grp, dtype=dtype)

        if with_attmaps:
            forward_projector_with_att.SetInput(1, src_img_normedToTotalCounts)

            # proj att+PVE
            forward_projector_with_att.SetSigmaZero(sigma0_psf)
            forward_projector_with_att.SetAlpha(alpha_psf)
            forward_projector_with_att.Update()
            output_forward_PVE = forward_projector_with_att.GetOutput()
            output_forward_PVE.DisconnectPipeline()
            if fov_is_set:
                fov_maskmult.SetInput2(output_forward_PVE)
                output_forward_PVE = fov_maskmult.GetOutput()


            save_me(img=output_forward_PVE, ftype=opt.type, output_folder=opt.output_folder, src_ref=source_ref,
                    ref="PVE_att", grp=grp, dtype=dtype)

            # proj noise(att+PVE)
            output_forward_PVE_array = itk.array_from_image(output_forward_PVE).astype(dtype=dtype)
            noisy_projection_array = np.random.poisson(lam=output_forward_PVE_array, size=output_forward_PVE_array.shape).astype(dtype=np.float64)
            save_me(array=noisy_projection_array, ftype=opt.type, output_folder=opt.output_folder, src_ref=source_ref,
                    ref="PVE_att_noisy", grp=grp, dtype=np.uint16, img_like=output_forward_PVE)


            # proj att+PVfree
            forward_projector_with_att.SetSigmaZero(0)
            forward_projector_with_att.SetAlpha(0)
            forward_projector_with_att.Update()
            output_forward_PVfree_att = forward_projector_with_att.GetOutput()
            output_forward_PVfree_att.DisconnectPipeline()
            if fov_is_set:
                fov_maskmult.SetInput2(output_forward_PVfree_att)
                output_forward_PVfree_att = fov_maskmult.GetOutput()

            save_me(img=output_forward_PVfree_att, ftype=opt.type, output_folder=opt.output_folder, src_ref=source_ref,
                    ref="PVfree_att", grp=grp, dtype=dtype)



        else:
            # proj PVE
            forward_projector.SetSigmaZero(sigma0_psf)
            forward_projector.SetAlpha(alpha_psf)
            forward_projector.Update()
            output_forward_PVE = forward_projector.GetOutput()
            output_forward_PVE.DisconnectPipeline()
            if fov_is_set:
                fov_maskmult.SetInput2(output_forward_PVE)
                output_forward_PVE = fov_maskmult.GetOutput()

            save_me(img=output_forward_PVE, ftype=opt.type, output_folder=opt.output_folder, src_ref=source_ref,
                    ref="PVE", grp=grp, dtype=dtype)


            # proj noise(PVE)
            output_forward_PVE_array = itk.array_from_image(output_forward_PVE).astype(dtype=dtype)
            noisy_projection_array = np.random.poisson(lam=output_forward_PVE_array, size=output_forward_PVE_array.shape).astype(dtype=np.float64)

            save_me(array=noisy_projection_array, ftype=opt.type, output_folder=opt.output_folder, src_ref=source_ref,
                    ref="PVE_noisy", grp=grp, dtype=np.uint16, img_like=output_forward_PVE)


        if opt.rec_fp>0:
            print('rec_fp...')
            constant_image = rtk.ConstantImageSource[imageType].New()
            constant_image.SetSpacing(vSpacing_recpfp)
            constant_image.SetOrigin(vOffset_recfp)
            constant_image.SetSize([int(s) for s in vSize_recfp])
            constant_image.SetConstant(1)
            output_rec = constant_image.GetOutput()

            OSEMType = rtk.OSEMConeBeamReconstructionFilter[imageType, imageType]
            osem = OSEMType.New()
            osem.SetInput(0, output_rec)
            osem.SetGeometry(geometry)
            osem.SetNumberOfIterations(opt.rec_fp)
            osem.SetNumberOfProjectionsPerSubset(15)
            osem.SetBetaRegularization(0)

            FP = osem.ForwardProjectionType_FP_ZENG
            BP = osem.BackProjectionType_BP_ZENG
            osem.SetSigmaZero(sigma0_psf)
            osem.SetAlphaPSF(alpha_psf)
            osem.SetForwardProjectionFilter(FP)
            osem.SetBackProjectionFilter(BP)

            if with_attmaps:
                osem.SetInput(2, attmap_rec_fp)

            forward_projector_rec_fp_att = rtk.ZengForwardProjectionImageFilter.New()
            forward_projector_rec_fp_att.SetInput(0, output_proj.GetOutput())
            forward_projector_rec_fp_att.SetGeometry(geometry)
            forward_projector_rec_fp_att.SetSigmaZero(0)
            forward_projector_rec_fp_att.SetAlpha(0)

            forward_projector_rec_fp = rtk.ZengForwardProjectionImageFilter.New()
            forward_projector_rec_fp.SetInput(0, output_proj.GetOutput())
            forward_projector_rec_fp.SetGeometry(geometry)
            forward_projector_rec_fp.SetSigmaZero(0)
            forward_projector_rec_fp.SetAlpha(0)


            output_forward_PVE_noisy = itk.image_from_array(noisy_projection_array.astype(dtype=np.float32))
            output_forward_PVE_noisy.CopyInformation(output_forward_PVE)
            osem.SetInput(1, output_forward_PVE_noisy)
            osem.Update()
            rec_volume = osem.GetOutput()
            rec_volume.DisconnectPipeline()

            # save rec_fp
            save_me(img=rec_volume, ftype=opt.type, output_folder=opt.output_folder, src_ref=source_ref,
                    ref="rec", grp=grp, dtype=dtype)

            # forward_projs rec_fp_att
            forward_projector_rec_fp_att.SetInput(1, rec_volume)
            forward_projector_rec_fp_att.SetInput(2, attmap_rec_fp)
            forward_projector_rec_fp_att.Update()
            output_rec_fp_att = forward_projector_rec_fp_att.GetOutput()
            if fov_is_set:
                fov_maskmult.SetInput2(output_rec_fp_att)
                output_rec_fp_att = fov_maskmult.GetOutput()

            save_me(img=output_rec_fp_att, ftype=opt.type, output_folder=opt.output_folder, src_ref=source_ref,
                    ref="rec_fp_att", grp=grp, dtype=dtype)

            # forward_projs rec_fp
            forward_projector_rec_fp.SetInput(1, rec_volume)
            forward_projector_rec_fp.Update()
            output_rec_fp = forward_projector_rec_fp.GetOutput()
            if fov_is_set:
                fov_maskmult.SetInput2(output_rec_fp)
                output_rec_fp = fov_maskmult.GetOutput()

            save_me(img=output_rec_fp, ftype=opt.type, output_folder=opt.output_folder, src_ref=source_ref,
                    ref="rec_fp", grp=grp, dtype=dtype)


        if with_attmaps:
            dataset_infos['src_refs'].append([source_ref,attmap_ref])
        else:
            dataset_infos['src_refs'].append(source_ref)

    print(dataset_infos['src_refs'])

    dataset_params_fn = os.path.join(opt.output_folder, 'dataset_infos.txt')
    dataset_infos_file = open(dataset_params_fn,'a')
    dataset_infos_file.writelines([str(u)+'\n' for u in dataset_infos['src_refs'] ])
    dataset_infos_file.close()

    tf = time.time()
    elapsed_time = round(tf - t0)
    elapsed_time_min = round(elapsed_time/60)
    dataset_infos['elapsed_time_s'] = elapsed_time
    print(f'Total time elapsed for data generation : {elapsed_time_min} min    (i.e. {elapsed_time} s)')
    print(f'Including {time_src} s for src creation')

    formatted_dataset_infos = json.dumps(dataset_infos, indent=4)
    output_info_json = os.path.join(opt.output_folder, f'dataset_infos_{current_date}_{source_ref}.json')
    jsonfile = open(output_info_json, "w")
    jsonfile.write(formatted_dataset_infos)
    jsonfile.close()



if __name__ == '__main__':
    opt = parser.parse_args()
    generate(opt=opt)
