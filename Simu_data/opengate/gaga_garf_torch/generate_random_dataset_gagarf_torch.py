import glob

import itk
import numpy as np
import argparse
import os
import random
import string
from itk import RTK as rtk
import time
import json
from scipy.spatial.transform import Rotation as Rot
from scipy.interpolate import RegularGridInterpolator
from skimage.morphology import convex_hull_image
import torch
from tqdm import tqdm

from pathlib import Path
import sys
path_root = Path(__file__).parents[4]
sys.path.append(str(path_root))
#
from PVE_data.Analytical_data.parameters import get_psf_params,get_detector_params


from gaga_garf.cgan_source import CGANSOURCE,ConditionsDataset
from gaga_garf.garf_detector import GARF,DetectorPlane,project_on_plane


def get_dtype(opt_dtype):
    if opt_dtype=='float64':
        return np.float64
    elif opt_dtype=='float32':
        return np.float32
    elif opt_dtype=='float16' or opt_dtype=='half':
        return np.float16
    elif opt_dtype=='uint16':
        return np.uint16
    elif opt_dtype=='uint64' or opt_dtype=='uint':
        return np.uint

def strParamToArray(str_param):
    array_param = np.array(str_param.split(','))
    array_param = array_param.astype(np.float64)
    if len(array_param) == 1:
        array_param = np.array([array_param[0].astype(np.float64)] * 3)
    return array_param[::-1]

letters = string.ascii_uppercase

def chooseRandomRef(Nletters):
    source_ref = ''.join(random.choice(letters) for _ in range(Nletters))
    return source_ref

def generate_ellipse(X,Y,Z, center, min_radius,max_radius, prop_radius):
    if prop_radius=='uniform':
        radius = np.random.rand(3) * (max_radius - min_radius) + min_radius
    elif prop_radius=='squared_inv':
        radius = min_radius / (1 + np.random.rand(3) * (min_radius/max_radius - 1))

    rotation_angles = np.random.rand(3) * 2 * np.pi
    rot = Rot.from_rotvec([[rotation_angles[0], 0, 0], [0, rotation_angles[1], 0], [0, 0, rotation_angles[2]]])
    rotation_matrices = rot.as_matrix()
    rot_matrice = rotation_matrices[0].dot((rotation_matrices[1].dot(rotation_matrices[2])))
    lesion = ((((      (X - center[0]) * rot_matrice[0, 0] + (Y - center[1]) * rot_matrice[0, 1] + (
                                              Z - center[2]) * rot_matrice[0, 2]) ** 2 / (radius[0] ** 2) +
                                  ((X - center[0]) * rot_matrice[1, 0] + (Y - center[1]) * rot_matrice[1, 1] + (
                                              Z - center[2]) * rot_matrice[1, 2]) ** 2 / (radius[1] ** 2) +
                                  ((X - center[0]) * rot_matrice[2, 0] + (Y - center[1]) * rot_matrice[2, 1] + (
                                              Z - center[2]) * rot_matrice[2, 2]) ** 2 / (radius[2] ** 2) < 1)
                                ).astype(float))
    return lesion

def generate_cylinder(X,Y,Z, center, min_radius,max_radius, prop_radius):
    if prop_radius=='uniform':
        radius = np.random.rand(3) * (max_radius - min_radius) + min_radius
    elif prop_radius=='squared_inv':
        radius = min_radius / (1 + np.random.rand(3) * (min_radius/max_radius - 1))

    rotation = np.random.rand() * 2 * np.pi
    rotation_angles = np.random.rand(3) * 2 * np.pi
    rotation_cyl = Rot.from_rotvec(rotation_angles)

    XYZ = np.array([X.ravel(), Y.ravel(), Z.ravel()]).transpose()
    # apply rotation
    XYZrot = rotation_cyl.apply(XYZ)
    # return to original shape of meshgrid
    Xrot = XYZrot[:, 0].reshape(X.shape)
    Yrot = XYZrot[:, 1].reshape(X.shape)
    Zrot = XYZrot[:, 2].reshape(X.shape)

    lesion =((((((Xrot - center[0]) * np.cos(rotation)
                                        - (Yrot - center[1]) * np.sin(rotation)) / radius[0]) ** 2 +
                                      (((Xrot - center[0]) * np.sin(rotation)
                                        + (Yrot - center[1]) * np.cos(rotation)) / radius[1]) ** 2) < 1) *
                           (np.abs(Zrot-center[2])<radius[2])
                                     ).astype(float)
    return lesion

def generate_bg_cylinder(X,Y,Z,activity, center, radius_xzy):
    rotation = np.random.rand() * 2 * np.pi
    background_array = (activity) * ((((((X - center[0]) * np.cos(rotation)
                                        - (Z - center[1]) * np.sin(rotation)) / radius_xzy[0]) ** 2 +
                                      (((X - center[0]) * np.sin(rotation)
                                        + (Z - center[1]) * np.cos(rotation)) / radius_xzy[1]) ** 2) < 1) *
                                     (np.abs(Y - center[2]) < radius_xzy[2])
                                     ).astype(float)
    return background_array

def generate_sphere(center,X,Y,Z,min_radius, max_radius, prop_radius):
    if prop_radius=='uniform':
        radius = np.random.rand() * (max_radius - min_radius) + min_radius
    elif prop_radius=='squared_inv':
        radius = min_radius / (1 + np.random.rand() * (min_radius/max_radius - 1))

    return ((((X - center[0]) / radius) ** 2 + ((Y - center[1]) / radius) ** 2 + ((Z - center[2]) / radius) ** 2) < 1).astype(float)

def generate_convex(X,Y,Z,center,min_radius,max_radius, prop_radius):
    if prop_radius=='uniform':
        radius = np.random.rand(3) * (max_radius - min_radius) + min_radius
    elif prop_radius=='squared_inv':
        radius = min_radius / (1 + np.random.rand(3) * (min_radius/max_radius - 1))

    N = 30
    theta,phi = np.pi*np.random.rand(N), 2*np.pi*np.random.rand(N)
    vertices_x,vertices_y,vertices_z = radius[0]*np.sin(theta)*np.cos(phi)+center[0],\
                                       radius[1]*np.sin(theta)*np.sin(phi)+center[1],\
                                       radius[2]*np.cos(theta) + center[2]
    lesion = np.zeros_like(X,dtype=bool)
    id_center_x = np.searchsorted(X[:,0,0], vertices_x)
    id_center_y = np.searchsorted(Y[0,:,0], vertices_y)
    id_center_z = np.searchsorted(Z[0,0,:], vertices_z)

    id_center_x[id_center_x==len(X[:,0,0])]= id_center_x[id_center_x==len(X[:,0,0])] - 1
    id_center_y[id_center_y==len(Y[0,:,0])]= id_center_y[id_center_y==len(Y[0,:,0])] - 1
    id_center_z[id_center_z==len(Z[0,0,:])]= id_center_z[id_center_z==len(Z[0,0,:])] - 1

    lesion[id_center_x, id_center_y, id_center_z] = True
    lesion = convex_hull_image(lesion).astype(float)
    return lesion



def sample_activity(min_r,max_r,lbda,with_bg):
    if with_bg:
        S = 1 / (max_r + np.log(np.random.rand()) / lbda)
        if 1/S < min_r:
            return 1/min_r
        else:
            return S
    else:
        S = np.random.rand()*(max_r-min_r)+min_r
        return S

def random_3d_function(a0, xx, yy, zz, M):
    # Coarser grid for function generation
    N = 64
    size_x, size_y, size_z = N,N,N
    # Define the range of the 3D grid (adjust as needed)
    x0 = np.linspace(xx[0,0,0], xx[-1,0,0], size_x)
    y0 = np.linspace(yy[0,0,0], yy[0,-1,0], size_y)
    z0 = np.linspace(zz[0,0,0], zz[0,0,-1], size_z)
    period = xx[-1,0,0]-xx[0,0,0]
    xx0, yy0, zz0 = np.meshgrid(x0, y0, z0, indexing='ij')

    # Generate random Fourier coefficients
    coeffs_real = 2*np.random.rand(2*M+1,2*M+1,2*M+1)-1
    coeffs_imag = 2*np.random.rand(2*M+1,2*M+1,2*M+1)-1
    coeffs = coeffs_real + 1j * coeffs_imag

    # Compute the Fourier Transform
    coarse_f = np.zeros_like(xx0, dtype=np.float64)
    for m_x in range(-M,M+1):
        for m_y in range(-M,M+1):
            for m_z in range(-M,M+1):
                coarse_f += np.real(coeffs[m_x+M, m_y+M, m_z+M]\
                                    * np.exp(2j * np.pi * (m_x * xx0 + m_y * yy0 + m_z * zz0)/period)\
                               / (m_x**2 + m_y**2 + m_z**2)) if (m_x,m_y,m_z)!=(0,0,0) else 0
    coarse_f += a0

    interp = RegularGridInterpolator((x0, y0, z0), coarse_f)
    interpolated_values = interp((xx, yy, zz))
    return interpolated_values

def save_me(img=None,array=None,ftype=None,output_folder=None, src_ref=None, ref=None, grp=None, dtype=None, img_like=None):
    if ftype=="h5":
        if ((array is None) and (img is not None)):
            array=itk.array_from_image(img)
        dset_ref = grp.create_dataset(ref, array.shape, dtype=dtype)
        dset_ref[:, :, :] = array
    elif ftype in ["mhd", "mha"]:
        filename = os.path.join(output_folder, f'{src_ref}_{ref}.{ftype}')
        if ((array is not None) and (img is None)):
            output_img = itk.image_from_array(array)
            output_img.SetSpacing(img_like.GetSpacing())
            output_img.SetOrigin(img_like.GetOrigin())
            itk.imwrite(output_img, filename)
        elif ((array is None) and (img is not None)):
            itk.imwrite(img, filename)
        else:
            print("ERROR : give at leat array or img (not both)")
            exit(0)
    elif ftype=="npy":
        filename = os.path.join(output_folder, f'{src_ref}_{ref}.{ftype}')

        if ((array is not None) and (img is None)):
            np.save(filename, array)
        elif ((array is None) and (img is not None)):
            array=itk.array_from_image(img)
            np.save(filename, array)
        else:
            print("ERROR : give at leat array or img (not both)")
            exit(0)
    else:
        print("ERROR : wrong output type")
        exit(0)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--nb_data','-n', type = int, required = True, help = 'number of desired data = (src,projPVE,projPVfree)')
parser.add_argument('--attmap', type=str, required=True)
parser.add_argument('--type', default = 'mha', help = "Create mha, mhd,npy image")
parser.add_argument('--dtype', default = 'float64', help = "if npy, image dtype")
parser.add_argument('--min_radius', default = 4,type = float, help = 'minimum radius of the random spheres')
parser.add_argument('--max_radius', default = 32,type = float, help = 'max radius of the random spheres')
parser.add_argument('--prop_radius', default = "uniform", choices=['uniform', 'squared_inv'], help = 'proportion of radius between min/max')
parser.add_argument('--min_ratio', default = 1/1000,type = float, help = 'min bg:src ratio. If no background, it is the min activity')
parser.add_argument('--max_ratio', default = 1/8,type = float, help = 'max bg:src ratio. If no background, it is the max activity')
parser.add_argument('--min_activity', default = 10, type = float, help = "minimum activity in MBq. Then, N_min = A_min * 1e6 * 20s * efficiency ")
parser.add_argument('--max_activity', default = 100, type = float, help = "maximal activity in MBq. Then, N_min = A_max * 1e6 * 20s * efficiency")
parser.add_argument('--nspheres', default = 1,type = int, help = 'max number of spheres to generate on each source')
parser.add_argument('--background', action= 'store_true', help = 'If you want background add --background')
parser.add_argument('--sphere',type = float, default = 0, help = "if --sphere p, activity sources are spheres with proba p")
parser.add_argument('--ellipse',type = float, default = 0, help = "if --ellipse p, activity sources are ellipses with proba p")
parser.add_argument('--cylinder',type = float, default = 0, help = "if --cylinder p, activity sources are cylinders with proba p")
parser.add_argument('--convex',type = float, default = 0, help = "if --convex p, activity sources are convexs with proba p")
parser.add_argument('--grad_act',action ="store_true", help = "if --grad_act, hot sources are not homogeneous")
parser.add_argument('--nproj',type = int, default = None, help = 'if no geom, precise nb of proj angles')
parser.add_argument('--sid',type = float, default = None, help = 'if no geom, precise detector-to-isocenter distance (mm)')
parser.add_argument('--fov', type=str,help="FOV (mm,mm) of the detector. Should be in the format --fov 532,388 ")
parser.add_argument("--pthgaga", type=str)
parser.add_argument("--pthgarf", type=str)
parser.add_argument("-b", "--batchsize", type=float, default=100000)
parser.add_argument('--output_folder','-o', default = './dataset', help = " Absolute or relative path to the output folder")
parser.add_argument('--save_src',action ="store_true", help = "if you want to also save the source that will be forward projected")
parser.add_argument('--lesion_mask',action ="store_true", help = "if you want to also save the source that will be forward projected")
parser.add_argument('--rec_fp',action="store_true", help = "noisy projections are reconstructed with 1 osem-rm iter and forward-projected w/o rm to obtain ABCDE_rec_fp.mha")
parser.add_argument("-v", "--verbose", action="store_true")
def generate(opt):
    print(opt)
    current_date = time.strftime("%d_%m_%Y_%Hh_%Mm_%Ss", time.localtime())
    opt.date = current_date
    dataset_infos = vars(opt)

    t0 = time.time()

    sigma0_psf, alpha_psf,efficiency = get_psf_params(machine="siemens-intevo")
    dataset_infos['sigma0_psf'] = sigma0_psf
    dataset_infos['alpha_psf'] = alpha_psf
    dataset_infos['efficiency'] = efficiency

    size_proj,spacing_proj = get_detector_params(machine="siemens-intevo")
    print(f'size / spacing derived from spect_system (siemens-intevo) : size={size_proj}    spacing={spacing_proj}')
    dataset_infos['size_proj']=size_proj
    dataset_infos['spacing_proj']=spacing_proj

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

    min_ratio, Max_ratio = opt.min_ratio, opt.max_ratio
    if opt.background:
        background_radius_x_mean, background_radius_z_mean,background_radius_y_mean = 200, 120,300
        background_radius_x_std, background_radius_z_std,background_radius_y_std = 20, 10, 100
        dataset_infos['bg_shape_params'] = {'mean_xzy': f'({background_radius_x_mean},{background_radius_z_mean},{background_radius_y_mean})',
                                            'std_xzy': f'({background_radius_x_std},{background_radius_z_std},{background_radius_y_std})'}

        R = 100 # proba ratio max/min : p(M)/p(m)=R
        lbda = np.log(R) / (Max_ratio - min_ratio)
    else:
        lbda = None

    p_sphere,p_ellipse,p_cylinder,p_convex = opt.sphere,opt.ellipse,opt.cylinder,opt.convex
    assert(p_sphere+p_ellipse+p_cylinder+p_convex==1)


    min_activity,max_activity = opt.min_activity, opt.max_activity
    min_count= int(min_activity * 1e6 * 20 * efficiency)
    max_count= int(max_activity * 1e6 * 20 * efficiency)

    print(f'Activity between {min_activity} MBq and {max_activity} MBq --> nb of counts between {min_count} and {max_count}')

    print(json.dumps(dataset_infos, indent = 3))

    dataset_infos['src_refs']=[]
    attmap = itk.imread(opt.attmap, pixel_type=pixelType)

    attmap_np = itk.array_from_image(attmap)
    vSpacing = np.array(attmap.GetSpacing())[::-1]
    vSize = np.array(attmap_np.shape)
    vOffset = np.array(attmap.GetOrigin())[::-1]
    # matrix settings
    lengths = vSize * vSpacing
    lspaceX = np.linspace(-vSize[0] * vSpacing[0] / 2, vSize[0] * vSpacing[0] / 2, vSize[0])
    lspaceY = np.linspace(-vSize[1] * vSpacing[1] / 2, vSize[1] * vSpacing[1] / 2, vSize[1])
    lspaceZ = np.linspace(-vSize[2] * vSpacing[2] / 2, vSize[2] * vSpacing[2] / 2, vSize[2])
    X, Y, Z = np.meshgrid(lspaceX, lspaceY, lspaceZ, indexing='ij')

    time_src=0

    # GAN
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    gan_info = {}
    gan_info['pth_filename'] = opt.pthgaga
    gan_info['batchsize'] = opt.batchsize
    gan_info['device'] = device
    print(device)

    cgan_source = CGANSOURCE(gan_info)
    nprojs = opt.nproj
    l_angles = torch.linspace(0, 2*torch.pi, nprojs+1)[:-1]
    dist_to_crystal = 29.104800000000004
    l_detectorsPlanes = []
    for angle in l_angles:
        det_plane = DetectorPlane(size=565.511, device=device, center0=[0,0, -opt.sid], rot_angle=angle,dist_to_crystal=dist_to_crystal) #FIXME (center)
        l_detectorsPlanes.append(det_plane)
    garf_ui = {}
    # garf_ui['pth_filename'] = os.path.join(paths.current, "pths/arf_5x10_9.pth")
    garf_ui['pth_filename'] = opt.pthgarf
    garf_ui['batchsize'] = opt.batchsize
    garf_ui['device'] = device
    garf_ui['nprojs'] = len(l_detectorsPlanes)

    for n in range(opt.nb_data):
        # Random output filename
        source_ref = chooseRandomRef(Nletters=5)

        if opt.verbose:
            print(source_ref)

        forward_projector = rtk.ZengForwardProjectionImageFilter.New()
        forward_projector.SetInput(0, output_proj.GetOutput())
        forward_projector.SetGeometry(geometry)

        forward_projector_with_att = rtk.ZengForwardProjectionImageFilter.New()
        forward_projector_with_att.SetInput(0, output_proj.GetOutput())
        forward_projector_with_att.SetGeometry(geometry)

        forward_projector_with_att.SetInput(2, attmap)


        time_src_0=time.time()

        src_array = np.zeros_like(X)

        if opt.lesion_mask:
            lesion_array=np.zeros_like(X)

        if opt.background:
            # background = cylinder with revolution axis = Y
            background_array = np.zeros_like(X)

            background_array[attmap_np>0.01]=1


            if opt.grad_act:
                M = 5
                background_array = background_array * random_3d_function(a0 = 10,xx = X,yy=Y,zz=Z,M=M)/10
                background_array[background_array<0] = 0

            src_array += background_array


        random_nb_of_sphers = np.random.randint(1,opt.nspheres+1)
        if opt.grad_act:
            M = 8
            rndm_grad_act = random_3d_function(a0=10, xx=X, yy=Y, zz=Z, M=M)/10

            # rndm_grad_act scaled between 0.5 and 1.5.
            rndm_grad_act_0_1 =  (rndm_grad_act - rndm_grad_act.min()) / (rndm_grad_act.max() - rndm_grad_act.min())
            min_scale, max_scale = 0.5, 1.5
            rndm_grad_act_scaled = rndm_grad_act_0_1 * (max_scale - min_scale) + min_scale


        for s in range(random_nb_of_sphers):
            random_activity = sample_activity(min_r=min_ratio,max_r=Max_ratio,lbda=lbda,with_bg=opt.background)

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

            if opt.lesion_mask:
                lesion_array+=lesion

            if opt.grad_act:
                rndm_grad_act_scaled_scaled = rndm_grad_act_scaled / np.mean(rndm_grad_act_scaled[lesion>0])
                lesion = lesion * (rndm_grad_act_scaled_scaled*random_activity)
            else:
                lesion = random_activity * lesion

            src_array += lesion

        time_src+=(time.time() - time_src_0)
        if opt.verbose:
            print('fp...')





        total_counts_per_proj = round(np.random.rand() * (max_count - min_count) + min_count)
        print(f"{total_counts_per_proj} ({total_counts_per_proj/(1e6 * 20 * efficiency)} MBq)")
        src_array_normedToTotalCounts = src_array / np.sum(src_array) * total_counts_per_proj * spacing_proj**2 / (vSpacing[0]*vSpacing[1]*vSpacing[2])

        src_img_normedToTotalCounts = itk.image_from_array(src_array_normedToTotalCounts.astype(np.float32))
        src_img_normedToTotalCounts.SetSpacing(vSpacing[::-1])
        src_img_normedToTotalCounts.SetOrigin(vOffset[::-1])



        # saving of source 3D image
        if opt.save_src:
            save_me(img=src_img_normedToTotalCounts, ftype=opt.type, output_folder=opt.output_folder, src_ref=source_ref,
                    ref="src", grp=None, dtype=dtype)

        ########### GAN
        activity = int(total_counts_per_proj / efficiency / 20)
        print(f"ACT : {activity}")
        dataset = ConditionsDataset(activity=activity,
                                    cgan_src=cgan_source,
                                    source_fn=os.path.join(opt.output_folder, f"{source_ref}_src.{opt.type}"),
                                    save_cond=False)
        batch_size = int(float(opt.batchsize))
        n_batchs = int(activity // batch_size)
        garf_ui['output_fn'] = os.path.join(opt.output_folder, f"{source_ref}_gagarf.mhd")
        garf_detector = GARF(user_info=garf_ui)
        N_primaries = activity
        N=0
        pbar = tqdm(total=N_primaries)

        with torch.no_grad():
            for _ in range(n_batchs):
                gan_input_z_cond = dataset.get_batch_torch(batch_size)
                N += batch_size

                gan_input_z_cond = gan_input_z_cond.to(device)
                fake = cgan_source.generate(gan_input_z_cond)

                fake = fake[fake[:, 0] > 0.100]
                dirn = torch.sqrt(fake[:, 4] ** 2 + fake[:, 5] ** 2 + fake[:, 6] ** 2)
                fake[:, 4:7] = fake[:, 4:7] / dirn[:, None]

                beta = (fake[:, 1:4] * fake[:, 4:7]).sum(dim=1)
                R1, R2 = 610, opt.sid - 50
                alpha = beta - torch.sqrt(beta ** 2 + R2 ** 2 - R1 ** 2)
                fake[:, 1:4] = fake[:, 1:4] - alpha[:, None] * fake[:, 4:7]

                for proj_i, plane_i in enumerate(l_detectorsPlanes):
                    batch_arf_i = plane_i.get_intersection(batch=fake)
                    garf_detector.apply(batch_arf_i, proj_i)

                pbar.update(batch_size)
            garf_detector.save_projection()
        ###########


        if opt.lesion_mask:
            lesion_mask = (lesion_array > 0).astype(np.float32)
            lesion_mask_img = itk.image_from_array(lesion_mask)
            lesion_mask_img.CopyInformation(src_img_normedToTotalCounts)
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
                    ref = "lesion_mask_fp", grp = None, dtype = dtype, img_like = output_forward_lesion_mask)


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
                ref="PVfree", grp=None, dtype=dtype)

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
                ref="PVE_att", grp=None, dtype=dtype)

        # proj noise(att+PVE)
        # FIXME, FIXME, FIXME
        output_forward_PVE_array = itk.array_from_image(output_forward_PVE).astype(dtype=dtype)
        noisy_projection_array = np.random.poisson(lam=output_forward_PVE_array, size=output_forward_PVE_array.shape).astype(dtype=np.float64)
        save_me(array=noisy_projection_array, ftype=opt.type, output_folder=opt.output_folder, src_ref=source_ref,
                ref="PVE_att_noisy", grp=None, dtype=dtype, img_like=output_forward_PVE)


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
                ref="PVfree_att", grp=None, dtype=dtype)



        if opt.rec_fp:
            print('rec_fp...')

            constant_image = rtk.ConstantImageSource[imageType].New()
            constant_image.SetSpacing(vSpacing)
            constant_image.SetOrigin(vOffset)
            constant_image.SetSize([int(s) for s in vSize])
            constant_image.SetConstant(1)
            output_rec = constant_image.GetOutput()

            OSEMType = rtk.OSEMConeBeamReconstructionFilter[imageType, imageType]
            osem = OSEMType.New()
            osem.SetInput(0, output_rec)
            osem.SetGeometry(geometry)
            osem.SetNumberOfIterations(1)
            osem.SetNumberOfProjectionsPerSubset(15)
            osem.SetBetaRegularization(0)

            FP = osem.ForwardProjectionType_FP_ZENG
            BP = osem.BackProjectionType_BP_ZENG
            osem.SetSigmaZero(sigma0_psf)
            osem.SetAlphaPSF(alpha_psf)
            osem.SetForwardProjectionFilter(FP)
            osem.SetBackProjectionFilter(BP)


            osem.SetInput(2, attmap)

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
                    ref="rec", grp=None, dtype=dtype)

            # forward_projs
            forward_projector_rec_fp.SetInput(1, rec_volume)
            forward_projector_rec_fp.Update()
            output_rec_fp = forward_projector_rec_fp.GetOutput()
            if fov_is_set:
                fov_maskmult.SetInput2(output_rec_fp)
                output_rec_fp = fov_maskmult.GetOutput()

            save_me(img=output_rec_fp, ftype=opt.type, output_folder=opt.output_folder, src_ref=source_ref,
                    ref="rec_fp", grp=None, dtype=dtype)


            forward_projector_rec_fp.SetInput(2, attmap)
            forward_projector_rec_fp.Update()
            output_rec_fp_att = forward_projector_rec_fp.GetOutput()
            if fov_is_set:
                fov_maskmult.SetInput2(output_rec_fp_att)
                output_rec_fp_att = fov_maskmult.GetOutput()

            save_me(img=output_rec_fp_att, ftype=opt.type, output_folder=opt.output_folder, src_ref=source_ref,
                    ref="rec_fp_att", grp=None, dtype=dtype)



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
