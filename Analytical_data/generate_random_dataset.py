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



from parameters import get_psf_params

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

def save_one_file(array, filename,ftype,img_like):
    if ftype != 'npy':
        output_img = itk.image_from_array(array)
        output_img.SetSpacing(img_like.GetSpacing())
        output_img.SetOrigin(img_like.GetOrigin())
        itk.imwrite(output_img, filename)
    else:
        np.save(filename, array)


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--nb_data','-n', type = int, required = True, help = 'number of desired data = (src,projPVE,projPVfree)')
parser.add_argument('--size_volume', type = str, default = "150", help = 'Size of the desired image i.e. number of voxels per dim')
parser.add_argument('--spacing_volume', type = str, default = "4", help = 'Spacing of the desired image i.e phyisical length of a voxels (mm)')
parser.add_argument('--size_proj', type = int, default = 128, help = 'Size of the desired projections')
parser.add_argument('--spacing_proj', type = float, default = 4.41806, help = 'Spacing of the desired projection. Ex intevo : 2.3976')
parser.add_argument('--type', default = 'mha', help = "Create mha, mhd,npy image")
parser.add_argument('--dtype', default = 'float64', help = "if npy, image dtype")
parser.add_argument('--like', default = None, help = "Instead of specifying spacing/size, you can specify an image as a metadata model")
parser.add_argument('--min_radius', default = 4,type = float, help = 'minimum radius of the random spheres')
parser.add_argument('--max_radius', default = 32,type = float, help = 'max radius of the random spheres')
parser.add_argument('--prop_radius', default = "uniform", choices=['uniform', 'squared_inv'], help = 'proportion of radius between min/max')
parser.add_argument('--min_ratio', default = 1/1000,type = float, help = 'min bg:src ratio. If no background, it is the min activity')
parser.add_argument('--max_ratio', default = 1/8,type = float, help = 'max bg:src ratio. If no background, it is the max activity')
parser.add_argument('--min_counts', default = 2e4, type = float, help = "minimum number of counts per proj (noise level)")
parser.add_argument('--max_counts', default = 1e5, type = float, help = "maximum number of counts per proj (noise level)")
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
parser.add_argument('--attenuationmap', '-a',default = None, help = 'path to the attenuation map file')
parser.add_argument('--output_folder','-o', default = './dataset', help = " Absolute or relative path to the output folder")
parser.add_argument('--spect_system', default = "ge-discovery", choices=['ge-discovery', 'siemens-intevo'], help = 'SPECT system simulated for PVE projections')
parser.add_argument('--save_src',action ="store_true", help = "if you want to also save the source that will be forward projected")
parser.add_argument('--noise',action ="store_true", help = "Add Poisson noise ONLY to ProjPVE")
parser.add_argument('--merge',action="store_true", help = "If --merge, the 3 (or 2) projections are stored in the same file ABCDE(_noisy)_PVE_PVfree.mha. In this order : noisy, PVE, PVfree")
parser.add_argument('--rec_fp',action="store_true", help = "noisy projections are reconstructed with 1 osem-rm iter and forward-projected w/o rm to obtain ABCDE_rec_fp.mha")
parser.add_argument("-v", "--verbose", action="store_true")
def generate(opt):
    print(opt)
    current_date = time.strftime("%d_%m_%Y_%Hh_%Mm_%Ss", time.localtime())
    opt.date = current_date
    dataset_infos = vars(opt)

    t0 = time.time()

    sigma0_psf, alpha_psf = get_psf_params(opt.spect_system)
    dataset_infos['sigma0_psf'] = sigma0_psf
    dataset_infos['alpha_psf'] = alpha_psf

    # get output image parameters
    if opt.like is not None:
        im_like = itk.imread(opt.like)
        vSpacing = np.array(im_like.GetSpacing())
        vSize = np.array(itk.size(im_like))
        vOffset = np.array(im_like.GetOrigin())
    else:
        vSize = strParamToArray(opt.size_volume).astype(int)
        vSpacing = strParamToArray(opt.spacing_volume)
        vOffset = [(-sp*size + sp)/2 for (sp,size) in zip(vSpacing,vSize)]

    # matrix settings
    lengths = vSize*vSpacing
    lspaceX = np.linspace(-vSize[0] * vSpacing[0] / 2, vSize[0] * vSpacing[0] / 2, vSize[0])
    lspaceY = np.linspace(-vSize[1] * vSpacing[1] / 2, vSize[1] * vSpacing[1] / 2, vSize[1])
    lspaceZ = np.linspace(-vSize[2] * vSpacing[2] / 2, vSize[2] * vSpacing[2] / 2, vSize[2])

    X,Y,Z = np.meshgrid(lspaceX,lspaceY,lspaceZ, indexing='ij')

    offset = (-opt.spacing_proj * opt.size_proj + opt.spacing_proj) / 2 #proj offset

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

    # Projections infos
    dtype = get_dtype(opt.dtype)
    pixelType = itk.F
    imageType = itk.Image[pixelType, 3]
    output_spacing = [opt.spacing_proj,opt.spacing_proj, 1]
    offset = (-opt.spacing_proj * opt.size_proj + opt.spacing_proj) / 2
    output_offset = [offset, offset, (-nproj+1)/2]
    output_image = rtk.ConstantImageSource[imageType].New()
    output_image.SetSpacing(output_spacing)
    output_image.SetOrigin(output_offset)
    output_image.SetSize([opt.size_proj, opt.size_proj, nproj])
    output_image.SetConstant(0.)


    forward_projector = rtk.ZengForwardProjectionImageFilter.New()
    forward_projector.SetInput(0, output_image.GetOutput())
    forward_projector.SetGeometry(geometry)


    if opt.attenuationmap is not None:
        attenuation_image = itk.imread(opt.attenuationmap, itk.F)
        forward_projector.SetInput(2, attenuation_image)
        forward_projector.SetInput(2, attenuation_image)

    if opt.rec_fp:
        vSpacing_recpfp = np.array([4.6875, 4.6875, 4.6875])
        vSize_recfp = np.array([128,128,128])
        vOffset_recfp = [(-sp * size + sp) / 2 for (sp, size) in zip(vSpacing_recpfp, vSize_recfp)]

        # vSpacing_recpfp = vSpacing
        # vSize_recfp = vSize
        # vOffset_recfp = vOffset

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
        osem.SetNumberOfIterations(1)
        osem.SetNumberOfProjectionsPerSubset(15)
        osem.SetBetaRegularization(0)

        if opt.attenuationmap is not None:
            osem.SetInput(2, attenuation_image)

        FP = osem.ForwardProjectionType_FP_ZENG
        BP = osem.BackProjectionType_BP_ZENG
        osem.SetSigmaZero(sigma0_psf)
        osem.SetAlpha(alpha_psf)
        osem.SetForwardProjectionFilter(FP)
        osem.SetBackProjectionFilter(BP)

        forward_projector_rec_fp = rtk.ZengForwardProjectionImageFilter.New()
        forward_projector_rec_fp.SetInput(0, output_image.GetOutput())
        forward_projector_rec_fp.SetGeometry(geometry)
        forward_projector_rec_fp.SetSigmaZero(0)
        forward_projector_rec_fp.SetAlpha(0)

    min_ratio, Max_ratio = opt.min_ratio, opt.max_ratio
    if opt.background:
        background_radius_x_mean, background_radius_z_mean,background_radius_y_mean = 200, 120,lengths[1]/2
        background_radius_x_std, background_radius_z_std,background_radius_y_std = 20, 10, 100
        dataset_infos['bg_shape_params'] = {'mean_xzy': f'({background_radius_x_mean},{background_radius_z_mean},{background_radius_y_mean})',
                                            'std_xzy': f'({background_radius_x_std},{background_radius_z_std},{background_radius_y_std})'}

        R = 100 # proba ratio max/min : p(M)/p(m)=R
        lbda = np.log(R) / (Max_ratio - min_ratio)
    else:
        lbda = None

    p_sphere,p_ellipse,p_cylinder,p_convex = opt.sphere,opt.ellipse,opt.cylinder,opt.convex
    assert(p_sphere+p_ellipse+p_cylinder+p_convex==1)


    total_counts_in_proj_min,total_counts_in_proj_max = opt.min_counts, opt.max_counts
    print(f'Total counts in projections between {total_counts_in_proj_min} and {total_counts_in_proj_max}')

    print(json.dumps(dataset_infos, indent = 3))

    for n in range(opt.nb_data):
        # Random output filename
        source_ref = chooseRandomRef(Nletters=5)
        while os.path.exists(os.path.join(opt.output_folder, f'{source_ref}_PVE.{opt.type}')):
            source_ref = chooseRandomRef(Nletters=5)

        if opt.verbose:
            print(source_ref)


        src_array = np.zeros_like(X)

        if opt.background:
            # background = cylinder with revolution axis = Y
            background_array = np.zeros_like(X)
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


        random_nb_of_sphers = np.random.randint(1,opt.nspheres)
        if opt.grad_act:
            M = 8
            rndm_grad_act = random_3d_function(a0=0, xx=X, yy=Y, zz=Z, M=M)

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

            if opt.grad_act:
                lesion = lesion * (((1-random_activity)/np.min(rndm_grad_act)) * rndm_grad_act + random_activity)
            else:
                lesion = random_activity * lesion

            src_array += lesion

        if opt.verbose:
            print('fp...')

        total_counts_in_proj = np.random.randint(total_counts_in_proj_min,total_counts_in_proj_max)
        src_array_normedToTotalCounts = src_array / np.sum(src_array) * total_counts_in_proj * opt.spacing_proj**2 / (vSpacing[0]*vSpacing[1]*vSpacing[2])

        src_img_normedToTotalCounts = itk.image_from_array(src_array_normedToTotalCounts.astype(np.float32))
        src_img_normedToTotalCounts.SetSpacing(vSpacing[::-1])
        src_img_normedToTotalCounts.SetOrigin(vOffset)



        # saving of source 3D image
        if opt.save_src:
            src_img = itk.image_from_array(src_array.astype(np.float32))
            src_img.SetSpacing(vSpacing[::-1])
            src_img.SetOrigin(vOffset)
            source_path = os.path.join(opt.output_folder,f'{source_ref}.mhd')
            itk.imwrite(src_img,source_path)


        #compute fowardprojections :
        #proj PVE
        forward_projector.SetInput(1, src_img_normedToTotalCounts)

        forward_projector.SetSigmaZero(sigma0_psf)
        forward_projector.SetAlpha(alpha_psf)
        forward_projector.Update()
        output_forward_PVE = forward_projector.GetOutput()
        output_forward_PVE.DisconnectPipeline()
        output_forward_PVE_array = itk.array_from_image(output_forward_PVE).astype(dtype=dtype)

        #proj PVfree
        forward_projector.SetSigmaZero(0)
        forward_projector.SetAlpha(0)
        forward_projector.Update()
        output_forward_PVfree = forward_projector.GetOutput()
        output_forward_PVfree.DisconnectPipeline()
        output_forward_PVfree_array = itk.array_from_image(output_forward_PVfree).astype(dtype=dtype)

        if opt.noise:
            noisy_projection_array = np.random.poisson(lam=output_forward_PVE_array, size=output_forward_PVE_array.shape).astype(dtype=dtype)

            if opt.rec_fp:
                print('rec_fp...')
                output_forward_PVE_noisy = itk.image_from_array(noisy_projection_array.astype(dtype=np.float32))
                output_forward_PVE_noisy.CopyInformation(output_forward_PVE)
                osem.SetInput(1, output_forward_PVE_noisy)
                osem.Update()
                rec_fp_volume = osem.GetOutput()
                rec_fp_volume.DisconnectPipeline()
                forward_projector_rec_fp.SetInput(1, rec_fp_volume)
                forward_projector_rec_fp.Update()
                output_rec_fp = forward_projector_rec_fp.GetOutput()
                output_rec_fp_array = itk.array_from_image(output_rec_fp)

        # Write projections :
        if opt.merge:
            if opt.noise:
                if opt.rec_fp:
                    output_filename_merged = os.path.join(opt.output_folder,f'{source_ref}_noisy_PVE_PVfree_rec_fp.{opt.type}')
                    output_forward_merged_array = np.concatenate((noisy_projection_array,output_forward_PVE_array, output_forward_PVfree_array,output_rec_fp_array), axis=0)
                else:
                    output_filename_merged = os.path.join(opt.output_folder,f'{source_ref}_noisy_PVE_PVfree.{opt.type}')
                    output_forward_merged_array = np.concatenate((noisy_projection_array,output_forward_PVE_array, output_forward_PVfree_array), axis=0)
            else:
                output_filename_merged = os.path.join(opt.output_folder, f'{source_ref}_PVE_PVfree.{opt.type}')
                output_forward_merged_array = np.concatenate((output_forward_PVE_array,output_forward_PVfree_array), axis=0)
            if opt.type!='npy':
                output_forward_merged = itk.image_from_array(output_forward_merged_array)
                output_forward_merged.SetSpacing(output_forward_PVE.GetSpacing())
                output_forward_merged.SetOrigin(output_forward_PVE.GetOrigin())
                itk.imwrite(output_forward_merged,output_filename_merged)
            else:
                np.save(output_filename_merged,output_forward_merged_array)
        else:
            output_filename_PVfree = os.path.join(opt.output_folder, f'{source_ref}_PVfree.{opt.type}')
            output_filename_PVE = os.path.join(opt.output_folder, f'{source_ref}_PVE.{opt.type}')

            save_one_file(array=output_forward_PVfree_array,filename=output_filename_PVfree,ftype=opt.type,img_like=output_forward_PVE)
            save_one_file(array=output_forward_PVE_array,filename=output_filename_PVE,ftype=opt.type,img_like=output_forward_PVE)
            if opt.noise:
                output_filename_PVE_noisy = os.path.join(opt.output_folder, f'{source_ref}_PVE_noisy.{opt.type}')
                save_one_file(array=noisy_projection_array, filename=output_filename_PVE_noisy, ftype=opt.type,img_like=output_forward_PVE)
                if opt.rec_fp:
                    output_filename_rec_fp = os.path.join(opt.output_folder, f'{source_ref}_rec_fp.mhd')
                    save_one_file(array=output_rec_fp_array, filename=output_filename_rec_fp, ftype=opt.type,img_like=output_forward_PVE)

    tf = time.time()
    elapsed_time = round(tf - t0)
    elapsed_time_min = round(elapsed_time/60)
    dataset_infos['elapsed_time_s'] = elapsed_time
    print(f'Total time elapsed for data generation : {elapsed_time_min} min    (i.e. {elapsed_time} s)')

    formatted_dataset_infos = json.dumps(dataset_infos, indent=4)
    output_info_json = os.path.join(opt.output_folder, f'dataset_infos_{current_date}_{source_ref}.json')
    jsonfile = open(output_info_json, "w")
    jsonfile.write(formatted_dataset_infos)
    jsonfile.close()

if __name__ == '__main__':
    opt = parser.parse_args()
    generate(opt=opt)
