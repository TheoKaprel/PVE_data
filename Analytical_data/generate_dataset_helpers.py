#!/usr/bin/env python3

import numpy as np
import random
import string
from scipy.spatial.transform import Rotation as Rot
from scipy.interpolate import RegularGridInterpolator
from skimage.morphology import convex_hull_image
import itk
import os


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



def chooseRandomRef(Nletters):
    source_ref = ''.join(random.choice(string.ascii_uppercase) for _ in range(Nletters))
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

def random_3d_function_(a0, xx, yy, zz, M):
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
    coeffs = np.random.normal(0, 1,(2*M+1,2*M+1,2*M+1))
    phases = 2*np.pi*np.random.rand(2*M+1,2*M+1,2*M+1) - np.pi

    # Compute the Fourier Transform
    coarse_f = np.zeros_like(xx0, dtype=np.float64)
    for m_x in range(-M,M+1):
        for m_y in range(-M,M+1):
            for m_z in range(-M,M+1):
                coarse_f += coeffs[m_x+M, m_y+M, m_z+M]*np.cos(2*np.pi * (m_x*xx0 + m_y*yy0+m_z*zz0)/period + phases[m_x+M, m_y+M, m_z+M])\
                            / (m_x**2 + m_y**2 + m_z**2) if (m_x,m_y,m_z)!=(0,0,0) else 0
    coarse_f += a0

    interp = RegularGridInterpolator((x0, y0, z0), coarse_f)
    interpolated_values = interp((xx, yy, zz))
    return interpolated_values

def save_me(img=None,array=None,ftype=None,output_folder=None, src_ref=None, ref=None, dtype=None, img_like=None):
    if ftype in ["mhd", "mha"]:
        filename = os.path.join(output_folder, f'{src_ref}_{ref}.{ftype}')
        if ((array is not None) and (img is None)):
            output_img = itk.image_from_array(array.astype(dtype))
            output_img.SetSpacing(img_like.GetSpacing())
            output_img.SetOrigin(img_like.GetOrigin())
            itk.imwrite(output_img, filename)
        elif ((array is None) and (img is not None)):
            array = itk.array_from_image(img)
            array = array.astype(dtype)
            imggg = itk.image_from_array(array)
            imggg.CopyInformation(img)
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
