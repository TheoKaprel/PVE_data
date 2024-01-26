#!/usr/bin/env python3
import time

import matplotlib.pyplot as plt
from garf_helpers import *
import garf
import numpy as np
import torch
import itk

class GARF:
    def __init__(self, user_info):
        self.batchsize = user_info['batchsize']
        self.pth_filename = user_info['pth_filename']
        self.output_fn = user_info['output_fn']
        self.device = user_info['device']
        self.nprojs = user_info['nprojs']

        self.size=256
        self.spacing=2.3976
        self.image_size=[2 * self.nprojs, 256,256]
        self.image_spacing = [self.spacing, self.spacing, 1]

        self.zeros = torch.zeros((self.image_size[1], self.image_size[2])).to(self.device)

        self.degree = np.pi / 180
        self.init_garf()

        self.t_image_from_coord,self.time_nn_predict = 0,0
        self.t_preprocess_arf,self.t_postprocess_arf = 0,0
        self.t_remove,self.t_jsp = 0,0

    def init_garf(self):
        # load the pth file
        self.nn, self.model = garf.load_nn(
            self.pth_filename, verbose=False
        )
        self.model = self.model.to(self.device)


        # size and spacing (2D)
        self.model_data = self.nn["model_data"]

        self.x_mean = torch.tensor(self.model_data['x_mean'], device = self.device)
        self.x_std = torch.tensor(self.model_data['x_std'], device=self.device)
        if ('rr' in self.model_data):
            self.rr = self.model_data['rr']
        else:
            self.rr = self.model_data['RR']


        # output image: nb of energy windows times nb of runs (for rotation)
        self.nb_ene = self.model_data["n_ene_win"]
        # size and spacing in 3D

        # create output image as np array
        # self.output_image = np.zeros(self.image_size, dtype=np.float64)
        self.output_image = torch.zeros(tuple(self.image_size)).to(self.device)
        # compute offset
        self.psize = [self.size * self.spacing, self.size * self.spacing]

        self.hsize = np.divide(self.psize, 2.0)
        self.offset = [self.image_spacing[0] / 2.0, self.image_spacing[1] / 2.0]

        print('--------------------------------------------------')
        print(f'size {self.size}')
        print(f'spacing {self.spacing}')
        print(f'image size {self.image_size}')
        print(f'image spacing {self.image_spacing}')
        print(f'psize : {self.psize}')
        print(f'psize : {self.psize}')

    def apply(self,batch, proj_i):
        t_preprocess_arf_0 = time.time()
        x = batch.clone()

        x[:,2] = torch.arccos(batch[:,3]) / self.degree
        x[:,3] = torch.arccos(batch[:,2]) / self.degree
        ax = x[:, 2:5]  # two angles and energy

        self.t_preprocess_arf+=(time.time() - t_preprocess_arf_0)

        time_nn_predict_0 = time.time()
        w = self.nn_predict(self.model, self.nn["model_data"], ax)

        # ww = w[w[:,0]<0.5]
        # if len(ww)>0:
        #     print(ww)

        # w = torch.bernoulli(w)
        # w = w.multinomial(1,replacement=True)

        self.time_nn_predict += (time.time() - time_nn_predict_0)

        # positions
        # x_np = x.cpu().numpy()
        t_postprocess_arf_0 = time.time()
        x_np = x
        # angles = x_np[:, 2:4]
        # t = self.compute_angle_offset(angles, self.distance_to_crystal)

        cx = x_np[:, 0:2]
        # cx = cx + t
        coord = (cx + (self.size-1)*self.spacing/2) / self.spacing
        vu = torch.round(coord).to(int)

        self.t_jsp+=(time.time() - t_postprocess_arf_0)

        t_remove_0 = time.time()
        # vu, w_pred = self.remove_out_of_image_boundaries(vu, w, self.image_size)
        self.t_remove+=(time.time() - t_remove_0)

        self.t_postprocess_arf+=(time.time() - t_postprocess_arf_0)


        t0=time.time()
        # do nothing if there is no hit in the image
        if vu.shape[0] != 0:
            # PW
            temp = self.zeros.fill_(0)
            temp = self.image_from_coordinates_2(temp, vu,w[:,2])
            # temp = self.image_from_coordinates_2(temp, vu,(w==2).to(torch.float32))
            self.output_image[proj_i,:,:]= self.output_image[proj_i,:,:] + temp
            # SW
            temp = self.zeros.fill_(0)
            temp = self.image_from_coordinates_2(temp, vu,w[:,1])
            # temp = self.image_from_coordinates_2(temp, vu,(w==1).to(torch.float32))
            self.output_image[proj_i+self.nprojs,:,:]= self.output_image[proj_i+self.nprojs,:,:] + temp

        self.t_image_from_coord+=(time.time() - t0)

    def nn_predict(self,model, model_data, x):
        '''
        Apply the NN to predict y from x
        '''

        # apply input model normalisation
        x = (x - self.x_mean) / self.x_std

        # torch encapsulation
        # x = x.astype('float32')
        # vx = Variable(torch.from_numpy(x)).type(dtypef)
        # vx = torch.from_numpy(x).to(device=self.device)

        vx = x.float()

        # predict values
        vy_pred = model(vx)

        # convert to numpy and normalize probabilities

        # y_pred = vy_pred.data.cpu().numpy()
        # y_pred = y_pred.astype(np.float64)
        y_pred = vy_pred
        y_pred = self.normalize_logproba(y_pred)
        y_pred = self.normalize_proba_with_russian_roulette(y_pred, 0, self.rr)
        # y_pred = y_pred.data.cpu().numpy()

        return y_pred

    def normalize_logproba(self,x):
        '''
        Convert un-normalized log probabilities to normalized ones (0-100%)
        Not clear how to deal with exp overflow ?
        (https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/)
        '''

        # exb = torch.exp(x)
        # exb_sum = torch.sum(exb, dim=1)
        # # divide if not equal at zero
        # p = torch.divide(exb.T, exb_sum,
        #               out=torch.zeros_like(exb.T)).T

        # check (should be equal to 1.0)
        # check = np.sum(p, axis=1)
        # print(check)

        b=x.amax(dim=1,keepdim=True)
        exb=torch.exp(x-b)
        exb_sum=torch.sum(exb,dim=1)
        p = torch.divide(exb.T,exb_sum,out=torch.zeros_like(exb.T)).T

        return p

    def compute_angle_offset(self,angles, length):
        '''
        compute the x,y offset according to the angle
        '''

        angles_rad = (angles)*np.pi/180
        cos_theta = torch.cos(angles_rad[:, 0])
        cos_phi = torch.cos(angles_rad[:, 1])

        tx = length * cos_phi    ## yes see in Gate_NN_ARF_Actor, line "phi = acos(dir.x())/degree;"
        ty = length * cos_theta  ## yes see in Gate_NN_ARF_Actor, line "theta = acos(dir.y())/degree;"
        t = torch.column_stack((tx, ty))

        return t



    def image_from_coordinates(self,img, u, v, w_pred):
        '''
        Convert an array of pixel coordinates u,v (int) and corresponding weight
        into an image
        '''

        # convert to int16
        u = u.astype(np.int16)
        v = v.astype(np.int16)

        # create a 32bit view of coordinate arrays to unite pairs of x,y into
        # single integer
        uv32 = np.vstack((u, v)).T.ravel().view(dtype=np.int32)

        # nb of energy windows
        nb_ene = len(w_pred[0])

        # sum up values for pixel coordinates which occur multiple times
        ch = []
        for i in range(1, nb_ene):
            a = np.bincount(uv32, weights=w_pred[:, i])
            ch.append(a)

        # init image
        img.fill(0.0)

        # create range array which goes along with the arrays returned by bincount
        # (see man for np.bincount)
        uv32Bins = np.arange(np.amax(uv32) + 1, dtype=np.int32)

        # this will generate many 32bit values corresponding to 16bit value pairs
        # lying outside of the image -> see conditions below

        # generate 16bit view to convert back and reshape
        uv16Bins = uv32Bins.view(dtype=np.uint16)
        hs = int((uv16Bins.size / 2))
        uv16Bins = uv16Bins.reshape((hs, 2))

        # fill image using index broadcasting
        # Important: the >0 condition is to avoid outside elements.
        tiny = 0  ## FIXME
        for i in range(1, nb_ene):
            chx = ch[i - 1]
            img[i, uv16Bins[chx > tiny, 0], uv16Bins[chx > tiny, 1]] = chx[chx > tiny]
        # end
        return img

    def image_from_coordinates_2(self, img, vu,w):
        # for uu,vv,ww in zip(vu[:,0],vu[:,1],w):
        #     img[uu,vv]+=ww
        img_r = img.ravel()
        ind_r = vu[:,1]*img.shape[0]+vu[:,0]
        img_r.put_(index=ind_r,source=w,accumulate=True)
        img=img_r.reshape_as(img)
        return img

    def remove_out_of_image_boundaries(self,vu, w_pred, size):
        '''
        Remove values out of the images (<0 or > size)
        '''
        len_0 = vu.shape[0]
        # index = torch.where((vu[:,0]>=0)
        #                     & (vu[:,1]>=0)
        #                     & (vu[:,0]< size[2])
        #                     & (vu[:,1]<size[1]))[0]
        # vu = vu[index]
        # w_pred = w_pred[index]

        vu_ = vu[(vu[:,0]>=0) & (vu[:,1]>=0) & (vu[:,0]< size[2]) & (vu[:,1]<size[1])]
        w_pred_ = w_pred[(vu[:,0]>=0) & (vu[:,1]>=0) & (vu[:,0]< size[2]) & (vu[:,1]<size[1])]

        if (len_0 - vu.shape[0]>0):
            print('Remove points out of the image: {} values removed sur {}'.format(len_0 - vu.shape[0], len_0))

        return vu_, w_pred_



    # -----------------------------------------------------------------------------
    def normalize_proba_with_russian_roulette(self,w_pred, channel, rr):
        '''
        Consider rr times the values for the energy windows channel
        '''
        # multiply column 'channel' by rr
        w_pred[:, channel] *= rr
        # normalize
        p_sum = torch.sum(w_pred, dim=1, keepdim=True)
        w_pred = w_pred / p_sum
        # check
        # p_sum = np.sum(w_pred, axis=1)
        # print(p_sum)
        return w_pred

    def save_projection(self):
        print(f'TIME FOR PREPROC: {self.t_preprocess_arf}')
        print(f'TIME FOR POSTROC: {self.t_postprocess_arf}')
        print(f'dont {self.t_remove} pour remove et {self.t_jsp} pour avant remove')
        print(f'TIME FOR NN PREDICT: {self.time_nn_predict}')
        print(f'TIME FOR IND TO COORD : {self.t_image_from_coord}')

        # convert to itk image
        output_image_array = self.output_image.cpu().numpy()
        self.output_image_itk = itk.image_from_array(output_image_array)

        # set spacing and origin like DigitizerProjectionActor
        spacing = self.image_spacing
        spacing = np.array([spacing[0], spacing[1], 1])
        size = np.array(self.image_size)
        size[0] = self.image_size[2]
        size[2] = self.image_size[0]
        origin = -size / 2.0 * spacing + spacing / 2.0
        origin[2] = 0
        self.output_image_itk.SetSpacing(spacing)
        self.output_image_itk.SetOrigin(origin)

        # convert double to float
        # InputImageType = itk.Image[itk.D, 3]
        # OutputImageType = itk.Image[itk.F, 3]
        # castImageFilter = itk.CastImageFilter[InputImageType, OutputImageType].New()
        # castImageFilter.SetInput(self.output_image_itk)
        # castImageFilter.Update()
        # self.output_image_itk = castImageFilter.GetOutput()


        itk.imwrite(self.output_image_itk, self.output_fn)
        print(f'Output projection saved in : {self.output_fn}')


def get_rot_matrix(theta):
    theta = torch.tensor([theta])
    return torch.tensor([[torch.cos(theta), 0, torch.sin(theta)],
                         [0, 1, 0],
                         [-torch.sin(theta), 0, torch.cos(theta)]])

# def get_rot_matrix(theta):#FIXME
#     theta = torch.tensor([theta])
#     return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
#                          [torch.sin(theta), torch.cos(theta), 0],
#                          [0, 0, 1]])

class DetectorPlane:
    def __init__(self, size, device, center0, rot_angle,dist_to_crystal):
        self.device = device
        self.M = get_rot_matrix(rot_angle)
        self.Mt = get_rot_matrix(-rot_angle).to(device)
        self.center =  torch.matmul(self.M,torch.tensor(center0).float()).to(device)
        self.normal = -self.center / torch.norm(self.center)
        self.dd = torch.matmul(self.center, self.normal)
        self.size = size
        self.dist_to_crystal = dist_to_crystal


    def get_intersection(self,batch):
        energ0 = batch[:, 0:1]
        pos0 = batch[:,1:4]
        dir0 = batch[:,4:7]

        dir_produit_scalaire = torch.sum(dir0*self.normal,dim=1)
        t = (self.dd - torch.sum(pos0*self.normal,dim=1))/dir_produit_scalaire

        pos_xyz = dir0*t[:,None] + pos0

        pos_xyz_rot = torch.matmul(pos_xyz, self.Mt.t()) #FIXME
        dir_xyz_rot = torch.matmul(dir0, self.Mt.t()) #FIXME
        # pos_xy_rot = torch.matmul(pos_xyz, self.Mt[[0,2], :].t())
        # dir_xy_rot = torch.matmul(dir0, self.Mt[[0,2], :].t())

        pos_xyz_rot_crystal = pos_xyz_rot + (self.dist_to_crystal/dir_xyz_rot[:,2:3]) * dir_xyz_rot
        pos_xy_rot_crystal = pos_xyz_rot_crystal[:,0:2]
        dir_xy_rot = dir_xyz_rot[:,0:2]

        indexes_to_keep = torch.where((dir_produit_scalaire<0) &
                                      (t>0) &
                                      (pos_xy_rot_crystal.abs().max(dim=1)[0] < self.size/2)
                                      )[0]

        batch_arf = torch.concat((pos_xy_rot_crystal[indexes_to_keep,:],
                                  dir_xy_rot[indexes_to_keep,:],
                                  energ0[indexes_to_keep,:]),dim=1)

        return batch_arf

    def get_intersection__(self,batch):
        energ0 = batch[:, 0:1]
        pos0 = batch[:,1:4]
        dir0 = batch[:,4:7]

        dir_produit_scalaire = torch.tensordot(dir0,self.normal,dims=1)

        keep = (dir_produit_scalaire<0)
        pos0,dir0,energ0 = pos0[keep],dir0[keep],energ0[keep]

        t= (self.dd - torch.tensordot(pos0,self.normal, dims=1))/dir_produit_scalaire[keep]

        keep = (t>0)
        pos0, dir0, energ0,t = pos0[keep], dir0[keep], energ0[keep],t[keep]

        pos_xyz = dir0*t[:,None] + pos0

        pos_xy_rot = torch.matmul(self.Mt, pos_xyz.t()).t()[:,0:2]
        dir_xy_rot = torch.matmul(self.Mt, dir0.t()).t()[:,0:2]

        pos_xy_rot_crystal = pos_xy_rot + 75 * dir_xy_rot

        keep=((pos_xy_rot.abs().max(dim=1)[0] < self.size / 2)
                & (pos_xy_rot_crystal.abs().max(dim=1)[0] < self.size/2))

        pos0, dir0, energ0 = pos0[keep,0:2], dir0[keep,0:2], energ0[keep]
        return torch.cat((pos0, dir0, energ0),dim=1)



def project_on_plane(x, plane, image_plane_size_mm, debug=False):
    """
    Project the x points (Ekine X Y Z dX dY dZ)
    on the image plane defined by plane_U, plane_V, plane_center, plane_normal
    """

    # print(f"Projection of {len(x)} particles on the plane")
    # print(f"Plane size is {image_plane_size_mm} mm")

    # shorter variable names

    # n is the normal plane, duplicated n times
    # n = plane["plane_normal"][0 : len(x)]
    n = np.tile(plane["plane_normal"],(len(x),1))

    # c0 is the center of the plane, duplicated n times
    # c0 = plane["plane_center"][0 : len(x)]
    c0 = np.tile(plane["plane_center"], (len(x), 1))

    # r is the rotation matrix of the plane, according to the current rotation angle (around Y)
    # r = plane["rotation"][0 : len(x)]
    r = plane["rotation"]

    # p is the set of points position generated by the GAN
    p = x[:, 1:4]

    # u is the set of points direction generated by the GAN
    u = x[:, 4:7]

    # w is the set of vectors from all points to the plane center
    w = p - c0

    # project to plane
    ## dot product : out = (x*y).sum(-1)
    # https://rosettacode.org/wiki/Find_the_intersection_of_a_line_with_a_plane#Python
    # http://geomalgorithms.com/a05-_intersect-1.html
    # https://github.com/pytorch/pytorch/issues/18027
    ndotu = (n * u).sum(-1)  # dot product between normal plane (n) and direction (u)
    si = (
        -(n * w).sum(-1) / ndotu
    )  # dot product between normal plane and vector from plane to point (w)

    # only positive (direction to the plane)
    mask = si > 0
    mw = w[mask]
    mu = u[mask]
    mc0 = c0[mask]
    mn = n[mask]
    mx = x[mask]
    mp = p[mask]
    msi = si[mask]
    mnb = len(msi)
    # print(f"Remove negative direction, remains {mnb}/{len(x)}")

    # si is a (nb) size vector, expand it to (nb x 3)
    msi = np.array([msi] * 3).T

    # intersection between point-direction and plane
    psi = mp + msi * mu

    # apply the inverse of the rotation
    ri = r.T
    # psip = ri.apply(psi)  # - offset
    psip = np.dot(ri,psi.T).T  # - offset

    # remove out of plane (needed ??)
    sizex = image_plane_size_mm[0] / 2.0
    sizey = image_plane_size_mm[1] / 2.0
    mask1 = psip[:, 0] < sizex
    mask2 = psip[:, 0] > -sizex
    mask3 = psip[:, 1] < sizey
    mask4 = psip[:, 1] > -sizey
    m = mask1 & mask2 & mask3 & mask4
    psip = psip[m]
    psi = psi[m]
    mp = mp[m]
    mu = mu[m]
    mx = mx[m]
    mc0 = mc0[m]
    nb = len(psip)
    # print(f"Remove points that are out of detector, remains {nb}/{len(x)}")

    # reshape results
    pu = psip[:, 0].reshape((nb, 1))  # u
    pv = psip[:, 1].reshape((nb, 1))  # v
    y = np.concatenate((pu, pv), axis=1)

    # rotate direction according to the plane
    # mup = ri.apply(mu)
    mup = np.dot(ri,mu.T).T
    norm = np.linalg.norm(mup, axis=1, keepdims=True)
    mup = mup / norm
    dx = mup[:, 0]
    dy = mup[:, 1]

    # FIXME -> clip arcos -1;1 ?

    # convert direction into theta/phi
    # theta is acos(dy)
    # phi is acos(dx)
    theta = np.degrees(np.arccos(dy)).reshape((nb, 1))
    phi = np.degrees(np.arccos(dx)).reshape((nb, 1))
    y = np.concatenate((y, theta), axis=1)
    y = np.concatenate((y, phi), axis=1)

    # concat the E
    E = mx[:, 0].reshape((nb, 1))
    data = np.concatenate((y, E), axis=1)

    return data
