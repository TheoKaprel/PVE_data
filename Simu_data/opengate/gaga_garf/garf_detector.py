#!/usr/bin/env python3

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

        self.size = 128
        self.spacing = 4.41806
        self.image_size = [self.nprojs, 128, 128]
        self.image_spacing = [self.spacing, self.spacing, 1]

        self.distance_to_crystal = 75
        self.degree = np.pi / 180
        self.init_garf()

    def init_garf(self):
        # load the pth file
        self.nn, self.model = garf.load_nn(
            self.pth_filename, gpu="auto", verbose=False
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
        self.output_image = np.zeros(self.image_size, dtype=np.float64)
        # compute offset
        self.psize = [self.size * self.spacing,self.size * self.spacing]

        self.hsize = np.divide(self.psize, 2.0)
        self.offset = [self.image_spacing[0] / 2.0, self.image_spacing[1] / 2.0]


    def apply(self,batch, proj_i):
        x = batch.clone()
        x[:,2] = torch.arccos(batch[:,3]) / self.degree
        x[:,3] = torch.arccos(batch[:,2]) / self.degree

        ax = x[:, 2:5]  # two angles and energy
        w = self.nn_predict(self.model, self.nn["model_data"], ax)

        # positions
        x_np = x.cpu().numpy()
        angles = x_np[:, 2:4]
        t = garf.compute_angle_offset(angles, self.distance_to_crystal)
        cx = x_np[:, 0:2]
        cx = cx + t
        coord = (cx + self.hsize - self.offset) / self.image_spacing[0:2]
        coord = np.around(coord).astype(int)
        v = coord[:, 0]
        u = coord[:, 1]
        u, v, w_pred = garf.remove_out_of_image_boundaries(u, v, w, self.image_size)
        # print('------')
        # print(u[:10])
        # print(v[:10])
        # print(w_pred[:10])
        # print('------')

        # do nothing if there is no hit in the image
        if u.shape[0] != 0:
            temp = np.zeros([self.image_size[1], self.image_size[2]], dtype=np.float64)
            temp = self.image_from_coordinates_2(temp, u,v,w_pred[:,2])

            # temp = self.image_from_coordinates(temp, u, v, w_pred)[2,:,:] # STORE ONLY  PRIMARY window
            # self.output_image[proj_i,:,:] = (
            #         # self.output_image[proj_i,:,:] + temp
            #         self.output_image[proj_i,:,:] + temp
            # )
            self.output_image[proj_i,:,:]= self.output_image[proj_i,:,:] + temp


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

        y_pred = y_pred.data.cpu().numpy()

        return y_pred

    def normalize_logproba(self,x):
        '''
        Convert un-normalized log probabilities to normalized ones (0-100%)
        Not clear how to deal with exp overflow ?
        (https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/)
        '''
        exb = torch.exp(x)
        exb_sum = torch.sum(exb, dim=1)
        # divide if not equal at zero
        p = torch.divide(exb.T, exb_sum,
                      out=torch.zeros_like(exb.T)).T
        # check (should be equal to 1.0)
        # check = np.sum(p, axis=1)
        # print(check)
        return p

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

    def image_from_coordinates_2(self, img, u,v,w):

        for uu,vv,ww in zip(u,v,w):
            img[uu,vv]+=ww

        return img


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
        # convert to itk image
        self.output_image = itk.image_from_array(self.output_image)

        # set spacing and origin like DigitizerProjectionActor
        spacing = self.image_spacing
        spacing = np.array([spacing[0], spacing[1], 1])
        size = np.array(self.image_size)
        size[0] = self.image_size[2]
        size[2] = self.image_size[0]
        origin = -size / 2.0 * spacing + spacing / 2.0
        origin[2] = 0
        self.output_image.SetSpacing(spacing)
        self.output_image.SetOrigin(origin)

        # convert double to float
        InputImageType = itk.Image[itk.D, 3]
        OutputImageType = itk.Image[itk.F, 3]
        castImageFilter = itk.CastImageFilter[InputImageType, OutputImageType].New()
        castImageFilter.SetInput(self.output_image)
        castImageFilter.Update()
        self.output_image = castImageFilter.GetOutput()


        itk.imwrite(self.output_image, self.output_fn)
        print(f'Output projection saved in : {self.output_fn}')


def get_rot_matrix(theta):
    theta = torch.tensor([theta])
    return torch.tensor([[torch.cos(theta), 0, torch.sin(theta)],
                         [0, 1, 0],
                         [-torch.sin(theta), 0, torch.cos(theta)]])


class DetectorPlane:
    def __init__(self, size, device, center0, rot_angle):
        self.device = device
        self.M = get_rot_matrix(rot_angle)
        self.Mt = get_rot_matrix(-rot_angle).to(device)
        self.center =  torch.matmul(self.M,torch.tensor(center0).float()).to(device)
        self.normal = -self.center / torch.norm(self.center)
        self.dd = torch.matmul(self.center, self.normal)
        self.size = size


    def get_intersection(self,batch):
        energ0 = batch[:, 0:1]
        pos0 = batch[:,1:4]
        dir0 = batch[:,4:7]

        dir_produit_scalaire = torch.tensordot(dir0,self.normal,dims=1)
        keep = (dir_produit_scalaire<0)
        t= (self.dd - torch.tensordot(pos0,self.normal, dims=1))/dir_produit_scalaire
        keep = keep & (t>0)
        pos_xyz = dir0*t[:,None] + pos0

        pos_xyz_rot = torch.matmul(self.Mt, pos_xyz.t()).t()
        dir_rot = torch.matmul(self.Mt, dir0.t()).t()

        indexes_to_keep = ((keep) &
                           (torch.abs(pos_xyz_rot[:,0])<self.size/2) &
                           (torch.abs(pos_xyz_rot[:, 1]) < self.size / 2)
                           )

        pos_xy_rot_keep = pos_xyz_rot[indexes_to_keep,0:2]
        dir_to_keep = dir_rot[indexes_to_keep,0:2]
        energ_to_keep = energ0[indexes_to_keep,:]

        batch_arf = torch.concat((pos_xy_rot_keep,
                               dir_to_keep,
                               energ_to_keep),dim=1) # pos, dir, energy

        return batch_arf
