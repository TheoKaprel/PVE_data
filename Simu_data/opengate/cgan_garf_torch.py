#!/usr/bin/env python3

import argparse
import opengate.contrib.phantom_nema_iec_body as gate_iec
import numpy as np
import pathlib
import os
from pathlib import Path
from box import Box
import itk
from garf_helpers import *
import gaga_phsp as gaga
import garf
import scipy
import torch
from torch.autograd import Variable

import matplotlib.pyplot as plt
import time

class CGANSOURCE:
    def __init__(self, user_info):
        self.batchsize = int(float(user_info['batchsize']))
        self.pth_filename = user_info['pth_filename']
        self.device = user_info['device']
        self.init_gan()

        self.condition_time = 0
        self.gan_time = 0

    def init_gan(self):
        self.gan_info = Box()
        g = self.gan_info
        g.params, g.G, g.D, g.optim, g.dtypef = gaga.load(
            self.pth_filename, "auto", verbose=False
        )

        self.z_rand = self.get_z_rand(g.params)

        # normalize the conditional vector
        xmean = g.params["x_mean"][0]
        xstd = g.params["x_std"][0]
        xn = g.params["x_dim"]
        cn = len(g.params["cond_keys"])
        self.ncond = cn
        # mean and std for cond only
        self.xmeanc = xmean[xn - cn: xn]
        self.xstdc = xstd[xn - cn: xn]
        # mean and std for non cond
        self.xmeannc = xmean[0: xn - cn]
        self.xstdnc = xstd[0: xn - cn]

        self.z_dim = g.params["z_dim"]
        self.x_dim = g.params["x_dim"]

    def get_z_rand(self,params):
        if "z_rand_type" in params:
            if params["z_rand_type"] == "rand":
                return torch.rand
            if params["z_rand_type"] == "randn":
                return torch.randn
        if "z_rand" in params:
            if params["z_rand"] == "uniform":
                return torch.rand
            if params["z_rand"] == "normal":
                return torch.randn
        params["z_rand_type"] = "randn"
        return torch.randn

    def generate_condition(self, n):
        pass

    def generate_samples(self, cond):
        cond = (cond - self.xmeanc) / self.xstdc
        z = self.z_rand(self.batchsize,self.z_dim).to(device = self.device)
        condx = torch.from_numpy(cond).to(device = self.device).view(self.batchsize,self.ncond)
        z = torch.cat((z.float(), condx.float()), dim=1)

        fake = self.gan_info.G(z)
        fake = fake.cpu().data.numpy()
        fake = (fake * self.xstdnc) + self.xmeannc
        return fake


    def generate(self):
        n = int(float(self.batchsize))
        condition_t0 = time.time()
        cond = self.generate_condition(n)
        condition_t1 = time.time()
        self.condition_time+=(condition_t1-condition_t0)

        generation_t0 = time.time()
        fake = self.generate_samples(cond)
        generation_t1 = time.time()
        self.gan_time+=(generation_t1-generation_t0)

        # fake = gaga.generate_samples2(
        #     g.params,
        #     g.G,
        #     g.D,
        #     n=n,
        #     batch_size=n,
        #     normalize=False,
        #     to_numpy=True,
        #     cond=cond,
        #     silence=True,
        # )
        return fake


class GARF:
    def __init__(self, user_info):
        self.batchsize = user_info['batchsize']
        self.pth_filename = user_info['pth_filename']
        self.output_fn = user_info['output_fn']
        self.device = user_info['device']

        self.image_size = [128, 128]
        self.image_spacing = [4.41806, 4.41806]
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

        # output image: nb of energy windows times nb of runs (for rotation)
        self.nb_ene = self.model_data["n_ene_win"]
        # size and spacing in 3D
        self.image_size = [self.nb_ene, self.image_size[0], self.image_size[1]]
        self.image_spacing = [self.image_spacing[0], self.image_spacing[1], 1]
        # create output image as np array
        self.output_size = [self.nb_ene, self.image_size[1], self.image_size[2]]
        self.output_image = np.zeros(self.output_size, dtype=np.float64)
        # compute offset
        self.psize = [
            self.image_size[1] * self.image_spacing[0],
            self.image_size[2] * self.image_spacing[1],
        ]
        self.hsize = np.divide(self.psize, 2.0)
        self.offset = [self.image_spacing[0] / 2.0, self.image_spacing[1] / 2.0]


    def apply(self,batch):
        x = np.copy(batch)
        x[:,2] = np.arccos(batch[:,3]) / self.degree
        x[:,3] = np.arccos(batch[:,2]) / self.degree

        ax = x[:, 2:5]  # two angles and energy
        w = self.nn_predict(self.model, self.nn["model_data"], ax)

        # positions
        angles = x[:, 2:4]
        t = garf.compute_angle_offset(angles, self.distance_to_crystal)
        cx = x[:, 0:2]
        cx = cx + t
        coord = (cx + self.hsize - self.offset) / self.image_spacing[0:2]
        coord = np.around(coord).astype(int)
        v = coord[:, 0]
        u = coord[:, 1]
        u, v, w_pred = garf.remove_out_of_image_boundaries(u, v, w, self.image_size)

        # do nothing if there is no hit in the image
        if u.shape[0] != 0:
            temp = np.zeros(self.image_size, dtype=np.float64)
            temp = garf.image_from_coordinates(temp, u, v, w_pred)
            # add to previous, at the correct slice location
            # the slice is : current_ene_window + run_id * nb_ene_windows

            self.output_image[0:self.nb_ene] = (
                    self.output_image[0:self.nb_ene] + temp
            )

    def nn_predict(self,model, model_data, x):
        '''
        Apply the NN to predict y from x
        '''

        x_mean = model_data['x_mean']
        x_std = model_data['x_std']
        if ('rr' in model_data):
            rr = model_data['rr']
        else:
            rr = model_data['RR']

        # apply input model normalisation
        x = (x - x_mean) / x_std

        # torch encapsulation
        x = x.astype('float32')
        # vx = Variable(torch.from_numpy(x)).type(dtypef)
        vx = torch.from_numpy(x).to(device=self.device)

        # predict values
        vy_pred = model(vx)

        # convert to numpy and normalize probabilities

        y_pred = vy_pred.data.cpu().numpy()
        y_pred = y_pred.astype(np.float64)
        y_pred = self.normalize_logproba(y_pred)
        y_pred = self.normalize_proba_with_russian_roulette(y_pred, 0, rr)

        # return
        return y_pred

    def normalize_logproba(self,x):
        '''
        Convert un-normalized log probabilities to normalized ones (0-100%)
        Not clear how to deal with exp overflow ?
        (https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/)
        '''
        exb = np.exp(x)
        exb_sum = np.sum(exb, axis=1)
        # divide if not equal at zero
        p = np.divide(exb.T, exb_sum,
                      out=np.zeros_like(exb.T),
                      where=exb_sum != 0).T
        # check (should be equal to 1.0)
        # check = np.sum(p, axis=1)
        # print(check)
        return p

    # -----------------------------------------------------------------------------
    def normalize_proba_with_russian_roulette(self,w_pred, channel, rr):
        '''
        Consider rr times the values for the energy windows channel
        '''
        # multiply column 'channel' by rr
        w_pred[:, channel] *= rr
        # normalize
        p_sum = np.sum(w_pred, axis=1, keepdims=True)
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


class DetectorPlane:
    def __init__(self,sid, size):
        self.sid = sid
        self.normal = np.array([0,0,-1])
        self.center = np.array([0,0,sid])
        self.size = size

    def get_intersection(self,batch):
        pos0 = batch[:,1:4]
        dir0 = batch[:,4:7]
        t= (self.sid - np.dot(pos0,self.normal))/(np.dot(dir0,self.normal))
        pos_xyz = dir0*(t[:,None])
        pos_xyz += pos0

        indexes_to_keep = ((pos_xyz[:,0] > -self.size/2) &
                          (pos_xyz[:,0] < self.size/2) &
                          (pos_xyz[:,1] > -self.size/2) &
                          (pos_xyz[:,1] < self.size/2))
        pos_xyz = pos_xyz[indexes_to_keep]
        batch_to_keep = batch[indexes_to_keep]
        batch_arf = np.hstack((pos_xyz[:, 0:2],
                               batch_to_keep[:, 4:6],
                               batch_to_keep[:, 0:1]))

        return batch_arf

def recons_ideal(output_fn, particles):
    c = scipy.constants.speed_of_light * 1000  # in mm
    positions,directions,times, energies = particles[:,1:4],particles[:,4:7],particles[:,7:8],particles[:,0:1]
    emissions = np.zeros_like(positions)
    for pos, dir, t, E, p in zip(positions, directions, times, energies, emissions):
        l = t * c
        p += pos + l * -dir

    size = np.array((256, 256, 256)).astype(int)
    spacing = np.array([2, 2, 2])
    offset = -size * spacing / 2.0 + spacing / 2.0
    print(f"Image size, spacing, offset: {size} {spacing} {offset}")
    pix = np.rint((positions - offset) / spacing).astype(int)

    # remove values out of the image fov
    print(f"Number of events after E selection: {len(pix)}")
    for i in [0, 1, 2]:
        pix = pix[(pix[:, i] < size[i]) & (pix[:, i] > 0)]
    print(f"Number of events in the image FOV:  {len(pix)}")


    print(f"Output file: {output_fn}")

    # create the image
    a = np.zeros(size)
    for x in pix:
        a[x[0], x[1], x[2]] += 1
    img = itk.image_from_array(a)
    #
    img.SetSpacing(spacing.tolist())
    img.SetOrigin(offset)
    itk.imwrite(img, output_fn)


def main():
    print(args)
    paths = Box()
    paths.current = pathlib.Path(__file__).parent.resolve()

    t0 = time.time()

    # activity parameters
    total_activity = int(float(args.activity))
    spheres_diam = [10, 13, 17, 22, 28, 37]
    spheres_activity_concentration_ratio = [6, 5, 4, 3, 2, 1]
    # initialisation for conditional
    spheres_radius = [x / 2.0 for x in spheres_diam]
    spheres_centers,spheres_volumes = gate_iec.get_default_sphere_centers_and_volumes()
    spheres_activity_ratio = [spheres_activity_concentration_ratio[k]*spheres_volumes[k] for k in range(6)]
    total2 = sum(spheres_activity_ratio)
    spheres_activity_ratio = [r / total2 for r in spheres_activity_ratio]
    print("Activity Ratio ", spheres_activity_ratio)

    print(f"Total activity {total_activity} Bq")
    # unique (reproducible) random generator
    rs = gate.get_rnd_seed(123456)
    def generate_condition(n):
        n_samples = gate_iec.get_n_samples_from_ratio(n, spheres_activity_ratio)
        cond = gate_iec.generate_pos_dir_spheres(
            spheres_centers, spheres_radius, n_samples, shuffle=True, rs=rs
        )
        return cond

    gan_info = {}
    gan_info['pth_filename'] = os.path.join(paths.current, "pths/test001_GP_0GP_10_50000.pth")
    gan_info['batchsize'] = args.batchsize
    gan_info['device'] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    cgan_source = CGANSOURCE(gan_info)
    cgan_source.generate_condition = generate_condition

    print('PARAMS KEYS : ')
    print(cgan_source.gan_info.params.keys())
    print()

    # print(cgan_source.gan_info.params)
    print("CONDITIONS : ")
    print(cgan_source.gan_info.params['cond_keys'])


    print("KEYS : ")
    keys_list = cgan_source.gan_info.params['keys_list']
    print(keys_list)

    detectorPlane = DetectorPlane(sid = 380,size = 565.51168)


    garf_ui = {}
    garf_ui['pth_filename'] = os.path.join(paths.current, "pths/arf_5x10_9.pth")
    garf_ui['batchsize'] = args.batchsize
    garf_ui['output_fn'] = os.path.join(args.output,"proj.mhd")
    garf_ui['device'] = gan_info['device']
    garf_detector = GARF(user_info=garf_ui)

    # debug
    # fake = cgan_source.generate()
    # print(f"{fake.shape[0]} generated photons")
    # fake = fake[fake[:, 0] > 0.01] # keep exiting particles only
    # recons_ideal(output_fn=os.path.join(args.output, "recons_ideal.mhd"),particles=fake)

    #
    # if args.v:
    #     fig,ax = plt.subplots(3,3)
    #     axes = ax.ravel()
    #     for k in range(8):
    #         axes[k].hist(fake[:,k], bins = 100)
    #         axes[k].set_title(keys_list[k])
    #     plt.show()

    # print(f"{fake.shape[0]} exiting the phspace")


    intsction_time = 0
    generation_time = 0
    detection_time = 0

    generated_particles = 0
    while generated_particles < total_activity:
        ready_to_garf = None
        batch_count = 0
        while (batch_count==0) or (ready_to_garf.shape[0]<args.batchsize):
            generation_t0 = time.time()
            fake = cgan_source.generate()
            generation_t1 = time.time()
            generation_time+=(generation_t1-generation_t0)
            # print(f"{fake.shape[0]} generated photons")
            fake = fake[fake[:, 0] > 0.01] # keep exiting particles only
            intsction_t0 = time.time()
            batch_arf = detectorPlane.get_intersection(batch=fake)
            intsction_t1 = time.time()
            intsction_time+=(intsction_t1-intsction_t0)
            arf_key_list= ['pos_x', 'pos_y', 'dir_x', 'dir_y', 'energy']
            if batch_count==0:
                ready_to_garf = batch_arf
                if args.v:
                    fig, ax = plt.subplots(3, 2)
                    axes = ax.ravel()
                    for k in range(5):
                        axes[k].hist(batch_arf[:, k], bins=100)
                        axes[k].set_title(arf_key_list[k])
                    plt.show()
            else:
                ready_to_garf = np.vstack((ready_to_garf,batch_arf))
            batch_count+=1

        # print(f"{batch_arf.shape[0]} intersecting the detector")

        detection_t0 = time.time()
        garf_detector.apply(batch_arf)
        detection_t1 = time.time()
        detection_time +=(detection_t1-detection_t0)

        generated_particles+=cgan_source.batchsize * batch_count
        print(f'{generated_particles/total_activity * 100} % ...')


    garf_detector.save_projection()

    t1 = time.time()


    print(f"TOTAL TIME ELASPED : {t1-t0} s")
    print('Including : ')
    print(f'     *  {generation_time} s for particle generation')
    print(f'        Including : ')
    print(f'                °{cgan_source.condition_time} s for conditions')
    print(f'                °{cgan_source.gan_time} s for gan')
    print(f'     *  {intsction_time} s for ray tracing and intersecting particle selection')
    print(f'     *  {detection_time} s for detection')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-a","--activity", type = float, default = 100)
    parser.add_argument("-b","--batchsize", type = float, default = 100000)
    parser.add_argument("-o", "--output", type = str)
    parser.add_argument('-v', action="store_true")

    args = parser.parse_args()

    main()
