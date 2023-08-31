#!/usr/bin/env python3

import argparse
import opengate.contrib.phantom_nema_iec_body as gate_iec
import numpy as np
import pathlib
import os
from box import Box
import itk
from garf_helpers import *
import gaga_phsp as gaga
import garf
import scipy
import torch
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
        g.params, g.G, _, __, ___ = gaga.load(
            self.pth_filename, "auto", verbose=False
        )
        g.G = g.G.to(self.device)
        g.G.eval()

        self.z_rand = self.get_z_rand(g.params)

        # normalize the conditional vector
        xmean = torch.tensor(g.params["x_mean"][0], device=self.device)
        xstd = torch.tensor(g.params["x_std"][0], device=self.device)
        xn = g.params["x_dim"]
        cn = len(g.params["cond_keys"])
        self.ncond = cn
        # mean and std for cond only
        self.xmeanc = xmean[xn - cn: xn]
        self.xstdc = xstd[xn - cn: xn]
        # mean and std for non cond
        self.xmeannc = xmean[0: xn - cn]
        self.xstdnc = xstd[0: xn - cn]
        print(f"mean nc : {self.xmeannc}")
        print(f"mean c : {self.xmeanc}")
        print(f"std nc : {self.xstdnc}")
        print(f"std c : {self.xstdc}")

        self.z_dim = g.params["z_dim"]
        self.x_dim = g.params["x_dim"]

        print(f"zdim : {self.z_dim}")
        print(f"xdim : {self.x_dim}")

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
        condx = torch.from_numpy(cond).to(device=self.device)
        condx = (condx - self.xmeanc) / self.xstdc
        z = self.z_rand(condx.shape[0],self.z_dim).to(device = self.device)
        # condx = torch.from_numpy(cond).to(device = self.device).view(self.batchsize,self.ncond)

        z = torch.cat((z.float(), condx.float()), dim=1)
        fake = self.gan_info.G(z)
        # fake = fake.cpu().data.numpy()

        fake = (fake * self.xstdnc) + self.xmeannc
        return fake


    def generate(self, n):

        condition_t0 = time.time()
        cond = self.generate_condition(n)
        condition_t1 = time.time()
        self.condition_time+=(condition_t1-condition_t0)

        generation_t0 = time.time()
        fake = self.generate_samples(cond)
        generation_t1 = time.time()
        self.gan_time+=(generation_t1-generation_t0)

        return fake.float()


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

        self.x_mean = torch.tensor(self.model_data['x_mean'], device = self.device)
        self.x_std = torch.tensor(self.model_data['x_std'], device=self.device)
        if ('rr' in self.model_data):
            self.rr = self.model_data['rr']
        else:
            self.rr = self.model_data['RR']


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


class DetectorPlane:
    def __init__(self,sid, size, device):
        self.device = device
        self.sid = sid
        self.normal = torch.tensor([0,0,-1], device=self.device).float()
        self.center = np.array([0,0,sid])
        self.size = size


    def get_intersection(self,batch):
        pos0 = batch[:,1:4]
        dir0 = batch[:,4:7]
        dir_produit_scalaire = torch.tensordot(dir0,self.normal,dims=1)
        t= (self.sid - torch.tensordot(pos0,self.normal, dims=1))/dir_produit_scalaire
        pos_xyz = dir0*t[:,None] + pos0

        indexes_to_keep = (
                # (batch[:,0]>0.01) &
                            (pos_xyz[:,0] > -self.size/2) &
                            (pos_xyz[:,0] < self.size/2) &
                            (pos_xyz[:,1] > -self.size/2) &
                            (pos_xyz[:,1] < self.size/2))

        pos_xyz = pos_xyz[indexes_to_keep]
        batch_to_keep = batch[indexes_to_keep]
        batch_arf = torch.concat((pos_xyz[:, 0:2],
                               batch_to_keep[:, 4:6],
                               batch_to_keep[:, 0:1]),dim=1)

        return batch_arf


def project_on_plane_torch(x, plane, image_plane_size_mm):
    """
    Project the x points (Ekine X Y Z dX dY dZ)
    on the image plane defined by plane_U, plane_V, plane_center, plane_normal
    """
    # n is the normal plane, duplicated n times


    n = plane["plane_normal"]

    # c0 is the center of the plane, duplicated n times
    c0 = plane["plane_center"]

    # r is the rotation matrix of the plane, according to the current rotation angle (around Y)

    # p is the set of points position generated by the GAN
    p = x[:, 1:4]

    # u is the set of points direction generated by the GAN
    u = x[:, 4:7]

    # w is the set of vectors from all points to the plane center
    w = p - c0

    # dot product between normal plane (n) and direction (u)
    ndotu = (n * u).sum(-1)

    # dot product between normal plane and vector from plane to point (w)
    si = (-(n * w).sum(-1) / ndotu)

    # only positive (direction to the plane)
    mask = si > 0
    mu = u[mask]
    mx = x[mask]
    mp = p[mask]
    msi = si[mask]

    # si is a (nb) size vector, expand it to (nb x 3)
    msi = msi.expand((3,len(msi))).T
    # intersection between point-direction and plane
    psip = mp + msi * mu + c0

    # apply the inverse of the rotation
    # psip = torch.from_numpy(r.apply(psi))

    # remove out of plane (needed ??)
    sizex = image_plane_size_mm[0] / 2.0
    sizey = image_plane_size_mm[1] / 2.0
    mask1 = psip[:, 0] < sizex
    mask2 = psip[:, 0] > -sizex
    mask3 = psip[:, 1] < sizey
    mask4 = psip[:, 1] > -sizey
    m = mask1 & mask2 & mask3 & mask4
    psip = psip[m]
    mu = mu[m]
    mx = mx[m]
    nb = len(psip)

    # reshape results
    pu = psip[:, 0].reshape((nb, 1))  # u
    pv = psip[:, 1].reshape((nb, 1))  # v
    # y = np.concatenate((pu, pv), axis=1)

    # rotate direction according to the plane
    # mup = torch.from_numpy(r.apply(mu))
    norm = torch.norm(mu, dim=1, keepdim=True)
    mup = mu / norm
    dx = mup[:, 0]
    dy = mup[:, 1]

    # concat the E
    E = mx[:, 0].reshape((nb, 1))
    dx = dx.reshape((nb, 1))
    dy = dy.reshape((nb, 1))
    data = torch.cat((pu,pv,dx,dy,E),dim=1)

    return data


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

    t0 = time.time()
    print(args)
    paths = Box()
    paths.current = pathlib.Path(__file__).parent.resolve()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')

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
    gan_info['device'] = device
    print(device)
    cpu = torch.device('cpu')

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

    detectorPlane = DetectorPlane(sid = 380,size = 565.51168, device=device)

    garf_ui = {}
    garf_ui['pth_filename'] = os.path.join(paths.current, "pths/arf_5x10_9.pth")
    garf_ui['batchsize'] = args.batchsize
    garf_ui['output_fn'] = os.path.join(args.output,"proj.mhd")
    garf_ui['device'] = device
    garf_detector = GARF(user_info=garf_ui)

    intsction_time = 0
    generation_time = 0
    selection_time = 0
    detection_time = 0
    selec_time,mask_time = 0,0

    generated_particles = 0
    init_time = time.time() - t0
    with torch.no_grad():
        while generated_particles < total_activity:
            ready_to_garf = None
            batch_count = 0
            while (batch_count==0) or (ready_to_garf.shape[0]<args.batchsize):
                generation_t0 = time.time()
                fake = cgan_source.generate(n=cgan_source.batchsize)
                generated_particles += cgan_source.batchsize
                generation_t1 = time.time()


                generation_time+=(generation_t1-generation_t0)
                selection_t0 = time.time()

                fake = fake[fake[:,0]>0.01]

                # fig,ax = plt.subplots(2,4)
                # axes = ax.ravel()
                # zeros = fake.cpu().numpy()
                # l = ['KineticEnergy',
                #      'PrePosition_X', 'PrePosition_Y', 'PrePosition_Z',
                #      'PreDirection_X', 'PreDirection_Y', 'PreDirection_Z',
                #      'TimeFromBeginOfEvent']
                # for p in range(8):
                #     axes[p].hist(zeros[:,p],bins = 100)
                #     axes[p].set_title(l[p])
                # plt.show()

                selection_time +=(time.time() - selection_t0)

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
                    # ready_to_garf = np.vstack((ready_to_garf,batch_arf))
                    ready_to_garf = torch.vstack((ready_to_garf,batch_arf))
                batch_count+=1

            detection_t0 = time.time()
            garf_detector.apply(batch_arf)
            detection_t1 = time.time()
            detection_time +=(detection_t1-detection_t0)


            print(f'{generated_particles/total_activity * 100} % ...')


    save_t0= time.time()
    garf_detector.save_projection()
    save_time = time.time() - save_t0

    t1 = time.time()


    print(f"TOTAL TIME ELASPED : {t1-t0} s")
    print('Including : ')
    print(f'     *  {init_time} s for initialization')
    print(f'     *  {generation_time} s for particle generation')
    print(f'        Including : ')
    print(f'                °{cgan_source.condition_time} s for conditions')
    print(f'                °{cgan_source.gan_time} s for gan')
    print(f'     *  {selection_time} s for non-null energy particle selection')
    print(f'        Including : ')
    print(f'                °{mask_time} s for mask')
    print(f'                °{selec_time} s for selection')
    print(f'     *  {intsction_time} s for ray tracing and intersecting particle selection')
    print(f'     *  {detection_time} s for detection')
    print(f'     * {save_time} s for proj saving')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-a","--activity", type = float, default = 100)
    parser.add_argument("-b","--batchsize", type = float, default = 100000)
    parser.add_argument("-o", "--output", type = str)
    parser.add_argument('-v', action="store_true")

    args = parser.parse_args()

    main()
