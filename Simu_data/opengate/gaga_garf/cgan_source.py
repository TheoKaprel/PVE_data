#!/usr/bin/env python3


from box import Box
import torch
import time
from torch.utils.data import Dataset
import opengate as gate

import opengate.contrib.phantom_nema_iec_body as gate_iec
import gaga_phsp as gaga


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


    def generate(self, z):
        fake = self.gan_info.G(z)
        fake = (fake * self.xstdnc) + self.xmeannc
        return fake.float()


class ConditionsDataset(Dataset):
    def __init__(self, activity, cgan_src):
        self.total_activity = int(float(activity))
        spheres_diam = [10, 13, 17, 22, 28, 37]
        spheres_activity_concentration_ratio = [6, 5, 4, 3, 2, 1]
        # initialisation for conditional
        self.spheres_radius = [x / 2.0 for x in spheres_diam]
        self.spheres_centers, self.spheres_volumes = gate_iec.get_default_sphere_centers_and_volumes()
        spheres_activity_ratio = [spheres_activity_concentration_ratio[k] * self.spheres_volumes[k] for k in range(6)]
        total2 = sum(spheres_activity_ratio)
        self.spheres_activity_ratio = [r / total2 for r in spheres_activity_ratio]
        print("Activity Ratio ", self.spheres_activity_ratio)
        print(f"Total activity {self.total_activity} Bq")

        self.xmeanc = cgan_src.xmeanc.cpu()
        self.xstdc = cgan_src.xstdc.cpu()
        self.z_dim = cgan_src.z_dim
        self.z_rand = cgan_src.z_rand

        self.all_conditions = self.generate_condition(n=self.total_activity)

    def generate_condition(self,n):
        n_samples = gate_iec.get_n_samples_from_ratio(n, self.spheres_activity_ratio)
        cond = gate_iec.generate_pos_dir_spheres(
            self.spheres_centers, self.spheres_radius, n_samples, shuffle=True)
        return cond

    def __len__(self):
        return self.total_activity


    def __getitem__(self, idx):
        condx = torch.from_numpy(self.all_conditions[idx,:])
        condx = (condx - self.xmeanc) / self.xstdc
        z = self.z_rand(self.z_dim)
        return z,condx