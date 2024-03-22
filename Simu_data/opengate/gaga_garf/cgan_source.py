#!/usr/bin/env python3


from box import Box
import torch
import itk
import numpy as np
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
        g.params, g.G, _, __ = gaga.load(
            self.pth_filename, "auto"
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


class VoxelizerSourcePDFSamplerTorch:
    """
    This is an alternative to GateSPSVoxelsPosDistribution (c++)
    It is needed because the cond voxel source is used on python side.

    There are two versions, version 2 is much slower (do not use)
    """

    def __init__(self, itk_image, version=1):
        self.image = itk_image
        self.version = version
        # get image in np array
        self.imga = itk.array_view_from_image(itk_image)
        imga = self.imga

        # image sizes
        lx = self.imga.shape[0]
        ly = self.imga.shape[1]
        lz = self.imga.shape[2]

        # normalized pdf
        pdf = imga.ravel(order="F")
        self.pdf = pdf / pdf.sum()

        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.pdf = torch.from_numpy(self.pdf).to()

        # create grid of indices
        [x_grid, y_grid, z_grid] = torch.meshgrid(
            torch.arange(lx).to(self.device), torch.arange(ly).to(self.device), torch.arange(lz).to(self.device), indexing="ij"
        )

        # list of indices
        self.xi, self.yi, self.zi = (
            x_grid.permute(2, 1, 0).contiguous().view(-1),
            y_grid.permute(2, 1, 0).contiguous().view(-1),
            z_grid.permute(2, 1, 0).contiguous().view(-1),
        )

    def sample_indices(self, n):
        indices = torch.multinomial(self.pdf, num_samples=n,replacement=True)
        i = self.xi[indices]
        j = self.yi[indices]
        k = self.zi[indices]
        return i, j, k


class ConditionsDataset:
    def __init__(self, activity, cgan_src, source_fn,save_cond=False):
        self.total_activity = int(float(activity))
        source = itk.imread(source_fn)

        source_array=itk.array_from_image(source)
        self.source_size = np.array(source_array.shape)
        self.source_spacing = np.array(source.GetSpacing())
        self.source_origin = np.array(source.GetOrigin())
        self.offset = (self.source_size-1)*self.source_spacing/2

        self.save_cond = save_cond

        if save_cond:
            self.condition_img = np.zeros(self.source_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.sampler = VoxelizerSourcePDFSamplerTorch(source)
        self.xmeanc = cgan_src.xmeanc.to(self.device)
        self.xstdc = cgan_src.xstdc.to(self.device)
        self.z_dim = cgan_src.z_dim
        self.z_rand = cgan_src.z_rand
        self.offset = torch.from_numpy(self.offset).to(self.device).flip(0)
        print(self.offset)

        self.max_radius = 0


    def save_conditions(self, fn):
        condition_img_itk = itk.image_from_array(self.condition_img)
        condition_img_itk.SetSpacing(self.source_spacing)
        condition_img_itk.SetOrigin(self.source_origin)
        itk.imwrite(condition_img_itk, fn)

    def generate_isotropic_directions_torch(self,n):
        min_theta = torch.tensor([0], device=self.device)
        max_theta = torch.tensor([torch.pi], device=self.device)
        min_phi = torch.tensor([0], device=self.device)
        max_phi = 2 * torch.tensor([torch.pi], device=self.device)

        u = torch.rand(n, device=self.device)
        costheta = torch.cos(min_theta) - u * (torch.cos(min_theta) - torch.cos(max_theta))
        sintheta = torch.sqrt(1 - costheta ** 2)

        v = torch.rand(n, device=self.device)
        phi = min_phi + (max_phi - min_phi) * v
        sinphi = torch.sin(phi)
        cosphi = torch.cos(phi)

        px = -sintheta * cosphi
        py = -sintheta * sinphi
        pz = -costheta

        return torch.column_stack((px,py,pz))

    def get_batch(self, n):
        i,j,k = self.sampler.sample_indices(n=n)

        if self.save_cond:
            for ii,jj,kk in zip(i,j,k):
                id_i,id_j,id_k=ii.cpu().numpy(),jj.cpu().numpy(),kk.cpu().numpy()
                self.condition_img[id_i,id_j,id_k]+=1

        # half pixel size
        hs = self.source_spacing / 2.0
        # sample within the voxel
        rx = torch.rand(n,device=self.device)*2*hs[0] - hs[0]
        ry = torch.rand(n,device=self.device)*2*hs[1] - hs[1]
        rz = torch.rand(n,device=self.device)*2*hs[2] - hs[2]
        # warning order np is z,y,x while itk is x,y,z
        x = self.source_spacing[0] * i + rz
        y = self.source_spacing[1] * j + ry
        z = self.source_spacing[2] * k + rx

        p =  torch.column_stack((z,y,x)) - self.offset
        dir = self.generate_isotropic_directions_torch(n)
        condx = torch.column_stack((p,dir))

        condx = (condx - self.xmeanc) / self.xstdc
        z_rand = self.z_rand((n,self.z_dim),device=self.device)
        gan_input_z_cond = torch.cat((z_rand, condx), dim=1).float()

        r = torch.sqrt(p[0]**2 + p[2] **2)
        if r.max()>self.max_radius:
            self.max_radius = r.max()
            print(r.mean())

        return gan_input_z_cond
