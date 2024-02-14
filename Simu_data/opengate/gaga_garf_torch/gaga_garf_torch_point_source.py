#!/usr/bin/env python3

import argparse
import pathlib
import os
import numpy as np
import matplotlib.pyplot as plt
from box import Box
import torch
import time
from tqdm import tqdm
from gaga_garf.garf_detector import GARF,DetectorPlane


def sample_pos_R(max_radius,n):
    phi = torch.rand(n)*2*torch.pi
    costheta = 2*torch.rand(n)  - 1
    u = torch.rand(n)

    theta = torch.arccos(costheta)
    r = max_radius * (u)**(1/3)
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)
    return torch.column_stack((x,y,z))

def generate_point_source_torch(n,device, E0 = 0.1405):
    min_theta = torch.tensor([0], device=device)
    max_theta = torch.tensor([torch.pi], device=device)
    min_phi = torch.tensor([0], device=device)
    max_phi = 2 * torch.tensor([torch.pi], device=device)

    u = torch.rand(n, device=device)
    costheta = torch.cos(min_theta) - u * (torch.cos(min_theta) - torch.cos(max_theta))
    sintheta = torch.sqrt(1 - costheta ** 2)

    v = torch.rand(n, device=device)
    phi = min_phi + (max_phi - min_phi) * v
    sinphi = torch.sin(phi)
    cosphi = torch.cos(phi)

    dx = -sintheta * cosphi
    dy = -sintheta * sinphi
    dz = -costheta

    directions = torch.column_stack((dx,dy,dz))

    positions = sample_pos_R(max_radius=1,n=n).to(device)
    energies = E0 * torch.ones_like(dx)
    times = torch.zeros_like(dx)

    return torch.column_stack((energies,positions,directions,times))


def main():
    t0 = time.time()
    print(args)
    paths = Box()
    paths.current = pathlib.Path(__file__).parent.resolve()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    nprojs = args.nprojs
    l_angles = torch.linspace(0, 2*torch.pi, nprojs+1)[:-1]
    dist_to_crystal = 29.104800000000004
    l_detectorsPlanes = []
    for angle in l_angles:
        det_plane = DetectorPlane(size=565.511, device=device, center0=[0,0, -args.sid], rot_angle=angle,dist_to_crystal=dist_to_crystal) #FIXME (center)
        # det_plane = DetectorPlane(size=565.511, device=device, center0=[0,args.sid,0], rot_angle=angle) #FIXME (center)
        l_detectorsPlanes.append(det_plane)

    garf_ui = {}
    garf_ui['pth_filename'] = args.pthgarf
    garf_ui['batchsize'] = args.batchsize
    garf_ui['device'] = device
    garf_ui['output_fn'] = os.path.join(args.folder, f"projs.mhd") if args.output is None else args.output
    garf_ui['nprojs'] = len(l_detectorsPlanes)
    garf_detector = GARF(user_info=garf_ui)

    batch_size = int(float(args.batchsize))
    n_batchs = int(float(args.activity)) // batch_size
    N_primaries = int(float(args.activity))

    N,M = 0,0

    pbar=tqdm(total=N_primaries)
    with torch.no_grad():
        for _ in range(n_batchs):
            N+=batch_size

            fake = generate_point_source_torch(n=batch_size,device=device,E0 = 0.1405)

            if args.debug:
                fig,ax=plt.subplots(3,4)
                axs=ax.ravel()
                fake_np=fake.cpu().numpy()
                for k in range(8):
                    axs[k].hist(fake_np[:,k],bins=100)
                dist=np.sqrt(fake_np[:,1]**2+fake_np[:,2]**2+fake_np[:,3]**2)
                axs[8].hist(dist,bins=100)
                dirn = np.sqrt(fake_np[:,4]**2+fake_np[:,5]**2+fake_np[:,6]**2)
                print(dirn[0])
                axs[9].hist(dirn,bins=1)
                plt.show()

                fig = plt.figure()
                ax_scatter = fig.add_subplot(projection='3d')
                ax_scatter.scatter(fake_np[:,1], fake_np[:,2], fake_np[:,3], s=2)

                for k,det in enumerate(l_detectorsPlanes):
                    cent=det.center.cpu().numpy()
                    ax_scatter.scatter(cent[0], cent[1], cent[2], s=int(k/10)+1, c='red')

                ax_scatter.set_xlabel('x')
                ax_scatter.set_ylabel('y')
                ax_scatter.set_zlabel('z')
                # plt.show()

            l_nc = []
            for proj_i, plane_i in enumerate(l_detectorsPlanes):
                batch_arf_i = plane_i.get_intersection(batch=fake)

                if (args.debug and proj_i==0):
                    print(fake.shape[0], batch_arf_i.shape[0])
                    sc_xy=batch_arf_i.cpu().numpy()
                    sc_z = args.sid*np.ones_like(sc_xy[:,0])
                    ax_scatter.scatter(sc_xy[:,0], sc_xy[:,1],sc_z ,s=5, c='black')

                if args.debug:
                    l_nc.append(batch_arf_i.shape[0])

                garf_detector.apply(batch_arf_i,proj_i)

            if args.debug:
                fig,ax = plt.subplots()
                ax.plot(l_nc)
                plt.show()

            pbar.update(batch_size)

    garf_detector.save_projection()


    print(f"TOTAL TIME : {time.time() - t0}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-a","--activity", type = float)
    parser.add_argument("--pthgarf", type=str)
    parser.add_argument("-b","--batchsize", type = float, default = 100000)
    parser.add_argument("-f", "--folder", type = str)
    parser.add_argument("-o", "--output", type = str)
    parser.add_argument("-n","--nprojs", type = int)
    parser.add_argument("--sid", type=float)
    parser.add_argument('--debug', action="store_true")
    args = parser.parse_args()
    main()
