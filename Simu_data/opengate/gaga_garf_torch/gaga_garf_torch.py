#!/usr/bin/env python3

import argparse
import pathlib
import os
import numpy as np
import matplotlib.pyplot as plt
from box import Box
import torch
import time
import itk
import scipy
from tqdm import tqdm
from gaga_garf.cgan_source import CGANSOURCE,ConditionsDataset
from gaga_garf.garf_detector import GARF,DetectorPlane


def update_ideal_recons(batch,recons,offset,spacing,size,e_min=0.001):
    batch=batch.cpu().numpy()
    c = scipy.constants.speed_of_light * 1000  # in mm

    # loop on x ; check energy
    positions = batch[:,1:4]
    directions = batch[:,4:7]

    times = batch[:,7] / 1e9
    energies = batch[:,0]

    # filter according to E ?
    # mask = energies > e_min
    mask = (energies>0.140) & (energies<0.141)
    positions = positions[mask]
    directions = directions[mask]
    times = times[mask]

    # output
    emissions = np.zeros_like(positions)

    for pos, dir, t, E, p in zip(positions, directions, times, energies, emissions):
        l = t * c
        p += pos + l * -dir

    pix = np.rint((emissions - offset) / spacing).astype(int)

    size=[size[2],size[1],size[0]]
    for i in [0, 1, 2]:
        pix = pix[(pix[:, i] < size[i]) & (pix[:, i] > 0)]

    for x in pix:
        recons[x[2], x[1], x[0]] += 1

    return recons

    # garf_ui['pth_filename'] = os.path.join(paths.current, "pths/arf_5x10_9.pth")

def main():
    t0 = time.time()
    print(args)
    paths = Box()
    paths.current = pathlib.Path(__file__).parent.resolve()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    gan_info = {}
    gan_info['pth_filename'] = args.pthgaga
    gan_info['batchsize'] = args.batchsize
    gan_info['device'] = device
    print(device)

    cgan_source = CGANSOURCE(gan_info)

    print('PARAMS KEYS : ')
    print(cgan_source.gan_info.params)
    print()

    print("CONDITIONS : ")
    print(cgan_source.gan_info.params['cond_keys'])

    print("KEYS : ")
    keys_list = cgan_source.gan_info.params['keys_list']
    print(keys_list)

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

    dataset = ConditionsDataset(activity=args.activity,cgan_src=cgan_source,source_fn=args.source,save_cond=args.save)
    batch_size = int(float(args.batchsize))
    n_batchs = int(float(args.activity)) // batch_size
    N_primaries = int(float(args.activity))

    t_condition_generation = 0
    t_gan_generation = 0
    t_intersection = 0
    t_backprojction =0
    t_apply = 0
    t_selection = 0
    t_save = 0

    if args.save:
        src=itk.imread(args.source)
        src_array=itk.array_from_image(src)
        recons_generated=np.zeros_like(src_array)
        size = np.array(recons_generated.shape)
        spacing = np.array(src.GetSpacing())
        offset = -size * spacing / 2.0 + spacing / 2.0


    N,M = 0,0



    pbar=tqdm(total=N_primaries)
    with torch.no_grad():
        for _ in range(n_batchs):
            t_condition_generation_0 = time.time()
            gan_input_z_cond = dataset.get_batch(batch_size)
            N+=batch_size
            t_condition_generation+=(time.time() - t_condition_generation_0)

            t_gan_generation_0 = time.time()
            gan_input_z_cond = gan_input_z_cond.to(device)
            fake = cgan_source.generate(gan_input_z_cond)
            t_gan_generation+=(time.time() - t_gan_generation_0)


            t_selection_0 = time.time()
            fake=fake[fake[:, 0] > 0.100]
            dirn = torch.sqrt(fake[:, 4] ** 2 + fake[:, 5] ** 2 + fake[:, 6] ** 2)
            fake[:,4:7]=fake[:,4:7]/dirn[:,None]

            t_selection += (time.time() - t_selection_0)

            if args.save:
                recons_generated = update_ideal_recons(batch=fake,
                                                       recons=recons_generated,
                                                       offset=offset[::-1],spacing=spacing,size=size,
                                                       e_min=0.140)

            # backproject a little bit: p2= p1 - alpha * d1
            # solved (avec inconnu=alpha) using ||p2||² = R2² puis equation degré 2 en alpha
            t_backprojction_0 = time.time()
            # beta=(fake[:,1:4] * fake[:,4:7]).sum(dim=1)
            # R1,R2 = 610,args.sid - 150
            # alpha= beta - torch.sqrt(beta**2 + R2**2-R1**2)
            # fake[:,1:4] = fake[:,1:4] - alpha[:,None]*fake[:,4:7]
            fake[:,1:4] = fake[:,1:4] - 530 * fake[:,4:7]
            t_backprojction+=(time.time() - t_backprojction_0)

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

                # p=dataset.generate_condition(1000)
                # ax_scatter.scatter(p[:,0], p[:,1], p[:,2], s=2, c = 'green')


                for k,det in enumerate(l_detectorsPlanes):
                    cent=det.center.cpu().numpy()
                    ax_scatter.scatter(cent[0], cent[1], cent[2], s=int(k/10)+1, c='red')

                ax_scatter.set_xlabel('x')
                ax_scatter.set_ylabel('y')
                ax_scatter.set_zlabel('z')
                # plt.show()



            l_nc = []

            for proj_i, plane_i in enumerate(l_detectorsPlanes):
                t_intersection_0 = time.time()
                batch_arf_i = plane_i.get_intersection(batch=fake)
                t_intersection += (time.time() - t_intersection_0)

                if (args.debug and proj_i==0):
                    print(fake.shape[0], batch_arf_i.shape[0])

                    sc_xy=batch_arf_i.cpu().numpy()
                    sc_z = args.sid*np.ones_like(sc_xy[:,0])
                    ax_scatter.scatter(sc_xy[:,0], sc_xy[:,1],sc_z ,s=5, c='black')

                if args.debug:
                    l_nc.append(batch_arf_i.shape[0])

                t_apply_0 = time.time()
                garf_detector.apply(batch_arf_i,proj_i)
                t_apply += (time.time() - t_apply_0)

            if args.debug:
                fig,ax = plt.subplots()
                ax.plot(l_nc)
                plt.show()

            pbar.update(batch_size)


    if args.save:
        dataset.save_conditions(fn = os.path.join(args.folder, f"conditions.mhd"))
        recons_generated_itk=itk.image_from_array(recons_generated)
        recons_generated_itk.CopyInformation(src)
        itk.imwrite(recons_generated_itk, os.path.join(args.folder, f"fake.mhd"))

    t_save_0 = time.time()
    garf_detector.save_projection()
    t_save+=(time.time() -t_save_0)


    print(f"TOTAL TIME : {time.time() - t0}")
    print(f"Cond GENERATION TIME: {t_condition_generation}")
    print(f"GENERATION TIME: {t_gan_generation}")
    print(f"SELECTION TIME : {t_selection}")
    print(f"BACKPROJECTION TIME: {t_backprojction}")
    print(f"INTERSECTION TIME : {t_intersection}")
    print(f"APPLY TIME : {t_apply}")
    print(f"SAVING TIME : {t_save}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-a","--activity", type = float)
    parser.add_argument("-s", "--source", type=str)
    parser.add_argument("--pthgaga", type=str)
    parser.add_argument("--pthgarf", type=str)
    parser.add_argument("-b","--batchsize", type = float, default = 100000)
    parser.add_argument("-f", "--folder", type = str)
    parser.add_argument("-o", "--output", type = str)
    parser.add_argument("-n","--nprojs", type = int)
    parser.add_argument("--sid", type=float, help = "source-to-isocenter distance ")
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--save', action="store_true", help="will save conditions, and generated positions (for debug)")

    args = parser.parse_args()

    main()
