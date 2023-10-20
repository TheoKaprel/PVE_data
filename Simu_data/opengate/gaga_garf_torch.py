#!/usr/bin/env python3

import argparse
import pathlib
import os
import numpy as np
import matplotlib.pyplot as plt
from box import Box
import torch
import time
from gaga_garf.cgan_source import CGANSOURCE,ConditionsDataset
from gaga_garf.garf_detector import GARF,DetectorPlane

def main():
    t0 = time.time()
    print(args)
    paths = Box()
    paths.current = pathlib.Path(__file__).parent.resolve()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    gan_info = {}
    gan_info['pth_filename'] = args.pth
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
    l_detectorsPlanes = []
    for angle in l_angles:
        l_detectorsPlanes.append(
            DetectorPlane(size=565.511, device=device, center0=[0, 0, 380], rot_angle=angle)
        )

    garf_ui = {}
    garf_ui['pth_filename'] = os.path.join(paths.current, "pths/arf_5x10_9.pth")
    garf_ui['batchsize'] = args.batchsize
    garf_ui['device'] = device
    garf_ui['output_fn'] = os.path.join(args.output, f"projs.mhd")
    garf_ui['nprojs'] = len(l_detectorsPlanes)
    garf_detector = GARF(user_info=garf_ui)

    dataset = ConditionsDataset(activity=args.activity,cgan_src=cgan_source,source_fn=args.source,save_cond=args.save_cond)
    batch_size = int(float(args.batchsize))
    n_batchs = int(float(args.activity)) // batch_size

    t_condition_generation = 0
    t_intersection = 0
    t_apply = 0
    t_selection = 0
    t_save = 0

    with torch.no_grad():
        for _ in range(n_batchs):
            t_condition_generation_0 = time.time()
            gan_input_z_cond = dataset.get_batch(batch_size)
            t_condition_generation+=(time.time() - t_condition_generation_0)

            gan_input_z_cond = gan_input_z_cond.to(device)
            fake = cgan_source.generate(gan_input_z_cond)

            if args.debug:
                fig,ax=plt.subplots(2,4)
                axs=ax.ravel()
                fake_np=fake.cpu().numpy()
                for k in range(fake_np.shape[1]-1):
                    axs[k].hist(fake_np[:,k],bins=100)
                dist=np.sqrt(fake_np[:,1]**2+fake_np[:,2]**2+fake_np[:,3]**2)
                axs[-1].hist(dist,bins=100)
                plt.show()

            t_selection_0 = time.time()
            # fake = fake[fake[:,0]>0.01]
            t_selection += (time.time() - t_selection_0)

            for proj_i, plane_i in enumerate(l_detectorsPlanes):
                t_intersection_0 = time.time()
                batch_arf_i = plane_i.get_intersection(batch=fake)
                if (args.debug and proj_i==0):
                    print(fake.shape[0], batch_arf_i.shape[0])
                t_intersection+=(time.time() - t_intersection_0)
                t_apply_0 = time.time()
                garf_detector.apply(batch_arf_i,proj_i)
                t_apply += (time.time() - t_apply_0)

    if args.save_cond:
        dataset.save_conditions(fn = os.path.join(args.output, f"conditions.mhd"))

    t_save_0 = time.time()
    garf_detector.save_projection()
    t_save+=(time.time() -t_save_0)

    print(f"TOTAL TIME : {time.time() - t0}")
    print(f"GENERATION TIME: {t_condition_generation}")
    print(f"SELECTION TIME : {t_selection}")
    print(f"INTERSECTION TIME : {t_intersection}")
    print(f"APPLY TIME : {t_apply}")
    print(f"SAVING TIME : {t_save}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-a","--activity", type = float, default = 100)
    parser.add_argument("-s", "--source", type=str)
    parser.add_argument("--pth", type=str)
    parser.add_argument("-b","--batchsize", type = float, default = 100000)
    parser.add_argument("-o", "--output", type = str)
    parser.add_argument("-n","--nprojs", type = int, default= 1)
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--save_cond', action="store_true")

    args = parser.parse_args()

    main()
