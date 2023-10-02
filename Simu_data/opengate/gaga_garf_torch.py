#!/usr/bin/env python3

import argparse
import pathlib
import os
from box import Box
import torch
import time
from torch.utils.data import DataLoader
from gaga_garf.cgan_source import CGANSOURCE,ConditionsDataset
from gaga_garf.garf_detector import GARF,DetectorPlane


def main():
    t0 = time.time()
    print(args)
    paths = Box()
    paths.current = pathlib.Path(__file__).parent.resolve()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    gan_info = {}
    gan_info['pth_filename'] = os.path.join(paths.current, "pths/test001_GP_0GP_10_50000.pth")
    gan_info['batchsize'] = args.batchsize
    gan_info['device'] = device
    print(device)

    cgan_source = CGANSOURCE(gan_info)

    print('PARAMS KEYS : ')
    # print(cgan_source.gan_info.params.keys())
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

    dataset = ConditionsDataset(activity=args.activity,cgan_src=cgan_source)

    dataloader = DataLoader(dataset, batch_size=int(float(args.batchsize)), shuffle=True,num_workers=8)

    t_intersection = 0
    t_apply = 0
    t_selection = 0
    t_save = 0

    with torch.no_grad():
        for z,cond in dataloader:
            gan_input = torch.cat((z, cond), dim=1).float()
            gan_input = gan_input.to(device)
            fake = cgan_source.generate(gan_input)
            t_selection_0 = time.time()
            fake = fake[fake[:,0]>0.01]
            t_selection += (time.time() - t_selection_0)

            for proj_i, plane_i in enumerate(l_detectorsPlanes):
                t_intersection_0 = time.time()
                batch_arf_i = plane_i.get_intersection(batch=fake)
                t_intersection+=(time.time() - t_intersection_0)
                t_apply_0 = time.time()
                garf_detector.apply(batch_arf_i,proj_i)
                t_apply += (time.time() - t_apply_0)



    garf_detector.save_projection()

    print(f"TOTAL TIME : {time.time() - t0}")
    print(f"SELECTION TIME : {t_selection}")
    print(f"INTERSECTION TIME : {t_intersection}")
    print(f"APPLY TIME : {t_apply}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-a","--activity", type = float, default = 100)
    parser.add_argument("-b","--batchsize", type = float, default = 100000)
    parser.add_argument("-o", "--output", type = str)
    parser.add_argument("-n","--nprojs", type = int, default= 1)
    parser.add_argument('--debug', action="store_true")

    args = parser.parse_args()

    main()
