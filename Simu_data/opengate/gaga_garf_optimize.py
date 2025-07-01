#!/usr/bin/env python3

import argparse
import opengate as gate
import torch
from gaga_phsp.spect_intevo_helpers import *
from gaga_phsp.gaga_helpers_tests import get_tests_folder
import opengate.tests.utility as utility
from pathlib import Path
import itk
import time

# from torchviz import make_dot, make_dot_from_trace

import sys
sys.setrecursionlimit(10000)
# torch.autograd.set_detect_anomaly(True)

def main():
    print(args)
    # units
    mm = gate.g4_units.mm
    Bq = gate.g4_units.Bq
    sec = gate.g4_units.second
    deg = gate.g4_units.deg


    # spect options
    simu = SpectIntevoSimulator('standalone_torch', "test004_main5_standalone_torch")
    simu.output_folder = Path(args.output_folder)
    simu.ct_image = args.ct  # (needed to position the source)
    simu.activity_image = args.like_img
    simu.radionuclide = args.radionuclide
    simu.gantry_angles = [(3 * k + 180) * deg for k in range(120)]
    # simu.gantry_angles = [(30 * k + 180) * deg for k in range(5)]
    simu.axis = args.axis
    simu.duration = 15 * sec
    simu.number_of_threads = 1
    simu.total_activity = args.activity * Bq
    # simu.visu = True

    simu.image_size = [128, 128]
    simu.image_spacing = [4.7951998710632 * mm , 4.7951998710632 * mm]

    simu.gaga_source.pth_filename = args.gan_pth
    simu.garf_detector.pth_filename = args.garf_pth
    simu.garf_detector.hit_slice_flag = False
    simu.radius = args.sid * mm

    simu.gaga_source.batch_size = int(args.batchsize)  # 5e5 best on nvidia linux
    simu.gaga_source.backward_distance = 150 * mm # ????
    simu.gaga_source.energy_threshold_MeV = 0.15
    simu.compile = args.compile
    # simu.gaga_source.gpu_mode = "cpu"
    # simu.garf_detector.gpu_mode = "cpu"



    # run the simulation
    simu.optim_initialize()

    measured_projections = itk.imread(args.projections)
    measured_projections_torch = torch.from_numpy(
        itk.array_from_image(measured_projections)).to(simu.gaga_source.current_gpu_device)


    like_img = itk.imread(args.like_img)
    like_img_array = itk.array_from_image(like_img)
    image_k_tensor = torch.ones_like(torch.from_numpy(like_img_array)).to(torch.float32).to(simu.gaga_source.current_gpu_device)
    image_k_tensor.requires_grad_(True)
    optimizer = torch.optim.Adam([image_k_tensor, ], lr=0.001)
    loss_fct = torch.nn.MSELoss()

    # with torch.no_grad():
    # src = torch.from_numpy(like_img_array)
    # src.requires_grad = True
    # output_projs = simu.optim_generate_projections_from_source(source_tensor=image_k_tensor)

    # itk.imwrite(itk.image_from_array(output_projs.detach().cpu().numpy()), "/export/home/tkaprelian/temp/output_projs_gaga_garf.mha")

    # loss = loss_fct(output_projs[:5, 4, :, :], measured_projections_torch[:5,:,:])
    # loss.backward()
    # optimizer.step()

    # make_dot(loss,show_attrs=True, show_saved=True).render(format="png", filename="torchviz")
    # exit(0)


    n_epochs = 10
    for epoch in range(n_epochs):
        t0_epoch = time.time()
        optimizer.zero_grad()
        output_projs = simu.optim_generate_projections_from_source(source_tensor = image_k_tensor)
        loss = loss_fct(output_projs[:, 4, :, :], measured_projections_torch[:, :, :])
        loss.backward()
        optimizer.step()
        print(f"[Epoch {epoch}/{n_epochs}] Loss = {loss.item():8.4f}            ({time.time()-t0_epoch:.4f} s)")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-a","--activity", type = float, default = 2e7)
    parser.add_argument("--like_img", type=str)
    parser.add_argument("--projections", type=str)
    parser.add_argument("--ct", type=str)
    parser.add_argument("--radionuclide", type=str, choices=['Tc99m', 'Lu177'])
    parser.add_argument("--batchsize", type=float)
    parser.add_argument("--gan_pth", type=str)
    parser.add_argument("--garf_pth", type=str)
    parser.add_argument("--sid", type=float, default = 280)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--axis", type=str)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    main()
