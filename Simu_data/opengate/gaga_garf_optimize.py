#!/usr/bin/env python3
import argparse
import opengate as gate
import torch
from gaga_phsp.spect_intevo_helpers import *
from pathlib import Path
import itk
import time
import os
import sys


sys.setrecursionlimit(10000)
# torch.autograd.set_detect_anomaly(True)

def main():
    print(args)
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
    if args.viz:
        # simu.gantry_angles = [180 * deg, (180 + 90) * deg]
        simu.gantry_angles = [(3 * k + 180) * deg for k in range(120)]
    else:
        simu.gantry_angles = [(3 * k + 180) * deg for k in range(120)]

    simu.axis = args.axis
    simu.duration = 15 * sec
    simu.number_of_threads = 1
    simu.total_activity = args.activity * Bq

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
    simu.gaga_source.gpu_mode = args.device
    simu.garf_detector.gpu_mode = args.device

    simu.optim_initialize()

    measured_projections = itk.imread(args.projections)
    measured_projections_torch = torch.from_numpy(
        itk.array_from_image(measured_projections)).to(simu.gaga_source.current_gpu_device)


    like_img = itk.imread(args.like_img)
    like_img_array = itk.array_from_image(like_img)
    # image_k_tensor = torch.ones_like(torch.from_numpy(like_img_array)).to(torch.float32).to(simu.gaga_source.current_gpu_device)
    image_k_tensor = 1+0.1*torch.rand_like(torch.from_numpy(like_img_array)).to(torch.float32).to(simu.gaga_source.current_gpu_device)
    image_k_tensor.requires_grad_(True)
    optimizer = torch.optim.Adam([image_k_tensor,], lr=args.lr)
    loss_fct = torch.nn.MSELoss()

    if args.viz==True:
        # from torchviz import make_dot

        src = torch.from_numpy(like_img_array).to(torch.float32).to(simu.gaga_source.current_gpu_device)
        # src.requires_grad = True
        with torch.no_grad():
            output_projs = simu.optim_generate_projections_from_source(source_tensor=src)
        itk.imwrite(itk.image_from_array(output_projs[:,4,:,:].detach().cpu().numpy()), os.path.join(args.output_folder, "output_projs_gaga_garf.mha"))
        # loss = loss_fct(output_projs[:2, 4, :, :], measured_projections_torch[:2,:,:])

        # make_dot(loss,show_attrs=True, show_saved=True).render(format="png", filename="torchviz")
        exit(0)

    else:
        n_epochs = args.nepochs
        for epoch in range(n_epochs):
            t0_epoch = time.time()
            optimizer.zero_grad()
            output_projs = simu.optim_generate_projections_from_source(source_tensor = image_k_tensor)

            print(f"{image_k_tensor.sum().item()=}   /   {output_projs[:, 4, :, :].sum().item()=}")

            if ddp:
                torch.distributed.all_reduce(output_projs, op=torch.distributed.ReduceOp.SUM)

            # normalization
            output_projs = output_projs[:,4,:,:]/output_projs[:,4,:,:].max() * measured_projections_torch.max()
            loss = loss_fct(output_projs, measured_projections_torch)

            print(f"({rank=}) Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MiB i.e. {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GiB")
            loss.backward()

            print("Image gradient abs mean: ", image_k_tensor.grad.abs().mean())
            print("Image gradient abs max: ", image_k_tensor.grad.abs().max())

            optimizer.step()

            print(f"[Epoch {epoch}/{n_epochs}] Loss = {loss.item():8.4f}            ({time.time()-t0_epoch:.4f} s)")
            rec_k = itk.image_from_array(image_k_tensor.detach().cpu().numpy())
            rec_k.CopyInformation(like_img)
            itk.imwrite(rec_k, os.path.join(args.output_folder, f"rec_{epoch}.mha"))

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
    parser.add_argument("--device", type=str, default = "auto")
    parser.add_argument("--sid", type=float, default = 280)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--axis", type=str)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--nepochs", type=int, default = 10)
    parser.add_argument("--lr", type=float, default = 0.001)
    parser.add_argument("--viz", action="store_true")
    args = parser.parse_args()

    host = os.uname()[1]
    if (host !='suillus'):
        import torch.distributed as dist
        import idr_torch

        # get distributed configuration from Slurm environment
        NODE_ID = os.environ['SLURM_NODEID']
        MASTER_ADDR = os.environ['MASTER_ADDR'] if ("MASTER_ADDR" in os.environ) else os.environ['HOSTNAME']

        # display info
        if idr_torch.rank == 0:
            print(">>> Training on ", len(idr_torch.nodelist), " nodes and ", idr_torch.world_size,
                  " processes, master node is ", MASTER_ADDR)
        print("- Process {} corresponds to GPU {} of node {}".format(idr_torch.rank, idr_torch.local_rank, NODE_ID))

        dist.init_process_group(backend='nccl',
                                init_method='env://',
                                world_size=idr_torch.world_size,
                                rank=idr_torch.rank)
        rank=idr_torch.rank

        torch.cuda.set_device(idr_torch.local_rank)
        if idr_torch.size>1:
            ddp = True
        else:
            ddp = False
    else:
        rank=0
        ddp = False

    main()
