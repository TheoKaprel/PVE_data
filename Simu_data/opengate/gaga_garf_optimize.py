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
from itk import RTK as rtk
import numpy as np


sys.setrecursionlimit(10000)

def main():
    print(args)
    # torch.autograd.set_detect_anomaly(True)
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
    if args.torchviz:
        simu.gantry_angles = [180 * deg, (180 + 90) * deg]
        simu.radius = [280*mm, 280*mm]
    else:
        if args.geom is not None:
            xmlReader = rtk.ThreeDCircularProjectionGeometryXMLFileReader.New()
            xmlReader.SetFilename(args.geom)
            xmlReader.GenerateOutputInformation()
            geometry = xmlReader.GetOutputObject()
            list_sid = list(geometry.GetSourceToIsocenterDistances())
            list_angles_rad = list(geometry.GetGantryAngles())
            simu.gantry_angles = [angle_rad * 360 / (2 * np.pi) * deg for angle_rad in list_angles_rad]
            simu.radius = [sid * mm for sid in list_sid]
        else:
            simu.gantry_angles = [(3 * k + 180) * deg for k in range(args.nprojs)]
            simu.radius = [args.sid * mm for _ in range(args.nprojs)]


    simu.axis = args.axis
    simu.duration = 15 * sec
    simu.number_of_threads = 1
    simu.total_activity = args.activity * Bq

    simu.image_size = [128, 128]
    simu.image_spacing = [4.7952 * mm , 4.7952 * mm]

    simu.gaga_source.pth_filename = args.gan_pth
    simu.garf_detector.pth_filename = args.garf_pth
    simu.garf_detector.hit_slice_flag = False


    simu.gaga_source.batch_size = int(args.batchsize)  # 5e5 best on nvidia linux
    simu.gaga_source.backward_distance = 330 * mm # ????
    simu.gaga_source.energy_threshold_MeV = 0.15
    simu.compile = args.compile
    simu.gaga_source.gpu_mode = args.device
    simu.garf_detector.gpu_mode = args.device

    simu.optim_initialize()

    measured_projections = itk.imread(args.projections)
    measured_projections_torch = torch.from_numpy(
        itk.array_from_image(measured_projections)).to(simu.gaga_source.current_gpu_device).to(torch.float64)


    like_img = itk.imread(args.like_img)
    like_img_array = itk.array_from_image(like_img)
    like_img_tensor = torch.from_numpy(like_img_array)
    # image_k_tensor = (20*torch.rand(like_img_tensor.shape,
    #                                dtype=torch.float32,
    #                                device = simu.gaga_source.current_gpu_device)).requires_grad_()
    image_k_tensor = (torch.ones(like_img_tensor.shape,
                                   dtype=torch.float32,
                                   device = simu.gaga_source.current_gpu_device)).requires_grad_()

    # optimizer = torch.optim.Adam([image_k_tensor,], lr=args.lr)

    if args.loss == "mse":
        loss_fct = torch.nn.MSELoss()
    elif args.loss=="poisson":
        loss_fct = torch.nn.PoissonNLLLoss(log_input=False, reduction="mean")


    if args.torchviz==True:
        from torchviz import make_dot

        src = torch.from_numpy(like_img_array).to(torch.float32).to(simu.gaga_source.current_gpu_device)
        src.requires_grad = True
        simu.garf_detector.detector_planes_subset = simu.garf_detector.detector_planes


        output_projs = simu.optim_generate_projections_from_source(source_tensor=src)
        output_projs = output_projs / output_projs.sum() * measured_projections_torch[:2, :, :].sum()
        loss = loss_fct(output_projs, measured_projections_torch[:2, :, :])
        make_dot(loss,show_attrs=True, show_saved=True).render(format="png", filename="torchviz")
        exit(0)

    elif args.fp==True:
        src = torch.from_numpy(like_img_array).to(torch.float32).to(simu.gaga_source.current_gpu_device)
        # simu.garf_detector.detector_planes_subset = simu.garf_detector.detector_planes
        subset_ids = [8 * k for k in range(120//8)]
        simu.garf_detector.detector_planes_subset = [simu.garf_detector.detector_planes[k] for k in subset_ids]

        with torch.no_grad():
            output_projs = simu.optim_generate_projections_from_source(source_tensor=src)
            n_event_per_voxels = simu.gaga_source.final_N_generated / (
                    image_k_tensor.shape[0] * image_k_tensor.shape[1] * image_k_tensor.shape[2])

            output_projs = output_projs / n_event_per_voxels
        output_projs_itk = itk.image_from_array(output_projs.detach().cpu().numpy())
        output_projs_itk.CopyInformation(measured_projections)
        itk.imwrite(output_projs_itk, os.path.join(args.output_folder, "output_projs_gaga_garf.mha"))
        exit(0)

    else:
        n_epochs = args.nepochs
        nprojs_per_subsets = 120//args.nsubsets


        losses_np = []
        np.save(os.path.join(args.output_folder,"losses.npy"), losses_np)

        for epoch in range(n_epochs):
            for subset in range(args.nsubsets):
                image_k_tensor.requires_grad_(True)
                if image_k_tensor.grad is not None:
                    image_k_tensor.grad.zero_()

                subset_ids = [subset+8*k for k in range(nprojs_per_subsets)]
                print(f"{subset_ids=}")

                # optimizer.zero_grad()

                t0_epoch = time.time()
                simu.garf_detector.detector_planes_subset = [simu.garf_detector.detector_planes[k] for k in subset_ids]
                output_projs = simu.optim_generate_projections_from_source(source_tensor = image_k_tensor)
                n_event_per_voxels = simu.gaga_source.final_N_generated / (image_k_tensor.shape[0]*image_k_tensor.shape[1]*image_k_tensor.shape[2])
                print(f"{n_event_per_voxels=}")
                output_projs = output_projs/n_event_per_voxels

                # normalization
                # output_projs = (output_projs/output_projs.sum()*measured_projections_torch[subset_ids,:,:]).sum()


                # loss = loss_fct(output_projs, measured_projections_torch[subset_ids,:,:])
                loss = (output_projs - measured_projections_torch[subset_ids,:,:] * torch.log(output_projs+1e-8)).sum()

                print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MiB i.e. {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GiB")
                loss.backward()

                # image_k_tensor = image_k_tensor - image_k_tensor*image_k_tensor.grad
                # mask_sensitivity = simu.garf_detector.sensitivity_image>0
                # image_k_tensor.grad[mask_sensitivity] = image_k_tensor.grad[mask_sensitivity] * image_k_tensor[mask_sensitivity]/simu.garf_detector.sensitivity_image[mask_sensitivity]
                CF = (4.7952**3)*0.001 * 1e6 * 15 * 0.11

                simu.garf_detector.sensitivity_image = simu.garf_detector.sensitivity_image / n_event_per_voxels * CF
                simu.garf_detector.sensitivity_image[simu.garf_detector.sensitivity_image < 1e-8] = 1


                with torch.no_grad():
                    update = image_k_tensor * image_k_tensor.grad / simu.garf_detector.sensitivity_image
                    image_k_tensor = image_k_tensor - update

                # optimizer.step()
                # print("Image gradient abs mean: ", image_k_tensor.grad.abs().mean())
                # print("Image gradient abs max: ", image_k_tensor.grad.abs().max())

                losses_np.append(loss.item())
                np.save(os.path.join(args.output_folder, "losses.npy"), losses_np)

                print(f"[Epoch {epoch+1}/{n_epochs}] [Subset {subset+1}/{args.nsubsets}] Loss = {loss.item():8.4f}            ({time.time()-t0_epoch:.4f} s)")

                proj_k = itk.image_from_array(output_projs.float().detach().cpu().numpy())
                proj_k.CopyInformation(measured_projections)
                itk.imwrite(proj_k, os.path.join(args.output_folder, f"proj_{epoch + 1}_{subset+1}.mha"))

                rec_k = itk.image_from_array(image_k_tensor.float().detach().cpu().numpy())
                rec_k.CopyInformation(like_img)
                itk.imwrite(rec_k, os.path.join(args.output_folder, f"rec_{epoch + 1}_{subset+1}.mha"))

                sens_k = itk.image_from_array(simu.garf_detector.sensitivity_image.float().detach().cpu().numpy())
                sens_k.CopyInformation(like_img)
                itk.imwrite(sens_k, os.path.join(args.output_folder, f"sens_{epoch + 1}_{subset+1}.mha"))


            rec_k = itk.image_from_array(image_k_tensor.float().detach().cpu().numpy())
            rec_k.CopyInformation(like_img)
            itk.imwrite(rec_k, os.path.join(args.output_folder, f"rec_{epoch+1}.mha"))

            output_projs_itk = itk.image_from_array(output_projs.float().detach().cpu().numpy())
            output_projs_itk.CopyInformation(measured_projections)
            itk.imwrite(output_projs_itk, os.path.join(args.output_folder, f"projs_{epoch+1}.mha"))

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
    parser.add_argument("--nprojs", type=int, default = 120)
    parser.add_argument("--geom", type=str)
    parser.add_argument("--nsubsets", type=int, default = 8)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--axis", type=str)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--nepochs", type=int, default = 10)
    parser.add_argument("--lr", type=float, default = 0.001)
    parser.add_argument("--loss", type=str, default = "mse", choices=["mse", "poisson"])
    parser.add_argument("--torchviz", action="store_true")
    parser.add_argument("--fp", action="store_true")
    args = parser.parse_args()

    main()
