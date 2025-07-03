#!/usr/bin/env python3

import argparse
import os
import torch
import sys
sys.setrecursionlimit(10000)

os.environ['LD_LIBRARY_PATH']+="/linkhome/rech/gencre01/uyo34ub/.local/lib/python3.11/site-packages/opengate_core.libs:"
os.environ['LD_PRELOAD']+="/linkhome/rech/gencre01/uyo34ub/.local/lib/python3.11/site-packages/opengate_core.libs/libG4processes-d7125d28.so:"

import opengate as gate

def main():
    print(args)

    gpu = torch.device("cuda")

    print("Code seems to run")
    print(f"Local GPU is : {gpu}")



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

    print(f"hello ... ?")
    host = os.uname()[1]
    if (host !='suillus'):
        print(f"hello {host}")
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
