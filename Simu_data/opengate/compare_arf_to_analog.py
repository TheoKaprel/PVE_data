#!/usr/bin/env python3

import argparse

import numpy as np
import uproot
import torch
from garf.garf_detector import load_nn,normalize_logproba,normalize_proba_with_russian_roulette
import matplotlib.pyplot as plt

def init_model(pth_fn,device):
    nn, model = load_nn(pth_fn, verbose=True, gpu_mode="auto")
    model = model.to(device)
    model.eval()
    model_data = nn["model_data"]
    md = model_data
    x_mean= torch.tensor(md["x_mean"], device=device)
    x_std = torch.tensor(md["x_std"], device=device)

    if "rr" in model_data:
        rr = model_data["rr"]
    if "RR" in model_data:
        rr = model_data["RR"]

    return model, x_mean,x_std,rr

def main():
    print(args)

    h = uproot.open(args.rootfile)
    a = h['ARF (training);1']
    dataset_array = a.arrays(library='numpy')
    E = dataset_array['E']
    theta = dataset_array['Theta']
    phi = dataset_array['Phi']
    true_window = dataset_array["window"]




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth1")
    parser.add_argument("--pth2")
    parser.add_argument("--rootfile")
    args = parser.parse_args()

    main()
