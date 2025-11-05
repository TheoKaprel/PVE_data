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
    mask = (np.abs(theta-90)<10)*(np.abs(phi-90)<10)
    E,theta,phi,true_window = E[mask],theta[mask],phi[mask],true_window[mask]



    data = np.concatenate((theta[:,None], phi[:,None], E[:,None], true_window[:,None]),axis=1)
    print(f"{data.shape}")
    mask_energy= (E >= 0.206) & (E <= 0.210)
    subset = data[mask_energy]

    distances = np.sqrt((subset[:, 0] - 90) ** 2 + (subset[:, 1] - 90) ** 2)

    nbins = 50  # adjust depending on your dataset
    bins = np.linspace(0, distances.max(), nbins + 1)

    # Step 4: Compute probability of class 5 in each bin
    probabilities = []
    bin_centers = []

    for i in range(nbins):
        bin_mask = (distances >= bins[i]) & (distances < bins[i + 1])
        bin_samples = subset[bin_mask,:]
        total = len(bin_samples)
        if total > 0:
            prob_class5 = np.sum(bin_samples[:, 3] == 5) / total
        else:
            prob_class5 = np.nan  # empty bin
        probabilities.append(prob_class5)
        bin_centers.append(0.5 * (bins[i] + bins[i + 1]))

    probabilities = np.array(probabilities)
    bin_centers = np.array(bin_centers)
    fig,ax = plt.subplots()
    ax.plot(bin_centers, probabilities, marker='o')
    ax.set_xlabel("Angular distance from (90°, 90°) [deg]")
    ax.set_ylabel("P(class = 5 | E ≈ 208 keV)")

    # fig,ax = plt.subplots()
    # ax.hist(l2_dists[(true_window==5)*(np.abs(E-0.208)<0.002)],bins=100)


    device = torch.device("cuda")
    model_1, x_mean_1, x_std_1, rr_1 = init_model(pth_fn=args.pth1, device=device)
    model_2, x_mean_2, x_std_2, rr_2 = init_model(pth_fn=args.pth2, device=device)
    for energy in [0.208]:
        N = 10000
        t_theta = torch.rand(N,1)*(95-85)+85
        t_phi = torch.rand(N,1)*(95-85)+85
        t_E = torch.ones(N,1)*energy

        t_theta,t_phi,t_E = t_theta.to(device),t_phi.to(device),t_E.to(device)
        batch = torch.cat((t_theta,t_phi,t_E),dim=1)


        # model 1

        x1 = (batch - x_mean_1) / x_std_1
        vx1 = x1.float()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            y_pred1 = model_1(vx1)
        y_pred1 = normalize_logproba(y_pred1)
        y_pred1 = normalize_proba_with_russian_roulette(y_pred1, 0, rr_1)

        # model 2

        x2 = (batch - x_mean_2) / x_std_2
        vx2 = x2.float()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            y_pred2 = model_2(vx2)
        y_pred2 = normalize_logproba(y_pred2)


        ax.scatter(((t_theta-90)**2+(t_phi-90)**2).sqrt().squeeze().detach().cpu().numpy(), y_pred1[:,5].detach().cpu().numpy(),
                   s=4,color='black', label="without RR")
        ax.scatter(((t_theta-90)**2+(t_phi-90)**2).sqrt().squeeze().detach().cpu().numpy(), y_pred2[:,1].detach().cpu().numpy(),
                   s=4,color='blue',label="with RR")
        ax.set_xlabel('L2 Distance to (90°,90°)')
        ax.set_ylabel(f'Proba to be detected in PW with Energy={energy*100} keV')
        ax.legend()

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth1")
    parser.add_argument("--pth2")
    parser.add_argument("--rootfile")
    args = parser.parse_args()

    main()
