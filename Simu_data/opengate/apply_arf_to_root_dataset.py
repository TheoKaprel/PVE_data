#!/usr/bin/env python3

import argparse

import numpy as np
import uproot
import torch
from garf.garf_detector import load_nn,normalize_logproba,normalize_proba_with_russian_roulette
import matplotlib.pyplot as plt

def nn_predict_torch(model, x, x_mean, x_std,rr):
    x = (x - x_mean) / x_std
    vx = x.float()

    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        y_pred = model(vx)

    y_pred = normalize_logproba(y_pred)
    # y_pred = normalize_proba_with_russian_roulette(y_pred, 0, rr)

    return y_pred

def main():
    print(args)

    fig,ax = plt.subplots(1,3)
    h = uproot.open(args.rootfile)
    a = h['ARF (training);1']
    dataset_array = a.arrays(library='numpy')
    E = dataset_array['E']
    theta = dataset_array['Theta']
    phi = dataset_array['Phi']
    true_window = dataset_array["window"]
    # n_ew = len(np.unique(true_window))
    n_ew = 2
    # dataset_input = torch.from_numpy(np.concatenate((theta[:,None], phi[:,None],E[:,None]),axis=1))

    # mask = (np.abs(theta-90)<10)*(np.abs(phi-90)<10)*(np.abs(E-0.208)<0.05)
    mask = (np.abs(theta-90)<10)*(np.abs(phi-90)<10)
    E,theta,phi,true_window = E[mask],theta[mask],phi[mask],true_window[mask]
    true_window[true_window!=5] = 0
    true_window[true_window==5] = 1

    Ntot = E.shape[0]
    for e in range(n_ew):
        print(f"Nb in EW_{e} : {(true_window==e).sum()} ({round((true_window==e).sum()/Ntot*100,3)}%)")

    dataset_input = torch.from_numpy(np.concatenate((phi[:,None], theta[:,None],E[:,None]),axis=1))
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


    nn, model = load_nn(args.pth, verbose=True, gpu_mode="auto")
    fig_tr,ax_tr = plt.subplots()
    epochs = [d['epoch'] for d in nn["optim"]['data']]
    training_loss = [d['train_loss'] for d in nn["optim"]['data']]
    ax_tr.plot(epochs,training_loss, '-o')
    ax_tr.set_xlabel("Epochs")
    ax_tr.set_ylabel("<Training loss>")

    model = model.to(device)
    model.eval()
    model_data = nn["model_data"]
    md = model_data
    x_mean = torch.tensor(md["x_mean"], device=device)
    x_std = torch.tensor(md["x_std"], device=device)
    # Russian roulette
    if "rr" in model_data:
        rr = model_data["rr"]
    if "RR" in model_data:
        rr = model_data["RR"]
    batch_size = int(1e5)
    predicted_window = None
    proba_window = None
    n_data = dataset_input.shape[0]
    current = 0
    while current < n_data:
        batch = dataset_input[current:min(n_data,current+batch_size)].to(device)
        with torch.no_grad():
            out = nn_predict_torch(model=model, x = batch,
                               x_mean=x_mean,x_std=x_std,rr=rr)


        predicted_window_batch = torch.multinomial(out,1)
        if predicted_window is not None:
            predicted_window = np.concatenate((predicted_window,predicted_window_batch.detach().cpu().numpy()[:,0]),axis=0)
            proba_window = np.concatenate((proba_window,out.detach().cpu().numpy()),axis=0)
        else:
            predicted_window = predicted_window_batch.detach().cpu().numpy()[:,0]
            proba_window = out.detach().cpu().numpy()
        current+=batch_size


    h1, xedges, yedges, img = ax[0].hist2d(theta[true_window == 1], phi[true_window == 1], bins=100, cmap='viridis', density=False)
    h2, _, _, img = ax[1].hist2d(theta[predicted_window==1], phi[predicted_window==1], bins=[xedges,yedges],cmap='viridis',density=False)
    # ax[0].imshow(h1)
    # ax[1].imshow(h2)
    cbar = fig.colorbar(img, ax=ax[1])

    ax[2].imshow(h1-h2)
    confusion_matrix = np.zeros((n_ew,n_ew))
    for i in range(n_ew):
        for j in range(n_ew):
            confusion_matrix[i,j] = ((true_window==i)*(predicted_window==j)).sum()
    # confusion_matrix[0,0] = 0
    fig_conf,ax_conf = plt.subplots()
    ax_conf.imshow(confusion_matrix)
    print(f'{confusion_matrix=}')



    fig_ds,ax_ds = plt.subplots(2,4)
    ax_ds = ax_ds.ravel()
    for k in range(n_ew):
        ax_ds[k].hist(E[true_window==k]*1000,bins=100,color="blue",alpha=0.7)
        ax_ds[k].hist(E[predicted_window==k]*1000,bins=100,color="red",alpha=0.7)
        ax_ds[k].set_title(f"Window {k}")
        ax_ds[k].set_xlabel("Energy (keV)")

    fig_ds,ax_ds = plt.subplots(2,4)
    ax_ds = ax_ds.ravel()
    print(f"{out.shape=}")
    for k in range(n_ew):
        ax_ds[k].hist(E*1000,bins=100,color="blue",alpha=0.7,weights=(true_window==k))
        ax_ds[k].hist(E*1000,bins=100,color="red",alpha=0.7,weights = proba_window[:,k])
        ax_ds[k].set_title(f"WEIGHTED Window {k}")
        ax_ds[k].set_xlabel("Energy (keV)")




    for class_i in range(n_ew):
        tp = ((true_window==class_i)*(predicted_window==class_i)).sum()
        fp = ((true_window!=class_i)*(predicted_window==class_i)).sum()
        fn = ((true_window==class_i)*(predicted_window!=class_i)).sum()
        precision = (tp)/(tp+fp)
        recall = (tp)/(tp+fn)
        F1 = 2 * (precision*recall)/(precision+recall)
        print("-------------------------")
        print(f"For class {class_i}: ")
        print(f"{precision=}")
        print(f"{recall=}")
        print(f"{F1=}")


    N = 1000
    t_theta = torch.rand(N,1)*(100-80)+80
    t_phi = torch.rand(N,1)*(100-80)+80
    t_E = torch.ones(N,1)*0.208

    t_theta,t_phi,t_E = t_theta.to(device),t_phi.to(device),t_E.to(device)
    batch = torch.cat((t_theta,t_phi,t_E),dim=1)
    print(batch.shape)
    out = nn_predict_torch(model=model, x=batch,
                           x_mean=x_mean, x_std=x_std, rr=rr)

    fig,ax = plt.subplots()
    ax.scatter(((t_theta-90)**2+(t_phi-90)**2).sqrt().squeeze().detach().cpu().numpy(), out[:,1].detach().cpu().numpy(),
               s=4,color='black')
    ax.set_xlabel('L2 Distance to (90°,90°)')
    ax.set_ylabel('Proba to be detected in PW')

    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--rootfile")
    parser.add_argument("--pth")
    parser.add_argument("--outputnpy")
    args = parser.parse_args()

    main()
