#!/usr/bin/env python3

import torch
import matplotlib.pyplot as plt


def gemini_differentiable_histogram(
        input_tensor: torch.Tensor,
        min_val: float,
        max_val: float,
        num_bins: int,
        sigma: float = None  # Standard deviation for Gaussian smoothing, if None, uses linear interpolation
) -> torch.Tensor:
    """
    Computes a differentiable histogram for a given input tensor.

    Args:
        input_tensor (torch.Tensor): The input data for which to compute the histogram.
        min_val (float): The minimum value for the histogram range.
        max_val (float): The maximum value for the histogram range.
        num_bins (int): The number of histogram bins.
        sigma (float, optional): Standard deviation for Gaussian kernel smoothing.
                                 If None, a linear interpolation (triangular kernel) is used.

    Returns:
        torch.Tensor: A 1D tensor representing the differentiable histogram.
    """
    if not (max_val > min_val):
        raise ValueError("max_val must be greater than min_val.")
    if num_bins <= 0:
        raise ValueError("num_bins must be positive.")

    bin_edges = torch.linspace(min_val, max_val, num_bins + 1, device=input_tensor.device, dtype=input_tensor.dtype)
    # print("BIN EDGES: {}".format(bin_edges))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # print("BIN CENTERS: {}".format(bin_centers))
    bin_width = (max_val - min_val) / num_bins

    # Flatten input to process all values
    flat_input = input_tensor.view(-1)

    # Calculate contribution for each input value to each bin
    # Expand dims for broadcasting: flat_input (N, 1), bin_centers (1, B)
    diff = flat_input.unsqueeze(1) - bin_centers.unsqueeze(0)  # (N, B)

    if sigma is None:  # Linear interpolation (triangular kernel)
        weights = torch.relu(1 - torch.abs(diff) / bin_width)
        # weights = ((1 - torch.abs(diff) / bin_width) >=0).float()
    else:  # Gaussian kernel
        weights = torch.exp(-(diff ** 2) / (2 * sigma ** 2))
        # Normalize weights for each input value so they sum to 1 across bins
        # This ensures each input value adds exactly 1 (or its own weight if input_tensor had weights)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)

    # Sum contributions into histogram bins
    histogram = torch.sum(weights, dim=0)

    return histogram


def sample_gumbel(shape, device='cpu', eps=1e-10):
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, tau=1.0, hard=False):
    # logits: (K,), N: number of samples
    gumbel_noise = sample_gumbel(logits.shape, device=logits.device)
    y = (logits + gumbel_noise) / tau
    y_soft = torch.nn.functional.softmax(y, dim=-1)

    if hard:
        y_hard = torch.zeros_like(y_soft)
        # y_hard.scatter_(1, y_soft.argmax(dim=-1, keepdim=True), 1.0)
        y_hard = torch.scatter(y_hard, -1, y_soft.argmax(dim=-1, keepdim=True), 1.0)
        # Straight-through estimator
        return (y_hard - y_soft).detach() + y_soft


src =torch.tensor([[0., 6., 1., 9., 8., 9., 8., 6., 5., 8.],
                   [0., 6., 8., 6., 8., 6., 3., 9., 4., 9.]], dtype=torch.float64)
src_est = torch.ones_like(src)
src_est.requires_grad = True
Npart = 1e4
li = torch.arange(0,10).to(torch.float64)

optimizer = torch.optim.Adam([src_est, ], lr=0.001)
loss_fct = torch.nn.MSELoss()

training_loss = []
for e in range(5000):
    optimizer.zero_grad()

    src_est_ = torch.nn.functional.relu(src_est,inplace=False)

    pdf = (src_est_ / src_est_.sum()).to(torch.float64)

    logits = torch.log(pdf + 1e-10)
    # logits = logits.repeat((int(Npart),1))
    logits = logits[None,:,:].repeat((int(Npart),1,1))
    samples = gumbel_softmax_sample(logits=logits,hard=True,tau=max(0.1, 1.0 * (0.999**e)))



    a =  samples @ li
    print(samples[:3,:,:])
    print(a.shape)
    print(a[:3,:])
    exit(0)
    x = gemini_differentiable_histogram(input_tensor=a+0.001,min_val=-0.5,max_val=9.5,num_bins=10,sigma = 0.5)
    # x = gemini_differentiable_histogram(input_tensor=a+0.001,min_val=-0.5,max_val=9.5,num_bins=10)
    x = x / x.sum() * src.sum()

    loss = loss_fct(x,src)
    loss.backward()

    optimizer.step()
    print(f"({e}) loss = {loss.item()} | {list(x.detach().cpu().numpy())}")
    training_loss.append(loss.item())


fig,ax = plt.subplots()
ax.plot(training_loss)
plt.show()