import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def sample_gumbel(shape, device='cpu', eps=1e-10):
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, tau=1.0, N=1, hard=False):
    # logits: (K,), N: number of samples
    K = logits.shape[0]
    gumbel_noise = sample_gumbel((N, K), device=logits.device)
    y = (logits.unsqueeze(0) + gumbel_noise) / tau
    y_soft = torch.nn.functional.softmax(y, dim=-1)

    if hard:
        y_hard = torch.zeros_like(y_soft)
        y_hard.scatter_(1, y_soft.argmax(dim=-1, keepdim=True), 1.0)
        # Straight-through estimator
        return (y_hard - y_soft).detach() + y_soft
    else:
        return y_soft

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# True target categorical distribution
true_probs = torch.tensor([0.1, 0.2, 0.05, 0.15, 0.5], device=device,requires_grad=True)

# Number of categories
K = len(true_probs)

# Learnable logits (initial guess)
logits = torch.randn(K, device=device, requires_grad=True)

# Optimizer
optimizer = torch.optim.Adam([logits], lr=0.1)

# Training loop
for epoch in range(10000):
    optimizer.zero_grad()

    # Generate many Gumbel-Softmax samples
    N = 1000  # number of samples
    # gumbel_soft_samples = F.gumbel_softmax(
    #     logits.expand(N, -1), tau=1, hard=True
    # )  # one-hot samples
    gumbel_soft_samples = gumbel_softmax_sample(logits,tau=1,hard=True,N =N)

    # Estimate sampled categorical distribution (average over axis 0)

    # sampled_probs = torch.where(gumbel_soft_samples > 0.9,
    #                             true_probs.expand_as(gumbel_soft_samples),
    #                             torch.zeros_like(gumbel_soft_samples)).mean(0)

    sampled_probs = gumbel_soft_samples.mean(0)
    # sampled_probs = sampled_probs / sampled_probs.sum()
    # Loss: match to target categorical distribution
    loss = F.mse_loss(sampled_probs, true_probs)
    loss.backward()
    optimizer.step()

    # Plot
    if epoch % 1000 == 0 or epoch == 999:
        with torch.no_grad():
            plt.figure(figsize=(6, 4))
            plt.bar(torch.arange(K).cpu(), true_probs.cpu(), alpha=0.6, label="True")
            plt.bar(torch.arange(K).cpu(), sampled_probs.cpu(), alpha=0.6, label="Sampled")
            plt.bar(torch.arange(K).cpu(), (logits.exp() / logits.exp().sum()).cpu(), alpha=0.6, label="logits")
            plt.title(f"Epoch {epoch}, Loss = {loss.item():.5f}")
            plt.xlabel("Category")
            plt.ylabel("Probability")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()