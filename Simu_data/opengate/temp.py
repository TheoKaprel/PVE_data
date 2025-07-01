import torch
import matplotlib.pyplot as plt


class HistoBin(torch.nn.Module):
    def __init__(self, locations, radius=.2, norm=True):
        super(HistoBin, self).__init__()

        self.locs = locations
        self.r = radius
        self.norm = norm

    def forward(self, x):

        counts = []

        for loc in self.locs:
            dist = torch.abs(x - loc)
            ct = torch.relu(self.r - dist).sum()
            counts.append(ct)

        out = torch.stack(counts)

        if self.norm:
            summ = out.sum() + .000001
            return (out / summ)
        return out

# x = torch.linspace(0,1,1000)
# distrib_true = x * torch.exp(-4*x)
# samples_id = torch.multinomial(distrib,Nsamples,replacement=True)
# samples_id = torch.distributions.Categorical(distrib,Nsamples,replacement=True)
# samples_id = torch.nn.functional.gumbel_softmax(torch.log(logits + 1e-10))

dev = torch.device("cuda")

Nx = 1000
x = torch.linspace(-5,5,Nx).to(dev)

mu,sigma = -2,1/2
dist =  1/sigma / torch.sqrt(torch.Tensor([2*torch.pi]).to(dev)) * torch.exp(-1/2 * ((x-mu)/sigma)**2)

Nsamples = 10000


sigma0,mu0 = torch.Tensor([1.]).to(dev),torch.Tensor([0.0]).to(dev)
sigma0.requires_grad=True
mu0.requires_grad=True
optimizer = torch.optim.Adam([sigma0,mu0],lr=0.01)
loss_fct = torch.nn.MSELoss()
hist2bins = HistoBin(locations=x+(x[1]-x[0])/2)
for e in range(1000):
    optimizer.zero_grad()
    samples_sN = torch.randn(Nsamples,device=dev)*sigma0 + mu0
    hist = hist2bins(samples_sN)
    hist = hist / hist.sum() / (x[1]-x[0])

    loss = loss_fct(hist,dist)
    loss.backward()
    optimizer.step()
    print(f"epoch {e}: {round(loss.item(),4)} mu0 = {mu0.item()} / sigma0 = {sigma0.item()}")

    if e%200==0:
        fig,ax = plt.subplots()
        ax.plot(x.detach().cpu().numpy(), dist.detach().cpu().numpy())
        ax.plot(x.detach().cpu().numpy(), hist.detach().cpu().numpy())
        plt.show()