import torch
import torch.nn as nn
import torch.distributions as D


class BananaShape(nn.Module):
    def __init__(self, dim, p=100, b=0.1):
        super().__init__()
        self.dim = dim
        if dim < 2:
            raise ValueError("Not enough dimensions")
        self.p = p
        self.b = b
        var = torch.ones(dim)
        var[0] = p
        self.base_dist = D.MultivariateNormal(torch.zeros(dim), torch.diag(var))

    def potential(self, x):
        return x[:, 0]**2/(2*self.p) + (x[:, 1] + self.b*x[:, 0]**2 - self.p*self.b)**2 + (x[:, 2:]**2/2).sum(dim=1)

    def sample(self, n):
        samples = self.base_dist.sample((n,))
        samples[:, 1] = samples[:, 1] + self.b*samples[:, 0]**2 - self.p*self.b
        return samples

    def log_prob(self, x: torch.Tensor):
        y = x
        y[:, 1] = y[:, 1] - self.b * y[:, 0] ** 2 + self.p * self.b
        return self.base_dist.log_prob(y)

    def grad_log(self, x: torch.Tensor):
        y = x.clone()
        sec = 2*(x[:, 1]+self.b*x[:, 0]**2-self.p*self.b)
        first = x[:, 0]/self.p + sec**2 * self.b * x[:, 0]
        y[:, 0] = first
        y[:, 1] = sec
        return -y






