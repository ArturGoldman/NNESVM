import math
import torch
import torch.nn as nn
import torch.distributions as D


"""
Each distribution class must have:
dim field: int, dimensionality of data
grad_log function: function which computes gradient of log density
"""


class MyDistribution(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def log_prob(self, x: torch.Tensor):
        """
        Calculates log density up to additive constant
        :param x: [n_batch, dimension]
        :return: log_prob [n_batch, ]
        """
        raise NotImplementedError

    def grad_log(self, x: torch.Tensor):
        """
        Calculates gradient of log density for given points
        Expects log_prob function to be working in differentiable manner
        :param x: [n_batch, dimension]
        :return: gradients [n_batch, dimension]
        """
        grads = []
        x.requires_grad = True
        for val in x:
            val = val.reshape(1, -1)
            out = self.log_prob(val)
            grad = torch.autograd.grad(out, val)
            grads.append(grad[0].squeeze(0))
        x.requires_grad = False
        return torch.stack(grads, dim=0)


class BananaShape(MyDistribution):
    def __init__(self, dim, p=100, b=0.1):
        super().__init__(dim)
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
        samples[:, 1] = samples[:, 1] - self.b*samples[:, 0]**2 + self.p*self.b
        return samples

    def log_prob(self, x: torch.Tensor):
        """
        y = x.clone()
        y[:, 1] = y[:, 1] + self.b * y[:, 0] ** 2 - self.p * self.b
        return self.base_dist.log_prob(y)
        """
        return -(x[:, 0]**2/(2*self.p) + (x[:, 1] + self.b*x[:, 0]**2 - self.p*self.b)**2/2 + (x[:, 2:]**2/2).sum(-1))

    def grad_log(self, x: torch.Tensor):
        y = x.clone()
        sec = x[:, 1]+self.b*x[:, 0]**2-self.p*self.b
        first = x[:, 0]/self.p + 2*sec * self.b * x[:, 0]
        y[:, 0] = first
        y[:, 1] = sec
        return -y


class GMM(MyDistribution):
    def __init__(self, dim, mu, sigma="I", rho=0.5):
        super().__init__(dim)
        if sigma == "I":
            cov_mat = torch.eye(dim)
        else:
            raise ValueError("Cov matrix type not recognised")
        mu_mat = torch.ones((2, dim))*mu
        mu_mat[1, :] = -mu_mat[1, :]
        cov_mat = torch.tile(cov_mat.unsqueeze(0), (2, 1, 1))
        mix = D.Categorical(torch.tensor([rho, 1-rho]))
        comp = D.MultivariateNormal(
            mu_mat, cov_mat)
        self.gmm = D.MixtureSameFamily(mix, comp)

    def log_prob(self, x):
        return self.gmm.log_prob(x)

    def sample(self, n):
        return self.gmm.sample((n,))


class Funnel(MyDistribution):
    def __init__(self, dim, a=1, b=0.5):
        super().__init__(dim)
        if self.dim % 2 != 0:
            raise ValueError("Please, specify even number of dimensions for Funnel distribution")
        self.a = a
        self.b = b

    def log_prob(self, x):
        logprob1 = -x[:, 0]**2/(2*self.a)

        """
        logprob2 = (
                -torch.exp(-2*self.b*x[:, 0]) *
                (x[:, 1:]**2).sum(-1)
                - math.log(self.dim)
                + (2*self.b*x[:, 0])
        )
        """

        logprob2 = (
                -torch.exp(-2*self.b*x[:, 0]) *
                (x[:, 1:]**2/2).sum(-1)
                - ((self.dim-1)*self.b*x[:, 0])
        )

        return logprob1 + logprob2

    def sample(self, n):
        all_c = torch.randn((n, self.dim))
        all_c[:, 0] = all_c[:, 0] * self.a**0.5
        all_c[:, 1:] = all_c[:, 1:]*(torch.exp(self.b*all_c[:, 0]))[:, None]
        return all_c
