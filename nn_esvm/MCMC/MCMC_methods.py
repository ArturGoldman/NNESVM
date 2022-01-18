import torch
import torch.nn as nn
from tqdm import tqdm
import multiprocessing


def ula_step(prev_point, dist, gamma):
    step = (2*gamma)**0.5*torch.randn(prev_point.size(1))
    return prev_point + gamma*dist.grad_log(prev_point)+step


def mala_step(prev_point, dist, gamma):
    step = (2*gamma)**0.5*torch.randn(prev_point.size(1))
    new_point = prev_point + gamma*dist.grad_log(prev_point)+step
    a = dist.log_prob(new_point)-dist.log_prob(prev_point)
    b = (-((prev_point - new_point - gamma * dist.grad_log(new_point))**2).sum() +
                  ((new_point - prev_point - gamma * dist.grad_log(prev_point))**2).sum())/(4*gamma)
    a = torch.exp(a + b)
    u = torch.rand(1)
    if u <= a:
        return new_point
    return prev_point


class GenMCMC(nn.Module):
    def __init__(self, dist, mcmc_type="ULA", gamma=0.1):
        super().__init__()
        # note, that grad log is passed, which is -nabla U (note the minus)
        self.dist = dist
        self.mcmc_type = mcmc_type
        self.gamma = gamma
        self.mapping = {
            "ULA": ula_step,
            "MALA": mala_step
        }

    def gen_samples(self, n_samples, dim, rseed=None):
        if rseed is not None:
            torch.manual_seed(rseed)
        prev_point = 5*torch.randn(dim).reshape(1, -1)
        samples = [prev_point]
        for i in tqdm(range(n_samples), desc="Generating samples"):
            new_point = self.mapping[self.mcmc_type](prev_point, self.dist, self.gamma)
            samples.append(new_point)
            prev_point = new_point
        return torch.cat(samples, dim=0)

    def generate_parallel_chains(self, n_samples, dim, T, rseed=926):
        nbcores = multiprocessing.cpu_count()
        ctx = multiprocessing.get_context('spawn')
        print("Total cores for multiprocessing", nbcores)
        multi = ctx.Pool(nbcores)
        res = multi.starmap(self.gen_samples,
                            [(n_samples,
                              dim,
                              rseed + i) for i in range(T)])
        return res
