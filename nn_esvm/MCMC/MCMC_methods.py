import torch
import torch.nn as nn
from tqdm import tqdm


def ula_step(prev_point, grad_log, gamma):
    step = (2*gamma)**0.5*torch.randn(prev_point.size(0))
    return prev_point + gamma*grad_log(prev_point)+step


class GenMCMC(nn.Module):
    def __init__(self, grad_log, mcmc_type="ULA", gamma=0.1):
        super().__init__()
        # note, that grad log is passed, which is -nabla U (note the minus)
        self.grad_log = grad_log
        self.mcmc_type = mcmc_type
        self.gamma = gamma
        self.mapping = {
            "ULA": ula_step,
        }

    def gen_samples(self, n_samples, dim, rseed=None):
        if rseed is not None:
            torch.manual_seed(rseed)
        prev_point = torch.randn(dim).reshape(1, -1)
        samples = [prev_point]
        for i in tqdm(range(n_samples), desc="Generating samples"):
            new_point = self.mapping[self.mcmc_type](prev_point, self.grad_log, self.gamma)
            samples.append(new_point)
            prev_point = new_point
        return torch.cat(samples, dim=0)

