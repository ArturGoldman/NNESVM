import torch.nn as nn
from nn_esvm.MCMC.MCMC_methods import GenMCMC


class DistributionDataset(nn.Module):

    def __init__(self, dist, mcmc_type, gamma, n_burn, n_clean, n_step=1, rseed=926):
        super().__init__()
        self.generator = GenMCMC(dist, mcmc_type, gamma)
        self.dist = dist
        chain = self.generator.gen_samples(n_burn+n_clean, dist.dim, rseed=rseed)
        self.chain = chain[n_burn::n_step]
        self.len = self.chain.size(0)

    def __getitem__(self, index: int):
        return self.chain[index]

    def __len__(self):
        return self.len
