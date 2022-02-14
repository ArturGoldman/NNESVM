import torch
import torch.nn as nn
from nn_esvm.MCMC.MCMC_methods import GenMCMC


class DistributionDataset(nn.Module):

    def __init__(self, dist, mcmc_type, gamma, n_burn, n_clean, n_step=1, rseed=926):
        super().__init__()
        generator = GenMCMC(dist, mcmc_type, gamma)
        chain = generator.gen_samples(n_burn, n_clean, rseed=rseed)
        self.chain = chain[::n_step]
        self.len = self.chain.size(0)

    def __getitem__(self, index: int):
        return self.chain[index]

    def __len__(self):
        return self.len


class FolderDataset(nn.Module):
    def __init__(self, folder_name, n_step=1, dist=None):
        super().__init__()
        checkpoint = torch.load(folder_name)
        chain = checkpoint["chains"]
        self.chain = chain[0, ::n_step]
        self.len = self.chain.size(0)

    def __getitem__(self, index: int):
        return self.chain[index]

    def __len__(self):
        return self.len
