from torch import Tensor
import random
import pandas as pd
import torch
import torch.nn as nn
import nn_esvm.utils.matrix_op as matop

# functions have to return tensor [n_batch, function_dim]


def cubic(x: Tensor):
    return (x**3).sum(dim=1, keepdim=True)


def quadratic(x: Tensor):
    return (x**2).sum(dim=1, keepdim=True)


def sec_coord(x: Tensor):
    return x[:, 1].reshape(-1, 1)


def sec_coord_square(x: Tensor):
    return (x[:, 1]**2).reshape(-1, 1)


class AvgLikelihood(object):
    def __init__(self, path_to_dset, train_ratio, intercept=True, sample_sz=None, stand_type="full", rseed=926):
        """
        Parameters should be put exactly as in LogReg distribution this function is used for
        """
        self.dset = pd.read_csv(path_to_dset)
        self.dset_sz = self.dset.shape[0]

        self.inds = list(range(self.dset_sz))
        random.seed(rseed)
        random.shuffle(self.inds)
        if sample_sz is not None:
            self.inds = self.inds[:sample_sz]
        self.tr_inds = self.inds[:int(train_ratio*len(self.inds))]
        self.test_inds = self.inds[int(train_ratio*len(self.inds)):]
        self.dset = self.dset.to_numpy()

        self.X_train = self.dset[self.tr_inds, :-1]
        self.X_test = self.dset[self.test_inds, :-1]
        _, self.X = matop.standartize(self.X_train, self.X_test, intercept)
        if stand_type == "full":
            _, self.X = matop.standartize(self.X_train, self.X_test, intercept)
        elif stand_type == "poor":
            _, self.X = matop.poor_standartize(self.X_train, self.X_test, intercept)
        else:
            raise RuntimeError('Passed standartization was not recognised')

        self.Y = torch.from_numpy(self.dset[self.test_inds, -1:]).float()

    def __call__(self, x: Tensor):
        logits = x.cpu() @ (self.Y * self.X).transpose(0, 1) + nn.functional.logsigmoid(-x.cpu() @ self.X.transpose(0, 1))
        return torch.exp(logits).mean(dim=1, keepdim=True).to(x.device)


